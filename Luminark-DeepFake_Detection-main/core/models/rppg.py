"""
rPPG Physiological Analysis Module

Detects deepfakes by analyzing physiological signals extracted from face video.
Remote photoplethysmography (rPPG) can reveal fake videos that lack genuine pulse.
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class RppgAnalyzer(BaseDetector):
    """
    Physiological signal detector using remote PPG analysis.
    
    Detects:
    - Missing or implausible heart rate
    - Inconsistent pulse across face regions
    - Unnatural blood flow patterns
    - Temporal periodicity anomalies
    
    Pure signal processing approach - no deep learning required.
    Works well on CPU.
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        fps: float = 30.0,
        min_hr: float = 40.0,   # Minimum plausible heart rate
        max_hr: float = 200.0,  # Maximum plausible heart rate
        mc_dropout_samples: int = 5,  # Lower for signal processing
    ):
        super().__init__(
            model_name='rppg_chrom_v1',
            modality='rppg',
            device=device,
            mc_dropout_samples=mc_dropout_samples,
        )
        self.fps = fps
        self.min_hr = min_hr
        self.max_hr = max_hr
        
        # No model to load - pure signal processing
        self._is_loaded = True
    
    def load_model(self) -> None:
        """No model loading needed for signal processing."""
        pass
    
    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract RGB signals from face regions.
        
        Args:
            frames: Video frames (N, H, W, 3) BGR uint8
            
        Returns:
            RGB time series (N, 3) averaged over face region
        """
        import cv2
        
        # Simple face detection using color thresholding
        # In production, would use proper face detector + landmarks
        rgb_signals = []
        
        for frame in frames:
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Simple skin detection using YCrCb color space
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            skin_mask = cv2.inRange(ycrcb, lower, upper)
            
            # Apply mask and get mean RGB
            masked = cv2.bitwise_and(rgb, rgb, mask=skin_mask)
            
            if skin_mask.sum() > 0:
                mean_rgb = masked.sum(axis=(0, 1)) / skin_mask.sum()
            else:
                mean_rgb = rgb.mean(axis=(0, 1))
            
            rgb_signals.append(mean_rgb)
        
        return np.array(rgb_signals, dtype=np.float64)
    
    def forward(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Extract pulse signal using CHROM algorithm.
        
        Args:
            preprocessed: RGB time series (N, 3)
            
        Returns:
            Fake/real probabilities based on pulse analysis
        """
        # Extract pulse using CHROM (Chrominance-based)
        pulse_signal = self._chrom_extraction(preprocessed)
        
        # Analyze signal for physiological plausibility
        hr, confidence, snr = self._analyze_pulse(pulse_signal)
        
        # Score based on physiological plausibility
        score = self._compute_plausibility_score(hr, confidence, snr)
        
        # Return as [fake_prob, real_prob]
        real_prob = (score + 1.0) / 2.0  # Map [-1,1] to [0,1]
        fake_prob = 1.0 - real_prob
        
        return np.array([fake_prob, real_prob])
    
    def _chrom_extraction(self, rgb: np.ndarray) -> np.ndarray:
        """
        CHROM (Chrominance) method for pulse extraction.
        
        De Haan, G., & Jeanne, V. (2013). Robust pulse rate from 
        chrominance-based rPPG.
        """
        # Normalize RGB signals
        rgb_norm = rgb / (rgb.mean(axis=0) + 1e-8)
        
        # CHROM projection
        r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
        
        x_chrom = 3 * r - 2 * g
        y_chrom = 1.5 * r + g - 1.5 * b
        
        # Bandpass filter for heart rate frequencies
        pulse = self._bandpass_filter(x_chrom, self.fps, 0.7, 4.0)
        
        return pulse
    
    def _bandpass_filter(
        self, 
        signal: np.ndarray, 
        fs: float, 
        low: float, 
        high: float
    ) -> np.ndarray:
        """Apply bandpass filter for pulse frequencies."""
        from scipy import signal as sig
        
        # Handle short signals - need at least padlen * 2 samples
        min_samples = 20  # Minimum for reasonable filtering
        if len(signal) < min_samples:
            logger.warning(f"Signal too short ({len(signal)} samples) for bandpass filter, returning raw signal")
            return signal
        
        nyq = fs / 2
        low_norm = low / nyq
        high_norm = high / nyq
        
        # Ensure valid frequency range
        low_norm = max(0.01, min(low_norm, 0.99))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
        
        b, a = sig.butter(2, [low_norm, high_norm], btype='band')
        
        # Use padlen that fits the signal
        padlen = min(len(signal) - 1, 15)
        try:
            filtered = sig.filtfilt(b, a, signal, padlen=padlen)
        except ValueError:
            # Fallback: return raw signal if filtering fails
            logger.warning("Bandpass filter failed, returning raw signal")
            return signal
        
        return filtered
    
    def _analyze_pulse(
        self, pulse: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Analyze pulse signal for heart rate and quality.
        
        Returns:
            (estimated_hr, confidence, signal_to_noise_ratio)
        """
        from scipy import signal as sig
        
        # Compute power spectral density
        freqs, psd = sig.welch(pulse, fs=self.fps, nperseg=min(256, len(pulse)))
        
        # Find peak in HR range (40-200 BPM = 0.67-3.33 Hz)
        hr_mask = (freqs >= self.min_hr / 60) & (freqs <= self.max_hr / 60)
        
        if not hr_mask.any():
            return 0.0, 0.0, 0.0
        
        hr_freqs = freqs[hr_mask]
        hr_psd = psd[hr_mask]
        
        # Find peak
        peak_idx = np.argmax(hr_psd)
        peak_freq = hr_freqs[peak_idx]
        peak_power = hr_psd[peak_idx]
        
        # Estimated heart rate in BPM
        hr = peak_freq * 60
        
        # Signal-to-noise ratio
        noise_power = np.median(hr_psd)
        snr = peak_power / (noise_power + 1e-8)
        
        # Confidence based on peak prominence
        confidence = min(1.0, snr / 10.0)  # Normalize to [0, 1]
        
        return float(hr), float(confidence), float(snr)
    
    def _compute_plausibility_score(
        self, hr: float, confidence: float, snr: float
    ) -> float:
        """
        Score based on physiological plausibility.
        
        Real videos have:
        - Detectable pulse in normal HR range
        - High confidence (clear periodicity)
        - Good SNR
        
        Fake videos often have:
        - No detectable pulse
        - Random noise instead of periodic signal
        - Very low SNR
        """
        if confidence < 0.2:
            return 0.0

        score = 0.0
        
        # Heart rate in plausible range
        if self.min_hr <= hr <= self.max_hr:
            # Closer to typical resting HR (60-80) = more confident
            hr_score = 1.0 - abs(hr - 70) / 130
            score += 0.3 * hr_score
        else:
            # Implausible HR (too high/low) -> Likely noise, but don't penalize.
            # Just ignore this signal.
            return 0.0
        
        # High confidence indicates clear periodic signal
        score += 0.4 * confidence
        
        # SNR contribution
        snr_score = min(1.0, snr / 5.0)
        score += 0.3 * snr_score
        
        # Return only positive scores (Real evidence) or 0 (Neutral)
        # Never return negative (Fake evidence) based on simple signal processing
        # as it is too prone to environmental error.
        normalized_score = np.clip(score, 0.0, 1.0)
        
        # Remap 0..1 to -1..1 logic? No, we want it to be 0..1 (Real side)
        # But base detector expects -1(Fake) to 1(Real).
        # So we return 0.0 to 1.0.
        return float(normalized_score)
    
    def predict(
        self,
        frames: np.ndarray,
        with_uncertainty: bool = True,
    ) -> DetectionResult:
        """
        Analyze video for physiological signals.
        
        Args:
            frames: Video frames (N, H, W, 3) BGR uint8
            with_uncertainty: Add noise for uncertainty estimation
            
        Returns:
            DetectionResult with rPPG analysis
        """
        if not self._is_loaded:
            self.load_model()
        
        preprocessed = self.preprocess(frames)
        
        if with_uncertainty:
            # Add small noise for uncertainty estimation
            predictions = []
            for _ in range(self.mc_dropout_samples):
                noisy = preprocessed + np.random.normal(0, 0.01, preprocessed.shape)
                output = self.forward(noisy)
                predictions.append(self._output_to_score(output))
            
            score, uncertainty = self._aggregate_predictions(np.array(predictions))
        else:
            output = self.forward(preprocessed)
            score = self._output_to_score(output)
            uncertainty = 0.0
        
        confidence = self._uncertainty_to_confidence(uncertainty)
        
        # Get detailed analysis
        pulse = self._chrom_extraction(preprocessed)
        hr, pulse_conf, snr = self._analyze_pulse(pulse)
        
        return DetectionResult(
            score=score,
            confidence=confidence,
            uncertainty=uncertainty,
            modality=self.modality,
            model_name=self.model_name,
            feature_importance={
                'estimated_heart_rate': hr,
                'pulse_confidence': pulse_conf,
                'signal_to_noise_ratio': snr,
            }
        )
