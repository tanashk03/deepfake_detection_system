"""
Lip-Sync Consistency Module

Detects audio-visual synchronization issues between lip movements and speech.
Key indicator for face-swap and lip-sync deepfakes.
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class LipsyncAnalyzer(BaseDetector):
    """
    Audio-visual synchronization detector.
    
    Detects:
    - Lip movement / audio timing mismatches
    - Unnatural mouth shapes for phonemes
    - Temporal desynchronization patterns
    
    Uses cross-correlation and learned embeddings.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        sync_window_ms: float = 100.0,  # Sync tolerance window
        mc_dropout_samples: int = 10,
    ):
        super().__init__(
            model_name='syncnet_lite_v1',
            modality='lipsync',
            device=device,
            mc_dropout_samples=mc_dropout_samples,
        )
        self.model_path = model_path
        self.sync_window_ms = sync_window_ms
        
        self._torch = None
    
    def load_model(self) -> None:
        """Load SyncNet-style embedding models."""
        import torch
        self._torch = torch
        
        logger.info(f"Loading lip-sync model on {self.device}")
        
        self._audio_encoder, self._visual_encoder = self._build_model()
        
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self._audio_encoder.load_state_dict(state_dict['audio'])
            self._visual_encoder.load_state_dict(state_dict['visual'])
        
        self._audio_encoder.eval().to(self.device)
        self._visual_encoder.eval().to(self.device)
        
        self._enable_mc_dropout()
        
        logger.info("Lip-sync model loaded successfully")
    
    def _build_model(self) -> Tuple:
        """
        Build lightweight SyncNet-style encoders.
        
        Maps audio and visual streams to shared embedding space.
        """
        import torch.nn as nn
        
        class AudioEncoder(nn.Module):
            """Encodes MFCC audio features to embedding."""
            def __init__(self, embed_dim=128):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(13, 64, 3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.2),
                    
                    nn.Conv1d(64, 128, 3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.2),
                    
                    nn.Conv1d(128, embed_dim, 3, padding=1),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                )
                
            def forward(self, x):
                return self.encoder(x)
        
        class VisualEncoder(nn.Module):
            """Encodes mouth region frames to embedding."""
            def __init__(self, embed_dim=128):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv3d(3, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
                    nn.BatchNorm3d(32),
                    nn.ReLU(),
                    nn.MaxPool3d((1, 2, 2)),
                    nn.Dropout3d(0.2),
                    
                    nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(),
                    nn.MaxPool3d((1, 2, 2)),
                    nn.Dropout3d(0.2),
                    
                    nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
                    nn.BatchNorm3d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d(1),
                    nn.Flatten(),
                    
                    nn.Linear(128, embed_dim),
                )
                
            def forward(self, x):
                return self.encoder(x)
        
        return AudioEncoder(), VisualEncoder()
    
    def _enable_mc_dropout(self) -> None:
        """Enable dropout for MC estimation."""
        for model in [self._audio_encoder, self._visual_encoder]:
            for module in model.modules():
                if 'Dropout' in module.__class__.__name__:
                    module.train()
    
    def preprocess(
        self, inputs: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess audio and visual streams.
        
        Args:
            inputs: Dict with 'audio' (1D signal) and 'frames' (N, H, W, 3)
            
        Returns:
            (audio_features, visual_features)
        """
        import cv2
        import librosa
        
        audio = inputs['audio']
        frames = inputs['frames']
        
        # Audio: extract MFCCs (13 coefficients for lip-sync)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        
        mfcc = librosa.feature.mfcc(
            y=audio, sr=16000, n_mfcc=13,
            hop_length=160, n_fft=512
        )
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        audio_features = mfcc[np.newaxis, ...]  # (1, 13, time)
        
        # Visual: extract mouth regions
        mouth_crops = []
        for frame in frames:
            # Simplified mouth region extraction (center-bottom of frame)
            # In production, would use face landmarks
            h, w = frame.shape[:2]
            mouth_region = frame[int(h*0.5):int(h*0.9), int(w*0.3):int(w*0.7)]
            
            # Resize to fixed size
            resized = cv2.resize(mouth_region, (48, 48))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 127.5 - 1.0
            mouth_crops.append(normalized)
        
        # Stack: (T, H, W, C) -> (C, T, H, W)
        visual_stack = np.stack(mouth_crops, axis=0)
        visual_features = np.transpose(visual_stack, (3, 0, 1, 2))
        visual_features = visual_features[np.newaxis, ...]  # (1, C, T, H, W)
        
        return audio_features, visual_features
    
    def forward(self, preprocessed: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute audio-visual sync score.
        
        Args:
            preprocessed: (audio_features, visual_features)
            
        Returns:
            Sync score as [fake_prob, real_prob]
        """
        audio_feat, visual_feat = preprocessed
        
        with self._torch.no_grad():
            audio_tensor = self._torch.from_numpy(audio_feat).float().to(self.device)
            visual_tensor = self._torch.from_numpy(visual_feat).float().to(self.device)
            
            # Get embeddings
            audio_embed = self._audio_encoder(audio_tensor)
            visual_embed = self._visual_encoder(visual_tensor)
            
            # Normalize embeddings
            audio_embed = audio_embed / (audio_embed.norm(dim=1, keepdim=True) + 1e-8)
            visual_embed = visual_embed / (visual_embed.norm(dim=1, keepdim=True) + 1e-8)
            
            # Cosine similarity
            similarity = (audio_embed * visual_embed).sum(dim=1)
            
            # Convert to probabilities
            # High similarity = in sync = probably real
            real_prob = (similarity.item() + 1.0) / 2.0  # Map [-1,1] to [0,1]
            fake_prob = 1.0 - real_prob
            
            return np.array([fake_prob, real_prob])
    
    def compute_sync_offset(
        self, 
        audio: np.ndarray, 
        frames: np.ndarray,
        fps: float = 30.0,
        sr: int = 16000,
    ) -> Tuple[float, float]:
        """
        Estimate audio-visual synchronization offset.
        
        Returns:
            (offset_ms, confidence)
        """
        import librosa
        
        # Audio energy envelope
        audio_envelope = np.abs(audio)
        audio_envelope = np.convolve(
            audio_envelope, 
            np.ones(int(sr * 0.02)) / int(sr * 0.02), 
            mode='same'
        )
        
        # Resample to video frame rate
        audio_at_fps = librosa.resample(
            audio_envelope, 
            orig_sr=sr, 
            target_sr=int(fps)
        )[:len(frames)]
        
        # Visual motion (simplified - mouth movement proxy)
        visual_motion = []
        prev_frame = None
        for frame in frames:
            gray = np.mean(frame, axis=2)
            if prev_frame is not None:
                diff = np.abs(gray - prev_frame).mean()
                visual_motion.append(diff)
            prev_frame = gray
        visual_motion = np.array([0] + visual_motion)
        
        # Cross-correlation
        min_len = min(len(audio_at_fps), len(visual_motion))
        audio_at_fps = audio_at_fps[:min_len]
        visual_motion = visual_motion[:min_len]
        
        # Normalize
        audio_at_fps = (audio_at_fps - audio_at_fps.mean()) / (audio_at_fps.std() + 1e-8)
        visual_motion = (visual_motion - visual_motion.mean()) / (visual_motion.std() + 1e-8)
        
        # Compute cross-correlation
        correlation = np.correlate(visual_motion, audio_at_fps, mode='full')
        lags = np.arange(-len(audio_at_fps) + 1, len(audio_at_fps))
        
        # Find peak
        peak_idx = np.argmax(correlation)
        offset_frames = lags[peak_idx]
        offset_ms = offset_frames / fps * 1000
        
        # Confidence based on peak prominence
        peak_value = correlation[peak_idx]
        confidence = min(1.0, peak_value / (len(audio_at_fps) * 0.5))
        
        return float(offset_ms), float(max(0, confidence))
    
    def predict(
        self,
        inputs: dict,
        with_uncertainty: bool = True,
        check_offset: bool = True,
    ) -> DetectionResult:
        """
        Full lip-sync analysis.
        
        Args:
            inputs: Dict with 'audio' and 'frames'
            with_uncertainty: Run MC dropout
            check_offset: Compute sync offset
            
        Returns:
            DetectionResult with sync analysis
        """
        if not self._is_loaded:
            self.load_model()
            self._is_loaded = True
        
        preprocessed = self.preprocess(inputs)
        
        if with_uncertainty:
            predictions = []
            for _ in range(self.mc_dropout_samples):
                output = self.forward(preprocessed)
                predictions.append(self._output_to_score(output))
            score, uncertainty = self._aggregate_predictions(np.array(predictions))
        else:
            output = self.forward(preprocessed)
            score = self._output_to_score(output)
            uncertainty = 0.0
        
        confidence = self._uncertainty_to_confidence(uncertainty)
        
        result = DetectionResult(
            score=score,
            confidence=confidence,
            uncertainty=uncertainty,
            modality=self.modality,
            model_name=self.model_name,
        )
        
        # Add offset analysis
        if check_offset:
            offset_ms, offset_conf = self.compute_sync_offset(
                inputs['audio'], inputs['frames']
            )
            
            # Penalize large offsets
            if abs(offset_ms) > self.sync_window_ms:
                penalty = min(0.5, abs(offset_ms) / 500.0)
                result.score = max(-1.0, result.score - penalty)
            
            result.feature_importance = {
                'sync_offset_ms': offset_ms,
                'offset_confidence': offset_conf,
                'in_sync_window': abs(offset_ms) <= self.sync_window_ms,
            }
        
        return result
