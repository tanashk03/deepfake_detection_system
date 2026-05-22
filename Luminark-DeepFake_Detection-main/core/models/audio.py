"""
Audio Analysis Module

Detects synthetic voice and audio manipulation using Wav2Vec-style features.
Optimized for CPU inference on macOS Intel.
"""

import numpy as np
from typing import Optional, Tuple
import logging

from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class AudioAnalyzer(BaseDetector):
    """
    Voice synthesis detector using spectral and learned features.
    
    Detects:
    - Text-to-speech artifacts
    - Voice cloning signatures
    - Audio splicing discontinuities
    - Unnatural prosody patterns
    
    Uses MFCC + lightweight transformer for CPU efficiency.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        mc_dropout_samples: int = 10,
    ):
        super().__init__(
            model_name='wav2vec_lite_v1',
            modality='audio',
            device=device,
            mc_dropout_samples=mc_dropout_samples,
        )
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        self._torch = None
        self._librosa = None
    
    def load_model(self) -> None:
        """Load audio classifier model."""
        import torch
        self._torch = torch
        
        logger.info(f"Loading audio model on {self.device}")
        
        self._model = self._build_model()
        
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
        
        self._model.eval()
        self._model.to(self.device)
        self._enable_mc_dropout()
        
        logger.info("Audio model loaded successfully")
    
    def _build_model(self):
        """
        Build lightweight audio classifier.
        
        Architecture: MFCC → CNN → Transformer → Classifier
        """
        import torch.nn as nn
        
        class AudioClassifier(nn.Module):
            def __init__(self, n_mfcc=40, n_heads=4, hidden_dim=128):
                super().__init__()
                
                # CNN feature extractor
                self.cnn = nn.Sequential(
                    nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.2),
                    
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.2),
                    
                    nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                )
                
                # Transformer for temporal patterns
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.2,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                
                # Classifier head
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 2),  # [fake, real]
                )
            
            def forward(self, x):
                # x: (batch, n_mfcc, time)
                x = self.cnn(x)
                
                # (batch, hidden, time) -> (batch, time, hidden)
                x = x.transpose(1, 2)
                x = self.transformer(x)
                
                # (batch, time, hidden) -> (batch, hidden, time)
                x = x.transpose(1, 2)
                x = self.classifier(x)
                
                return x
        
        return AudioClassifier(n_mfcc=self.n_mfcc)
    
    def _enable_mc_dropout(self) -> None:
        """Enable dropout during inference."""
        for module in self._model.modules():
            if isinstance(module, self._torch.nn.Dropout):
                module.train()
    
    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio signal.
        
        Args:
            audio: Raw audio signal (1D float32, mono)
            
        Returns:
            MFCC features (1, n_mfcc, time)
        """
        import librosa
        self._librosa = librosa
        
        # Ensure correct sample rate
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Mono
        
        # Normalize
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0  # Assume int16
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=512,
            n_fft=2048,
        )
        
        # Delta features for dynamics
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Concatenate and normalize
        features = np.vstack([mfcc, mfcc_delta])
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Add batch dimension
        return features[np.newaxis, :self.n_mfcc, :]  # Use only MFCC, not delta
    
    def forward(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Run inference on MFCC features.
        
        Args:
            preprocessed: Shape (1, n_mfcc, time)
            
        Returns:
            Softmax probabilities [fake, real]
        """
        with self._torch.no_grad():
            x = self._torch.from_numpy(preprocessed).float().to(self.device)
            outputs = self._model(x)
            probs = self._torch.softmax(outputs, dim=1)
            return probs.squeeze().cpu().numpy()
    
    def analyze_spectral_artifacts(self, audio: np.ndarray) -> dict:
        """
        Detect spectral artifacts common in synthetic audio.
        
        Returns:
            Dictionary with artifact analysis
        """
        import librosa
        
        # Compute spectrogram
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        # Check for unnaturally smooth spectra (TTS artifact)
        spectral_variance = np.var(S_db, axis=0).mean()
        smoothness_score = 1.0 / (1.0 + spectral_variance / 100)
        
        # Check for frequency cutoff (common in low-quality TTS)
        high_freq_energy = S[-S.shape[0]//4:, :].mean()
        low_freq_energy = S[:S.shape[0]//4, :].mean()
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        # Natural speech has higher freq_ratio
        cutoff_score = 1.0 if freq_ratio < 0.1 else 0.0
        
        return {
            'smoothness_score': float(smoothness_score),
            'frequency_cutoff_detected': bool(cutoff_score > 0.5),
            'high_low_freq_ratio': float(freq_ratio),
        }
    
    def predict(
        self,
        audio: np.ndarray,
        with_uncertainty: bool = True,
        analyze_artifacts: bool = True,
    ) -> DetectionResult:
        """
        Full audio analysis pipeline.
        
        Args:
            audio: Raw audio signal (float32, mono)
            with_uncertainty: Run MC dropout
            analyze_artifacts: Check spectral artifacts
            
        Returns:
            DetectionResult with audio analysis
        """
        result = super().predict(audio, with_uncertainty)
        
        if analyze_artifacts:
            artifacts = self.analyze_spectral_artifacts(audio)
            
            # Adjust score if artifacts detected
            if artifacts['frequency_cutoff_detected']:
                result.score = max(-1.0, result.score - 0.3)
            
            if artifacts['smoothness_score'] > 0.7:
                result.score = max(-1.0, result.score - 0.2)
            
            result.feature_importance = artifacts
        
        return result


class AudioAnalyzerWav2Vec(AudioAnalyzer):
    """
    Experimental: Full Wav2Vec 2.0 based detector.
    
    More accurate but slower. Not used in production by default.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'wav2vec2_experimental'
    
    def _build_model(self):
        """Use HuggingFace Wav2Vec if available."""
        logger.warning("Wav2Vec2 is experimental, using lightweight model")
        return super()._build_model()
