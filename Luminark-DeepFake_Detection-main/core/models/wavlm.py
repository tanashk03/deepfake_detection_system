"""
WavLM Audio Backbone

Uses pre-trained Microsoft WavLM model from HuggingFace for state-of-the-art
audio deepfake detection.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

class WavLMAnalyzer(BaseDetector):
    """
    SOTA Audio Deepfake Detector using WavLM backbone.
    
    WavLM is pre-trained on large-scale mixed speech data and excels at 
    detecting synthetic voice artifacts and unseen spoofing attacks.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        model_path: str = None, # Legacy compatibility
        sample_rate: int = 16000, # Legacy compatibility
        device: str = "cpu",
        mc_dropout_samples: int = 10,
        **kwargs
    ):
        super().__init__(
            model_name="wavlm_sota",
            modality="audio",
            device=device,
            mc_dropout_samples=mc_dropout_samples,
        )
        self.hf_model_name = model_name
        self.processor = None
        self.classifier = None
        
    def load_model(self) -> None:
        """Load WavLM model and classification head."""
        logger.info(f"Loading WavLM model: {self.hf_model_name}")
        
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.hf_model_name)
            self.backbone = WavLMModel.from_pretrained(self.hf_model_name)
            
            # Freeze backbone usually, but here we might want to fine-tune
            # For inference-only now, eval mode
            self.backbone.eval()
            self.backbone.to(self.device)
            
            # Classification head (simple MLP on top of pooled embeddings)
            hidden_size = self.backbone.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)  # [fake, real]
            ).to(self.device)
            
            # In a real scenario, we would load fine-tuned weights for the classifier here
            # For this demo, we initialize with reasonable weights or dummy if not trained
            self._init_weights()
            
            # Check for fine-tuned weights
            project_root = Path(__file__).parent.parent.parent.resolve()
            ft_path = project_root / "models" / "audio_finetuned.pt"
            
            logger.info(f"Looking for WavLM weights at: {ft_path}")
            
            if ft_path.exists():
                logger.info(f"Loading CUSTOM fine-tuned weights from {ft_path}")
                self.classifier.load_state_dict(torch.load(ft_path, map_location=self.device))
            else:
                logger.warning(f"No fine-tuned weights found at {ft_path}, using initialized weights.")
            
            self._is_loaded = True
            logger.info("WavLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load WavLM: {e}")
            raise
            
    def _init_weights(self):
        """Initialize classifier weights (simulated pre-training)."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """Process audio raw waveform to inputs."""
        # Resample is handled by caller (assuming 16k)
        # Normalize
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=16000 * 5  # 5 seconds max
        )
        return inputs.input_values.to(self.device)

    def forward(self, input_values: torch.Tensor) -> np.ndarray:
        """Run inference."""
        with torch.no_grad():
            outputs = self.backbone(input_values)
            # Average pooling over time dimension
            # last_hidden_state: (batch, seq_len, hidden)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy().squeeze()

    def predict(self, audio: np.ndarray, with_uncertainty: bool = True) -> DetectionResult:
        # Override predict to handle loading
        if not self._is_loaded:
            self.load_model()
            
        return super().predict(audio, with_uncertainty)
