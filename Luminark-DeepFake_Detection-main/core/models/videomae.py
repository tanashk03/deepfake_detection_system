"""
VideoMAE Backbone

Uses pre-trained VideoMAE model from HuggingFace for state-of-the-art
video deepfake detection.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
from pathlib import Path
from transformers import VideoMAEImageProcessor, VideoMAEModel
from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

class VideoMAEAnalyzer(BaseDetector):
    """
    SOTA Video Deepfake Detector using VideoMAE backbone.
    
    VideoMAE uses masked autoencoders to learn robust spatiotemporal features
    from video clips, outperforming traditional 3D-CNNs.
    """
    
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        model_path: str = None, # Legacy compatibility
        frame_sample_rate: int = 5, # Legacy compatibility
        device: str = "cpu",
        mc_dropout_samples: int = 10,
        **kwargs
    ):
        super().__init__(
            model_name="videomae_sota",
            modality="video",
            device=device,
            mc_dropout_samples=mc_dropout_samples,
        )
        self.hf_model_name = model_name
        self.processor = None
        self.classifier = None
        
    def load_model(self) -> None:
        """Load VideoMAE model and classification head."""
        logger.info(f"Loading VideoMAE model: {self.hf_model_name}")
        
        try:
            self.processor = VideoMAEImageProcessor.from_pretrained(self.hf_model_name)
            self.backbone = VideoMAEModel.from_pretrained(self.hf_model_name)
            
            self.backbone.eval()
            self.backbone.to(self.device)
            
            # Classification head
            hidden_size = self.backbone.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)  # [fake, real]
            ).to(self.device)
            
            # Check for fine-tuned weights
            # Resolve absolute path relative to project root (2 levels up from core/models/videomae.py)
            project_root = Path(__file__).parent.parent.parent.resolve()
            ft_path = project_root / "models" / "video_finetuned.pt"
            
            logger.info(f"Looking for weights at: {ft_path}")
            
            self._init_weights()
            if ft_path.exists():
                logger.info(f"Loading CUSTOM fine-tuned weights from {ft_path}")
                self.classifier.load_state_dict(torch.load(ft_path, map_location=self.device))
            else:
                logger.warning(f"No fine-tuned weights found at {ft_path}, using initialized weights.")
            self._is_loaded = True
            logger.info("VideoMAE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VideoMAE: {e}")
            raise

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def preprocess(self, frames: np.ndarray) -> dict:
        """
        Process video frames.
        Args:
            frames: (T, H, W, C) numpy array, BGR (from opencv)
        """
        import cv2
        
        # VideoMAE expects 16 frames usually. 
        # We sample 16 frames uniformly from the input
        num_frames = 16
        input_frames = []
        
        total_frames = frames.shape[0]
        if total_frames < num_frames:
            # Pad
            indices = np.arange(total_frames)
            indices = np.pad(indices, (0, num_frames - total_frames), mode='edge')
        else:
            # Sample
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            
        sampled_frames = frames[indices]
        
        # Convert BGR to RGB and list of arrays
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled_frames]
        
        inputs = self.processor(list(rgb_frames), return_tensors="pt")
        return inputs.to(self.device)

    def forward(self, inputs) -> np.ndarray:
        """Run inference."""
        with torch.no_grad():
            outputs = self.backbone(**inputs)
            # VideoMAE output: last_hidden_state (batch, seq_len, hidden)
            # Pool
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy().squeeze()
            
    def predict(self, frames: np.ndarray, with_uncertainty: bool = True) -> DetectionResult:
        if not self._is_loaded:
            self.load_model()
        return super().predict(frames, with_uncertainty)
