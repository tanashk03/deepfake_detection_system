"""
Temporal Detection Model - Video Vision Transformer
Detects temporal inconsistencies across video frames typical in deepfakes.
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEConfig
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class TemporalConsistencyDetector(nn.Module):
    """
    Temporal detector using Video Vision Transformer to identify
    inconsistencies across video frames typical in deepfakes.
    """

    def __init__(self, sequence_length: int = 16, num_classes: int = 2, pretrained: bool = True):
        super(TemporalConsistencyDetector, self).__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # Video Vision Transformer
        if pretrained:
            try:
                self.video_transformer = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
                hidden_size = self.video_transformer.config.hidden_size
            except Exception as e:
                logger.warning(f"Failed to load pretrained VideoMAE: {e}. Using random initialization.")
                config = VideoMAEConfig(
                    image_size=224, patch_size=16, num_channels=3,
                    num_frames=sequence_length, hidden_size=768,
                    num_hidden_layers=12, num_attention_heads=12
                )
                self.video_transformer = VideoMAEModel(config)
                hidden_size = config.hidden_size
        else:
            config = VideoMAEConfig(
                image_size=224, patch_size=16, num_channels=3,
                num_frames=sequence_length, hidden_size=768,
                num_hidden_layers=12, num_attention_heads=12
            )
            self.video_transformer = VideoMAEModel(config)
            hidden_size = config.hidden_size

        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal detector."""
        # Process through Video Transformer
        outputs = self.video_transformer(video_frames)
        frame_features = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Apply temporal attention
        attended_features, _ = self.temporal_attention(
            frame_features, frame_features, frame_features
        )

        # Global temporal pooling
        temporal_features = attended_features.mean(dim=1)

        # Classification
        return self.classifier(temporal_features)

    def analyze_temporal_consistency(self, video_frames: torch.Tensor) -> Dict[str, np.ndarray]:
        """Analyze temporal consistency metrics for explainability."""
        with torch.no_grad():
            outputs = self.video_transformer(video_frames)
            frame_features = outputs.last_hidden_state

            # Compute frame-to-frame similarity
            similarities = []
            for i in range(frame_features.size(1) - 1):
                sim = torch.cosine_similarity(
                    frame_features[:, i, :], frame_features[:, i+1, :], dim=1
                )
                similarities.append(sim.cpu().numpy())

            return {
                'frame_similarities': similarities,
                'consistency_score': np.mean(similarities) if similarities else 0.0
            }

def create_temporal_detector(sequence_length: int = 16, num_classes: int = 2) -> TemporalConsistencyDetector:
    """Factory function to create a temporal detector model."""
    return TemporalConsistencyDetector(sequence_length=sequence_length, num_classes=num_classes)
