"""
Conv3D Temporal Detector - EXACT match to Kaggle trained weights.

From weight analysis:
- conv.0 = Conv3d(3, 64, kernel=(3,7,7)) with 'conv.' prefix (NOT 'conv3d.')
- conv.1 = BatchNorm3d(64)
- conv.4 = Conv3d(64, 128, kernel=(3,3,3))
- conv.5 = BatchNorm3d(128)
- conv.8 = Conv3d(128, 256, kernel=(3,3,3))
- conv.9 = BatchNorm3d(256)
- fc.1 = Linear(256, 128)
- fc.4 = Linear(128, 2)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class Conv3DTemporalDetector(nn.Module):
    """
    Conv3D temporal detector with EXACT trained weight dimensions.
    
    CRITICAL: Uses 'conv.' prefix (not 'conv3d.') despite being Conv3D!
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Conv3D layers with 'conv.' prefix (EXACT match to trained weights)
        self.conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),  # conv.0
            nn.BatchNorm3d(64),                                                             # conv.1
            nn.ReLU(),                                                                      # conv.2
            nn.MaxPool3d(kernel_size=(1, 2, 2)),                                           # conv.3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),                                  # conv.4
            nn.BatchNorm3d(128),                                                            # conv.5
            nn.ReLU(),                                                                      # conv.6
            nn.MaxPool3d(kernel_size=2),                                                   # conv.7
            nn.Conv3d(128, 256, kernel_size=3, padding=1),                                 # conv.8
            nn.BatchNorm3d(256),                                                            # conv.9
            nn.ReLU(),                                                                      # conv.10
            nn.AdaptiveAvgPool3d(1)                                                        # conv.11
        )
        
        # FC layers with EXACT indices
        self.fc = nn.Sequential(
            nn.Flatten(),            # fc.0 (no weights)
            nn.Linear(256, 128),     # fc.1
            nn.ReLU(),               # fc.2
            nn.Dropout(0.5),         # fc.3
            nn.Linear(128, num_classes)  # fc.4
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input formats
        if len(x.shape) == 4:
            # Single frame: (B, C, H, W) -> (B, C, 1, H, W)
            x = x.unsqueeze(2)
        elif len(x.shape) == 5 and x.shape[1] != 3:
            # (B, T, C, H, W) -> (B, C, T, H, W) for Conv3d
            x = x.permute(0, 2, 1, 3, 4)
        
        features = self.conv(x)
        logits = self.fc(features)
        return logits


def create_temporal_efficientnet_detector() -> Conv3DTemporalDetector:
    """Factory function to create temporal detector."""
    return Conv3DTemporalDetector(num_classes=2)
