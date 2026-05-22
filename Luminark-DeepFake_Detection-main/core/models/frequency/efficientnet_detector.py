"""
EfficientNet-B0 Frequency Detector - EXACT match to Kaggle trained weights.

Same fc structure as spatial: fc.0=(256,1280), fc.3=(2,256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging

logger = logging.getLogger(__name__)


class EfficientNetFrequencyDetector(nn.Module):
    """
    EfficientNet-B0 frequency detector with FFT preprocessing.
    EXACT weight dimensions matching trained model.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        # EXACT trained dimensions
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),    # fc.0
            nn.ReLU(),               # fc.1
            nn.Dropout(0.4),         # fc.2
            nn.Linear(256, num_classes)  # fc.3
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT preprocessing
        fft = torch.fft.rfft2(x)
        mag = torch.abs(fft).to(torch.float32)
        mag = torch.log1p(mag)
        mag = F.interpolate(mag, size=(224, 224), mode='bilinear', align_corners=False)
        
        features = self.backbone(mag)
        logits = self.fc(features)
        return logits


def create_frequency_efficientnet_detector() -> EfficientNetFrequencyDetector:
    """Factory function to create frequency detector."""
    return EfficientNetFrequencyDetector(num_classes=2, pretrained=False)
