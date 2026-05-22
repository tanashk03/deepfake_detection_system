"""
EfficientNet-B0 Spatial Detector - EXACT match to Kaggle trained weights.

From weight analysis:
- fc.0 = Linear(1280, 256)  NOT 512!
- fc.3 = Linear(256, 2)     Final classifier

So architecture is: backbone → Linear(1280,256) → ReLU → Dropout → Linear(256,2)
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)


class EfficientNetB0Detector(nn.Module):
    """
    EfficientNet-B0 spatial detector with EXACT weight dimensions.
    
    Trained weights:
    - fc.0.weight: (256, 1280)
    - fc.0.bias: (256,)
    - fc.3.weight: (2, 256)
    - fc.3.bias: (2,)
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        
        # EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        # Classifier head with EXACT trained dimensions
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),   # fc.0 - matches (256, 1280)
            nn.ReLU(),              # fc.1
            nn.Dropout(0.4),        # fc.2
            nn.Linear(256, num_classes)  # fc.3 - matches (2, 256)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.fc(features)
        return logits


def create_efficientnet_detector() -> EfficientNetB0Detector:
    """Factory function to create spatial detector."""
    return EfficientNetB0Detector(num_classes=2, pretrained=False)
