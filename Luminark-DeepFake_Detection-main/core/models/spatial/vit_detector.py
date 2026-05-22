"""
Spatial Detection Model - Vision Transformer + Xception Hybrid
Detects spatial artifacts in deepfake videos using advanced attention mechanisms.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import torchvision.models as models
from typing import Optional, Dict, Any
import logging
import timm

logger = logging.getLogger(__name__)

class HybridSpatialDetector(nn.Module):
    """
    Hybrid spatial detector combining Vision Transformer and Xception architectures
    for comprehensive spatial artifact detection in deepfake videos.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.3):
        super(HybridSpatialDetector, self).__init__()

        # Vision Transformer Configuration
        self.vit_config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )

        # Initialize Vision Transformer
        if pretrained:
            try:
                self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", config=self.vit_config)
                logger.info("✅ Loaded pretrained Vision Transformer")
            except Exception as e:
                logger.warning(f"Failed to load pretrained ViT: {e}. Using random initialization.")
                self.vit = ViTModel(self.vit_config)
        else:
            self.vit = ViTModel(self.vit_config)

        # Xception backbone using timm (PyTorch Image Models)
        if pretrained:
            try:
                self.xception = timm.create_model('xception', pretrained=True, num_classes=0)
                logger.info("✅ Loaded pretrained Xception from timm")
            except Exception as e:
                logger.warning(f"Failed to load pretrained Xception: {e}. Using random initialization.")
                self.xception = timm.create_model('xception', pretrained=False, num_classes=0)
        else:
            self.xception = timm.create_model('xception', pretrained=False, num_classes=0)

        # Feature dimensions
        vit_feature_dim = self.vit_config.hidden_size  # 768
        xception_feature_dim = self.xception.num_features  # Typically 2048 for Xception

        logger.info(f"ViT feature dim: {vit_feature_dim}, Xception feature dim: {xception_feature_dim}")

        # Feature fusion and classification layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(vit_feature_dim + xception_feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 4),

            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.fusion_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid spatial detector."""
        # Vision Transformer branch
        vit_outputs = self.vit(x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Xception branch (timm models return features directly when num_classes=0)
        xception_features = self.xception(x)

        # Concatenate features
        combined_features = torch.cat([vit_features, xception_features], dim=1)

        # Classification
        output = self.fusion_layers(combined_features)
        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights from Vision Transformer for visualization."""
        with torch.no_grad():
            vit_outputs = self.vit(x, output_attentions=True)
            attention_weights = vit_outputs.attentions[-1]  # Last layer
            attention_weights = attention_weights.mean(dim=1)  # Average across heads
            attention_weights = attention_weights[:, 0, 1:]  # CLS to patches

            # Reshape to spatial dimensions
            patch_size = int(attention_weights.size(1) ** 0.5)
            attention_weights = attention_weights.view(-1, patch_size, patch_size)

        return attention_weights


def create_spatial_detector(num_classes: int = 2, pretrained: bool = True) -> HybridSpatialDetector:
    """Factory function to create a spatial detector model."""
    logger.info(f"Creating spatial detector with {num_classes} classes (pretrained={pretrained})")
    return HybridSpatialDetector(num_classes=num_classes, pretrained=pretrained)
