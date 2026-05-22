"""
Frequency Domain Detection Model
Analyzes FFT and DCT characteristics disturbed by deepfake generation.
"""

import torch
import torch.nn as nn
import torch.fft
import cv2
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class FrequencyDomainDetector(nn.Module):
    """
    Frequency domain detector analyzing FFT and DCT characteristics
    that are disturbed by deepfake generation processes.
    """

    def __init__(self, num_classes: int = 2, input_size: int = 224):
        super(FrequencyDomainDetector, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # FFT analysis branch
        self.fft_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8)
        )

        # DCT analysis branch
        self.dct_branch = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8)
        )

        # Feature fusion and classification
        total_features = 256 * 64 + 128 * 64  # FFT + DCT features

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def extract_frequency_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract frequency domain features from input."""
        batch_size = x.size(0)

        # FFT analysis
        fft_features = torch.fft.fft2(x)
        fft_magnitude = torch.log(torch.abs(fft_features) + 1e-8)

        # DCT approximation using FFT (PyTorch doesn't have native DCT)
        # Convert to grayscale
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        else:
            gray = x[:, 0]

        # Use real FFT as DCT-like features
        gray_unsqueezed = gray.unsqueeze(1)  # (B, 1, H, W)
        dct_features = torch.fft.rfft2(gray_unsqueezed, dim=(-2, -1))
        dct_features = torch.log(torch.abs(dct_features) + 1e-8)

        return {
            'fft_magnitude': fft_magnitude,
            'dct': dct_features
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frequency domain detector."""
        # Extract frequency features
        freq_features = self.extract_frequency_features(x)

        # Process through branches
        fft_processed = self.fft_branch(freq_features['fft_magnitude'])
        dct_processed = self.dct_branch(freq_features['dct'])

        # Flatten features
        fft_flat = fft_processed.view(fft_processed.size(0), -1)
        dct_flat = dct_processed.view(dct_processed.size(0), -1)

        # Concatenate features
        combined_features = torch.cat([fft_flat, dct_flat], dim=1)

        # Classification
        return self.classifier(combined_features)

def create_frequency_detector(num_classes: int = 2) -> FrequencyDomainDetector:
    """Factory function to create a frequency detector model."""
    return FrequencyDomainDetector(num_classes=num_classes)
