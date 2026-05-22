"""
CNN + BiLSTM Physiological Detector - EXACT match to Kaggle trained weights.

From weight analysis:
- cnn.0: Conv2d(3, 32, 3, 3)
- cnn.1: BatchNorm2d(32)
- cnn.4: Conv2d(32, 64, 3, 3)
- cnn.5: BatchNorm2d(64)
- lstm: BiLSTM(input_size=3136, hidden_size=128)
  - 3136 = 7 * 7 * 64 (after two 2x2 maxpools on 224->28->7, then flatten)
- fc.0: Linear(256, 64)  (256 = 128*2 bidirectional)
- fc.3: Linear(64, 2)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CNNBiLSTMPhysiologicalDetector(nn.Module):
    """
    CNN + BiLSTM physiological detector with EXACT trained weight dimensions.
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # CNN layers with EXACT indices
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # cnn.0
            nn.BatchNorm2d(32),                              # cnn.1
            nn.ReLU(),                                       # cnn.2
            nn.MaxPool2d(4),                                 # cnn.3 - 224->56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # cnn.4
            nn.BatchNorm2d(64),                              # cnn.5
            nn.ReLU(),                                       # cnn.6
            nn.MaxPool2d(8)                                  # cnn.7 - 56->7
        )
        # After CNN: 64 channels x 7 x 7 = 3136 features per frame
        
        # BiLSTM with EXACT trained dimensions
        self.lstm = nn.LSTM(
            input_size=3136,   # 64 * 7 * 7 = 3136 (EXACT from trained weights)
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )
        
        # FC layers with EXACT indices
        self.fc = nn.Sequential(
            nn.Linear(256, 64),      # fc.0 (256 = 128*2 bidirectional)
            nn.ReLU(),               # fc.1
            nn.Dropout(0.3),         # fc.2
            nn.Linear(64, num_classes)  # fc.3
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input shapes
        if len(x.shape) == 4:
            # Single frame: (B, C, H, W) -> (B, 1, C, H, W)
            x = x.unsqueeze(1)
        
        B, T, C, H, W = x.shape
        
        # Process frames through CNN
        x = x.view(B * T, C, H, W)
        cnn_out = self.cnn(x)  # (B*T, 64, 7, 7)
        cnn_out = cnn_out.view(B, T, -1)  # (B, T, 3136)
        
        # BiLSTM
        lstm_out, _ = self.lstm(cnn_out)  # (B, T, 256)
        lstm_out = lstm_out[:, -1, :]  # Last timestep: (B, 256)
        
        # Classify
        logits = self.fc(lstm_out)
        return logits


def create_physiological_cnn_detector() -> CNNBiLSTMPhysiologicalDetector:
    """Factory function to create physiological detector."""
    return CNNBiLSTMPhysiologicalDetector(num_classes=2)
