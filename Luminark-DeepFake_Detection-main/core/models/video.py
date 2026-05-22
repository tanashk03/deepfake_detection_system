"""
Video Analysis Module

Detects visual manipulation artifacts in video frames using XceptionNet.
Optimized for CPU inference on macOS Intel.
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

from .base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class VideoAnalyzer(BaseDetector):
    """
    Face manipulation detector using XceptionNet-style architecture.
    
    Detects:
    - Blending artifacts around face boundaries
    - Unnatural texture patterns
    - Temporal inconsistencies between frames
    - Geometric distortions
    
    Optimized for CPU inference with quantization support.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        input_size: Tuple[int, int] = (299, 299),
        frame_sample_rate: int = 5,
        mc_dropout_samples: int = 10,
    ):
        super().__init__(
            model_name='xception_v1',
            modality='video',
            device=device,
            mc_dropout_samples=mc_dropout_samples,
        )
        self.model_path = model_path
        self.input_size = input_size
        self.frame_sample_rate = frame_sample_rate
        
        # Lazy imports for faster startup
        self._torch = None
        self._transforms = None
    
    def load_model(self) -> None:
        """Load XceptionNet model weights."""
        import torch
        import torch.nn as nn
        self._torch = torch
        
        logger.info(f"Loading video model on {self.device}")
        
        # Build lightweight XceptionNet-inspired architecture
        self._model = self._build_model()
        
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
        
        self._model.eval()
        self._model.to(self.device)
        
        # Enable dropout for MC inference
        self._enable_mc_dropout()
        
        logger.info("Video model loaded successfully")
    
    def _build_model(self):
        """
        Build simplified XceptionNet for CPU inference.
        
        Uses depthwise separable convolutions for efficiency.
        """
        import torch.nn as nn
        
        class SeparableConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3):
                super().__init__()
                self.depthwise = nn.Conv2d(
                    in_channels, in_channels, kernel_size,
                    padding=kernel_size//2, groups=in_channels
                )
                self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
                
            def forward(self, x):
                return self.pointwise(self.depthwise(x))
        
        class XceptionBlock(nn.Module):
            def __init__(self, in_channels, out_channels, dropout=0.2):
                super().__init__()
                self.conv1 = SeparableConv2d(in_channels, out_channels)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = SeparableConv2d(out_channels, out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.pool = nn.MaxPool2d(3, stride=2, padding=1)
                self.dropout = nn.Dropout2d(dropout)
                
                # Skip connection
                self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)
                
            def forward(self, x):
                residual = self.skip(x)
                
                x = self.conv1(x)
                x = self.bn1(x)
                x = nn.functional.relu(x)
                x = self.dropout(x)
                
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.pool(x)
                
                return nn.functional.relu(x + residual)
        
        class LightweightXception(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Entry flow
                self.entry = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
                
                # Middle flow (reduced for CPU)
                self.middle = nn.Sequential(
                    XceptionBlock(64, 128),
                    XceptionBlock(128, 256),
                    XceptionBlock(256, 512),
                )
                
                # Exit flow
                self.exit = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2),  # [fake, real]
                )
                
            def forward(self, x):
                x = self.entry(x)
                x = self.middle(x)
                x = self.exit(x)
                return x
        
        return LightweightXception()
    
    def _enable_mc_dropout(self) -> None:
        """Enable dropout during inference for MC estimation."""
        for module in self._model.modules():
            if isinstance(module, (self._torch.nn.Dropout, self._torch.nn.Dropout2d)):
                module.train()
    
    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess video frames for inference.
        
        Args:
            frames: Shape (N, H, W, 3) uint8 BGR frames
            
        Returns:
            Tensor-ready array (N, 3, 299, 299) float32
        """
        import cv2
        
        processed = []
        for frame in frames:
            # Resize
            resized = cv2.resize(frame, self.input_size)
            
            # BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [-1, 1]
            normalized = (rgb.astype(np.float32) / 127.5) - 1.0
            
            # HWC to CHW
            chw = np.transpose(normalized, (2, 0, 1))
            processed.append(chw)
        
        return np.stack(processed, axis=0)
    
    def forward(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed frames.
        
        Args:
            preprocessed: Shape (N, 3, 299, 299)
            
        Returns:
            Softmax probabilities [fake, real]
        """
        with self._torch.no_grad():
            x = self._torch.from_numpy(preprocessed).float().to(self.device)
            
            # Process each frame
            outputs = self._model(x)
            
            # Softmax and average across frames
            probs = self._torch.softmax(outputs, dim=1)
            avg_probs = probs.mean(dim=0)
            
            return avg_probs.cpu().numpy()
    
    def analyze_temporal_consistency(
        self, frames: np.ndarray
    ) -> Tuple[float, List[int]]:
        """
        Check for temporal inconsistencies between frames.
        
        Returns:
            (consistency_score, suspicious_frame_indices)
        """
        import cv2
        
        suspicious_frames = []
        prev_frame = None
        flow_magnitudes = []
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_magnitudes.append(np.mean(magnitude))
            
            prev_frame = gray
        
        if len(flow_magnitudes) < 2:
            return 1.0, []
        
        # Detect sudden jumps in motion (potential splices)
        flow_magnitudes = np.array(flow_magnitudes)
        mean_flow = np.mean(flow_magnitudes)
        std_flow = np.std(flow_magnitudes)
        
        threshold = mean_flow + 2 * std_flow
        for i, mag in enumerate(flow_magnitudes):
            if mag > threshold:
                suspicious_frames.append(i + 1)  # +1 because flow is between frames
        
        # Consistency score: lower if more suspicious frames
        consistency = 1.0 - (len(suspicious_frames) / len(flow_magnitudes))
        
        return float(consistency), suspicious_frames
    
    def predict(
        self,
        frames: np.ndarray,
        with_uncertainty: bool = True,
        check_temporal: bool = True,
    ) -> DetectionResult:
        """
        Full video analysis pipeline.
        
        Args:
            frames: Video frames (N, H, W, 3) BGR uint8
            with_uncertainty: Run MC dropout
            check_temporal: Check temporal consistency
            
        Returns:
            DetectionResult with video analysis
        """
        # Run base prediction
        result = super().predict(frames, with_uncertainty)
        
        # Add temporal consistency check
        if check_temporal and len(frames) > 1:
            temporal_score, suspicious = self.analyze_temporal_consistency(frames)
            
            # Adjust score if temporal issues found
            if temporal_score < 0.8:
                penalty = (0.8 - temporal_score) * 0.5
                result.score = max(-1.0, result.score - penalty)
                
            result.feature_importance = {
                'temporal_consistency': temporal_score,
                'suspicious_frames': suspicious,
            }
        
        return result


# Experimental variant (isolated)
class VideoAnalyzerEfficientNet(VideoAnalyzer):
    """
    Experimental: EfficientNet-based detector.
    
    Not used in production pipeline.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'efficientnet_b0_experimental'
    
    def _build_model(self):
        """Build EfficientNet-B0 architecture."""
        import torch.nn as nn
        
        # Placeholder - would use timm or torchvision in production
        logger.warning("EfficientNet is experimental, using base Xception")
        return super()._build_model()
