"""
Animation/Cartoon/Anime/CGI Detection

Detects if video content is animated, cartoon, anime, or CGI-generated.
Uses color analysis, edge detection, and feature extraction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AnimationDetector:
    """
    Detects animated, cartoon, anime, and CGI content.
    
    Returns:
        - is_animated: bool, confidence: float
    
    Techniques:
        1. Color saturation analysis - Animated content has higher saturation
        2. Edge uniformity - Cartoon/anime edges are more uniform
        3. Color simplification - Animated content uses limited color palette
        4. Texture flatness - Animated has flatter textures than real video
    """
    
    # Thresholds (tuned for production)
    SATURATION_THRESHOLD = 0.65  # Animated has higher saturation
    EDGE_UNIFORMITY_THRESHOLD = 0.58  # More uniform edges in animation
    COLOR_PALETTE_THRESHOLD = 0.45  # Fewer unique colors in animation
    TEXTURE_FLATNESS_THRESHOLD = 0.60  # Flatter texture in animation
    
    # Voting thresholds
    MIN_INDICATORS_FOR_ANIMATION = 2  # At least 2 indicators must fire
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize animation detector.
        
        Args:
            device: 'cpu' or 'cuda' (not used, kept for API compatibility)
        """
        self.device = device
        self._initialized = True
    
    def detect(
        self, 
        frames: np.ndarray,
        sample_rate: int = 5,
    ) -> Tuple[bool, float]:
        """
        Detect if frames contain animated/cartoon/anime/CGI content.
        
        Args:
            frames: (N, H, W, 3) numpy array with frames, BGR or RGB
            sample_rate: Process every Nth frame to speed up
            
        Returns:
            (is_animated: bool, confidence: float [0, 1])
        """
        if frames is None or len(frames) == 0:
            return False, 0.0
        
        # Sample frames evenly
        if len(frames) > 100:
            indices = np.linspace(0, len(frames)-1, min(100, len(frames)), dtype=int)
            sampled_frames = frames[indices]
        else:
            sampled_frames = frames[::sample_rate] if sample_rate > 1 else frames
        
        if len(sampled_frames) == 0:
            return False, 0.0
        
        # Run detectors on sampled frames
        animation_indicators = []
        
        for frame in sampled_frames:
            # Ensure BGR format
            if frame.shape[2] == 3:
                # Assume BGR from OpenCV
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                
                indicators = self._analyze_frame(frame)
                animation_indicators.append(indicators)
        
        if not animation_indicators:
            return False, 0.0
        
        # Aggregate results across frames
        avg_indicators = {
            'high_saturation': np.mean([ind['high_saturation'] for ind in animation_indicators]),
            'uniform_edges': np.mean([ind['uniform_edges'] for ind in animation_indicators]),
            'color_simplification': np.mean([ind['color_simplification'] for ind in animation_indicators]),
            'texture_flatness': np.mean([ind['texture_flatness'] for ind in animation_indicators]),
        }
        
        logger.debug(f"Animation indicators (averaged): {avg_indicators}")
        
        # Count how many indicators suggest animation
        firing_indicators = sum(1 for v in avg_indicators.values() if v > 0.5)
        
        # Determine if animated
        is_animated = firing_indicators >= self.MIN_INDICATORS_FOR_ANIMATION
        
        # Confidence: average of all indicators if animation detected
        if is_animated:
            confidence = np.mean(list(avg_indicators.values()))
        else:
            # If not detected as animation, confidence is low
            confidence = 1.0 - np.mean(list(avg_indicators.values()))
        
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        logger.info(
            f"Animation Detection: is_animated={is_animated}, confidence={confidence:.3f}, "
            f"indicators_firing={firing_indicators}"
        )
        
        return is_animated, confidence
    
    def _analyze_frame(self, frame: np.ndarray) -> dict:
        """
        Analyze single frame for animation indicators.
        
        Args:
            frame: (H, W, 3) BGR image, uint8
            
        Returns:
            dict with indicator flags (0.0 or 1.0)
        """
        indicators = {}
        
        # 1. Color Saturation Analysis
        saturation_score = self._analyze_saturation(frame)
        indicators['high_saturation'] = 1.0 if saturation_score > self.SATURATION_THRESHOLD else 0.0
        
        # 2. Edge Uniformity Analysis
        edge_uniformity = self._analyze_edge_uniformity(frame)
        indicators['uniform_edges'] = 1.0 if edge_uniformity > self.EDGE_UNIFORMITY_THRESHOLD else 0.0
        
        # 3. Color Palette Analysis
        palette_simplification = self._analyze_color_palette(frame)
        indicators['color_simplification'] = 1.0 if palette_simplification > self.COLOR_PALETTE_THRESHOLD else 0.0
        
        # 4. Texture Flatness Analysis
        texture_flatness = self._analyze_texture_flatness(frame)
        indicators['texture_flatness'] = 1.0 if texture_flatness > self.TEXTURE_FLATNESS_THRESHOLD else 0.0
        
        return indicators
    
    def _analyze_saturation(self, frame: np.ndarray) -> float:
        """
        Analyze color saturation.
        
        Animated/cartoon content tends to have higher saturation.
        Returns score [0, 1] where 1 = high saturation (animation-like)
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel (S in HSV)
        saturation = hsv[:, :, 1].astype(float) / 255.0
        
        # Calculate mean saturation
        mean_saturation = np.mean(saturation)
        
        # Percentage of pixels with high saturation (>150/255 ≈ 0.59)
        high_sat_pixels = np.sum(saturation > 0.59) / saturation.size
        
        # Score: combination of mean saturation and high-sat pixel percentage
        score = (mean_saturation + high_sat_pixels) / 2.0
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _analyze_edge_uniformity(self, frame: np.ndarray) -> float:
        """
        Analyze edge uniformity.
        
        Animated/cartoon has more uniform, sharp edges.
        Returns score [0, 1] where 1 = uniform edges (animation-like)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate slightly to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Edge uniformity: measure line thickness and straightness
        # Animated edges tend to be uniform thickness
        # Compare edge pattern consistency
        
        # Calculate edge density
        edge_pixels = np.sum(edges_dilated > 0)
        edge_density = edge_pixels / edges_dilated.size
        
        # For animation detection, we look at line continuity
        # Apply morphological closing to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Count connected components - lower = more uniform
        num_labels, _ = cv2.connectedComponents(edges_closed)
        
        # Normalize: fewer components = more uniform edges
        # If very few components, edges are uniform
        component_ratio = 1.0 / max(1, num_labels / 100.0)  # Normalized
        
        # Score combination
        score = (edge_density + min(component_ratio, 1.0)) / 2.0
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _analyze_color_palette(self, frame: np.ndarray) -> float:
        """
        Analyze color palette simplification.
        
        Animated/cartoon has simplified color palette (fewer unique colors).
        Returns score [0, 1] where 1 = simple palette (animation-like)
        """
        # Reduce colors to simulate posterization
        # Quantize to 64 levels (standard for animated content)
        quantized = (frame // 64) * 64
        
        # Count unique colors
        reshaped = quantized.reshape(-1, 3)
        unique_colors = len(np.unique(reshaped, axis=0))
        
        # For a 480p frame: 720x480 = 345,600 pixels
        # Real video: 50,000-200,000 unique colors
        # Animated: 1,000-20,000 unique colors
        
        # Calculate color complexity ratio
        # Lower ratio = simpler palette = more animation-like
        max_possible_colors = frame.shape[0] * frame.shape[1]
        color_ratio = unique_colors / max_possible_colors
        
        # Invert: high complexity (real) → 0, low complexity (animated) → 1
        score = 1.0 - np.clip(color_ratio, 0.0, 1.0)
        
        return float(score)
    
    def _analyze_texture_flatness(self, frame: np.ndarray) -> float:
        """
        Analyze texture smoothness/flatness.
        
        Animated/CGI content has flatter textures with less noise.
        Returns score [0, 1] where 1 = flat texture (animation-like)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        
        # Calculate local variance (Laplacian variance)
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        variance = np.var(laplacian)
        
        # Normalize variance
        # Real video: variance ≈ 500-2000
        # Animated: variance ≈ 100-500
        variance_normalized = np.clip(variance / 2000.0, 0.0, 1.0)
        
        # Invert: high variance (real) → 0, low variance (animated) → 1
        flatness_score = 1.0 - variance_normalized
        
        # Additional check: local contrast
        # Real faces have fine skin texture; animated doesn't
        kernel_size = 15
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        contrast = np.std(gray - blurred)
        
        # Normalize contrast
        contrast_normalized = np.clip(contrast / 0.15, 0.0, 1.0)
        
        # Invert: high contrast (real) → 0, low contrast (animated) → 1
        contrast_score = 1.0 - contrast_normalized
        
        # Combined score
        score = (flatness_score + contrast_score) / 2.0
        
        return float(np.clip(score, 0.0, 1.0))
    
    def predict(self, frames: np.ndarray, with_uncertainty: bool = True) -> dict:
        """
        Compatibility method for pipeline integration.
        
        Args:
            frames: (N, H, W, 3) video frames
            with_uncertainty: Include uncertainty estimate
            
        Returns:
            dict with keys: is_animated, confidence, uncertainty, reason
        """
        is_animated, confidence = self.detect(frames, sample_rate=5)
        
        # Uncertainty inversely related to confidence
        uncertainty = 1.0 - confidence if confidence > 0.5 else 0.5
        
        reason = (
            "Detected animated/cartoon/anime/CGI content characteristics"
            if is_animated else
            "Content appears to be real video"
        )
        
        return {
            'is_animated': is_animated,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'reason': reason,
        }
