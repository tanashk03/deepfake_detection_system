"""
Explainability (XAI) Package for Luminark

Provides visual and textual explanations for detection decisions.

Includes:
- Grad-CAM for video attention maps
- SHAP-style feature attribution for audio
- Textual rationale generation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExplanationArtifact:
    """Compact XAI artifact (PII-free)."""
    
    # Video attention (downsized)
    attention_map: Optional[np.ndarray] = None  # (H', W') float
    attention_summary: str = ""  # "High attention on lower face region"
    
    # Audio attribution
    audio_features: Dict[str, float] = None  # Feature importance scores
    audio_summary: str = ""
    
    # Textual rationale
    rationale: str = ""
    
    # Metadata
    modality_scores: Dict[str, float] = None
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (no raw arrays)."""
        return {
            'attention_summary': self.attention_summary,
            'audio_summary': self.audio_summary,
            'rationale': self.rationale,
            'modality_scores': self.modality_scores,
            'confidence': self.confidence,
        }


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for video.
    
    Highlights regions that contribute most to the decision.
    
    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
    """
    
    def __init__(self, model, target_layer: str = 'layer4'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of layer to visualize
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def generate(
        self,
        input_tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image (1, C, H, W)
            target_class: Class to explain (default: predicted class)
            
        Returns:
            Attention heatmap (H, W) normalized to [0, 1]
        """
        import torch
        import torch.nn.functional as F
        
        self.model.eval()
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Compute Grad-CAM
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]
        
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap
    
    def summarize_attention(self, heatmap: np.ndarray) -> str:
        """Generate textual summary of attention regions."""
        h, w = heatmap.shape
        
        # Divide into regions
        top = heatmap[:h//3, :].mean()
        middle = heatmap[h//3:2*h//3, :].mean()
        bottom = heatmap[2*h//3:, :].mean()
        
        left = heatmap[:, :w//3].mean()
        center = heatmap[:, w//3:2*w//3].mean()
        right = heatmap[:, 2*w//3:].mean()
        
        regions = {
            'upper face': top,
            'mid face': middle,
            'lower face': bottom,
        }
        
        max_region = max(regions, key=regions.get)
        
        if regions[max_region] > 0.5:
            return f"High attention on {max_region} region"
        else:
            return "Distributed attention across face"


class AudioFeatureAttribution:
    """
    Simple feature attribution for audio analysis.
    
    Uses perturbation-based importance estimation.
    """
    
    FEATURE_NAMES = [
        'spectral_centroid',
        'spectral_bandwidth',
        'spectral_rolloff',
        'zero_crossing_rate',
        'mfcc_variance',
        'pitch_stability',
        'formant_consistency',
    ]
    
    def compute_importance(
        self,
        features: Dict[str, float],
        prediction_score: float,
    ) -> Dict[str, float]:
        """
        Estimate feature importance.
        
        Simple approach: features far from typical values are more important.
        """
        # Typical ranges for real speech
        typical_ranges = {
            'spectral_centroid': (1500, 3500),
            'spectral_bandwidth': (1000, 2500),
            'spectral_rolloff': (3000, 8000),
            'zero_crossing_rate': (0.02, 0.1),
            'mfcc_variance': (5, 20),
        }
        
        importance = {}
        
        for name, value in features.items():
            if name in typical_ranges:
                low, high = typical_ranges[name]
                if value < low:
                    importance[name] = abs(value - low) / (high - low)
                elif value > high:
                    importance[name] = abs(value - high) / (high - low)
                else:
                    importance[name] = 0.1
            else:
                importance[name] = 0.2
        
        # Normalize
        total = sum(importance.values()) + 1e-8
        return {k: v / total for k, v in importance.items()}
    
    def summarize(self, importance: Dict[str, float], is_fake: bool) -> str:
        """Generate summary of audio analysis."""
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if is_fake:
            if 'spectral_centroid' in [f[0] for f in top_features]:
                return "Unusual spectral characteristics detected in audio."
            elif 'pitch_stability' in [f[0] for f in top_features]:
                return "Pitch patterns inconsistent with natural speech."
            else:
                return "Audio shows characteristics of synthetic generation."
        else:
            return "Audio characteristics consistent with natural speech."


class RationaleGenerator:
    """
    Generate human-readable explanations from analysis results.
    """
    
    TEMPLATES = {
        'FAKE_HIGH_CONF': [
            "Multiple indicators strongly suggest this content has been manipulated.",
            "Analysis reveals clear signs of synthetic generation.",
        ],
        'FAKE_MED_CONF': [
            "Several indicators suggest potential manipulation.",
            "Some characteristics are inconsistent with authentic content.",
        ],
        'REAL_HIGH_CONF': [
            "Content appears authentic across all analysis dimensions.",
            "No manipulation indicators detected.",
        ],
        'REAL_MED_CONF': [
            "Content appears largely authentic.",
            "Minor anomalies detected but within normal variation.",
        ],
        'INCONCLUSIVE': [
            "Unable to reach confident determination.",
            "Content quality or characteristics prevent reliable analysis.",
        ],
    }
    
    def generate(
        self,
        verdict: str,
        confidence: int,
        modality_summaries: Dict[str, str],
        top_modality: str,
    ) -> str:
        """
        Generate one-paragraph rationale.
        
        Args:
            verdict: REAL/FAKE/INCONCLUSIVE
            confidence: 0-100
            modality_summaries: Per-modality explanations
            top_modality: Most influential modality
            
        Returns:
            Human-readable paragraph
        """
        # Select template
        if verdict == 'INCONCLUSIVE':
            key = 'INCONCLUSIVE'
        elif verdict == 'FAKE':
            key = 'FAKE_HIGH_CONF' if confidence >= 80 else 'FAKE_MED_CONF'
        else:
            key = 'REAL_HIGH_CONF' if confidence >= 80 else 'REAL_MED_CONF'
        
        template = np.random.choice(self.TEMPLATES[key])
        
        # Add modality-specific detail
        if top_modality in modality_summaries:
            detail = modality_summaries[top_modality]
            if detail:
                template = f"{template} {detail}"
        
        return template


def generate_explanation(
    result,  # FusionResult or EnsembleResult
    video_heatmap: Optional[np.ndarray] = None,
    audio_features: Optional[Dict[str, float]] = None,
) -> ExplanationArtifact:
    """
    Generate complete explanation artifact.
    
    Args:
        result: Detection result
        video_heatmap: Optional Grad-CAM output
        audio_features: Optional audio feature dict
        
    Returns:
        ExplanationArtifact (PII-free, compact)
    """
    generator = RationaleGenerator()
    
    # Compute modality summaries
    summaries = {}
    if video_heatmap is not None:
        gradcam = GradCAM(None)  # Mock for summary only
        summaries['video'] = gradcam.summarize_attention(video_heatmap)
    
    if audio_features:
        audio_attr = AudioFeatureAttribution()
        importance = audio_attr.compute_importance(
            audio_features,
            result.score if hasattr(result, 'score') else 0
        )
        summaries['audio'] = audio_attr.summarize(
            importance,
            is_fake=result.verdict == 'FAKE'
        )
    
    # Determine top modality
    contributions = getattr(result, 'modality_contributions', {})
    top_modality = max(contributions, key=contributions.get) if contributions else 'video'
    
    # Generate rationale
    rationale = generator.generate(
        verdict=result.verdict,
        confidence=result.confidence,
        modality_summaries=summaries,
        top_modality=top_modality,
    )
    
    return ExplanationArtifact(
        attention_map=video_heatmap,
        attention_summary=summaries.get('video', ''),
        audio_features=audio_features,
        audio_summary=summaries.get('audio', ''),
        rationale=rationale,
        modality_scores=contributions,
        confidence=result.confidence / 100,
    )
