"""
Late Fusion Ensemble

Combines multimodal detection results into a single verdict with uncertainty.
Production default pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .base import DetectionResult
from ..utils.uncertainty import (
    ensemble_uncertainty,
    agreement_score,
    calibrate_confidence,
    should_abstain,
)

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Final fused result from all modalities."""
    
    # Core outputs (user-facing)
    verdict: str  # 'REAL', 'FAKE', 'INCONCLUSIVE'
    confidence: int  # 0-100
    explanation: str
    
    # Analysis details
    score: float  # -1.0 to +1.0
    uncertainty: float
    
    # Per-modality contributions (internal, not exposed to users)
    modality_contributions: Dict[str, float]
    
    # Raw detection results for debugging
    raw_results: Optional[Dict[str, 'DetectionResult']] = None
    
    # Processing metadata
    processing_time_ms: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to user-facing dictionary."""
        return {
            'verdict': self.verdict,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'processing_time_ms': self.processing_time_ms,
        }
    
    def to_internal_dict(self) -> dict:
        """Full dictionary with internal details."""
        return {
            'verdict': self.verdict,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'score': round(self.score, 4),
            'uncertainty': round(self.uncertainty, 4),
            'modality_contributions': {
                k: round(v, 4) for k, v in self.modality_contributions.items()
            },
            'processing_time_ms': self.processing_time_ms,
        }


class LateFusionEnsemble:
    """
    Production-default late fusion ensemble.
    
    Combines results from multiple modality-specific detectors
    using weighted voting with uncertainty-aware calibration.
    """
    
    # Modality weights for different content types
    WEIGHTS = {
        'talking_head': {
            'video': 0.05,
            'videomae': 0.25,
            'audio': 0.01,
            'rppg': 0.01,
            'lipsync': 0.01,
            'spatial': 0.25,
            'temporal': 0.07,
            'frequency': 0.25,
            'physiological': 0.10
        },
        'silent_video': {
            'video': 0.05,
            'videomae': 0.25,
            'audio': 0.0,
            'rppg': 0.01,
            'lipsync': 0.01,
            'spatial': 0.25,
            'temporal': 0.08,
            'frequency': 0.25,
            'physiological': 0.10
        },
        'audio_only': {
            'video': 0.0,
            'videomae': 0.0,
            'audio': 0.90,
            'rppg': 0.0,
            'lipsync': 0.10,
            'spatial': 0.0,
            'temporal': 0.0,
            'frequency': 0.0,
            'physiological': 0.0
        },
    }
    
    # Thresholds (deprecated - using dynamic thresholds now)
    FAKE_THRESHOLD = -0.42  # Only call FAKE if score is very negative
    REAL_THRESHOLD = -0.40  # Call REAL for anything above this (includes most real videos)
    
    # Updated thresholds for better accuracy (v2)
    FAKE_THRESHOLD_V2 = 0.08   # Positive scores indicate FAKE
    REAL_THRESHOLD_V2 = -0.08  # Negative scores indicate REAL
    INCONCLUSIVE_MARGIN_V2 = 0.15  # Range [-0.08, 0.08] is INCONCLUSIVE
    
    def __init__(
        self,
        content_type: str = 'talking_head',
        custom_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize fusion ensemble.
        
        Args:
            content_type: 'talking_head', 'silent_video', or 'audio_only'
            custom_weights: Override default weights (internal use only)
        """
        self.content_type = content_type
        
        if custom_weights:
            self.weights = custom_weights
        else:
            self.weights = self.WEIGHTS.get(content_type, self.WEIGHTS['talking_head'])
    
    def fuse(
        self,
        results: Dict[str, DetectionResult],
        processing_time_ms: Optional[float] = None,
    ) -> FusionResult:
        """
        Fuse multimodal detection results.
        
        Args:
            results: Dict mapping modality -> DetectionResult
            processing_time_ms: Total processing time
            
        Returns:
            FusionResult with verdict, confidence, and explanation
        """
        # Filter to available modalities
        available = {k: v for k, v in results.items() if k in self.weights}
        
        if not available:
            return self._create_inconclusive("No detection results available")
        
        # Compute weighted fusion
        fused_score, uncertainty = self._weighted_fusion(available)
        print(f"\\n=== FUSION: fused_score={fused_score:.4f}, uncertainty={uncertainty:.4f} ===", flush=True)
        
        # Compute modality contributions
        contributions = self._compute_contributions(available)
        
        # Determine verdict
        verdict = self._score_to_verdict(fused_score, uncertainty)
        
        # === IMPROVED CONFIDENCE CALCULATION (v2) ===
        # Confidence is higher when:
        # 1. Models agree on the verdict
        # 2. Fused score is far from decision boundary
        # 3. Uncertainty is low
        # 4. Multiple strong indicators
        
        # Get scores from high-weight modalities
        high_weight_modalities = [
            'videomae', 'spatial', 'temporal', 'frequency', 'physiological'
        ]
        high_weight_scores = [
            results[m].score for m in high_weight_modalities 
            if m in results
        ]
        
        # Check model agreement on verdict direction
        if high_weight_scores:
            all_negative = all(s < 0 for s in high_weight_scores)
            all_positive = all(s > 0 for s in high_weight_scores)
            models_agree = all_negative or all_positive
        else:
            models_agree = False
        
        # Distance from decision boundary (0.0)
        distance_from_zero = abs(fused_score)
        
        # Base confidence from score magnitude
        # Confidence increases with distance from 0
        magnitude_confidence = min(0.95, distance_from_zero * 2.5)
        
        # Uncertainty penalty
        uncertainty_penalty = uncertainty * 0.3
        
        # Agreement bonus
        agreement_bonus = 0.08 if models_agree else 0.0
        
        # Combine: base confidence adjusted by penalties/bonuses
        base_conf = magnitude_confidence - uncertainty_penalty + agreement_bonus
        
        # Ensure within bounds
        confidence = np.clip(base_conf, 0.45, 0.99)
        
        confidence_int = int(confidence * 100)
        print(f">>> CONFIDENCE v2: {confidence_int}% (agree={models_agree}, mag={distance_from_zero:.3f}, unc={uncertainty:.3f})", flush=True)
        
        # Generate explanation
        explanation = self._generate_explanation(available, verdict, contributions)
        
        return FusionResult(
            verdict=verdict,
            confidence=confidence_int,
            explanation=explanation,
            score=fused_score,
            uncertainty=uncertainty,
            modality_contributions=contributions,
            raw_results=results,
            processing_time_ms=processing_time_ms,
        )
    
    def _weighted_fusion(
        self, results: Dict[str, DetectionResult]
    ) -> Tuple[float, float]:
        """
        Compute weighted average score with uncertainty.
        
        Uses both weights and per-modality confidence.
        """
        scores = []
        effective_weights = []
        uncertainties = []
        
        for modality, result in results.items():
            base_weight = self.weights.get(modality, 0.0)
            
            if base_weight > 0:
                # Adjust weight by modality confidence
                effective_weight = base_weight * result.confidence
                
                scores.append(result.score)
                effective_weights.append(effective_weight)
                uncertainties.append(result.uncertainty)
        
        if not scores:
            return 0.0, 1.0  # Maximum uncertainty
        
        # Normalize weights
        total_weight = sum(effective_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in effective_weights]
        else:
            normalized_weights = [1.0 / len(scores)] * len(scores)
        
        # Weighted mean score
        fused_score = sum(s * w for s, w in zip(scores, normalized_weights))
        
        # Log fusion calculation
        import logging
        logger = logging.getLogger(__name__)
        logger.info("📊 FUSION CALCULATION:")
        for (modality, result), weight in zip(results.items(), normalized_weights):
            if modality in ['spatial', 'temporal', 'frequency', 'physiological']:
                logger.info(f"   {modality}: score={result.score:.3f}, weight={weight:.3f}, contrib={result.score*weight:.3f}")
        logger.info(f"   FUSED SCORE: {fused_score:.3f}")
        
        # Combined uncertainty (weighted + disagreement)
        weighted_uncertainty = sum(u * w for u, w in zip(uncertainties, normalized_weights))
        
        # Add penalty for disagreement between modalities
        agreement = agreement_score(scores)
        disagreement_penalty = (1.0 - agreement) * 0.2
        
        total_uncertainty = min(1.0, weighted_uncertainty + disagreement_penalty)
        
        return float(fused_score), float(total_uncertainty)
    
    def _compute_contributions(
        self, results: Dict[str, DetectionResult]
    ) -> Dict[str, float]:
        """
        Compute how much each modality contributed to the verdict.
        
        Returns normalized contribution weights.
        """
        contributions = {}
        total = 0.0
        
        for modality, result in results.items():
            base_weight = self.weights.get(modality, 0.0)
            
            # Contribution = weight * confidence * abs(score)
            # Higher absolute score = stronger signal
            contribution = base_weight * result.confidence * abs(result.score)
            contributions[modality] = contribution
            total += contribution
        
        # Normalize to sum to 1
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        
        return contributions
    
    def _score_to_verdict(self, score: float, uncertainty: float) -> str:
        """
        Convert fused score to verdict with updated thresholds.
        
        Updated thresholds (v2):
        - Positive score → FAKE
        - Negative score → REAL
        - Scores near 0 with high uncertainty → INCONCLUSIVE
        """
        
        # High uncertainty → abstain
        if uncertainty > 0.70:
            return 'INCONCLUSIVE'
        
        # Score convention: positive = FAKE, negative = REAL
        THRESHOLD = 0.02
        
        if score > THRESHOLD:
            return 'FAKE'
        elif score < -THRESHOLD:
            return 'REAL'
        else:
            return 'INCONCLUSIVE'
    
    def _generate_explanation(
        self,
        results: Dict[str, DetectionResult],
        verdict: str,
        contributions: Dict[str, float],
    ) -> str:
        """
        Generate human-readable explanation.
        
        Does NOT expose model names, thresholds, or technical details.
        """
        explanations = []
        
        # Get top contributors
        sorted_modalities = sorted(
            contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for modality, contribution in sorted_modalities[:2]:  # Top 2
            if contribution < 0.1:
                continue
                
            result = results[modality]
            
            if modality == 'video':
                if result.score < -0.2:
                    explanations.append(
                        "Visual analysis detected manipulation artifacts in facial regions"
                    )
                elif result.score > 0.2:
                    explanations.append(
                        "Face appears natural with consistent texture and lighting"
                    )
            
            elif modality == 'audio':
                if result.score < -0.2:
                    explanations.append(
                        "Audio shows characteristics of synthetic speech"
                    )
                elif result.score > 0.2:
                    explanations.append(
                        "Voice patterns appear natural and unmodified"
                    )
            
            elif modality == 'spatial':
                if result.score < -0.2:
                    explanations.append("Spatial analysis found inconsistent noise patterns")
                elif result.score > 0.2:
                    explanations.append("Image structure is consistent")

            elif modality == 'temporal':
                if result.score < -0.2:
                    explanations.append("Temporal irregularities detected across frames")
                elif result.score > 0.2:
                    explanations.append("Motion appears fluid and natural")

            elif modality == 'frequency':
                if result.score < -0.2:
                    explanations.append("Frequency domain (FFT) artifacts detected")
                elif result.score > 0.2:
                    explanations.append("Frequency spectrum matches real video statistics")

            elif modality == 'physiological':
                if result.score < -0.2:
                    explanations.append("Abnormal physiological signs detected (pulse/breathing)")
                elif result.score > 0.2:
                    explanations.append("bio-signals (rPPG) appear authentic")
            
            elif modality == 'rppg':
                if result.score < -0.2:
                    explanations.append(
                        "No detectable physiological signals found"
                    )
                elif result.score > 0.2:
                    explanations.append(
                        "Normal physiological patterns detected"
                    )
            
            elif modality == 'lipsync':
                if result.score < -0.2:
                    explanations.append(
                        "Lip movements do not align with speech"
                    )
                elif result.score > 0.2:
                    explanations.append(
                        "Audio and lip movements are synchronized"
                    )
        
        if not explanations:
            if verdict == 'INCONCLUSIVE':
                return "Unable to reach a confident determination. Evidence is ambiguous."
            elif verdict == 'REAL':
                return "No manipulation indicators found across analyzed modalities."
            else:
                return "Multiple indicators suggest this content may be manipulated."
        
        return ". ".join(explanations) + "."
    
    def _create_inconclusive(self, reason: str) -> FusionResult:
        """Create an INCONCLUSIVE result with given reason."""
        return FusionResult(
            verdict='INCONCLUSIVE',
            confidence=0,
            explanation=reason,
            score=0.0,
            uncertainty=1.0,
            modality_contributions={},
        )


# Experimental variants (isolated)
class EarlyFusionEnsemble:
    """
    Experimental: Feature-level fusion before classification.
    
    Not used in production.
    """
    
    def __init__(self):
        logger.warning("EarlyFusionEnsemble is experimental, use LateFusionEnsemble")
    
    def fuse(self, features: Dict[str, np.ndarray]) -> FusionResult:
        raise NotImplementedError("Experimental - not implemented")


class AttentionFusionEnsemble:
    """
    Experimental: Attention-based cross-modal fusion.
    
    Not used in production.
    """
    
    def __init__(self):
        logger.warning("AttentionFusionEnsemble is experimental, use LateFusionEnsemble")
    
    def fuse(self, results: Dict[str, DetectionResult]) -> FusionResult:
        raise NotImplementedError("Experimental - not implemented")
