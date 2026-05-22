"""
Adaptive Ensemble with Uncertainty Weighting

Meta-learner that combines modality predictions using:
1. Learned confidence weights
2. Ensemble variance (epistemic uncertainty)
3. Temperature-scaled calibration

Reference:
    Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty", NeurIPS 2017
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .base import DetectionResult
from ..utils.calibration import TemperatureScaler, compute_ece

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from adaptive ensemble."""
    score: float              # Combined score [-1, 1]
    confidence: int           # Calibrated confidence [0, 100]
    uncertainty: float        # Ensemble uncertainty [0, 1]
    verdict: str              # REAL / FAKE / INCONCLUSIVE
    explanation: str          # Human-readable
    modality_contributions: Dict[str, float]
    processing_time_ms: Optional[float] = None
    
    # Internal diagnostics
    ensemble_variance: float = 0.0
    agreement_score: float = 0.0
    abstention_triggered: bool = False


class AdaptiveEnsemble:
    """
    Uncertainty-aware multimodal fusion.
    
    Key features:
    - Variance-weighted combination
    - Temperature-calibrated confidence
    - Automatic abstention on high disagreement
    - Content-type adaptive weights
    """
    
    # Base weights by content type
    CONTENT_WEIGHTS = {
        'talking_head': {'video': 0.35, 'audio': 0.25, 'rppg': 0.15, 'lipsync': 0.25},
        'silent_video': {'video': 0.50, 'audio': 0.10, 'rppg': 0.30, 'lipsync': 0.10},
        'audio_only': {'video': 0.0, 'audio': 0.60, 'rppg': 0.0, 'lipsync': 0.40},
        'default': {'video': 0.40, 'audio': 0.25, 'rppg': 0.15, 'lipsync': 0.20},
    }
    
    # Thresholds
    ABSTENTION_UNCERTAINTY_THRESHOLD = 0.35
    ABSTENTION_DISAGREEMENT_THRESHOLD = 0.8
    VERDICT_THRESHOLDS = {'fake': 0.3, 'real': -0.3}
    
    def __init__(
        self,
        content_type: str = 'default',
        use_calibration: bool = True,
        edge_mode: bool = True,
    ):
        """
        Initialize adaptive ensemble.
        
        Args:
            content_type: One of 'talking_head', 'silent_video', 'audio_only', 'default'
            use_calibration: Apply temperature scaling
            edge_mode: Conservative thresholds for edge deployment
        """
        self.content_type = content_type
        self.base_weights = self.CONTENT_WEIGHTS.get(content_type, self.CONTENT_WEIGHTS['default'])
        self.use_calibration = use_calibration
        self.edge_mode = edge_mode
        
        # Per-modality temperature scalers
        self.scalers = {m: TemperatureScaler() for m in self.base_weights.keys()}
        
        # Learned weight adjustments (meta-learner)
        self.weight_adjustments = {m: 1.0 for m in self.base_weights.keys()}
    
    def combine(
        self,
        results: Dict[str, DetectionResult],
        verbose: bool = False,
    ) -> EnsembleResult:
        """
        Combine modality results with uncertainty weighting.
        
        Args:
            results: {modality_name: DetectionResult}
            verbose: Log detailed diagnostics
            
        Returns:
            EnsembleResult with combined verdict
        """
        if not results:
            return self._empty_result()
        
        # Extract scores and uncertainties
        scores = {}
        uncertainties = {}
        confidences = {}
        
        for modality, result in results.items():
            scores[modality] = result.score
            uncertainties[modality] = result.uncertainty
            confidences[modality] = result.confidence
        
        # Compute adaptive weights
        weights = self._compute_adaptive_weights(scores, uncertainties)
        
        # Variance-weighted combination
        combined_score, ensemble_var = self._variance_weighted_combine(
            scores, uncertainties, weights
        )
        
        # Agreement score
        agreement = self._compute_agreement(scores)
        
        # Calibrated confidence
        calibrated_conf = self._compute_calibrated_confidence(
            combined_score, ensemble_var, agreement
        )
        
        # Determine verdict with abstention
        verdict, abstained = self._determine_verdict(
            combined_score, ensemble_var, agreement
        )
        
        # Contribution breakdown
        contributions = self._compute_contributions(scores, weights)
        
        # Generate explanation
        explanation = self._generate_explanation(
            verdict, scores, contributions, agreement
        )
        
        if verbose:
            logger.info(f"Ensemble: score={combined_score:.3f}, "
                       f"var={ensemble_var:.3f}, agree={agreement:.3f}, "
                       f"verdict={verdict}")
        
        return EnsembleResult(
            score=float(combined_score),
            confidence=calibrated_conf,
            uncertainty=float(ensemble_var),
            verdict=verdict,
            explanation=explanation,
            modality_contributions=contributions,
            ensemble_variance=float(ensemble_var),
            agreement_score=float(agreement),
            abstention_triggered=abstained,
        )
    
    def _compute_adaptive_weights(
        self,
        scores: Dict[str, float],
        uncertainties: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute uncertainty-adjusted weights.
        
        Lower uncertainty → higher weight.
        """
        weights = {}
        total = 0.0
        
        for modality in scores.keys():
            base = self.base_weights.get(modality, 0.1)
            adj = self.weight_adjustments.get(modality, 1.0)
            unc = uncertainties.get(modality, 0.5)
            
            # Inverse uncertainty weighting
            inv_unc = 1.0 / (unc + 0.1)
            
            w = base * adj * inv_unc
            weights[modality] = w
            total += w
        
        # Normalize
        if total > 0:
            weights = {m: w / total for m, w in weights.items()}
        
        return weights
    
    def _variance_weighted_combine(
        self,
        scores: Dict[str, float],
        uncertainties: Dict[str, float],
        weights: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Combine scores using variance-weighted averaging.
        
        Returns:
            (combined_score, ensemble_variance)
        """
        if not scores:
            return 0.0, 1.0
        
        score_array = np.array(list(scores.values()))
        weight_array = np.array([weights.get(m, 0) for m in scores.keys()])
        unc_array = np.array(list(uncertainties.values()))
        
        # Weighted mean
        combined = np.sum(score_array * weight_array)
        
        # Ensemble variance (disagreement)
        weighted_var = np.sum(weight_array * (score_array - combined) ** 2)
        
        # Add aleatoric uncertainty
        mean_aleatoric = np.mean(unc_array)
        total_uncertainty = np.sqrt(weighted_var + mean_aleatoric ** 2)
        
        return float(combined), float(total_uncertainty)
    
    def _compute_agreement(self, scores: Dict[str, float]) -> float:
        """
        Compute agreement score among modalities.
        
        1.0 = perfect agreement (all same sign and magnitude)
        0.0 = complete disagreement
        """
        if len(scores) < 2:
            return 1.0
        
        score_array = np.array(list(scores.values()))
        
        # Sign agreement
        signs = np.sign(score_array)
        sign_agreement = abs(signs.mean())
        
        # Magnitude consistency
        if score_array.std() > 0:
            cv = score_array.std() / (abs(score_array.mean()) + 0.1)
            magnitude_agreement = 1.0 / (1.0 + cv)
        else:
            magnitude_agreement = 1.0
        
        return float(0.6 * sign_agreement + 0.4 * magnitude_agreement)
    
    def _compute_calibrated_confidence(
        self,
        score: float,
        variance: float,
        agreement: float,
    ) -> int:
        """
        Compute calibrated confidence percentage.
        """
        # Base confidence from score magnitude
        base_conf = min(1.0, abs(score) * 1.2)
        
        # Penalize high variance
        var_penalty = 1.0 - min(1.0, variance)
        
        # Boost for high agreement
        agreement_boost = 0.5 + 0.5 * agreement
        
        # Combine
        confidence = base_conf * var_penalty * agreement_boost
        
        # Calibrate if enabled
        if self.use_calibration and hasattr(self, '_global_temp'):
            confidence = confidence / self._global_temp
        
        return int(np.clip(confidence * 100, 0, 100))
    
    def _determine_verdict(
        self,
        score: float,
        variance: float,
        agreement: float,
    ) -> Tuple[str, bool]:
        """
        Determine verdict with abstention logic.
        
        Returns:
            (verdict, abstention_triggered)
        """
        # Check abstention conditions
        should_abstain = (
            variance > self.ABSTENTION_UNCERTAINTY_THRESHOLD or
            agreement < (1.0 - self.ABSTENTION_DISAGREEMENT_THRESHOLD)
        )
        
        if self.edge_mode:
            # More conservative on edge
            should_abstain = should_abstain or abs(score) < 0.2
        
        if should_abstain:
            return 'INCONCLUSIVE', True
        
        if score > self.VERDICT_THRESHOLDS['fake']:
            return 'FAKE', False
        elif score < self.VERDICT_THRESHOLDS['real']:
            return 'REAL', False
        else:
            return 'INCONCLUSIVE', False
    
    def _compute_contributions(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute per-modality contribution to final score."""
        if not scores:
            return {}
        
        total_weighted = sum(abs(s) * weights.get(m, 0) 
                           for m, s in scores.items())
        
        if total_weighted == 0:
            return {m: 1.0 / len(scores) for m in scores}
        
        return {
            m: abs(s) * weights.get(m, 0) / total_weighted
            for m, s in scores.items()
        }
    
    def _generate_explanation(
        self,
        verdict: str,
        scores: Dict[str, float],
        contributions: Dict[str, float],
        agreement: float,
    ) -> str:
        """Generate human-readable explanation."""
        modality_names = {
            'video': 'Visual analysis',
            'audio': 'Audio analysis',
            'rppg': 'Physiological signals',
            'lipsync': 'Lip synchronization',
        }
        
        # Sort by contribution
        sorted_modalities = sorted(
            contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        parts = []
        for modality, contrib in sorted_modalities[:2]:
            if contrib < 0.1:
                continue
            
            name = modality_names.get(modality, modality)
            score = scores.get(modality, 0)
            
            if score > 0.2:
                parts.append(f"{name} indicates manipulation")
            elif score < -0.2:
                parts.append(f"{name} appears authentic")
        
        if not parts:
            if verdict == 'INCONCLUSIVE':
                return "Unable to reach confident determination."
            return "Analysis complete."
        
        explanation = '. '.join(parts) + '.'
        
        if agreement < 0.5:
            explanation += " Note: Modalities show some disagreement."
        
        return explanation
    
    def _empty_result(self) -> EnsembleResult:
        """Return empty result when no inputs."""
        return EnsembleResult(
            score=0.0,
            confidence=0,
            uncertainty=1.0,
            verdict='INCONCLUSIVE',
            explanation='No analysis data available.',
            modality_contributions={},
        )
    
    def set_escalation_threshold(self, threshold: float) -> None:
        """
        Set uncertainty threshold for cloud escalation.
        
        Args:
            threshold: Uncertainty value above which to escalate (0.0-1.0)
        """
        self.ABSTENTION_UNCERTAINTY_THRESHOLD = threshold
    
    def should_escalate_to_cloud(self, result: EnsembleResult) -> bool:
        """
        Determine if this result should be escalated to cloud processing.
        
        Args:
            result: Ensemble result to check
            
        Returns:
            True if should escalate for higher-fidelity analysis
        """
        return (
            result.abstention_triggered or
            result.uncertainty > self.ABSTENTION_UNCERTAINTY_THRESHOLD or
            result.verdict == 'INCONCLUSIVE'
        )
