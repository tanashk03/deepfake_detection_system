"""
Uncertainty Estimation Utilities

Provides calibrated uncertainty estimation for deepfake detection models.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UncertaintyMetrics:
    """Container for various uncertainty measures."""
    
    # Aleatoric: inherent data uncertainty
    aleatoric: float
    
    # Epistemic: model uncertainty (reducible with more data)  
    epistemic: float
    
    # Total uncertainty
    total: float
    
    # Calibrated confidence
    calibrated_confidence: float
    
    def to_dict(self) -> dict:
        return {
            'aleatoric': round(self.aleatoric, 4),
            'epistemic': round(self.epistemic, 4),
            'total': round(self.total, 4),
            'calibrated_confidence': round(self.calibrated_confidence, 4),
        }


def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Compute entropy of probability distribution.
    
    Higher entropy = higher uncertainty.
    """
    probs = np.clip(probabilities, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def compute_predictive_entropy(mc_predictions: np.ndarray) -> float:
    """
    Compute predictive entropy from MC dropout samples.
    
    Args:
        mc_predictions: Shape (n_samples, n_classes) probability predictions
        
    Returns:
        Predictive entropy (total uncertainty)
    """
    mean_probs = np.mean(mc_predictions, axis=0)
    return compute_entropy(mean_probs)


def compute_mutual_information(mc_predictions: np.ndarray) -> float:
    """
    Compute mutual information (epistemic uncertainty) from MC samples.
    
    MI = H[y|x] - E[H[y|x,w]] 
       = predictive_entropy - mean_entropy
    """
    # Predictive entropy (total)
    predictive_entropy = compute_predictive_entropy(mc_predictions)
    
    # Mean of individual entropies (aleatoric)
    individual_entropies = [compute_entropy(p) for p in mc_predictions]
    mean_entropy = np.mean(individual_entropies)
    
    # Mutual information (epistemic)
    mi = predictive_entropy - mean_entropy
    return float(max(0.0, mi))


def decompose_uncertainty(
    mc_predictions: np.ndarray
) -> Tuple[float, float, float]:
    """
    Decompose total uncertainty into aleatoric and epistemic components.
    
    Args:
        mc_predictions: Shape (n_samples, n_classes)
        
    Returns:
        (aleatoric, epistemic, total) uncertainties
    """
    # Total uncertainty (predictive entropy)
    total = compute_predictive_entropy(mc_predictions)
    
    # Epistemic uncertainty (mutual information)
    epistemic = compute_mutual_information(mc_predictions)
    
    # Aleatoric uncertainty (expected entropy)
    aleatoric = total - epistemic
    
    return float(aleatoric), float(epistemic), float(total)


def calibrate_confidence(
    raw_confidence: float,
    uncertainty: float,
    temperature: float = 1.5,
) -> float:
    """
    Apply temperature scaling and uncertainty adjustment for calibration.
    
    Args:
        raw_confidence: Model's raw confidence (0-1)
        uncertainty: Estimated uncertainty
        temperature: Calibration temperature (>1 = more conservative)
        
    Returns:
        Calibrated confidence score
    """
    # Temperature scaling
    scaled = raw_confidence ** (1.0 / temperature)
    
    # Uncertainty penalty
    penalty = np.exp(-2.0 * uncertainty)
    
    calibrated = scaled * penalty
    return float(np.clip(calibrated, 0.0, 1.0))


def ensemble_uncertainty(
    predictions: List[float],
    weights: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Compute uncertainty from ensemble of model predictions.
    
    Args:
        predictions: List of scores from different models
        weights: Optional weights for each model
        
    Returns:
        (weighted_mean, weighted_std)
    """
    preds = np.array(predictions)
    
    if weights is None:
        weights = np.ones(len(predictions)) / len(predictions)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
    
    # Weighted mean
    mean = float(np.sum(weights * preds))
    
    # Weighted standard deviation
    variance = float(np.sum(weights * (preds - mean) ** 2))
    std = float(np.sqrt(variance))
    
    return mean, std


def agreement_score(predictions: List[float], threshold: float = 0.0) -> float:
    """
    Compute agreement between ensemble members.
    
    Returns fraction of models that agree on the verdict.
    """
    verdicts = [1 if p > threshold else -1 for p in predictions]
    
    n_positive = sum(1 for v in verdicts if v > 0)
    n_negative = len(verdicts) - n_positive
    
    # Agreement is max fraction on same side
    agreement = max(n_positive, n_negative) / len(verdicts)
    return float(agreement)


def should_abstain(
    confidence: float,
    uncertainty: float,
    min_confidence: float = 0.6,
    max_uncertainty: float = 0.3,
) -> bool:
    """
    Determine if the model should abstain (return INCONCLUSIVE).
    
    Abstain when confidence is too low or uncertainty too high.
    """
    return confidence < min_confidence or uncertainty > max_uncertainty
