"""
Temperature Scaling Calibration

Post-hoc calibration for neural network confidence scores.

Reference:
    Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration evaluation metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    reliability_diagram: dict  # Binned confidence vs accuracy
    
    def is_well_calibrated(self, threshold: float = 0.05) -> bool:
        """Check if ECE is below threshold."""
        return self.ece < threshold


class TemperatureScaler:
    """
    Temperature scaling for probability calibration.
    
    Learns a single temperature parameter T to scale logits:
        calibrated_prob = softmax(logits / T)
    
    Simple, effective, preserves accuracy.
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        self.temperature = initial_temperature
        self._fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iters: int = 100,
    ) -> float:
        """
        Fit temperature parameter on validation set.
        
        Args:
            logits: Model logits (N, C) or (N,) for binary
            labels: True labels (N,)
            lr: Learning rate for gradient descent
            max_iters: Maximum iterations
            
        Returns:
            Optimal temperature
        """
        # Ensure 2D logits
        if logits.ndim == 1:
            logits = np.stack([1 - logits, logits], axis=1)
        
        temperature = self.temperature
        
        for i in range(max_iters):
            # Compute scaled probabilities
            scaled_logits = logits / temperature
            probs = self._softmax(scaled_logits)
            
            # NLL loss
            nll = self._nll_loss(probs, labels)
            
            # Gradient of NLL w.r.t. temperature
            grad = self._temperature_gradient(logits, labels, temperature)
            
            # Update
            temperature = max(0.1, temperature - lr * grad)
            
            if i % 20 == 0:
                logger.debug(f"Iter {i}: T={temperature:.4f}, NLL={nll:.4f}")
        
        self.temperature = temperature
        self._fitted = True
        
        logger.info(f"Temperature scaling: T = {temperature:.4f}")
        return temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits (N, C) or (N,)
            
        Returns:
            Calibrated probabilities
        """
        if logits.ndim == 1:
            logits = np.stack([1 - logits, logits], axis=1)
        
        scaled = logits / self.temperature
        return self._softmax(scaled)
    
    def calibrate_confidence(self, confidence: float) -> float:
        """
        Calibrate a single confidence score.
        
        For binary classification:
            calibrated = sigmoid(logit(confidence) / T)
        """
        # Convert confidence to logit
        eps = 1e-8
        confidence = np.clip(confidence, eps, 1 - eps)
        logit = np.log(confidence / (1 - confidence))
        
        # Scale and convert back
        scaled_logit = logit / self.temperature
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return float(calibrated)
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    @staticmethod
    def _nll_loss(probs: np.ndarray, labels: np.ndarray) -> float:
        """Negative log-likelihood loss."""
        n = len(labels)
        log_probs = np.log(probs[np.arange(n), labels.astype(int)] + 1e-10)
        return -log_probs.mean()
    
    def _temperature_gradient(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        temperature: float,
    ) -> float:
        """Compute gradient of NLL w.r.t. temperature."""
        n = len(labels)
        scaled = logits / temperature
        probs = self._softmax(scaled)
        
        # dNLL/dT = (1/T^2) * sum_i (z_yi - sum_c z_c * p_c)
        correct_logits = logits[np.arange(n), labels.astype(int)]
        weighted_sum = (logits * probs).sum(axis=1)
        
        grad = (1 / temperature**2) * (weighted_sum - correct_logits).mean()
        return grad


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    
    Args:
        confidences: Predicted confidences (N,)
        predictions: Predicted classes (N,)
        labels: True labels (N,)
        n_bins: Number of bins
        
    Returns:
        CalibrationMetrics with ECE, MCE, and reliability diagram
    """
    n = len(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    mce = 0.0
    reliability = {'confidence': [], 'accuracy': [], 'count': []}
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        
        # Samples in this bin
        mask = (confidences > lower) & (confidences <= upper)
        bin_count = mask.sum()
        
        if bin_count == 0:
            continue
        
        # Average confidence and accuracy in bin
        bin_conf = confidences[mask].mean()
        bin_acc = (predictions[mask] == labels[mask]).mean()
        
        # ECE contribution
        ece += (bin_count / n) * abs(bin_acc - bin_conf)
        mce = max(mce, abs(bin_acc - bin_conf))
        
        reliability['confidence'].append(float(bin_conf))
        reliability['accuracy'].append(float(bin_acc))
        reliability['count'].append(int(bin_count))
    
    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        reliability_diagram=reliability,
    )


class EnsembleCalibrator:
    """
    Calibrate multiple models with per-model temperature scaling.
    """
    
    def __init__(self, modality_names: List[str]):
        self.modalities = modality_names
        self.scalers = {name: TemperatureScaler() for name in modality_names}
    
    def fit_all(
        self,
        modality_logits: dict,  # {modality: logits}
        labels: np.ndarray,
    ) -> dict:
        """Fit temperature for each modality."""
        temperatures = {}
        
        for modality, logits in modality_logits.items():
            if modality in self.scalers:
                t = self.scalers[modality].fit(logits, labels)
                temperatures[modality] = t
        
        return temperatures
    
    def calibrate_result(self, modality: str, confidence: float) -> float:
        """Calibrate confidence from a specific modality."""
        if modality in self.scalers and self.scalers[modality]._fitted:
            return self.scalers[modality].calibrate_confidence(confidence)
        return confidence
