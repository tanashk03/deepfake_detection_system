"""
Ensemble Meta-Learner
Combines predictions from all detector modules using neural approaches.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class AttentionBasedFusion(nn.Module):
    """Attention mechanism for combining detector outputs."""

    def __init__(self, num_detectors: int, hidden_dim: int = 64):
        super(AttentionBasedFusion, self).__init__()

        self.attention_layers = nn.Sequential(
            nn.Linear(num_detectors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_detectors),
            nn.Softmax(dim=-1)
        )

    def forward(self, detector_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention-based fusion to detector outputs."""
        batch_size, num_detectors, num_classes = detector_outputs.shape

        # Compute attention weights based on confidence
        detector_confidences = torch.softmax(detector_outputs, dim=-1).max(dim=-1)[0]
        attention_weights = self.attention_layers(detector_confidences)

        # Apply weighted combination
        weighted_outputs = torch.sum(
            detector_outputs * attention_weights.unsqueeze(-1), dim=1
        )

        return weighted_outputs, attention_weights

class DeepfakeEnsemble(nn.Module):
    """
    Meta-learning ensemble that combines predictions from all detector modules
    using attention-based fusion and uncertainty quantification.
    """

    def __init__(self, num_detectors: int = 4, num_classes: int = 2, use_uncertainty: bool = True):
        super(DeepfakeEnsemble, self).__init__()

        self.num_detectors = num_detectors
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty

        # Attention-based fusion
        self.attention_fusion = AttentionBasedFusion(num_detectors)

        # Neural meta-learner
        self.neural_meta = nn.Sequential(
            nn.Linear(num_detectors * num_classes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

        # Uncertainty estimation
        if use_uncertainty:
            self.uncertainty_estimator = nn.Sequential(
                nn.Linear(num_detectors * num_classes, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.uncertainty_estimator = None

        # Performance tracking
        self.detector_performance = nn.Parameter(
            torch.ones(num_detectors) / num_detectors, requires_grad=False
        )

    def forward(self, detector_outputs: List[torch.Tensor], return_uncertainty: bool = False, 
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble meta-learner."""
        # Convert to probabilities
        detector_probs = [torch.softmax(output, dim=-1) for output in detector_outputs]
        stacked_predictions = torch.stack(detector_probs, dim=1)

        # Apply attention-based fusion
        weighted_outputs, attention_weights = self.attention_fusion(stacked_predictions)

        # Flatten for neural meta-learner
        flattened = stacked_predictions.view(stacked_predictions.size(0), -1)

        # Neural meta-learning
        neural_output = self.neural_meta(flattened)

        # Combine outputs
        alpha, beta = 0.7, 0.3
        final_output = alpha * neural_output + beta * weighted_outputs

        # Prepare result
        result = {'predictions': final_output}

        # Add uncertainty if requested
        if return_uncertainty and self.uncertainty_estimator is not None:
            uncertainty = self.uncertainty_estimator(flattened)
            result['uncertainty'] = uncertainty

        # Add attention weights if requested
        if return_attention:
            result['attention_weights'] = attention_weights
            result['detector_performance'] = self.detector_performance

        return result

    def predict_with_confidence(self, detector_outputs: List[torch.Tensor], 
                              confidence_threshold: float = 0.8) -> Dict[str, torch.Tensor]:
        """Make predictions with confidence estimates."""
        with torch.no_grad():
            result = self.forward(detector_outputs, return_uncertainty=True, return_attention=True)

            predictions = result['predictions']
            probabilities = torch.softmax(predictions, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]

            # Quality indicators
            high_confidence_mask = confidence >= confidence_threshold

            return {
                'predictions': predictions,
                'probabilities': probabilities, 
                'confidence': confidence,
                'high_confidence': high_confidence_mask,
                'uncertainty': result.get('uncertainty'),
                'attention_weights': result.get('attention_weights')
            }

def create_ensemble_model(num_detectors: int = 4, num_classes: int = 2) -> DeepfakeEnsemble:
    """Factory function to create an ensemble model."""
    return DeepfakeEnsemble(num_detectors=num_detectors, num_classes=num_classes)
