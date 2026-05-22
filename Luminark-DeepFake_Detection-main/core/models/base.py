"""
Base Model Interface

All detection models inherit from BaseDetector and implement a common interface.
Includes uncertainty estimation via Monte Carlo dropout or ensemble variance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DetectionResult:
    """Result from a single detection model."""
    
    # Core outputs
    score: float  # -1.0 (fake) to +1.0 (real)
    confidence: float  # 0.0 to 1.0 (how certain the model is)
    
    # Uncertainty estimation
    uncertainty: float  # Standard deviation of predictions
    
    # Metadata
    modality: str  # 'video', 'audio', 'rppg', 'lipsync'
    model_name: str
    
    # Optional detailed outputs
    raw_probabilities: Optional[np.ndarray] = None
    feature_importance: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'score': round(self.score, 4),
            'confidence': round(self.confidence, 4),
            'uncertainty': round(self.uncertainty, 4),
            'modality': self.modality,
            'model_name': self.model_name,
        }


class BaseDetector(ABC):
    """
    Abstract base class for all detection models.
    
    All detectors must implement:
    - preprocess(): Prepare input data
    - forward(): Run inference
    - predict(): Full pipeline with uncertainty
    """
    
    def __init__(
        self,
        model_name: str,
        modality: str,
        device: str = 'cpu',
        mc_dropout_samples: int = 10,
    ):
        self.model_name = model_name
        self.modality = modality
        self.device = device
        self.mc_dropout_samples = mc_dropout_samples
        self._model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model weights. Called lazily on first inference."""
        pass
    
    @abstractmethod
    def preprocess(self, input_data) -> np.ndarray:
        """
        Preprocess raw input into model-ready format.
        
        Args:
            input_data: Raw input (video frames, audio signal, etc.)
            
        Returns:
            Preprocessed numpy array ready for inference
        """
        pass
    
    @abstractmethod
    def forward(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Run single forward pass.
        
        Args:
            preprocessed: Output from preprocess()
            
        Returns:
            Raw model output (logits or probabilities)
        """
        pass
    
    def predict(self, input_data, with_uncertainty: bool = True) -> DetectionResult:
        """
        Full prediction pipeline with uncertainty estimation.
        
        Args:
            input_data: Raw input data
            with_uncertainty: Whether to run MC dropout for uncertainty
            
        Returns:
            DetectionResult with score, confidence, and uncertainty
        """
        # Lazy model loading
        if not self._is_loaded:
            self.load_model()
            self._is_loaded = True
        
        # Preprocess
        preprocessed = self.preprocess(input_data)
        
        if with_uncertainty and self.mc_dropout_samples > 1:
            # Monte Carlo dropout for uncertainty estimation
            predictions = self._mc_dropout_inference(preprocessed)
            score, uncertainty = self._aggregate_predictions(predictions)
        else:
            # Single forward pass
            output = self.forward(preprocessed)
            score = self._output_to_score(output)
            uncertainty = 0.0
        
        # Convert uncertainty to confidence
        confidence = self._uncertainty_to_confidence(uncertainty)
        
        return DetectionResult(
            score=score,
            confidence=confidence,
            uncertainty=uncertainty,
            modality=self.modality,
            model_name=self.model_name,
            raw_probabilities=output if not with_uncertainty else None,
        )
    
    def _mc_dropout_inference(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Run multiple forward passes with dropout enabled.
        
        Returns array of shape (mc_dropout_samples,) with scores.
        """
        predictions = []
        for _ in range(self.mc_dropout_samples):
            output = self.forward(preprocessed)
            score = self._output_to_score(output)
            predictions.append(score)
        return np.array(predictions)
    
    def _aggregate_predictions(
        self, predictions: np.ndarray
    ) -> Tuple[float, float]:
        """
        Aggregate MC dropout predictions into mean and uncertainty.
        
        Returns:
            (mean_score, std_uncertainty)
        """
        mean_score = float(np.mean(predictions))
        std_uncertainty = float(np.std(predictions))
        return mean_score, std_uncertainty
    
    def _output_to_score(self, output: np.ndarray) -> float:
        """
        Convert raw model output to score in [-1, 1] range.
        
        Default: assumes output is [prob_fake, prob_real] and computes
        score = prob_real - prob_fake
        """
        if len(output.shape) > 1:
            output = output.squeeze()
        
        if output.shape[0] == 2:
            # Binary classification: [fake, real]
            prob_real = float(output[1])
            prob_fake = float(output[0])
            return prob_real - prob_fake
        else:
            # Single output, assume sigmoid
            prob = float(output[0])
            return 2 * prob - 1  # Map [0,1] to [-1,1]
    
    def _uncertainty_to_confidence(self, uncertainty: float) -> float:
        """
        Convert uncertainty (std) to confidence score [0, 1].
        
        Higher uncertainty = lower confidence.
        Uses exponential decay: confidence = exp(-k * uncertainty)
        """
        k = 3.0  # Decay rate
        confidence = np.exp(-k * uncertainty)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, modality={self.modality})"
