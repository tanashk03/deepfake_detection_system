"""
MLflow Tracking for Luminark

Local-first experiment tracking for deepfake detection.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Tracking disabled.")


class LuminarkTracker:
    """
    Experiment tracking for Luminark inference.
    
    Local-first: uses local file store by default.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "luminark-detection",
    ):
        """
        Initialize tracker.
        
        Args:
            tracking_uri: MLflow tracking URI (default: local ./mlruns)
            experiment_name: Experiment name
        """
        self.enabled = MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        
        if not self.enabled:
            return
        
        # Set tracking URI (local by default)
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            f"file://{Path.cwd() / 'mlruns'}"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            mlflow.set_experiment(experiment_name)
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            logger.info(f"MLflow tracking enabled: {self.tracking_uri}")
        except Exception as e:
            logger.warning(f"Could not set up MLflow experiment: {e}")
            self.enabled = False
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Start a new tracking run.
        
        Args:
            run_name: Optional run name
            tags: Optional tags
            
        Returns:
            Run ID or None
        """
        if not self.enabled:
            return None
        
        try:
            run = mlflow.start_run(
                run_name=run_name or f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=tags or {},
            )
            return run.info.run_id
        except Exception as e:
            logger.warning(f"Could not start MLflow run: {e}")
            return None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Could not log params: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Could not log metrics: {e}")
    
    def log_inference(
        self,
        video_path: str,
        result: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Log a complete inference run.
        
        Args:
            video_path: Path to analyzed video
            result: Inference result dict
            model_config: Optional model configuration
            
        Returns:
            Run ID or None
        """
        if not self.enabled:
            return None
        
        run_id = self.start_run(
            run_name=f"inference_{Path(video_path).stem}",
            tags={
                "video_name": Path(video_path).name,
                "verdict": result.get("verdict", "unknown"),
            }
        )
        
        if not run_id:
            return None
        
        try:
            # Log params
            params = {
                "video_path": str(video_path),
                **(model_config or {})
            }
            self.log_params(params)
            
            # Log metrics
            metrics = {
                "confidence": float(result.get("confidence", 0)),
                "score": float(result.get("score", 0)),
                "uncertainty": float(result.get("uncertainty", 0)),
            }
            
            if result.get("processing_time_ms"):
                metrics["processing_time_ms"] = float(result["processing_time_ms"])
            
            # Log modality contributions
            contributions = result.get("modality_contributions", {})
            for modality, value in contributions.items():
                metrics[f"contrib_{modality}"] = float(value)
            
            self.log_metrics(metrics)
            
            # Log verdict as artifact
            mlflow.log_text(
                result.get("explanation", ""),
                "explanation.txt"
            )
            
            return run_id
            
        except Exception as e:
            logger.warning(f"Could not log inference: {e}")
            return None
        finally:
            self.end_run()
    
    def end_run(self) -> None:
        """End the current run."""
        if not self.enabled:
            return
        
        try:
            mlflow.end_run()
        except Exception:
            pass


# Global tracker instance
_tracker: Optional[LuminarkTracker] = None


def get_tracker() -> LuminarkTracker:
    """Get or create global tracker."""
    global _tracker
    if _tracker is None:
        _tracker = LuminarkTracker()
    return _tracker


def log_inference(video_path: str, result: Dict[str, Any]) -> Optional[str]:
    """Convenience function to log an inference."""
    return get_tracker().log_inference(video_path, result)
