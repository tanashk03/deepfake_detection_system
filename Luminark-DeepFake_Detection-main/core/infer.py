"""
Luminark Core Inference Pipeline

Main entry point for multimodal deepfake detection.
Single command → probability + uncertainty + per-modality contribution.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from .models.videomae import VideoMAEAnalyzer
from .models.wavlm import WavLMAnalyzer
from .models import (
    RppgAnalyzer,
    LipsyncAnalyzer,
    AnimationDetector,
    LateFusionEnsemble,
    FusionResult,
    DetectionResult,
)

# Custom Kaggle-trained models
from .models.spatial import create_efficientnet_detector as create_spatial_detector
from .models.temporal import create_temporal_efficientnet_detector as create_temporal_detector
from .models.frequency import create_frequency_efficientnet_detector as create_frequency_detector
from .models.physiological import create_physiological_cnn_detector as create_physiological_detector
from .models.ensemble import create_ensemble_model
import torch

VideoAnalyzer = VideoMAEAnalyzer
AudioAnalyzer = WavLMAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    
    # Device settings
    device: str = 'cpu'
    
    # Model settings
    video_model_path: Optional[str] = None
    audio_model_path: Optional[str] = None
    lipsync_model_path: Optional[str] = None
    
    # Processing settings
    frame_sample_rate: int = 5  # FPS for frame extraction
    max_frames: int = 150       # Max frames to process
    audio_sample_rate: int = 16000
    
    # Uncertainty settings
    mc_dropout_samples: int = 10
    with_uncertainty: bool = True
    
    # Content type for fusion weights
    content_type: str = 'talking_head'


class DeepfakeDetector:
    """
    Production deepfake detection pipeline.
    
    Orchestrates multimodal analysis and fusion for CPU inference.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize detector with optional configuration.
        """
        self.config = config or InferenceConfig()
        
        # Initialize detectors (lazy loading)
        self._animation_detector = None  # Animation/Cartoon detection
        self._spatial = None
        self._temporal = None
        self._frequency = None
        self._physiological = None
        
        # Keep existing for fallback/augmentation
        self._video_analyzer = None # VideoMAE
        self._audio_analyzer = None # WavLM
        self._rppg_analyzer = None
        self._lipsync_analyzer = None
        self._fusion = None
        self._ensemble = None
        
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of models."""
        if self._initialized:
            return
        
        logger.info("Initializing Luminark Multi-Modal Pipeline...")
        
        # --- 1. Custom Architectures ---
        logger.info("Loading Advanced Detectors (Spatial, Temporal, Frequency, Physiological)...")
        try:
            # Resolve project root (infer.py is in core/, so parent of parent)
            # Actually infer.py is in core/, so core -> luminark. 
            project_root = Path(__file__).parent.parent.resolve()
            models_dir = project_root / "models"
            
            logger.info(f"🔍 Looking for models in: {models_dir}")
            logger.info(f"🔍 Models found: {list(models_dir.glob('*.pt')) if models_dir.exists() else 'DIRECTORY NOT FOUND'}")
            
            self._spatial = create_spatial_detector().to(self.config.device)
            spatial_weights = models_dir / "spatial_finetuned.pt"
            if spatial_weights.exists():
                try:
                    self._spatial.load_state_dict(torch.load(spatial_weights, map_location=self.config.device))
                    logger.info("✅ Loaded trained spatial weights")
                except Exception as e:
                    logger.warning(f"Could not load spatial weights: {e}")
            self._spatial.eval()
            
            self._temporal = create_temporal_detector().to(self.config.device)
            temporal_weights = models_dir / "temporal_finetuned.pt"
            if temporal_weights.exists():
                try:
                    self._temporal.load_state_dict(torch.load(temporal_weights, map_location=self.config.device))
                    logger.info("✅ Loaded trained temporal weights")
                except Exception as e:
                    logger.warning(f"Could not load temporal weights: {e}")
            self._temporal.eval()
            
            self._frequency = create_frequency_detector().to(self.config.device)
            frequency_weights = models_dir / "frequency_finetuned.pt"
            if frequency_weights.exists():
                try:
                    self._frequency.load_state_dict(torch.load(frequency_weights, map_location=self.config.device))
                    logger.info("✅ Loaded trained frequency weights")
                except Exception as e:
                    logger.warning(f"Could not load frequency weights: {e}")
            self._frequency.eval()
            
            self._physiological = create_physiological_detector().to(self.config.device)
            # Load trained weights if available
            physio_weights_path = models_dir / "physiological_finetuned.pt"
            if physio_weights_path.exists():
                try:
                    self._physiological.load_state_dict(torch.load(physio_weights_path, map_location=self.config.device))
                    logger.info("✅ Loaded trained physiological weights")
                except Exception as e:
                    logger.warning(f"Could not load physiological weights: {e}")
            self._physiological.eval()
            
            # Neural Ensemble (Meta-Learner)
            self._ensemble = create_ensemble_model(num_detectors=5).to(self.config.device)
            self._ensemble.eval()
            
        except Exception as e:
            logger.warning(f"Failed to load custom models: {e}. Falling back to standard pipeline.")

        # --- 2. Standard Analyzers ---
        self._video_analyzer = VideoAnalyzer(
            model_path=self.config.video_model_path,
            device=self.config.device,
            frame_sample_rate=self.config.frame_sample_rate,
            mc_dropout_samples=self.config.mc_dropout_samples,
        )
        
        self._audio_analyzer = AudioAnalyzer(
            model_path=self.config.audio_model_path,
            device=self.config.device,
            sample_rate=self.config.audio_sample_rate,
            mc_dropout_samples=self.config.mc_dropout_samples,
        )
        
        self._rppg_analyzer = RppgAnalyzer(
            device=self.config.device,
            mc_dropout_samples=self.config.mc_dropout_samples // 2,
        )
        # self._rppg_analyzer = None
        
        self._lipsync_analyzer = LipsyncAnalyzer(
            model_path=self.config.lipsync_model_path,
            device=self.config.device,
            mc_dropout_samples=self.config.mc_dropout_samples,
        )
        # self._lipsync_analyzer = None
        
        # --- 3. VideoMAE Analyzer (SOTA from HuggingFace) ---
        try:
            self._videomae_analyzer = VideoMAEAnalyzer(
                device=self.config.device,
                mc_dropout_samples=self.config.mc_dropout_samples,
            )
            self._videomae_analyzer.load_model()
            logger.info("✅ VideoMAE analyzer loaded successfully")
        except Exception as e:
            logger.warning(f"❌ VideoMAE analyzer failed to load: {e}")
            self._videomae_analyzer = None
        
        self._fusion = LateFusionEnsemble(
            content_type=self.config.content_type
        )
        
        # --- 4. Animation/Cartoon Detector (NEW) ---
        try:
            self._animation_detector = AnimationDetector(device=self.config.device)
            logger.info("✅ Animation detector initialized successfully")
        except Exception as e:
            logger.warning(f"❌ Animation detector failed to initialize: {e}")
            self._animation_detector = None
        
        self._initialized = True
        logger.info("Pipeline initialized")
    
    def analyze(
        self,
        video_path: Union[str, Path],
        with_uncertainty: Optional[bool] = None,
        progress_callback: Optional[callable] = None,
    ) -> FusionResult:
        """
        Analyze a video file for deepfake indicators.
        """
        start_time = time.time()
        
        if progress_callback:
            progress_callback("Initializing analysis", 0)

        self._ensure_initialized()
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Extract video and audio
        if progress_callback:
            progress_callback("Extracting media frames", 10)
            
        frames, audio, fps = self._extract_media(video_path)
        
        # --- EDGE CASE HANDLING ---
        if len(frames) == 0:
            logger.error("No frames extracted from video")
            return self._fusion._create_inconclusive("Video file could not be read or is empty")
        
        if len(frames) < 5:
            logger.warning(f"Very short video: only {len(frames)} frames. Analysis may be unreliable.")
        
        if len(frames) > 300:
            logger.info(f"Long video ({len(frames)} frames). Subsampling for efficiency.")
            # Subsample to max 300 frames evenly distributed
            indices = np.linspace(0, len(frames)-1, 300, dtype=int)
            frames = frames[indices]
        
        # --- ANIMATION/CARTOON DETECTION (NEW - Early Check) ---
        if progress_callback:
            progress_callback("Checking for animation/cartoon content", 15)
        
        if self._animation_detector is not None:
            try:
                is_animated, animation_confidence = self._animation_detector.detect(frames, sample_rate=5)
                
                if is_animated:
                    # Animation detected - immediately return FAKE verdict
                    logger.warning(
                        f"⚠️  ANIMATION DETECTED: confidence={animation_confidence:.2%}"
                    )
                    processing_time_ms = (time.time() - start_time) * 1000
                    
                    # Return FAKE verdict with high confidence
                    return FusionResult(
                        verdict='FAKE',
                        confidence=95,  # Very high confidence for animation
                        explanation='Content appears to be animated, cartoon, anime, or CGI-generated. This is classified as FAKE.',
                        score=0.85,  # Strong FAKE signal
                        uncertainty=0.05,  # Very certain
                        modality_contributions={'animation': 1.0},
                        raw_results={'animation': DetectionResult(
                            score=0.85,
                            confidence=animation_confidence,
                            uncertainty=0.05,
                            modality='animation',
                            model_name='animation_detector'
                        )},
                        processing_time_ms=processing_time_ms,
                    )
            except Exception as e:
                logger.warning(f"Animation detection failed (non-blocking): {e}")
        
        # Determine content type
        has_audio = audio is not None and len(audio) > 0
        content_type = 'talking_head' if has_audio else 'silent_video'
        self._fusion.content_type = content_type
        self._fusion.weights = LateFusionEnsemble.WEIGHTS[content_type]
        
        # Run modality-specific analysis
        with_unc = with_uncertainty if with_uncertainty is not None else self.config.with_uncertainty
        
        results: Dict[str, DetectionResult] = {}
        
        # --- Run Standard Analyzers (Reliable Baseline) ---
        if progress_callback:
            progress_callback("Running standard models (Video/Audio)", 30)

        # These models can handle ANY resolution - preserve original quality!
        logger.info(f"Running standard analysis at original resolution ({frames.shape[1]}x{frames.shape[2]})...")
        try:
            results['video'] = self._video_analyzer.predict(frames, with_uncertainty=with_unc)
        except Exception as e:
            logger.warning(f"Video analyzer failed: {e}")
        
        # VideoMAE transformer analysis
        if self._videomae_analyzer is not None:
            try:
                videomae_result = self._videomae_analyzer.predict(frames, with_uncertainty=with_unc)
                videomae_result.score = -videomae_result.score
                results['videomae'] = videomae_result
                logger.debug(f"VideoMAE: score={videomae_result.score:.3f}")
            except Exception as e:
                logger.warning(f"VideoMAE analyzer failed: {e}")
            
        try:
            results['rppg'] = self._rppg_analyzer.predict(frames, with_uncertainty=with_unc)
        except Exception as e:
            logger.warning(f"rPPG analyzer failed: {e}")
        
        if has_audio:
            try:
                results['audio'] = self._audio_analyzer.predict(audio, with_uncertainty=with_unc)
            except Exception as e:
                logger.warning(f"Audio analyzer failed: {e}")
            try:
                results['lipsync'] = self._lipsync_analyzer.predict({'audio': audio, 'frames': frames}, with_uncertainty=with_unc)
            except Exception as e:
                logger.warning(f"Lipsync analyzer failed: {e}")

        # --- Run Advanced Detectors (Requires 224x224) ---
        if progress_callback:
            progress_callback("Running advanced spatial models", 60)

        # Note: These return raw tensors, mapping to DetectionResult
        import cv2 as cv2_local  # Local import for robustness
        
        logger.info("Preparing frames for advanced models (resizing to 224x224)...")
        try:
            # CRITICAL: Resize all frames to 224x224 ONLY for advanced models
            resized_frames = []
            for frame in frames:
                resized = cv2_local.resize(frame, (224, 224), interpolation=cv2_local.INTER_LINEAR)
                # Ensure RGB format
                if len(resized.shape) == 2:  # Grayscale
                    resized = cv2_local.cvtColor(resized, cv2_local.COLOR_GRAY2RGB)
                elif resized.shape[2] == 4:  # RGBA
                    resized = cv2_local.cvtColor(resized, cv2_local.COLOR_RGBA2RGB)
                else:  # BGR (standard OpenCV)
                    resized = cv2_local.cvtColor(resized, cv2_local.COLOR_BGR2RGB)
                resized_frames.append(resized)
            
            resized_frames = np.array(resized_frames)
            frames_tensor = torch.FloatTensor(resized_frames).permute(0, 3, 1, 2)  # NHWC -> NCHW
            frames_tensor = frames_tensor.float() / 255.0  # Normalize 0-1
            
            # ImageNet normalization (CRITICAL: matches Kaggle training)
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_tensor = (frames_tensor - imagenet_mean) / imagenet_std
            
            if self.config.device == 'cuda':
                frames_tensor = frames_tensor.cuda()
                
            # 1. Spatial (Single Frame Analysis - middle frame)
            try:
                with torch.no_grad():
                    middle_frame = frames_tensor[len(frames_tensor)//2].unsqueeze(0)
                    spatial_logits = self._spatial(middle_frame)
                    spatial_probs = torch.softmax(spatial_logits, dim=1)
                    spatial_score = float(spatial_probs[0][0] - spatial_probs[0][1])
                    logger.debug(f"Spatial: score={spatial_score:.3f}")
                    
                    results['spatial'] = DetectionResult(
                        score=spatial_score,
                        confidence=float(spatial_probs.max()),
                        uncertainty=0.5,
                        modality='spatial',
                        model_name='vit_xception'
                    )
            except Exception as e:
                logger.warning(f"Spatial model failed: {e}")

            # 2. Temporal (Multi-Frame Analysis)
            if len(frames_tensor) >= 16:
                try:
                    with torch.no_grad():
                        indices = torch.linspace(0, len(frames_tensor)-1, 16).long()
                        sampled = frames_tensor[indices].unsqueeze(0)
                        temporal_logits = self._temporal(sampled)
                        temporal_probs = torch.softmax(temporal_logits, dim=1)
                        temporal_score = float(temporal_probs[0][1] - temporal_probs[0][0])
                        logger.debug(f"Temporal: score={temporal_score:.3f}")
                        
                        results['temporal'] = DetectionResult(
                            score=temporal_score,
                            confidence=float(temporal_probs.max()),
                            uncertainty=0.5,
                            modality='temporal',
                            model_name='videomae_temporal'
                        )
                except Exception as e:
                    logger.warning(f"Temporal model failed: {e}")
            
            # 3. Frequency (FFT/DCT Analysis - single middle frame)
            try:
                with torch.no_grad():
                    middle_frame = frames_tensor[len(frames_tensor)//2].unsqueeze(0)
                    freq_logits = self._frequency(middle_frame)
                    freq_probs = torch.softmax(freq_logits, dim=1)
                    freq_score = float(freq_probs[0][0] - freq_probs[0][1])
                    logger.debug(f"Frequency: score={freq_score:.3f}")
                    
                    results['frequency'] = DetectionResult(
                        score=freq_score,
                        confidence=float(freq_probs.max()),
                        uncertainty=0.5, 
                        modality='frequency',
                        model_name='fft_dct'
                    )
            except Exception as e:
                logger.warning(f"Frequency model failed: {e}")

            # 4. Physiological (CNN-LSTM rPPG) - Requires 224x224 input
            try:
                # Prepare frames at 224x224 (model trained with this size)
                physio_frames_list = []
                for frame in frames[:16]: # Limit to 16 frames
                    resized_physio = cv2_local.resize(frame, (224, 224), interpolation=cv2_local.INTER_LINEAR)
                    # Convert BGR to RGB
                    resized_physio = cv2_local.cvtColor(resized_physio, cv2_local.COLOR_BGR2RGB)
                    physio_frames_list.append(resized_physio)
                
                physio_frames_np = np.array(physio_frames_list)
                physio_tensor = torch.FloatTensor(physio_frames_np).permute(0, 3, 1, 2) # NHWC -> NCHW
                physio_tensor = physio_tensor.float() / 255.0
                
                if self.config.device == 'cuda':
                    physio_tensor = physio_tensor.cuda()
                    
                with torch.no_grad():
                    # Unsqueeze to add batch dim: (1, T, C, H, W)
                    physio_input = physio_tensor.unsqueeze(0)
                    physio_logits = self._physiological(physio_input)
                    physio_probs = torch.softmax(physio_logits, dim=1)
                    physio_score = float(physio_probs[0][0] - physio_probs[0][1])
                    logger.debug(f"Physiological: score={physio_score:.3f}")
                    
                    results['physiological'] = DetectionResult(
                        score=physio_score,
                        confidence=float(physio_probs.max()),
                        uncertainty=0.5,
                        modality='physiological',
                        model_name='cnn_lstm_rppg'
                    )
            except Exception as e:
                logger.warning(f"Physiological model failed: {e}")
                
        except Exception as e:
            logger.error(f"Advanced detection failed (preprocessing): {e}")

        
        # Fuse results
        # Use Weighted Fusion for now as it handles dynamic keys (spatial/temporal etc)
        # Note: We need to update weights in fusion.py to handle new keys, 
        # or map them to existing ones.
        # For now, we integrate them as 'auxiliary' signals or fallback to legacy fusion
        # but with new inputs available in the explanation.
        
        if progress_callback:
            progress_callback("Fusing results", 90)

        logger.info("Fusing multimodal results...")
        processing_time_ms = (time.time() - start_time) * 1000
        
        fusion_result = self._fusion.fuse(results, processing_time_ms)
        
        logger.info(f"Analysis complete: {fusion_result.verdict} ({fusion_result.confidence}%)")
        
        return fusion_result
    
    def _extract_media(
        self, video_path: Path
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Extract frames and audio from video.
        
        Returns:
            (frames, audio, fps)
        """
        import cv2
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling
        sample_interval = max(1, int(fps / self.config.frame_sample_rate))
        
        frames = []
        frame_idx = 0
        
        while len(frames) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        frames_array = np.array(frames)
        
        # Extract audio
        audio = self._extract_audio(video_path)
        
        return frames_array, audio, fps
    
    def _extract_audio(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Extract audio from video file.
        
        Returns:
            Audio signal as numpy array or None if no audio
        """
        try:
            import librosa
            
            # Load audio at target sample rate
            audio, sr = librosa.load(
                str(video_path),
                sr=self.config.audio_sample_rate,
                mono=True
            )
            
            if len(audio) < 1000:  # Less than ~60ms
                return None
            
            return audio
            
        except Exception as e:
            logger.warning(f"Could not extract audio: {e}")
            return None
    
    def get_detailed_results(
        self,
        video_path: Union[str, Path],
    ) -> Dict:
        """
        Get detailed internal results (for debugging/development).
        
        Not for production user-facing output.
        """
        result = self.analyze(video_path)
        
        return {
            'fusion_result': result.to_internal_dict(),
            'config': {
                'device': self.config.device,
                'content_type': self.config.content_type,
                'mc_samples': self.config.mc_dropout_samples,
            }
        }


def analyze_video(
    video_path: str,
    device: str = 'cpu',
    with_uncertainty: bool = True,
) -> FusionResult:
    """
    Convenience function for single video analysis.
    
    Args:
        video_path: Path to video file
        device: 'cpu' or 'cuda'
        with_uncertainty: Run MC dropout
        
    Returns:
        FusionResult
    """
    config = InferenceConfig(
        device=device,
        with_uncertainty=with_uncertainty,
    )
    detector = DeepfakeDetector(config)
    return detector.analyze(video_path)


# CLI entry point
def main():
    """Command-line interface for deepfake detection."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Luminark Deepfake Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.infer video.mp4
  python -m core.infer video.mp4 --detailed
  python -m core.infer video.mp4 --no-uncertainty
        """
    )
    
    parser.add_argument(
        'video',
        type=str,
        help='Path to video file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for inference (default: cpu)'
    )
    
    parser.add_argument(
        '--no-uncertainty',
        action='store_true',
        help='Disable uncertainty estimation (faster)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed internal results'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    config = InferenceConfig(
        device=args.device,
        with_uncertainty=not args.no_uncertainty,
    )
    
    detector = DeepfakeDetector(config)
    
    if args.detailed:
        result = detector.get_detailed_results(args.video)
    else:
        fusion_result = detector.analyze(args.video)
        result = fusion_result.to_internal_dict()
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*60)
        print("LUMINARK DEEPFAKE DETECTION RESULT")
        print("="*60)
        print(f"\nVerdict:      {result['verdict']}")
        print(f"Confidence:   {result['confidence']}%")
        print(f"\nExplanation:  {result['explanation']}")
        print(f"\nScore:        {result['score']:.4f}")
        print(f"Uncertainty:  {result['uncertainty']:.4f}")
        
        if 'modality_contributions' in result and result['modality_contributions']:
            print("\nModality Contributions:")
            for modality, contrib in result['modality_contributions'].items():
                bar = "█" * int(contrib * 20)
                print(f"  {modality:10s}: {bar} {contrib:.2%}")
        
        if result.get('processing_time_ms'):
            print(f"\nProcessing Time: {result['processing_time_ms']:.0f}ms")
        
        print("="*60 + "\n")


if __name__ == '__main__':
    main()
