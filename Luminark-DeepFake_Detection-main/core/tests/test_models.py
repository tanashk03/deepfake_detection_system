"""
Unit tests for core detection models.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""
    
    def test_to_dict(self):
        from core.models.base import DetectionResult
        
        result = DetectionResult(
            score=0.75,
            confidence=0.85,
            uncertainty=0.12,
            modality='video',
            model_name='test_model',
        )
        
        d = result.to_dict()
        
        assert d['score'] == 0.75
        assert d['confidence'] == 0.85
        assert d['uncertainty'] == 0.12
        assert d['modality'] == 'video'
        assert d['model_name'] == 'test_model'


class TestBaseDetector:
    """Tests for BaseDetector abstract class."""
    
    def test_output_to_score_binary(self):
        from core.models.base import BaseDetector
        
        class MockDetector(BaseDetector):
            def load_model(self): pass
            def preprocess(self, x): return x
            def forward(self, x): return x
        
        detector = MockDetector('test', 'video')
        
        # [fake, real] probabilities
        output = np.array([0.3, 0.7])
        score = detector._output_to_score(output)
        
        # score = real - fake = 0.7 - 0.3 = 0.4
        assert abs(score - 0.4) < 1e-6
    
    def test_uncertainty_to_confidence(self):
        from core.models.base import BaseDetector
        
        class MockDetector(BaseDetector):
            def load_model(self): pass
            def preprocess(self, x): return x
            def forward(self, x): return x
        
        detector = MockDetector('test', 'video')
        
        # Low uncertainty = high confidence
        conf_low_unc = detector._uncertainty_to_confidence(0.1)
        conf_high_unc = detector._uncertainty_to_confidence(0.5)
        
        assert conf_low_unc > conf_high_unc
        assert 0 <= conf_low_unc <= 1
        assert 0 <= conf_high_unc <= 1


class TestUncertaintyUtils:
    """Tests for uncertainty estimation utilities."""
    
    def test_compute_entropy(self):
        from core.utils.uncertainty import compute_entropy
        
        # Uniform distribution has max entropy
        uniform = np.array([0.5, 0.5])
        entropy_uniform = compute_entropy(uniform)
        
        # Confident prediction has low entropy
        confident = np.array([0.99, 0.01])
        entropy_confident = compute_entropy(confident)
        
        assert entropy_uniform > entropy_confident
    
    def test_ensemble_uncertainty(self):
        from core.utils.uncertainty import ensemble_uncertainty
        
        # All agree
        predictions_agree = [0.8, 0.82, 0.79]
        mean_agree, std_agree = ensemble_uncertainty(predictions_agree)
        
        # Disagree
        predictions_disagree = [0.9, 0.1, 0.5]
        mean_disagree, std_disagree = ensemble_uncertainty(predictions_disagree)
        
        assert std_disagree > std_agree
    
    def test_agreement_score(self):
        from core.utils.uncertainty import agreement_score
        
        # All positive
        all_positive = [0.5, 0.6, 0.7]
        assert agreement_score(all_positive) == 1.0
        
        # Mixed
        mixed = [0.5, -0.5, 0.3]
        assert agreement_score(mixed) == 2/3


class TestVideoAnalyzer:
    """Tests for video analysis module."""
    
    def test_preprocess_shape(self):
        from core.models.video import VideoAnalyzer
        
        analyzer = VideoAnalyzer()
        
        # Mock frames: 10 frames, 480x640, BGR
        frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        
        processed = analyzer.preprocess(frames)
        
        # Should be (10, 3, 299, 299)
        assert processed.shape == (10, 3, 299, 299)
        assert processed.dtype == np.float32
    
    def test_temporal_consistency(self):
        from core.models.video import VideoAnalyzer
        
        analyzer = VideoAnalyzer()
        
        # Static frames (high consistency)
        static_frames = np.zeros((20, 100, 100, 3), dtype=np.uint8)
        score, suspicious = analyzer.analyze_temporal_consistency(static_frames)
        
        assert score >= 0.8
        assert len(suspicious) == 0


class TestAudioAnalyzer:
    """Tests for audio analysis module."""
    
    def test_preprocess_mfcc(self):
        from core.models.audio import AudioAnalyzer
        
        analyzer = AudioAnalyzer()
        
        # Mock 1 second of audio at 16kHz
        audio = np.random.randn(16000).astype(np.float32)
        
        processed = analyzer.preprocess(audio)
        
        # Should be (1, n_mfcc, time)
        assert processed.shape[0] == 1
        assert processed.shape[1] == analyzer.n_mfcc


class TestRppgAnalyzer:
    """Tests for rPPG analysis module."""
    
    def test_bandpass_filter(self):
        from core.models.rppg import RppgAnalyzer
        
        analyzer = RppgAnalyzer(fps=30.0)
        
        # Create signal with known frequencies
        t = np.linspace(0, 10, 300)  # 10 seconds at 30 fps
        signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz (60 BPM)
        
        filtered = analyzer._bandpass_filter(signal, 30.0, 0.7, 4.0)
        
        # Signal should pass through (1 Hz is in 0.7-4 Hz range)
        assert np.std(filtered) > 0
    
    def test_pulse_analysis(self):
        from core.models.rppg import RppgAnalyzer
        
        analyzer = RppgAnalyzer(fps=30.0)
        
        # Create mock pulse signal at ~72 BPM (1.2 Hz)
        t = np.linspace(0, 10, 300)
        pulse = np.sin(2 * np.pi * 1.2 * t)
        
        hr, conf, snr = analyzer._analyze_pulse(pulse)
        
        # Should detect HR around 72 BPM
        assert 60 <= hr <= 90


class TestLipsyncAnalyzer:
    """Tests for lip-sync analysis module."""
    
    def test_sync_offset_calculation(self):
        from core.models.lipsync import LipsyncAnalyzer
        
        analyzer = LipsyncAnalyzer()
        
        # Mock aligned audio and video
        audio = np.random.randn(16000)  # 1 second
        frames = np.random.randint(0, 255, (30, 100, 100, 3), dtype=np.uint8)
        
        offset, conf = analyzer.compute_sync_offset(audio, frames)
        
        # Should be within reasonable range
        assert -500 < offset < 500


class TestLateFusionEnsemble:
    """Tests for late fusion ensemble."""
    
    def test_fuse_all_modalities(self):
        from core.models.fusion import LateFusionEnsemble
        from core.models.base import DetectionResult
        
        fusion = LateFusionEnsemble(content_type='talking_head')
        
        results = {
            'video': DetectionResult(
                score=0.5, confidence=0.8, uncertainty=0.1,
                modality='video', model_name='test'
            ),
            'audio': DetectionResult(
                score=0.4, confidence=0.7, uncertainty=0.15,
                modality='audio', model_name='test'
            ),
            'rppg': DetectionResult(
                score=0.6, confidence=0.6, uncertainty=0.2,
                modality='rppg', model_name='test'
            ),
            'lipsync': DetectionResult(
                score=0.3, confidence=0.75, uncertainty=0.12,
                modality='lipsync', model_name='test'
            ),
        }
        
        fused = fusion.fuse(results)
        
        assert fused.verdict in ['REAL', 'FAKE', 'INCONCLUSIVE']
        assert 0 <= fused.confidence <= 100
        assert -1 <= fused.score <= 1
    
    def test_verdict_thresholds(self):
        from core.models.fusion import LateFusionEnsemble
        from core.models.base import DetectionResult
        
        fusion = LateFusionEnsemble()
        
        # Strong fake signals
        fake_results = {
            'video': DetectionResult(
                score=-0.8, confidence=0.9, uncertainty=0.05,
                modality='video', model_name='test'
            ),
        }
        
        fused = fusion.fuse(fake_results)
        assert fused.verdict == 'FAKE'
        
        # Strong real signals
        real_results = {
            'video': DetectionResult(
                score=0.8, confidence=0.9, uncertainty=0.05,
                modality='video', model_name='test'
            ),
        }
        
        fused = fusion.fuse(real_results)
        assert fused.verdict == 'REAL'
    
    def test_inconclusive_on_low_confidence(self):
        from core.models.fusion import LateFusionEnsemble
        from core.models.base import DetectionResult
        
        fusion = LateFusionEnsemble()
        
        # High uncertainty = inconclusive
        uncertain_results = {
            'video': DetectionResult(
                score=0.1, confidence=0.3, uncertainty=0.5,
                modality='video', model_name='test'
            ),
        }
        
        fused = fusion.fuse(uncertain_results)
        assert fused.verdict == 'INCONCLUSIVE'


class TestIntegration:
    """Integration tests for full pipeline."""
    
    @pytest.mark.slow
    def test_full_pipeline_with_mock_video(self):
        """Test full pipeline with synthetic data."""
        from core.infer import DeepfakeDetector, InferenceConfig
        import tempfile
        import cv2
        
        # Create a small test video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        # Write minimal video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (100, 100))
        
        for _ in range(30):  # 1 second
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Test detection
        config = InferenceConfig(
            device='cpu',
            with_uncertainty=False,  # Faster
            max_frames=10,
        )
        
        detector = DeepfakeDetector(config)
        result = detector.analyze(video_path)
        
        assert result.verdict in ['REAL', 'FAKE', 'INCONCLUSIVE']
        assert 0 <= result.confidence <= 100
        
        # Cleanup
        import os
        os.unlink(video_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
