"""
Tests for accuracy improvement modules.
"""

import pytest
import numpy as np


class TestCalibration:
    """Tests for temperature scaling calibration."""
    
    def test_temperature_scaler_init(self):
        from core.utils.calibration import TemperatureScaler
        
        scaler = TemperatureScaler(initial_temperature=1.5)
        assert scaler.temperature == 1.5
        assert scaler._fitted is False
    
    def test_calibrate_confidence(self):
        from core.utils.calibration import TemperatureScaler
        
        scaler = TemperatureScaler(initial_temperature=2.0)
        
        # High temp should reduce extreme confidences
        conf = 0.95
        calibrated = scaler.calibrate_confidence(conf)
        assert calibrated < conf  # Should be lower due to T > 1
    
    def test_compute_ece(self):
        from core.utils.calibration import compute_ece
        
        # Perfect calibration case
        n = 100
        confidences = np.linspace(0.1, 0.9, n)
        predictions = (confidences > 0.5).astype(int)
        labels = predictions.copy()  # Perfect predictions
        
        metrics = compute_ece(confidences, predictions, labels)
        
        assert 0 <= metrics.ece <= 1
        assert 0 <= metrics.mce <= 1


class TestAdaptiveEnsemble:
    """Tests for uncertainty-weighted fusion."""
    
    def test_ensemble_init(self):
        from core.models.adaptive_ensemble import AdaptiveEnsemble
        
        ensemble = AdaptiveEnsemble(content_type='talking_head')
        assert ensemble.content_type == 'talking_head'
        assert 'video' in ensemble.base_weights
    
    def test_combine_results(self):
        from core.models.adaptive_ensemble import AdaptiveEnsemble
        from core.models.base import DetectionResult
        
        ensemble = AdaptiveEnsemble()
        
        results = {
            'video': DetectionResult(
                score=0.6, confidence=80, uncertainty=0.1,
                modality='video', model_name='test'
            ),
            'audio': DetectionResult(
                score=0.4, confidence=70, uncertainty=0.2,
                modality='audio', model_name='test'
            ),
        }
        
        combined = ensemble.combine(results)
        
        assert combined.verdict in ['REAL', 'FAKE', 'INCONCLUSIVE']
        assert 0 <= combined.confidence <= 100
        assert len(combined.modality_contributions) == 2
    
    def test_abstention_on_high_uncertainty(self):
        from core.models.adaptive_ensemble import AdaptiveEnsemble
        from core.models.base import DetectionResult
        
        ensemble = AdaptiveEnsemble()
        
        # High uncertainty results
        results = {
            'video': DetectionResult(
                score=0.2, confidence=40, uncertainty=0.6,
                modality='video', model_name='test'
            ),
        }
        
        combined = ensemble.combine(results)
        
        # Should abstain due to high uncertainty
        assert combined.verdict == 'INCONCLUSIVE' or combined.abstention_triggered
    
    def test_escalation_decision(self):
        from core.models.adaptive_ensemble import AdaptiveEnsemble, EnsembleResult
        
        ensemble = AdaptiveEnsemble()
        
        # High uncertainty result
        result = EnsembleResult(
            score=0.1,
            confidence=50,
            uncertainty=0.5,
            verdict='INCONCLUSIVE',
            explanation='Test',
            modality_contributions={},
            abstention_triggered=True,
        )
        
        assert ensemble.should_escalate_to_cloud(result) is True


class TestAdversarialTraining:
    """Tests for adversarial training components."""
    
    def test_fgsm_attack(self):
        import torch
        import torch.nn as nn
        from core.train_adv import FGSMAttack
        
        # Simple model
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 2),
        )
        
        attack = FGSMAttack(model, eps=0.03)
        
        x = torch.rand(1, 3, 32, 32)
        y = torch.tensor([0])
        
        x_adv = attack.perturb(x, y)
        
        assert x_adv.shape == x.shape
        assert torch.max(torch.abs(x_adv - x)) <= 0.03 + 1e-6
    
    def test_pgd_attack(self):
        import torch
        import torch.nn as nn
        from core.train_adv import PGDAttack
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 2),
        )
        
        attack = PGDAttack(model, eps=0.03, steps=3)
        
        x = torch.rand(1, 3, 32, 32)
        y = torch.tensor([0])
        
        x_adv = attack.perturb(x, y)
        
        assert x_adv.shape == x.shape


class TestXAI:
    """Tests for explainability modules."""
    
    def test_explanation_artifact(self):
        from core.xai.explainability import ExplanationArtifact
        
        artifact = ExplanationArtifact(
            attention_summary='Test attention',
            audio_summary='Test audio',
            rationale='Test rationale',
            modality_scores={'video': 0.6, 'audio': 0.4},
            confidence=0.85,
        )
        
        d = artifact.to_dict()
        
        assert 'rationale' in d
        assert d['confidence'] == 0.85
    
    def test_rationale_generator(self):
        from core.xai.explainability import RationaleGenerator
        
        gen = RationaleGenerator()
        
        rationale = gen.generate(
            verdict='FAKE',
            confidence=90,
            modality_summaries={'video': 'High attention on lower face'},
            top_modality='video',
        )
        
        assert len(rationale) > 0
        assert isinstance(rationale, str)


class TestEvalReliable:
    """Tests for evaluation script."""
    
    def test_compute_auc(self):
        from core.eval_reliable import compute_auc
        
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])
        
        auc = compute_auc(y_true, y_scores)
        
        assert 0.9 <= auc <= 1.0  # Should be high for this case
    
    def test_compute_eer(self):
        from core.eval_reliable import compute_eer
        
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        eer = compute_eer(y_true, y_scores)
        
        assert 0 <= eer <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
