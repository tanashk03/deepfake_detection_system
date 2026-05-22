# Luminark Core ML Modules

> Multimodal deepfake detection with uncertainty estimation

## Quick Start

```bash
# Single command detection
python -m core.infer video.mp4
```

Output:
```
============================================================
LUMINARK DEEPFAKE DETECTION RESULT
============================================================

Verdict:      FAKE
Confidence:   87%

Explanation:  Visual analysis detected manipulation artifacts in facial regions. 
              Lip movements do not align with speech.

Score:        -0.6234
Uncertainty:  0.0891

Modality Contributions:
  video     : ████████████████ 42.15%
  audio     : ██████████ 25.33%
  lipsync   : ████████ 21.89%
  rppg      : ████ 10.63%

Processing Time: 45230ms
============================================================
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Video     │     │   Audio     │     │    rPPG     │     │  Lip-Sync   │
│  Analyzer   │     │  Analyzer   │     │  Analyzer   │     │  Analyzer   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  Late Fusion    │
                          │   Ensemble      │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Verdict + Unc.  │
                          └─────────────────┘
```

## Modules

| Module | File | Detection Target |
|--------|------|------------------|
| Video | `models/video.py` | Face manipulation artifacts |
| Audio | `models/audio.py` | Synthetic voice signatures |
| rPPG | `models/rppg.py` | Physiological signal absence |
| Lip-Sync | `models/lipsync.py` | Audio-visual desync |
| Fusion | `models/fusion.py` | Weighted ensemble |

## API Usage

```python
from core.infer import DeepfakeDetector, InferenceConfig

# Configure
config = InferenceConfig(
    device='cpu',              # or 'cuda'
    with_uncertainty=True,     # MC dropout
    content_type='talking_head'
)

# Detect
detector = DeepfakeDetector(config)
result = detector.analyze("video.mp4")

# Results
print(result.verdict)       # 'REAL', 'FAKE', 'INCONCLUSIVE'
print(result.confidence)    # 0-100
print(result.uncertainty)   # 0.0-1.0
print(result.explanation)   # Human-readable
```

## Uncertainty Estimation

All models use Monte Carlo dropout for uncertainty:

```python
# High confidence detection
{
    "verdict": "FAKE",
    "confidence": 92,
    "uncertainty": 0.04
}

# Uncertain - returns INCONCLUSIVE
{
    "verdict": "INCONCLUSIVE",
    "confidence": 38,
    "uncertainty": 0.42
}
```

## Testing

```bash
# Run all tests
pytest core/tests/ -v

# Fast tests only
pytest core/tests/ -v -m "not slow"
```

## File Structure

```
core/
├── __init__.py
├── infer.py              # Main pipeline + CLI
├── models/
│   ├── base.py           # BaseDetector interface
│   ├── video.py          # XceptionNet
│   ├── audio.py          # CNN-Transformer
│   ├── rppg.py           # CHROM pulse extraction
│   ├── lipsync.py        # SyncNet-style
│   └── fusion.py         # Late fusion ensemble
├── utils/
│   └── uncertainty.py    # Entropy, calibration
└── tests/
    └── test_models.py
```
