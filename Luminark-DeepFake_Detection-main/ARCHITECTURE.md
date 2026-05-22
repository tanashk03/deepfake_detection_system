# System Architecture

## Overview

Luminark uses a multimodal ensemble architecture that combines 6 specialized detection models through late fusion to achieve robust deepfake detection.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VIDEO INPUT                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Frame     │  │   Audio      │  │  Resize to   │  │  ImageNet       │  │
│  │   Extract   │  │   Extract    │  │   224×224    │  │  Normalize      │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌────────────────────────┬┴────────────────────────┐
          ▼                        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   SPATIAL MODEL  │    │  TEMPORAL MODEL  │    │ FREQUENCY MODEL  │
│  EfficientNet-B0 │    │    Conv3D        │    │ EfficientNet+FFT │
│  Single Frame    │    │   16 Frames      │    │   Single Frame   │
│  Weight: 25%     │    │   Weight: 7%     │    │   Weight: 25%    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
          │                        │                         │
          ▼                        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ PHYSIOLOGICAL    │    │   VIDEOMAE       │    │   AUDIO MODEL    │
│  CNN + BiLSTM    │    │  Transformer     │    │     WavLM        │
│  rPPG Signals    │    │  HuggingFace     │    │   (if audio)     │
│  Weight: 10%     │    │  Weight: 25%     │    │   Weight: 1%     │
└──────────────────┘    └──────────────────┘    └──────────────────┘
          │                        │                         │
          └────────────────────────┴─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LATE FUSION ENSEMBLE                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Weighted Score Aggregation:                                          │  │
│  │  final_score = Σ (weight_i × score_i) / Σ weight_i                   │  │
│  │                                                                       │  │
│  │  Score Convention: positive = FAKE, negative = REAL                   │  │
│  │  Threshold: |score| > 0.02 → classification, else INCONCLUSIVE       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                          │
│  ┌───────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Verdict  │  │ Confidence  │  │ Raw Score    │  │ Model Contributions│  │
│  │ FAKE/REAL │  │   0-100%    │  │  -1 to +1    │  │   Per-modality %   │  │
│  └───────────┘  └─────────────┘  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Model Details

### 1. Spatial Model (25%)
- **Architecture**: EfficientNet-B0 backbone with custom classifier
- **Input**: Single middle frame (224×224×3)
- **Detects**: Blending artifacts, texture inconsistencies, compression artifacts

### 2. VideoMAE Model (25%)
- **Architecture**: Vision Transformer (ViT) with masked autoencoder pretraining
- **Source**: `MCG-NJU/videomae-base` from HuggingFace
- **Input**: 16 sampled frames
- **Detects**: Spatiotemporal inconsistencies, unnatural motion patterns

### 3. Frequency Model (25%)
- **Architecture**: EfficientNet-B0 with FFT/DCT preprocessing
- **Input**: Frequency-domain representation of single frame
- **Detects**: GAN fingerprints, spectral artifacts, unnatural frequency distributions

### 4. Physiological Model (10%)
- **Architecture**: CNN feature extractor + Bidirectional LSTM
- **Input**: 16 frames (224×224×3)
- **Detects**: Abnormal pulse patterns, unnatural skin color variations

### 5. Temporal Model (7%)
- **Architecture**: 3D Convolutional Network
- **Input**: 16 sampled frames as video clip
- **Detects**: Temporal discontinuities, flickering, unnatural transitions

### 6. Audio Model (1-25%)
- **Architecture**: WavLM Transformer from Microsoft
- **Input**: Audio waveform at 16kHz
- **Detects**: Voice synthesis artifacts, unnatural prosody, audio-visual sync issues

## Data Flow

```
1. Video Upload → API Endpoint (/explain)
       │
2. Frame Extraction → OpenCV (5 FPS)
       │
3. Audio Extraction → Librosa (if available)
       │
4. Parallel Model Inference
   ├── Spatial: 1 frame → score
   ├── Temporal: 16 frames → score
   ├── Frequency: 1 frame (FFT) → score
   ├── Physiological: 16 frames → score
   ├── VideoMAE: 16 frames → score
   └── Audio: waveform → score (if available)
       │
5. Late Fusion → Weighted average
       │
6. Verdict Decision → Threshold comparison
       │
7. Response Generation → JSON with explanation
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
│  ┌─────────────────┐          ┌─────────────────────┐  │
│  │                 │          │                     │  │
│  │    Frontend     │◄────────►│      Backend        │  │
│  │    (React)      │   API    │     (FastAPI)       │  │
│  │    Port 3000    │          │     Port 8000       │  │
│  │                 │          │                     │  │
│  └─────────────────┘          └─────────────────────┘  │
│                                        │               │
│                                        ▼               │
│                               ┌─────────────────┐      │
│                               │   Model Files   │      │
│                               │   /models/*.pt  │      │
│                               └─────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.10 |
| ML Framework | PyTorch 2.0 |
| Vision Models | timm, transformers |
| Audio Processing | librosa, torchaudio |
| Frontend | React, TypeScript |
| Containerization | Docker, Docker Compose |
| Video Processing | OpenCV, FFmpeg |

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Accuracy | 95% |
| Inference Time | 2-5 seconds (CPU) |
| Memory Usage | ~2 GB |
| Supported Formats | MP4, AVI, MOV, WebM |
| Max Video Length | 60 seconds |
