<div align="center">

# 🛡️ Luminark

### AI-Powered Deepfake Video Detection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Detect manipulated videos with 94%+ accuracy using an ensemble of 6 neural network models**

[Live Demo](#demo) • [Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [API Reference](#api-reference)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Multi-Model Ensemble** | 6 specialized AI models voting on authenticity |
| 🎥 **Real-Time Analysis** | Process videos up to 500MB with live progress |
| 🔬 **Explainable AI** | Grad-CAM visualizations and per-model contributions |
| 🌓 **Modern UI** | Glassmorphism design with light/dark themes |
| 🔒 **Privacy-First** | Videos deleted immediately after analysis |
| ⚡ **Fast Inference** | GPU-accelerated or optimized CPU processing |

---

## 🧠 Model Architecture

Luminark uses an **ensemble fusion** approach with 6 specialized detection models:

```
                    ┌─────────────────┐
                    │   Input Video   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐      ┌──────────────┐     ┌───────────┐
   │ VideoMAE │      │ EfficientNet │     │  WavLM    │
   │ (Video)  │      │  (Spatial)   │     │  (Audio)  │
   └────┬─────┘      └──────┬───────┘     └─────┬─────┘
        │                   │                   │
   ┌────┴─────┐      ┌──────┴───────┐     ┌─────┴─────┐
   │ CNN-LSTM │      │ FFT/DCT      │     │  Lip-Sync │
   │(Temporal)│      │ (Frequency)  │     │ Detector  │
   └────┬─────┘      └──────┬───────┘     └─────┬─────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼───────┐
                    │ Fusion Layer  │
                    │(Weighted Vote)│
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │    Verdict    │
                    │  REAL / FAKE  │
                    └───────────────┘
```

| Model | Type | Detection Focus |
|-------|------|-----------------|
| **VideoMAE** | Transformer | Spatio-temporal patterns |
| **EfficientNet** | CNN | Frame-level artifacts |
| **CNN-LSTM** | Hybrid | Motion inconsistencies |
| **FFT/DCT** | Frequency | Compression artifacts |
| **WavLM** | Audio | Voice synthesis traces |
| **Lip-Sync** | Multimodal | Audio-visual mismatch |

---

## 🖥️ Demo

<div align="center">
<img src="docs/demo.gif" alt="Luminark Demo" width="800"/>
</div>

### Screenshots

| Light Theme | Dark Theme |
|-------------|------------|
| ![Light](docs/light.png) | ![Dark](docs/dark.png) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/IsVohi/Luminark-DeepFake_Detection.git
cd Luminark-DeepFake_Detection

# Option 1: Docker (Recommended)
docker compose up

# Option 2: Manual Setup
# Backend
pip install -r requirements/base.txt
pip install torch torchvision torchaudio
uvicorn backend.app:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

### Access
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/infer` | Analyze video (quick) |
| `POST` | `/explain` | Analyze with XAI details |

### Example Request

```bash
curl -X POST http://localhost:8000/infer \
  -H "X-API-Key: your_api_key" \
  -F "video=@test_video.mp4"
```

### Response

```json
{
  "verdict": "FAKE",
  "confidence": 0.94,
  "scores": {
    "spatial": 0.89,
    "temporal": 0.92,
    "frequency": 0.87,
    "audio": 0.96
  },
  "explanation": "High temporal inconsistency detected..."
}
```

---

## 📁 Project Structure

```
luminark/
├── backend/          # FastAPI server
│   ├── app.py        # Main application
│   └── sdk/          # Client SDK
├── core/             # ML pipeline
│   ├── models/       # Neural network definitions
│   ├── xai/          # Explainability (Grad-CAM)
│   └── infer.py      # Inference orchestration
├── frontend/         # React + Vite
│   ├── src/
│   │   ├── pages/    # Landing, Analyze, Docs
│   │   └── components/
│   └── public/
├── models/           # Trained weights (download separately)
├── infra/            # Docker, K8s, AWS configs
└── requirements/     # Python dependencies
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DEVICE` | `cpu` or `cuda` | `cpu` |
| `LUMINARK_API_KEYS` | Comma-separated API keys | - |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## 📊 Performance

Tested on DFDC and Celeb-DF v2 datasets:

| Metric | Score |
|--------|-------|
| Accuracy | 94.2% |
| AUC-ROC | 0.967 |
| F1 Score | 0.938 |
| Inference Time | ~3s/video |

---

## 🛠️ Tech Stack

**Backend**: Python, FastAPI, PyTorch, OpenCV, FFmpeg

**Frontend**: React 18, Vite, Framer Motion, Lucide Icons

**ML Models**: VideoMAE, EfficientNet, WavLM, CNN-LSTM

**DevOps**: Docker, Kubernetes, AWS Lambda

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

---

## 👤 Author

**Vikas Sharma**

[![GitHub](https://img.shields.io/badge/GitHub-IsVohi-181717?logo=github)](https://github.com/IsVohi)
[![Email](https://img.shields.io/badge/Email-tuesviki@gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:tuesviki@gmail.com)

---

<div align="center">
<sub>Built with ❤️ for a safer digital world</sub>
</div>
