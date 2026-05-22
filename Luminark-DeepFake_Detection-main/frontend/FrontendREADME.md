# Luminark Frontend

> Professional deepfake detection dashboard

## Quick Start

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Features

| Feature | Description |
|---------|-------------|
| Video Upload | Drag-drop or click to upload |
| Live Camera | WebRTC recording (5s clips) |
| Traffic Light | Visual verdict (green/yellow/red) |
| Explainability | Heatmap + modality breakdown |

## Architecture

```
src/
├── components/
│   ├── Header.jsx       # Navigation
│   ├── VideoUpload.jsx  # Drag-drop upload
│   ├── LiveCamera.jsx   # WebRTC capture
│   ├── VerdictDisplay.jsx # Traffic light
│   └── Explainability.jsx # Analysis details
├── services/
│   └── api.js           # Backend API
└── styles/
    ├── index.css        # Global styles
    └── App.css          # Layout
```

## Prerequisites

Backend must be running:

```bash
cd backend
uvicorn app:app --reload --port 8000
```

## Environment

The frontend proxies `/api/*` to `http://localhost:8000` via Vite config.

## Design

- Dark theme for professional appearance
- Calm colors to build trust
- Clear visual hierarchy
- Non-technical language
