#!/bin/bash
# =============================================================================
# Luminark - macOS Intel Bootstrap Script
# =============================================================================
# Prepares a clean development environment for Luminark on macOS Intel (x86_64)
# Run with: ./scripts/bootstrap_mac_intel.sh
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
NODE_VERSION="18"
VENV_NAME=".venv"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_step() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

check_architecture() {
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]]; then
        print_warning "This script is optimized for Intel (x86_64) Macs."
        print_warning "Detected architecture: $ARCH"
        read -p "Continue anyway? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            exit 1
        fi
    fi
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Main Setup
# -----------------------------------------------------------------------------

print_header "Luminark - macOS Intel Bootstrap"
echo "Setting up development environment..."

check_architecture

# Change to project root (script is in scripts/)
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Step 1: Check/Install Homebrew
# -----------------------------------------------------------------------------
print_header "Step 1: Homebrew"

if check_command brew; then
    print_step "Homebrew is installed"
else
    print_warning "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# -----------------------------------------------------------------------------
# Step 2: Install System Dependencies
# -----------------------------------------------------------------------------
print_header "Step 2: System Dependencies"

BREW_PACKAGES="cmake python@${PYTHON_VERSION} node@${NODE_VERSION} redis ffmpeg"

for pkg in $BREW_PACKAGES; do
    if brew list "$pkg" &>/dev/null; then
        print_step "$pkg already installed"
    else
        echo "Installing $pkg..."
        brew install "$pkg"
        print_step "$pkg installed"
    fi
done

# Ensure correct Python is in PATH
export PATH="/usr/local/opt/python@${PYTHON_VERSION}/bin:$PATH"

# -----------------------------------------------------------------------------
# Step 3: Create Python Virtual Environment
# -----------------------------------------------------------------------------
print_header "Step 3: Python Virtual Environment"

PYTHON_CMD="python${PYTHON_VERSION}"

if [[ -d "${VENV_NAME}" ]]; then
    print_warning "Virtual environment exists. Recreating..."
    rm -rf "${VENV_NAME}"
fi

$PYTHON_CMD -m venv "${VENV_NAME}"
print_step "Created virtual environment: ${VENV_NAME}"

# Activate venv
source "${VENV_NAME}/bin/activate"
print_step "Activated virtual environment"

# Upgrade pip
pip install --upgrade pip wheel setuptools
print_step "Upgraded pip, wheel, setuptools"

# -----------------------------------------------------------------------------
# Step 4: Install PyTorch (CPU-only for Intel Mac)
# -----------------------------------------------------------------------------
print_header "Step 4: PyTorch (CPU-only)"

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

print_step "Installed PyTorch 2.1.2 (CPU)"

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CPU: {torch.device(\"cpu\")}')"

# -----------------------------------------------------------------------------
# Step 5: Install Face Detection Dependencies
# -----------------------------------------------------------------------------
print_header "Step 5: Face Detection Dependencies"

# dlib requires cmake (already installed above)
pip install dlib
print_step "Installed dlib"

pip install face-recognition
print_step "Installed face-recognition"

# -----------------------------------------------------------------------------
# Step 6: Install Python Requirements
# -----------------------------------------------------------------------------
print_header "Step 6: Python Requirements"

# Create requirements directory if it doesn't exist
mkdir -p requirements

# Generate base requirements if not present
if [[ ! -f "requirements/base.txt" ]]; then
    cat > requirements/base.txt << 'EOF'
# Core ML (versions managed separately for macOS compatibility)
# torch, torchvision, torchaudio installed via CPU-specific index

# Vision & Audio Processing
opencv-python-headless~=4.8.0
librosa~=0.10.1
soundfile~=0.12.1

# Transformers & Models
transformers~=4.36.0
huggingface-hub~=0.20.0

# API Framework
fastapi~=0.109.0
uvicorn[standard]~=0.27.0
python-multipart~=0.0.6
pydantic~=2.5.0

# Task Queue
celery~=5.3.0
redis~=5.0.0

# Configuration
pyyaml~=6.0.0
python-dotenv~=1.0.0

# Utilities
numpy~=1.26.0
pillow~=10.2.0
httpx~=0.26.0
tqdm~=4.66.0
EOF
    print_step "Generated requirements/base.txt"
fi

if [[ ! -f "requirements/dev.txt" ]]; then
    cat > requirements/dev.txt << 'EOF'
-r base.txt

# Testing
pytest~=7.4.0
pytest-asyncio~=0.23.0
pytest-cov~=4.1.0
httpx~=0.26.0

# Code Quality
black~=24.1.0
ruff~=0.1.0
mypy~=1.8.0
pre-commit~=3.6.0

# Development
ipython~=8.20.0
watchfiles~=0.21.0
EOF
    print_step "Generated requirements/dev.txt"
fi

pip install -r requirements/dev.txt
print_step "Installed all Python requirements"

# -----------------------------------------------------------------------------
# Step 7: Create Project Directories
# -----------------------------------------------------------------------------
print_header "Step 7: Project Structure"

DIRECTORIES=(
    "src/core"
    "src/models/face"
    "src/models/audio"
    "src/models/multimodal"
    "src/features"
    "src/api/routes"
    "src/api/schemas"
    "src/api/middleware"
    "src/workers"
    "src/utils"
    "tests/unit"
    "tests/integration"
    "tests/fixtures/images"
    "tests/fixtures/videos"
    "tests/fixtures/audio"
    "configs"
    "weights"
    "data/cache"
    "data/uploads"
    "docker"
    "docs/architecture"
    "docs/api"
    "docs/deployment"
)

for dir in "${DIRECTORIES[@]}"; do
    mkdir -p "$dir"
    # Create __init__.py for Python packages under src/
    if [[ "$dir" == src/* ]]; then
        touch "$dir/__init__.py"
    fi
done

# Create .gitkeep files
touch weights/.gitkeep
touch data/.gitkeep

print_step "Created project directories"

# -----------------------------------------------------------------------------
# Step 8: Create Configuration Files
# -----------------------------------------------------------------------------
print_header "Step 8: Configuration Files"

# .env.example
if [[ ! -f ".env.example" ]]; then
    cat > .env.example << 'EOF'
# Luminark Environment Configuration

# Application
APP_ENV=development
APP_DEBUG=true
APP_SECRET_KEY=change-this-in-production

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# Model Settings
MODEL_DEVICE=cpu
MODEL_CACHE_DIR=./weights

# Logging
LOG_LEVEL=INFO
EOF
    print_step "Created .env.example"
fi

# Copy to .env if not exists
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    print_step "Created .env from template"
fi

# Default config
if [[ ! -f "configs/default.yaml" ]]; then
    cat > configs/default.yaml << 'EOF'
# Luminark Default Configuration

app:
  name: Luminark
  version: 0.1.0
  debug: false

server:
  host: 0.0.0.0
  port: 8000
  workers: 1

detection:
  # Default thresholds
  face_confidence: 0.8
  audio_confidence: 0.7
  fusion_weight_face: 0.6
  fusion_weight_audio: 0.4

models:
  device: cpu
  cache_dir: ./weights
  face_detector: xception
  audio_detector: wav2vec

processing:
  max_video_duration: 300  # seconds
  frame_sample_rate: 5     # frames per second
  batch_size: 8

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
EOF
    print_step "Created configs/default.yaml"
fi

# .gitignore
if [[ ! -f ".gitignore" ]]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
.eggs/
*.egg-info/
dist/
build/

# Environment
.env
.env.local

# IDE
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Project specific
weights/*
!weights/.gitkeep
data/*
!data/.gitkeep
*.log
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Node
node_modules/
frontend/dist/

# Docker
docker-compose.override.yml
EOF
    print_step "Created .gitignore"
fi

# pyproject.toml
if [[ ! -f "pyproject.toml" ]]; then
    cat > pyproject.toml << 'EOF'
[project]
name = "luminark"
version = "0.1.0"
description = "Production-grade real-time multimodal deepfake detection"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.ruff]
line-length = 88
target-version = "py310"
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
EOF
    print_step "Created pyproject.toml"
fi

# Makefile
if [[ ! -f "Makefile" ]]; then
    cat > Makefile << 'EOF'
.PHONY: help install dev test lint format run clean

help:
	@echo "Luminark Development Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make install    Install dependencies"
	@echo "  make dev        Start development server"
	@echo "  make test       Run test suite"
	@echo "  make lint       Run linters"
	@echo "  make format     Format code"
	@echo "  make clean      Clean build artifacts"

install:
	pip install -r requirements/dev.txt

dev:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf .coverage htmlcov/
EOF
    print_step "Created Makefile"
fi

# -----------------------------------------------------------------------------
# Step 9: Create Minimal API Stub
# -----------------------------------------------------------------------------
print_header "Step 9: API Stub"

if [[ ! -f "src/api/app.py" ]]; then
    cat > src/api/app.py << 'EOF'
"""Luminark FastAPI Application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Luminark",
    description="Real-time multimodal deepfake detection API",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Luminark", "status": "operational"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
    print_step "Created src/api/app.py"
fi

# -----------------------------------------------------------------------------
# Step 10: Verify Installation
# -----------------------------------------------------------------------------
print_header "Step 10: Verification"

echo "Running verification checks..."

# Python version
python --version
print_step "Python installed"

# PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"
print_step "PyTorch working"

# OpenCV
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
print_step "OpenCV working"

# FastAPI
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
print_step "FastAPI working"

# Face recognition (optional, may fail on some systems)
if python -c "import face_recognition" 2>/dev/null; then
    print_step "face-recognition working"
else
    print_warning "face-recognition not fully working (optional)"
fi

# -----------------------------------------------------------------------------
# Complete
# -----------------------------------------------------------------------------
print_header "Setup Complete! 🎉"

echo ""
echo "Your Luminark development environment is ready."
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  source .venv/bin/activate"
echo "  2. Start the API server:      make dev"
echo "  3. Visit:                      http://localhost:8000"
echo "  4. View API docs:              http://localhost:8000/docs"
echo ""
echo "For more information, see QuickStart.md"
echo ""
