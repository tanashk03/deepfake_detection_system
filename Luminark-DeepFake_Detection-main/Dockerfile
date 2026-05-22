# ============================================
# Luminark Backend - Hugging Face Spaces
# OPTIMIZED: Using pre-built PyTorch image
# ============================================

# Use official PyTorch CPU image (much faster than installing from scratch)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Switch to CPU-only mode in environment
ENV CUDA_VISIBLE_DEVICES=""
ENV MODEL_DEVICE=cpu

WORKDIR /app

# Install minimal system dependencies (skip OpenCV dev, use pip version)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    bzip2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (minimal, skip what PyTorch image already has)
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.26.2 \
    Pillow==10.1.0 \
    scipy==1.11.4 \
    dlib==19.24.2 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    transformers==4.36.0

# Copy application code
COPY core/ /app/core/
COPY backend/ /app/backend/

# Download dlib face predictor model
RUN mkdir -p /app/models && \
    curl -L -o /app/models/shape_predictor.dat.bz2 \
    "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /app/models/shape_predictor.dat.bz2 && \
    mv /app/models/shape_predictor.dat /app/models/shape_predictor_68_face_landmarks.dat

# Environment
ENV PYTHONPATH=/app
ENV LUMINARK_API_KEYS=lum_prod_key_secure_2026

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
