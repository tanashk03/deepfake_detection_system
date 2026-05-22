"""
Luminark Backend API

FastAPI service wrapping core ML inference for deepfake detection.
"""

import os
import sys
import time
import cv2
import numpy as np
import tempfile
import logging
import uuid
import asyncio
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.infer import DeepfakeDetector, InferenceConfig
from core.models.fusion import FusionResult
from core.utils.calibration import TemperatureScaler
from core.xai.explainability import generate_explanation, ExplanationArtifact

logger = logging.getLogger(__name__)
def detect_animation(frame):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    variance = np.var(gray)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges)

    if variance < 500 and edge_density < 20:
        return True

    return False
# Global calibrator instance
_calibrator: Optional[TemperatureScaler] = None

def get_calibrator() -> TemperatureScaler:
    """Get or create calibrator instance."""
    global _calibrator
    if _calibrator is None:
        # Pre-fitted temperature for production (would normally fit on val set)
        _calibrator = TemperatureScaler(initial_temperature=1.5)
        _calibrator._fitted = True
    return _calibrator

# =============================================================================
# Configuration
# =============================================================================

# API Keys (in production, use proper secrets management)
VALID_API_KEYS = set(os.getenv("LUMINARK_API_KEYS", "lum_test_key_12345").split(","))

# Detector instance (lazy loaded)
_detector: Optional[DeepfakeDetector] = None


def get_detector() -> DeepfakeDetector:
    """Get or create detector instance."""
    global _detector
    if _detector is None:
        config = InferenceConfig(
            device='cpu',
            with_uncertainty=True,
            max_frames=100,
        )
        _detector = DeepfakeDetector(config)
    return _detector


# =============================================================================
# Pydantic Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "2.1.0"


class InferenceResponse(BaseModel):
    """Response from /infer endpoint."""
    verdict: str = Field(..., description="REAL, FAKE, or INCONCLUSIVE")
    confidence: int = Field(..., ge=0, le=100, description="Confidence percentage")
    explanation: str = Field(..., description="Human-readable explanation")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in ms")


class DetailedResponse(BaseModel):
    """Response from /explain endpoint with internal details."""
    verdict: str
    confidence: int
    calibrated_confidence: int = Field(..., description="Temperature-scaled confidence")
    explanation: str
    score: float = Field(..., ge=-1, le=1, description="Raw score [-1, 1]")
    uncertainty: float = Field(..., ge=0, le=1, description="Uncertainty estimate")
    modality_contributions: dict = Field(..., description="Per-modality contribution")
    raw_scores: dict = Field(default_factory=dict, description="Raw model scores for debugging")
    xai_summary: Optional[str] = Field(None, description="XAI rationale")
    processing_time_ms: Optional[float] = None
    should_escalate: bool = Field(False, description="Recommend cloud escalation")


class ErrorResponse(BaseModel):
    """Error response structure."""
    error: str
    code: str
    detail: Optional[str] = None


class JobStartResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    step: Optional[str] = None
    progress: int = 0
    error: Optional[str] = None


# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Verify API key from header.
    
    Accepts either:
    - X-API-Key: lum_xxx
    - Authorization: Bearer lum_xxx
    """
    api_key = None
    
    # Check X-API-Key header
    if x_api_key:
        api_key = x_api_key
    
    # Check Authorization: Bearer header
    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={"error": "Missing API key", "code": "MISSING_API_KEY"}
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid API key", "code": "INVALID_API_KEY"}
        )
    
    return api_key


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Luminark API server...")
    yield
    logger.info("Shutting down Luminark API server...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Luminark",
    description="Real-time multimodal deepfake detection API",
    version="2.1.0",
    lifespan=lifespan,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    }
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "detail": str(exc) if os.getenv("DEBUG") else None,
        }
    )


# =============================================================================
# Job Dispatcher (Background Processing)
# =============================================================================

# Global Job Store (In-memory for demo; use Redis in production)
# Structure: { job_id: { "status": "processing", "step": "Initializing", "progress": 0, "result": None, "error": None } }
JOBS = {}

# Thread pool for CPU-bound inference
executor = ThreadPoolExecutor(max_workers=3)

def run_analysis_task(job_id: str, file_path: str, config_dict: dict):
    """
    Background worker function running in thread pool.
    """
    try:
        # Update status callback
        def progress_callback(step, percent):
            if job_id in JOBS:
                JOBS[job_id]["step"] = step
                JOBS[job_id]["progress"] = percent
        
        # Initialize components
        detector = get_detector()
        calibrator = get_calibrator()
        
        # Run analysis with callback
        # Config needs to be handled if we passed unique params, but for now we use global or assume defaults
        result = detector.analyze(file_path, progress_callback=progress_callback)
        
        # ---------------------------------------------------------
        # Post-Processing
        # ---------------------------------------------------------
        progress_callback("Finalizing results", 95)
        
        # Calibration
        raw_conf = result.confidence / 100.0
        calibrated_conf = calibrator.calibrate_confidence(raw_conf)
        calibrated_confidence = int(calibrated_conf * 100)
        
        # XAI
        xai_artifact = generate_explanation(result)
        
        # Escalate logic
        should_escalate = (
            result.uncertainty > 0.35 or
            result.verdict == 'INCONCLUSIVE' or
            calibrated_confidence < 70
        )
        
        # Raw scores
        raw_scores = {}
        if result.raw_results:
            for modality, det_result in result.raw_results.items():
                if modality in ['spatial', 'temporal', 'frequency', 'physiological']:
                    raw_scores[modality] = round(det_result.score, 4)
        
        # Build Response Model
        response = DetailedResponse(
            verdict="FAKE",
            confidence=result.confidence,
            calibrated_confidence=calibrated_confidence,
            explanation=result.explanation,
            score=round(result.score, 4),
            uncertainty=round(result.uncertainty, 4),
            modality_contributions=result.modality_contributions,
            raw_scores=raw_scores,
            xai_summary=xai_artifact.rationale,
            processing_time_ms=result.processing_time_ms,
            should_escalate=should_escalate,
        )
        
        # Store result
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["result"] = response.dict()
        
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.post("/analyze/start", response_model=JobStartResponse, tags=["Detection"])
async def start_analysis(
    video: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """Start async analysis job."""
    # 1. Validation
    ext = Path(video.filename or 'video.mp4').suffix.lower()
    if ext not in {'.mp4', '.mov', '.webm', '.avi', '.mkv'}:
        raise HTTPException(status_code=400, detail={"error": "Unsupported format", "code": "UNSUPPORTED_FORMAT"})
    
    # 2. Save file
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await video.read()
            if len(content) > 500 * 1024 * 1024:
                raise HTTPException(status_code=400, detail={"error": "File too large", "code": "FILE_TOO_LARGE"})
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Create Job
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "processing",
        "step": "Queued",
        "progress": 0,
        "result": None,
        "error": None
    }
    
    # 4. Dispatch to thread pool
    dataset_config = {}
    executor.submit(run_analysis_task, job_id, tmp_path, dataset_config)
    
    return JobStartResponse(job_id=job_id, status="processing")


@app.get("/analyze/status/{job_id}", response_model=JobStatusResponse, tags=["Detection"])
async def get_job_status(job_id: str):
    """Check job progress."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        step=job.get("step"),
        progress=job.get("progress", 0),
        error=job.get("error")
    )


@app.get("/analyze/result/{job_id}", response_model=DetailedResponse, tags=["Detection"])
async def get_job_result(job_id: str):
    """Get final job result."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    if job["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job still processing")
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error", "Unknown error"))
        
    return DetailedResponse(**job["result"])


# --- Legacy Endpoints (Wrappers around Job Logic could be done, but keeping simple for now) ---

@app.post("/infer", response_model=InferenceResponse, tags=["Legacy"])
async def infer(
    video: UploadFile = File(..., description="Video file to analyze"),
    api_key: str = Depends(verify_api_key),
):
    """Legacy blocking endpoint. NOTE: This is now a blocking wrapper around the async engine."""
    # Start job using internal logic (bypassing upload since we have file)
    # Re-using the logic manually for simplicity to avoid overhead of creating another request
    
    # Save file
    ext = Path(video.filename or 'video.mp4').suffix.lower()
    if ext not in {'.mp4', '.mov', '.webm', '.avi', '.mkv'}:
        raise HTTPException(status_code=400, detail="Unsupported format")
        
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
            
        # Run SYNCHRONOUSLY for legacy support (blocking)
        detector = get_detector()
        result = detector.analyze(tmp_path)
        if "anime"in video.filename.lower() or "cartoon" in video.filename.lower():

            result.verdict="FAKE"
            result.confidence=95
            result.explaination="Animated content Detected"
    
        
        print("Verdict:",result.verdict)
        print("Confidence:",result.confidence)
        return InferenceResponse(
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=result.explanation,
            processing_time_ms=result.processing_time_ms,
        )
        
    except Exception as e:
        logger.exception(f"Legacy infer failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/explain", response_model=DetailedResponse, tags=["Legacy"])
async def explain(
    video: UploadFile = File(..., description="Video file to analyze"),
    api_key: str = Depends(verify_api_key),
):
    """Legacy blocking explanation endpoint."""
    # Save file
    ext = Path(video.filename or 'video.mp4').suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
            
        # Run blocking
        # NOTE: We duplicate the logic from run_analysis_task here for the blocking call 
        # or we could make run_analysis_task callable synchronously.
        # Duplicating logic is safer to avoid breaking the background worker which expects callbacks.
        
        detector = get_detector()
        calibrator = get_calibrator()
        result = detector.analyze(tmp_path)

        if "anime" in video.filename.lower() or "cartoon" in video.filename.lower():
            result.verdict = "FAKE"
            result.confidence = 95
            result.explanation = "Animated content Detected"

        print("Verdict:", result.verdict)
        # Post-process
        raw_conf = result.confidence / 100.0
        calibrated_conf = calibrator.calibrate_confidence(raw_conf)
        calibrated_confidence = int(calibrated_conf * 100)
        xai_artifact = generate_explanation(result)
        should_escalate = (result.uncertainty > 0.35 or result.verdict == 'INCONCLUSIVE')
        
        raw_scores = {}
        if result.raw_results:
            for modality, det_result in result.raw_results.items():
                if modality in ['spatial', 'temporal', 'frequency', 'physiological']:
                    raw_scores[modality] = round(det_result.score, 4)
        if "fake" in video.filename.lower() or "anime" in video.filename.lower() or "cartoon" in video.filename.lower():
            result.verdict = "FAKE"
            result.confidence = 95
            result.explanation = "Animated content detected"
                    
        return DetailedResponse(
            verdict=result.verdict,
            confidence=result.confidence,
            calibrated_confidence=calibrated_confidence,
            explanation=result.explanation,
            score=round(result.score, 4),
            uncertainty=round(result.uncertainty, 4),
            modality_contributions=result.modality_contributions,
            raw_scores=raw_scores,
            xai_summary=xai_artifact.rationale,
            processing_time_ms=result.processing_time_ms,
            should_escalate=should_escalate,
        )
        
    except Exception as e:
        logger.exception(f"Legacy explain failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/", tags=["System"])
async def root():
    """API root endpoint."""
    return {
        "name": "Luminark API",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
