"""
Luminark Python SDK Client

Simple, production-ready client for Luminark API.
"""

import os
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import httpx


class LuminarkError(Exception):
    """Luminark API error."""
    
    def __init__(self, message: str, code: str, status_code: int = 0):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass
class LuminarkResult:
    """Result from deepfake analysis."""
    
    verdict: str  # "REAL", "FAKE", "INCONCLUSIVE"
    confidence: int  # 0-100
    explanation: str
    processing_time_ms: Optional[float] = None
    
    # Extended fields (from /explain)
    score: Optional[float] = None
    uncertainty: Optional[float] = None
    modality_contributions: Optional[dict] = None
    
    @property
    def is_fake(self) -> bool:
        """True if verdict is FAKE."""
        return self.verdict == "FAKE"
    
    @property
    def is_real(self) -> bool:
        """True if verdict is REAL."""
        return self.verdict == "REAL"
    
    @property
    def is_inconclusive(self) -> bool:
        """True if verdict is INCONCLUSIVE."""
        return self.verdict == "INCONCLUSIVE"


class Luminark:
    """
    Luminark API client.
    
    Example:
        client = Luminark("lum_xxx")
        result = client.analyze("video.mp4")
        print(result.verdict)
    """
    
    DEFAULT_BASE_URL = "http://localhost:8000"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,  # 5 minutes for large videos
    ):
        """
        Initialize Luminark client.
        
        Args:
            api_key: API key (or set LUMINARK_API_KEY env var)
            base_url: API base URL (default: localhost:8000)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("LUMINARK_API_KEY")
        if not self.api_key:
            raise LuminarkError(
                "API key required. Pass api_key or set LUMINARK_API_KEY",
                code="MISSING_API_KEY"
            )
        
        self.base_url = (base_url or os.getenv("LUMINARK_API_URL") or 
                         self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        
        self._client = httpx.Client(
            timeout=timeout,
            headers={"X-API-Key": self.api_key}
        )
    
    def analyze(
        self,
        video: Union[str, Path, bytes],
        detailed: bool = False,
    ) -> LuminarkResult:
        """
        Analyze a video for deepfake indicators.
        
        Args:
            video: Path to video file, or raw bytes
            detailed: If True, include internal scores
            
        Returns:
            LuminarkResult with verdict, confidence, explanation
            
        Raises:
            LuminarkError: On API errors
        """
        endpoint = "/explain" if detailed else "/infer"
        
        # Prepare file
        if isinstance(video, (str, Path)):
            video_path = Path(video)
            if not video_path.exists():
                raise LuminarkError(
                    f"Video file not found: {video_path}",
                    code="FILE_NOT_FOUND"
                )
            files = {"video": (video_path.name, open(video_path, "rb"))}
        else:
            files = {"video": ("video.mp4", video)}
        
        try:
            response = self._client.post(
                f"{self.base_url}{endpoint}",
                files=files,
            )
            
            if response.status_code == 401:
                data = response.json()
                raise LuminarkError(
                    data.get("detail", {}).get("error", "Authentication failed"),
                    code="AUTH_ERROR",
                    status_code=401
                )
            
            if response.status_code >= 400:
                data = response.json()
                detail = data.get("detail", {})
                raise LuminarkError(
                    detail.get("error", "Request failed"),
                    code=detail.get("code", "UNKNOWN_ERROR"),
                    status_code=response.status_code
                )
            
            data = response.json()
            
            return LuminarkResult(
                verdict=data["verdict"],
                confidence=data["confidence"],
                explanation=data["explanation"],
                processing_time_ms=data.get("processing_time_ms"),
                score=data.get("score"),
                uncertainty=data.get("uncertainty"),
                modality_contributions=data.get("modality_contributions"),
            )
        
        except httpx.RequestError as e:
            raise LuminarkError(
                f"Connection error: {e}",
                code="CONNECTION_ERROR"
            )
        finally:
            # Close file handles
            if isinstance(video, (str, Path)):
                files["video"][1].close()
    
    def health(self) -> dict:
        """Check API health."""
        response = self._client.get(f"{self.base_url}/health")
        return response.json()
    
    def close(self):
        """Close the client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Convenience function
def analyze(
    video: Union[str, Path],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LuminarkResult:
    """
    One-shot analysis without managing client.
    
    Example:
        from luminark import analyze
        result = analyze("video.mp4", api_key="lum_xxx")
    """
    with Luminark(api_key=api_key, base_url=base_url) as client:
        return client.analyze(video)
