"""
Luminark Python SDK

One-line deepfake detection.

Usage:
    from luminark import Luminark
    
    client = Luminark("your_api_key")
    result = client.analyze("video.mp4")
    
    print(result.verdict)      # "FAKE"
    print(result.confidence)   # 87
    print(result.explanation)  # "..."
"""

from .client import Luminark, LuminarkResult, LuminarkError

__version__ = "2.1.0"
__all__ = ["Luminark", "LuminarkResult", "LuminarkError"]
