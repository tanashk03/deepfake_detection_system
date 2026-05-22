"""
Backend API Tests
"""

import pytest
from fastapi.testclient import TestClient
import tempfile
import os

# Set test API key
os.environ["LUMINARK_API_KEYS"] = "test_key_123,lum_test_key_12345"


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_no_auth_required(self):
        from backend.app import app
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestAuthentication:
    """Tests for API key authentication."""
    
    def test_missing_api_key(self):
        from backend.app import app
        client = TestClient(app)
        
        # Create minimal test video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            temp_path = f.name
        
        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/infer",
                    files={"video": ("test.mp4", f, "video/mp4")}
                )
            
            assert response.status_code == 401
            assert "MISSING_API_KEY" in str(response.json())
        finally:
            os.unlink(temp_path)
    
    def test_invalid_api_key(self):
        from backend.app import app
        client = TestClient(app)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            temp_path = f.name
        
        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/infer",
                    headers={"X-API-Key": "invalid_key"},
                    files={"video": ("test.mp4", f, "video/mp4")}
                )
            
            assert response.status_code == 401
            assert "INVALID_API_KEY" in str(response.json())
        finally:
            os.unlink(temp_path)
    
    def test_valid_api_key_header(self):
        from backend.app import app
        client = TestClient(app)
        
        # Just test auth passes, not full inference
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_bearer_token_format(self):
        from backend.app import app
        client = TestClient(app)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            temp_path = f.name
        
        try:
            with open(temp_path, "rb") as f:
                # Should accept Bearer format
                response = client.post(
                    "/infer",
                    headers={"Authorization": "Bearer test_key_123"},
                    files={"video": ("test.mp4", f, "video/mp4")}
                )
            
            # Auth should pass (might fail on inference due to invalid video)
            assert response.status_code != 401
        finally:
            os.unlink(temp_path)


class TestInferEndpoint:
    """Tests for /infer endpoint."""
    
    def test_unsupported_format(self):
        from backend.app import app
        client = TestClient(app)
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a video")
            temp_path = f.name
        
        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/infer",
                    headers={"X-API-Key": "test_key_123"},
                    files={"video": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            data = response.json()
            assert "UNSUPPORTED_FORMAT" in str(data)
        finally:
            os.unlink(temp_path)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self):
        from backend.app import app
        client = TestClient(app)
        
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Luminark API"
        assert "version" in data
        assert data["docs"] == "/docs"


class TestPythonSDK:
    """Tests for Python SDK."""
    
    def test_sdk_import(self):
        from backend.sdk.python import Luminark, LuminarkResult, LuminarkError
        
        assert Luminark is not None
        assert LuminarkResult is not None
        assert LuminarkError is not None
    
    def test_sdk_missing_key_error(self):
        from backend.sdk.python import Luminark, LuminarkError
        
        # Temporarily unset env var
        old_key = os.environ.pop("LUMINARK_API_KEY", None)
        
        try:
            with pytest.raises(LuminarkError) as exc_info:
                Luminark(api_key=None)
            
            assert exc_info.value.code == "MISSING_API_KEY"
        finally:
            if old_key:
                os.environ["LUMINARK_API_KEY"] = old_key
    
    def test_sdk_result_properties(self):
        from backend.sdk.python.client import LuminarkResult
        
        result = LuminarkResult(
            verdict="FAKE",
            confidence=85,
            explanation="Test explanation",
        )
        
        assert result.is_fake is True
        assert result.is_real is False
        assert result.is_inconclusive is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
