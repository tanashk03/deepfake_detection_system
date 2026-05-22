# Luminark Backend

> FastAPI backend wrapping core ML inference

## Quick Start

```bash
# Start server
cd backend
uvicorn app:app --reload

# Test health
curl http://localhost:8000/health
```

## Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/infer` | POST | Yes | Analyze video |
| `/explain` | POST | Yes | Detailed analysis |

## Authentication

Include API key in header:

```bash
# X-API-Key header
curl -X POST http://localhost:8000/infer \
  -H "X-API-Key: lum_test_key_12345" \
  -F "video=@video.mp4"

# Or Bearer token
curl -X POST http://localhost:8000/infer \
  -H "Authorization: Bearer lum_test_key_12345" \
  -F "video=@video.mp4"
```

## Response Format

```json
{
  "verdict": "FAKE",
  "confidence": 87,
  "explanation": "Visual artifacts detected...",
  "processing_time_ms": 45230
}
```

---

## Python SDK

```bash
pip install httpx
```

### One-Line Usage

```python
from backend.sdk.python import Luminark

client = Luminark("lum_test_key_12345")
result = client.analyze("video.mp4")

print(result.verdict)      # "FAKE"
print(result.confidence)   # 87
print(result.is_fake)      # True
```

### With Details

```python
result = client.analyze("video.mp4", detailed=True)
print(result.uncertainty)  # 0.08
print(result.modality_contributions)
```

---

## JavaScript SDK

```bash
npm install form-data
```

### Usage

```javascript
const { Luminark } = require('./sdk/js');

const client = new Luminark('lum_test_key_12345');
const result = await client.analyze('./video.mp4');

console.log(result.verdict);     // "FAKE"
console.log(result.confidence);  // 87
console.log(result.isFake);      // true
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LUMINARK_API_KEYS` | Comma-separated valid keys |
| `LUMINARK_API_KEY` | SDK: default API key |
| `LUMINARK_API_URL` | SDK: custom API URL |

---

## Testing

```bash
pytest backend/tests/ -v
```

---

## File Structure

```
backend/
├── app.py              # FastAPI application
├── tests/
│   └── test_api.py     # API tests
└── sdk/
    ├── python/
    │   ├── __init__.py
    │   └── client.py
    └── js/
        ├── index.js
        ├── index.d.ts
        └── package.json
```
