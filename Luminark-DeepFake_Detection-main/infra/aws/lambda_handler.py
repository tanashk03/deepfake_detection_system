"""
AWS Lambda handler for Luminark backend.
Uses Mangum to adapt FastAPI application for Lambda runtime.
"""

from mangum import Mangum
import sys
import os

# Add the Lambda task root to Python path
sys.path.insert(0, os.environ.get('LAMBDA_TASK_ROOT', '/var/task'))

from backend.app import app

# Mangum adapter - converts API Gateway events to ASGI and back
# lifespan="off" because Lambda manages container lifecycle
handler = Mangum(app, lifespan="off")
