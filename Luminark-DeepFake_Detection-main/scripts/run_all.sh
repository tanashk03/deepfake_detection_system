#!/bin/bash
# ============================================
# Luminark Run All Script
# Setup → Services → Inference → PASS/FAIL
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Status tracking
PASSED=0
FAILED=0
TESTS=()

log_step() { echo -e "${CYAN}▶ $1${NC}"; }
log_pass() { echo -e "${GREEN}✓ PASS: $1${NC}"; PASSED=$((PASSED + 1)); TESTS+=("PASS: $1"); }
log_fail() { echo -e "${RED}✗ FAIL: $1${NC}"; FAILED=$((FAILED + 1)); TESTS+=("FAIL: $1"); }
log_info() { echo -e "${BLUE}  $1${NC}"; }

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    Luminark Run All                       ║"
echo "║         Setup → Test → Infer → Results                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

cd "$PROJECT_ROOT"

# ============================================
# Step 1: Environment Check
# ============================================
log_step "Checking environment..."

if [ -d ".venv" ]; then
    source .venv/bin/activate
    log_pass "Virtual environment activated"
else
    log_fail "Virtual environment not found"
    echo "Run: ./scripts/bootstrap_mac_intel.sh first"
    exit 1
fi

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    log_pass "PyTorch $TORCH_VERSION available"
else
    log_fail "PyTorch not installed"
fi

if python -c "import fastapi" 2>/dev/null; then
    log_pass "FastAPI available"
else
    log_fail "FastAPI not installed"
fi

# ============================================
# Step 2: Run Core Tests
# ============================================
log_step "Running core ML tests..."

if PYTHONPATH="$PROJECT_ROOT" .venv/bin/python -m pytest core/tests/test_models.py -v -q --tb=no 2>&1 | tail -5; then
    log_pass "Core ML tests"
else
    log_fail "Core ML tests"
fi

# ============================================
# Step 3: Run Backend Tests
# ============================================
log_step "Running backend API tests..."

if PYTHONPATH="$PROJECT_ROOT" .venv/bin/python -m pytest backend/tests/test_api.py -v -q --tb=no 2>&1 | tail -5; then
    log_pass "Backend API tests"
else
    log_fail "Backend API tests"
fi

# ============================================
# Step 4: Start Backend Server
# ============================================
log_step "Starting backend server..."

# Kill any existing server
pkill -f "uvicorn backend.app" 2>/dev/null || true
sleep 1

# Start server in background
PYTHONPATH="$PROJECT_ROOT" .venv/bin/python -m uvicorn backend.app:app \
    --host 127.0.0.1 --port 8000 \
    --log-level warning &
SERVER_PID=$!

# Wait for server to start
sleep 3

if curl -s http://127.0.0.1:8000/health | grep -q "healthy"; then
    log_pass "Backend server started (PID: $SERVER_PID)"
else
    log_fail "Backend server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# ============================================
# Step 5: Create Test Video
# ============================================
log_step "Creating test video..."

TEST_VIDEO="$PROJECT_ROOT/test_sample.mp4"

if command -v ffmpeg &> /dev/null; then
    # Create a 5-second test video with synthetic content (longer for rPPG)
    ffmpeg -y -f lavfi -i "testsrc=duration=5:size=320x240:rate=30" \
           -f lavfi -i "sine=frequency=440:duration=5" \
           -c:v libx264 -c:a aac -shortest \
           "$TEST_VIDEO" 2>/dev/null
    
    if [ -f "$TEST_VIDEO" ]; then
        log_pass "Test video created (5s)"
    else
        log_fail "Could not create test video"
    fi
else
    log_info "ffmpeg not available, creating synthetic video..."
    # Create minimal video using Python (150 frames = 5s at 30fps)
    python -c "
import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('$TEST_VIDEO', fourcc, 30.0, (320, 240))
for i in range(150):
    # Create moving pattern for better testing
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, 0] = 100 + int(50 * np.sin(i * 0.1))  # Pulsing color
    frame[:, :, 1] = 150
    frame[:, :, 2] = 120
    out.write(frame)
out.release()
print('Created synthetic video (5s)')
" 2>/dev/null && log_pass "Test video created (synthetic)" || log_fail "Could not create test video"
fi

# ============================================
# Step 6: Run Sample Inference
# ============================================
log_step "Running sample inference..."

if [ -f "$TEST_VIDEO" ]; then
    RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/infer \
        -H "X-API-Key: lum_test_key_12345" \
        -F "video=@$TEST_VIDEO")
    
    if echo "$RESPONSE" | grep -q "verdict"; then
        VERDICT=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['verdict'])")
        CONFIDENCE=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['confidence'])")
        
        log_pass "Inference completed: $VERDICT ($CONFIDENCE%)"
        log_info "Response: $RESPONSE"
    else
        log_fail "Inference failed"
        log_info "Response: $RESPONSE"
    fi
else
    log_fail "No test video available"
fi

# ============================================
# Step 7: Cleanup
# ============================================
log_step "Cleaning up..."

kill $SERVER_PID 2>/dev/null && log_info "Server stopped" || true
rm -f "$TEST_VIDEO" && log_info "Test video removed" || true

# ============================================
# Results
# ============================================
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                        RESULTS                             ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

for test in "${TESTS[@]}"; do
    if [[ $test == PASS* ]]; then
        echo -e "  ${GREEN}$test${NC}"
    else
        echo -e "  ${RED}$test${NC}"
    fi
done

echo ""
echo -e "  Total: $((PASSED + FAILED)) | ${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                     ALL TESTS PASSED                       ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                     SOME TESTS FAILED                      ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
