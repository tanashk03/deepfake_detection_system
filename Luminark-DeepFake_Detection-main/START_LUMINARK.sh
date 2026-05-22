#!/bin/bash

# Luminark Quick Start Script
# Run this to start the application locally

echo "🚀 Starting Luminark Deepfake Detection System..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Start backend in background
echo "📡 Starting backend on http://localhost:8000..."
cd backend
uvicorn app:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend on http://localhost:3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Luminark is starting!"
echo ""
echo "📌 Backend:  http://localhost:8000"
echo "📌 Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo '✋ Stopping Luminark...'; exit 0" INT

# Keep script running
wait
