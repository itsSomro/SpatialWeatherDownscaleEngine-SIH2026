#!/bin/bash
set -e

echo "========================================================="
echo "   Spatial Weather Downscaling Engine (SIH 2026)"
echo "   High-Resolution Microclimate AI Architecture"
echo "========================================================="

# 1. Start FastAPI Backend in background
echo "[1/2] Launching FastAPI Backend on http://0.0.0.0:8000..."
uvicorn api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 2. Wait for backend to be healthy
echo "Awaiting backend readiness..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8000/docs > /dev/null 2>&1; then
        echo "Backend health check passed! (attempt $i)"
        break
    fi
    sleep 1
done

# 3. Start Streamlit Frontend
echo "[2/2] Launching Streamlit UI on http://0.0.0.0:8501..."
streamlit run frontend/ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
FRONTEND_PID=$!

# Trap termination signals for clean shutdown
trap "echo 'Shutting down services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

echo "All services running. Press Ctrl+C to terminate."
wait -n
exit $?
