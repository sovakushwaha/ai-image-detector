#!/usr/bin/env bash
# Launch SOVA VERIFY H4 investor demo (local backend + frontend).
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
BACKEND_DIR="$DEMO_DIR/backend"
FRONTEND_DIR="$DEMO_DIR/frontend"
PY="${PROJECT_ROOT}/.venv/bin/python"
UVICORN="${PROJECT_ROOT}/.venv/bin/uvicorn"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: project venv not found at $PY"
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
fi

# Ensure FastAPI present in project venv
"$PY" -c "import fastapi, uvicorn" 2>/dev/null || {
  echo "Installing FastAPI into project .venv..."
  "$PY" -m pip install -q -r "$BACKEND_DIR/requirements.txt"
}

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down SOVA VERIFY demo..."
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI backend on http://127.0.0.1:8000 ..."
cd "$BACKEND_DIR"
export PYTHONPATH="$BACKEND_DIR:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
"$UVICORN" app.main:app --host 127.0.0.1 --port 8000 --log-level info &
BACKEND_PID=$!

# Wait for health
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "ERROR: backend exited early"
    exit 1
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "ERROR: backend health check failed"
  exit 1
fi

echo "Starting Vite frontend on http://127.0.0.1:5173 ..."
cd "$FRONTEND_DIR"
npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

sleep 2

echo ""
echo "============================================"
echo "  SOVA VERIFY — H4 Research Prototype"
echo "  Backend:  http://127.0.0.1:8000"
echo "  Frontend: http://127.0.0.1:5173"
echo "  API docs: http://127.0.0.1:8000/docs"
echo "============================================"
echo "Press Ctrl+C to stop both servers."
echo ""

wait
