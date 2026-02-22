#!/bin/bash
# Backend container entrypoint.
# Applies database migrations before starting the API server.
set -euo pipefail

echo "[entrypoint] Running database migrations..."
python scripts/run_migrations.py

echo "[entrypoint] Starting API server..."
exec uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
