#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8765}"
export PYTHONUNBUFFERED=1
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1
