#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building frontend (demo mode)..."
(cd frontend && NEXT_PUBLIC_DEMO_MODE=true npm run build)

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required to generate requirements.txt. Install with 'pip install uv'."
  exit 1
fi

echo "Generating requirements.txt..."
uv export --format requirements-txt --output-file requirements.txt

echo "Deploying to App Engine..."
gcloud config set project build-an-llm
gcloud app deploy app.yaml --verbosity=info