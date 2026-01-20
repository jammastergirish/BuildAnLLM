#!/bin/bash

clear

set -euo pipefail

# Playwright install (idempotent, checks for updates)
(cd frontend && npx playwright install)

echo "Running backend tests..."
uv run --extra test pytest

echo "Running frontend tests..."
(cd frontend && npm run test:run && npm run test:e2e)
