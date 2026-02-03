#!/usr/bin/env bash

clear && printf '\e[3J'

# Function to handle script termination
cleanup() {
    echo "Stopping backend..."
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID
    fi
    exit
}

# Trap SIGINT (Ctrl+C) to run cleanup
trap cleanup SIGINT

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install Node.js and npm to run the frontend."
    echo "You can download it from https://nodejs.org/"
    exit 1
fi

# Idempotent
echo "Installing/Updating uv..."
pip install uv

echo "Starting Backend..."
# Run uvicorn in the background
uv run uvicorn backend.app.main:app --reload &
BACKEND_PID=$!

echo "Backend started with PID $BACKEND_PID"

echo "Setting up Frontend..."
cd frontend

echo "Installing frontend dependencies..."
npm install

echo "Starting Frontend..."
npm run dev
