#!/bin/bash
# Quick start script for Gravity Agents

set -e

echo "==================================="
echo "Gravity Agents - Quick Start"
echo "==================================="

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is required. Install from https://nodejs.org"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install and start web environment
echo ""
echo "Setting up web environment..."
cd "$SCRIPT_DIR/web-env"

if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install
fi

# Start server in background
echo "Starting physics server..."
node server.js &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# Wait for server to be ready
echo "Waiting for server..."
sleep 2

# Check server health
if curl -s http://localhost:3000/health > /dev/null; then
    echo "Server is ready!"
else
    echo "Error: Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Set up Python environment
echo ""
echo "Setting up Python environment..."
cd "$SCRIPT_DIR/python-orchestrator"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Server running at http://localhost:3000"
echo ""
echo "To run experiments:"
echo "  cd python-orchestrator"
echo "  source venv/bin/activate"
echo "  python run_experiment.py --help"
echo ""
echo "To stop server:"
echo "  kill $SERVER_PID"
echo ""

# Keep script running to maintain server
echo "Press Ctrl+C to stop..."
wait $SERVER_PID
