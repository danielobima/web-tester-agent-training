#!/bin/bash
# Setup script for training environment

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install datasets beautifulsoup4 tqdm playwright pandas

echo "Installing Playwright browsers..."
playwright install chromium

echo "Setup complete. Use 'source venv/bin/activate' to use the environment."
