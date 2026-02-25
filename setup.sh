#!/bin/bash
# setup.sh - Automated environment setup for SpatialTranscriptFormer (Linux/HPC)

echo "--- SpatialTranscriptFormer Setup ---"

ENV_NAME="SpatialTranscriptFormer"

# Check if conda environment exists
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME' with Python 3.9..."
    conda create -n $ENV_NAME python=3.9 -y
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

echo "Installing/Updating package in editable mode..."
conda run -n $ENV_NAME pip install -e .[dev]

echo ""
echo "Setup Complete!"
echo "You can now use the following commands (after activating the environment):"
echo "  stf-download --help"
echo "  stf-split --help"
echo "  stf-build-vocab --help"
echo ""
echo "To run tests, use: ./test.sh"
