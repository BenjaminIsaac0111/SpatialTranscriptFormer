#!/bin/bash
# setup.sh - Automated environment setup for SpatialTranscriptFormer (Linux/HPC)

set -e

echo "--- SpatialTranscriptFormer Setup ---"

ENV_NAME="SpatialTranscriptFormer"

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "Error: conda was not found. Please ensure Conda is installed and in your PATH."
    exit 1
fi

# Check if conda environment exists
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME' with Python 3.9..."
    conda create -n $ENV_NAME python=3.9 -y
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

echo "Installing PyTorch (CUDA 11.8)..."
conda run -n $ENV_NAME pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing/Updating package in editable mode..."
conda run -n $ENV_NAME pip install -e .[dev]

echo "Checking Hugging Face authentication..."
# Temporarily disable exit on error for this check
set +e
HF_STATUS=$(conda run -n $ENV_NAME huggingface-cli whoami 2>&1)
HF_EXIT=$?
set -e

if [ $HF_EXIT -ne 0 ] || [[ "$HF_STATUS" == *"Not logged in"* ]]; then
    HF_NEED_LOGIN=true
else
    HF_NEED_LOGIN=false
    echo "Hugging Face authentication found: $HF_STATUS"
fi

echo ""
echo "========================================="
echo "             SETUP COMPLETE!             "
echo "========================================="
echo ""
echo "IMPORTANT: You must activate the environment before using the tools:"
echo "  conda activate $ENV_NAME"
echo ""

if [ "$HF_NEED_LOGIN" = true ]; then
    echo "------------------------------------------------------------"
    echo "DATASET ACCESS REQUIRES AUTHENTICATION"
    echo "The HEST-1k dataset on Hugging Face is gated. You must provide an access token."
    echo "Please do ONE of the following before downloading data:"
    echo "  Option A (Persistent): Run 'conda run -n $ENV_NAME huggingface-cli login' and paste your token."
    echo "  Option B (Temporary): Run 'export HF_TOKEN=your_token_here'"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "------------------------------------------------------------"
    echo ""
fi

echo "You can then use the following commands:"
echo "  stf-download --help"
echo "  stf-split --help"
echo "  stf-build-vocab --help"
echo ""
echo "To run tests, use: ./test.sh"
