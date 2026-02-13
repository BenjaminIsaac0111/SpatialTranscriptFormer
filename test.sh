#!/bin/bash
# test.sh - Run project tests (Linux/HPC)

echo "--- Running SpatialTranscriptFormer Tests ---"

ENV_NAME="SpatialTranscriptFormer"

conda run -n $ENV_NAME pytest tests/
