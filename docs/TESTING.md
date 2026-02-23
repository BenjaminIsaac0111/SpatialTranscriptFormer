# Testing Guide

This project includes a test suite to ensure the reliability of data download and processing scripts.

## Running Tests

To run all tests, ensure you have the `SpatialTranscriptFormer` environment activated:

```bash
conda activate SpatialTranscriptFormer
pytest tests/
```

Or using the provided PowerShell script:

```powershell
.\test.ps1
```

## Test Files

- `tests/test_download.py`: Unit tests for `download.py`. Verifies:
  - Metadata downloading and existence checks.
  - Sample filtering logic (by Organ, Disease, Technology).
  - Pattern generation for HEST subsets.
  - Unzipping logic for segmentation files.
  
- `tests/test_splitting_logic.py`: Tests for `splitting.py`. Verifies:
  - Patient-level splitting (train/val/test).
  - Leakage prevention (ensuring patients don't overlap between splits).

## Adding New Tests

When adding new functionality, please add corresponding tests in the `tests/` directory.
- Use `unittest` or `pytest` style tests.
- Mock external calls (like `huggingface_hub` or large file I/O) to keep tests fast and offline-capable where possible.
