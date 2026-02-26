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

## Contributor Guidelines: Adding New Tests

When adding new functionality, please add corresponding tests in the `tests/` directory.

- **Framework**: Use `unittest` or `pytest` style tests.
- **Mocking**: Mock external calls (like `huggingface_hub` or large file I/O) to keep tests fast and offline-capable where possible.
- **Discussion**: We are always looking for ways to improve our testing practices! If you have ideas for better test architecture, coverage strategies, or tooling, please feel free to open a discussion or issue.

### Mutation Testing

While standard unit tests ensure the code behaves as expected under specific conditions, they don't always guarantee the robustness of the tests themselves.

We strongly encourage **Mutation Testing** when contributing critical components. Mutation testing introduces small changes (mutations) into the source code and checks if your tests catch them (by failing). If a test still passes despite the mutated code, our tests may not be tight enough!

Our preferred method for mutation testing in Python is **[cosmic-ray](https://github.com/sixty-north/cosmic-ray)**.

To get started with `cosmic-ray`:

1. Install it via pip: `pip install cosmic-ray`
2. Initialize a configuration file for your module.
3. Run the mutation tests and review the survival report to strengthen your test suites.

If you find ways to automate or better integrate mutation testing into our CI pipeline, we would welcome those discussions!
