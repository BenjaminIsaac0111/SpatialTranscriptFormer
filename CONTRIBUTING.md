# Contributing to SpatialTranscriptFormer

Thank you for your interest in contributing! As a project at the intersection of deep learning and pathology, we value rigorous, well-tested contributions.

## Project Status

> [!IMPORTANT]
> This project is a **Work in Progress**. We are actively refining the core interaction logic and scaling behaviors. Expect breaking changes in the CLI and data schemas.

## Intellectual Property & Licensing

SpatialTranscriptFormer is protected under a **Proprietary Source Code License**.

- **Academic/Non-Profit**: We encourage contributions from the research community. Contributions made under an academic affiliation are generally welcome.
- **Commercial/For-Profit**: Contributions from commercial entities or individuals intended for profit-seeking use require a separate agreement.
- **Assignment**: By submitting a Pull Request, you agree that your contributions will be licensed under the project's existing license, granting the author the right to include them in both the open-access and proprietary versions of the software.

## Development Workflow

### 1. Environment Setup

Use the provided setup scripts to ensure a consistent development environment:

```bash
# Windows
.\setup.ps1

# Linux/HPC
bash setup.sh
```

### 2. Coding Standards

We use `black` for formatting and `flake8` for linting. Please ensure your code passes these checks before submitting.

```bash
black .
flake8 src/
```

### 3. Testing

All new features must include unit tests in the `tests/` directory. We use `pytest` for our test suite.

```bash
# Run all tests
.\test.ps1  # Windows
bash test.sh # Linux
```

## Pull Request Process

1. **Open an Issue**: For major changes, please open an issue first to discuss the design.
2. **Branching**: Work on a descriptive feature branch (e.g., `feature/pathway-attention-mask`).
3. **Documentation**: Update relevant files in `docs/` and the `README.md` if your change affects usage.
4. **Verification**: Ensure all CI checks (GitHub Actions) pass.

## Contact

For questions regarding commercial licensing or complex architectural changes, please contact the author directly.
