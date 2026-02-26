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
```

### 4. AI-Assisted Development

We welcome contributions developed with the assistance of AI tools (e.g., Copilot, ChatGPT, Claude, or agentic frameworks). However, to ensure the long-term maintainability and integrity of the project:

- **Ownership**: You are ultimately responsible for the code you submit. Do not commit code you do not fully understand.
- **Explainability**: During the review process, you must be able to explain the logic, design decisions, and any subtle side effects of the AI-suggested changes.
- **Verification**: AI-generated code must strictly follow our coding standards, naming conventions, and architectural patterns. It must be accompanied by robust tests (see our [Testing Guide](docs/TESTING.md)).

### 3. Testing & Quality Assurance

All new features must be accompanied by relevant tests in the `tests/` directory natively using `pytest`.

We highly encourage rigorous testing approaches such as **Mutation Testing** (via `cosmic-ray`) for critical model components to prevent surviving mutants.

For full details on our testing requirements, how to run the test suites locally, and our guidelines on mutation testing, please read the [Testing Guide](docs/TESTING.md).

## Pull Request Process

1. **Open an Issue**: For major changes, please open an issue first to discuss the design.
2. **Branching**: Work on a descriptive feature branch (e.g., `feature/pathway-attention-mask`).
3. **Documentation**: Update relevant files in `docs/` and the `README.md` if your change affects usage.
4. **Verification**: Ensure all CI checks (GitHub Actions) pass.

### Branch Protections

To maintain code quality and stability, the following protections are enforced on the `main` branch:

- **Require Pull Request Reviews**: All merges to `main` require at least one approval from a project maintainer.
- **Required Status Checks**: The `CI` workflow must pass successfully before a PR can be merged. This includes formatting checks (`black`) and the full test suite (`pytest`).
- **No Direct Pushes**: Pushing directly to `main` is disabled. All changes must go through the Pull Request process.
- **Linear History**: We prefer **Squash and Merge** to keep the `main` branch history clean and concise.

## Contact

For questions regarding commercial licensing or complex architectural changes, please contact the author directly.
