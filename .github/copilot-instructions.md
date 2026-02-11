# Copilot Instructions for TicketForge

## Project Overview

TicketForge is an AI-Powered DevOps ticket assignment system built as a monorepo. It automates ticket triaging and assignment by recommending optimal assignments based on engineer skills, past performances, and ticket requirements.

## Repository Structure

This is a **monorepo** using both **uv workspaces** (Python) and **npm workspaces** (JavaScript).

```
.
├── apps/
│   ├── training/         # ML Pipeline: Data scraping, ETL, and model training
│   ├── web-backend/      # API Service: FastAPI for model predictions & logic
│   └── web-frontend/     # User Interface: Astro-based dashboard
├── libs/
│   ├── ml-core/         # Shared Logic: Data schemas, scrapers, transforms
│   └── shared/          # Utilities: Logging, DB clients, constants
├── terraform/           # Infrastructure as Code
├── data/                # Local data (git-ignored)
├── models/              # Local model registry (git-ignored)
└── notebooks/           # R&D and data exploration
```

## Technology Stack

### Python
- **Python Version**: 3.12+
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (NOT pip)
- **Web Framework**: FastAPI (in web-backend)
- **ML Libraries**: scikit-learn, pandas, etc. (in training)
- **Testing**: pytest
- **Linting**: Ruff (formatting + linting)
- **Type Checking**: Pyright

### JavaScript/TypeScript
- **Runtime**: Node.js 22
- **Package Manager**: npm
- **Frontend Framework**: Astro (in web-frontend)

### Infrastructure
- **IaC**: Terraform
- **Task Runner**: [Just](https://github.com/casey/just) (Justfile)

## Development Workflow

### Installation
Always run from repo root:
```bash
just  # Installs all dependencies for all workspaces
```

### Adding Dependencies

**Python packages** - Navigate to the specific workspace directory:
```bash
cd apps/web-backend
uv add <package>  # For runtime dependencies
uv add --dev <package>  # For dev dependencies
```

**JavaScript packages** - Use npm workspaces:
```bash
npm install <package> -w apps/web-frontend
```

### Running Commands

Use the `Justfile` for all common tasks. Run `just --list` to see available commands.

**Important commands:**
- `just check` - Run all checks (linting, type checking, tests)
- `just pylint [path]` - Lint Python code (uses Ruff + Pyright)
- `just pytest [args]` - Run Python tests
- `just pycheck [path]` - Run all Python checks (lint + test)
- `just train` - Run ML training pipeline
- `just precommit` - Run pre-commit hooks

### Testing

- Always run tests before committing: `just pytest`
- Tests are configured in `pyproject.toml` under `[tool.pytest.ini_options]`
- Test timeout is 30 seconds by default
- Python paths include all workspace directories

### Code Quality Standards

#### Python Style Guide
- **Line Length**: 88 characters (Ruff default)
- **Indentation**: 2 spaces
- **Type Hints**: Required (enforced by Pyright)
- **Docstrings**: Google style (enforced by Ruff)
- **Import Style**: Absolute imports only (no relative imports)

#### Linting Rules
Ruff is configured to enforce:
- PEP 8 compliance (E, F)
- Documentation (D) - Google style
- Type annotations (ANN)
- Bug detection (B)
- Logging best practices (LOG)
- Error handling (TRY, EM)
- Import sorting (I)

**Exceptions:**
- D104: Missing docstring in public package
- D100: Missing docstring in public module
- D205: 1 blank line required between summary line and description
- TRY002: Create your own exception
- PLR2004: Magic value used in comparison

### Pre-commit Hooks

This project uses pre-commit hooks. They are automatically installed with `just`.

Run manually: `just precommit`

## Important Notes for AI Assistants

1. **Monorepo Awareness**: When modifying dependencies, always work in the correct workspace directory. Never run `uv add` from the repo root.

2. **Use Just Commands**: Always prefer `just` commands over direct tool invocation:
   - Use `just pylint` instead of `uv run ruff`
   - Use `just pytest` instead of `uv run pytest`

3. **Testing**: Always run tests after making changes to ensure nothing breaks.

4. **Type Safety**: All Python code requires type hints. Use Pyright for verification.

5. **Code Style**: Let Ruff handle formatting. Don't manually format code - run `just pylint` instead.

6. **Import Style**: Always use absolute imports (e.g., `from apps.training.module import func`, not `from .module import func`).

7. **Documentation**: Follow Google-style docstrings for all public functions and classes.

8. **Git Ignored Directories**:
   - `data/` - Local datasets
   - `models/` - Trained model files
   - Standard ignores: `__pycache__`, `.pytest_cache`, `node_modules/`, etc.

## Commit Message Format

Follow conventional commit format:
```
[#issue] type: description

Example: [#2] feat: implement GitHub issue scraper
```

Types: feat, fix, docs, style, refactor, test, chore

## Common Pitfalls to Avoid

1. Don't use `pip` - use `uv` for Python packages
2. Don't run `uv add` from repo root - navigate to the workspace first
3. Don't use relative imports - use absolute imports only
4. Don't skip type hints - Pyright is strict
5. Don't manually format - use `just pylint`
6. Don't modify working files without understanding workspace context
