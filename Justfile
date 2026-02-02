# Configure repository and install dependencies
default: install-deps

# install all 3rd party packages
[group: 'lang-agnostic']
install-deps:
  uv sync --all-packages
  npm i
  uv run pre-commit install

# Runs python tests. Any args are forwarded to pytest.
[group: 'python']
[positional-arguments]
pytest *args='':
  uv run pytest "$@"

# Runs python linting. Specify the directories/files to lint as positional args.
[group: 'python']
[positional-arguments]
pylint *args=".":
  uv run ruff check --fix "$@"
  uv run ruff format "$@"
  uv run pyright "$@"

# Run all python checks on particular files and directories
[group: 'python']
pycheck *args=".":
  just pylint "$@"
  just pytest "$@"

# Run pre-commit hooks
[group: 'lang-agnostic']
[positional-arguments]
precommit *args='run':
  uv run pre-commit "$@"



# runs all checks on the repo from repo-root
[group: 'lang-agnostic']
check:
  just pycheck .

# runs the training script
[group: 'data-pipeline']
[positional-arguments]
train *args='':
  uv run apps/training/training/cmd/train.py "$@"
