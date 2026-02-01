default: install-deps

# install all 3rd party packages
install-deps:
  uv sync --all-packages
  npm i
  uv run pre-commit install

# Runs python tests. Any args are forwarded to pytest.
[positional-arguments]
@pytest *args='':
  uv run pytest "$@"

# Runs python linting. Specify the directories/files to lint as positional args.
[positional-arguments]
@pylint *args=".":
  uv run ruff check --fix "$@"
  uv run ruff format "$@"
  uv run pyright "$@"

# Run all python checks on particular files and directories
pycheck *args=".":
  just pylint "$@"
  just pytest "$@"

# Run pre-commit hooks
[positional-arguments]
@precommit *args='run':
  uv run pre-commit "$@"



# runs all checks on the repo from repo-root
check:
  just pycheck .
