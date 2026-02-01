# install all 3rd party packages
default: install-deps

# install all 3rd party packages
install-deps:
  uv sync --all-packages
  npm i

# Runs python tests. Any args are forwarded to pytest.
[positional-arguments]
@pytest *args='': 
  uv run pytest "$@"

# Runs python linting. Specify the directory to lint with dir.
pylint dir=".": 
  uv run ruff check --fix {{dir}}
  uv run ruff format {{dir}}
  uv run pyright {{dir}}

# runs all checks on the repo
check-all:
  just pylint
  just pytest