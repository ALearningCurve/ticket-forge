# Ticket Forge
TicketForge is an AI-Powered DevOps ticket assignment system capable of automating the time-consuming manual process of assigning tickets. It can recommend optimal assignments based on engineer skills, past performances and ticket requirements.

## Overview
Currently, teams spend 15-20 minutes per ticket (or a backlog item) for triaging. To triage means to understand what a ticket is, what is its priority and thinking of a suitable engineer for that. Instead of thinking about how to resolve the ticket, we wasted time on triaging. 
TicketForge is an AI-Powered DevOps ticket assignment system capable of automating the time-consuming manual process of assigning tickets. It can recommend optimal assignments based on engineer skills, past performances and ticket requirements.

### Folder Structure

> [!NOTE]
> Each folder contains documentation in the form of a `README.md` for how to run the apps/libs etc (or will contain it if not present already).

```
.
├── apps
│   ├── training         # ML Pipeline: Data scraping, ETL, and model training jobs
│   ├── web-backend      # API Service: Serves model predictions & business logic
│   └── web-frontend     # User Interface: Dashboard for interacting with the model
├── libs
│   ├── ml-core          # Shared Logic: Data schemas, scrapers, and transforms
│   └── shared           # Utilities: Logging, DB clients, and global constants
├── terraform            # IaC: Cloud resource provisioning
├── data                 # Local Data: (Git-ignored) raw and processed datasets
├── models               # Local Model Registry: (Git-ignored) serialized weights/pickles
├── notebooks            # R&D: Data exploration and model prototyping
├── pyproject.toml       # Workspace Config: Links apps and libs via uv
├── uv.lock              # Pinned Python dependencies
├── package.json         # Node.js dependencies for JS projects
├── package-lock.json
├── LICENSE
└── README.md
```

## Installation
(will update once product is complete)

> [!IMPORTANT]
> This project uses [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/) and [npm workspaces](https://docs.npmjs.com/cli/v8/using-npm/workspaces) since this project is laid out like a monorepo. Make sure you are familiar with both before continuing (i.e. make sure you know where to run install and package add commands)!
> [!NOTE]

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), [node](https://nodejs.org/en/download), and npm 

2. Install project level dependencies
```sh
uv sync
npm i
```

3. Good to go!

## Usage

Will update once more code is added.

## Development

This project includes linting, type checking, and testing tools to ensure code quality.

### Running Tests

To run all tests:
```sh
uv run pytest
```

To run tests with verbose output:
```sh
uv run pytest -v
```

### Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting.

To check for linting issues:
```sh
uv run ruff check .
```

To automatically fix linting issues:
```sh
uv run ruff check --fix .
```

To format code:
```sh
uv run ruff format .
```

### Type Checking

This project uses [Pyright](https://github.com/microsoft/pyright) for static type checking.

To run type checking:
```sh
uv run pyright
```

To check specific directories:
```sh
uv run pyright libs/ apps/
```

### Running All Checks

To run all quality checks (linting, type checking, and tests):
```sh
uv run ruff check . && uv run pyright && uv run pytest
```

