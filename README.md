# Ticket Forge
TicketForge is an AI-Powered DevOps ticket assignment system capable of automating the time-consuming manual process of assigning tickets. It can recommend optimal assignments based on engineer skills, past performances and ticket requirements.

## Overview
Currently, teams spend 15-20 minutes per ticket (or a backlog item) for triaging. To triage means to understand what a ticket is, what is its priority and thinking of a suitable engineer for that. Instead of thinking about how to resolve the ticket, we wasted time on triaging. 
TicketForge is an AI-Powered DevOps ticket assignment system capable of automating the time-consuming manual process of assigning tickets. It can recommend optimal assignments based on engineer skills, past performances and ticket requirements.

### Folder Structure
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
├── models               # Model Registry: (Git-ignored) serialized weights/pickles
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

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), [node](https://nodejs.org/en/download), and npm 

```sh
uv sync
npm i
```

## Usage
(will update once product is complete)
