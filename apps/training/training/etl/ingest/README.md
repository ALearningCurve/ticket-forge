# GitHub Issue Ingestion

This module ingests closed GitHub issues with assignees and produces
a compressed JSON dataset for downstream processing.

## Setup
- Create a GitHub Personal Access Token (classic)
- Add it to a .env file as GITHUB_TOKEN

## Run
python scrape_github_issues.py
python csv_to_json.py

## Outputs
- data/github_issues/tickets_raw.csv
- data/github_issues/tickets_final.json.gz

> Note: data/ is git-ignored and artifacts are not committed.