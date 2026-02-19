# Data Ingestion

This directory contains scripts for ingesting data from external sources.

## GitHub Issue Scraping

### Active Scraper

**Use:** `scrape_github_issues_graphql.py`

This is the current production scraper that uses GitHub's GraphQL API for fast, comprehensive data collection. It captures:
- All 3 issue types (closed, open+assigned, open+unassigned)
- Assignment timestamps from timeline events
- All issues from Terraform, Ansible, and Prometheus repos

**Output:** `data/github_issues/all_tickets.json`

**Run:**
```bash
python apps/training/training/etl/ingest/scrape_github_issues_graphql.py
```

### Legacy Scrapers

- `scrape_github_issues.py` - Original REST API version (deprecated, slower)
- Other test scripts - For exploration only, not production use

## Usage

1. Ensure `GITHUB_TOKEN` is set in `.env` file
2. Run the GraphQL scraper to collect fresh data
3. Use the output (`all_tickets.json`) for transformation pipeline