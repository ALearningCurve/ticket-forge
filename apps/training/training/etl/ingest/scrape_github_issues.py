# scrape_github_issues.py

import os
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from github import Auth, Github
from shared.configuration import Paths
from shared.logging import get_logger

logger = get_logger(__name__)

# Load GitHub token
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


if not GITHUB_TOKEN:
  msg = "GITHUB_TOKEN missing. Add it to a .env file."
  logger.error(msg)
  raise RuntimeError(msg)

auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)

logger.info(f"✅ Authenticated as: {g.get_user().login}")

# Repos defining the issue board
REPOS = ["hashicorp/terraform", "ansible/ansible", "prometheus/prometheus"]

SCRAPE_LIMIT_PER_REPO = 300


def scrape_repo_closed_issues(repo_name: str, limit: int) -> list[dict[str, Any]]:
  """Scrape closed issues with assignees from a GitHub repository.

  Args:
      repo_name: The name of the repository to scrape.
      limit: The maximum number of issues to scrape.

  Returns:
      A list of dictionaries containing issue data.
  """
  logger.info(f"📦 Scraping CLOSED issues with assignees from {repo_name}")
  repo = g.get_repo(repo_name)

  issues = repo.get_issues(state="closed", sort="created", direction="desc")
  rows = []

  count = 0
  for issue in issues:
    # Skip pull requests
    if issue.pull_request:
      continue

    # Skip issues with NO assignee
    if issue.assignee is None:
      continue

    rows.append(
      {
        "id": f"{repo_name.replace('/', '_')}-{issue.number}",
        "repo": repo_name,
        "title": issue.title,
        "body": issue.body,
        "labels": ",".join([label.name for label in issue.labels]),
        "assignee": issue.assignee.login,
        "created_at": issue.created_at.isoformat(),
        "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
        "comments_count": issue.comments,
        "url": issue.html_url,
      }
    )

    count += 1
    if count >= limit:
      break

    # Be nice to GitHub API
    time.sleep(0.5)

  logger.info(f"  ✅ Scraped {len(rows)} closed issues with assignees")
  return rows


# --------------------
# Main execution
# --------------------
all_rows = []

for repo in REPOS:
  all_rows.extend(scrape_repo_closed_issues(repo, SCRAPE_LIMIT_PER_REPO))

DATA_DIR = Paths.data_root / "github_issues"
DATA_DIR.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(all_rows)
output_csv = DATA_DIR / "tickets_raw.csv"
df.to_csv(output_csv, index=False)

logger.info("🎉 Scraping complete")
logger.info(f"📄 tickets_raw.csv created at {output_csv} with {len(df)} total tickets")


# --------------------
# Assignee summary
# --------------------
logger.info("👤 Assignee summary:")

assignee_counts = df["assignee"].value_counts()

for assignee, count in assignee_counts.items():
  logger.info(f"  - {assignee}: {count} tickets")

logger.info(f"📊 Total unique assignees: {assignee_counts.size}")
