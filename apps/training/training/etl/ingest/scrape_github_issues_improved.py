"""GraphQL scraper for all issue types from all repos."""

import asyncio
import os

import httpx
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from shared.configuration import Paths
from tqdm import tqdm


# --- Data Model ---
class GitHubIssue(BaseModel):
  """Represents a GitHub issue."""

  id: str
  repo: str
  title: str
  body: str | None = ""
  labels: str
  assignee: str | None
  state: str
  issue_type: str  # "closed", "open_assigned", "open_unassigned"
  created_at: str
  assigned_at: str | None = None
  closed_at: str | None = None
  comments_count: int = Field(alias="comments")
  url: str = Field(alias="html_url")

  class Config:
    """Pydantic configuration."""

    populate_by_name = True


# --- Configuration ---
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
  msg = "GITHUB_TOKEN missing."
  raise RuntimeError(msg)

REPOS = [
  ("hashicorp", "terraform"),
  ("ansible", "ansible"),
  ("prometheus", "prometheus"),
]

HEADERS = {
  "Authorization": f"Bearer {GITHUB_TOKEN}",
  "Content-Type": "application/json",
}

GRAPHQL_URL = "https://api.github.com/graphql"


# --- GraphQL Query ---
def build_query(owner: str, name: str, state: str, cursor: str | None = None) -> dict:
  """Build GraphQL query for issues."""
  after_clause = f', after: "{cursor}"' if cursor else ""

  query = f"""
    query {{
      repository(owner: "{owner}", name: "{name}") {{
        issues(
            first: 100, 
            states: {state}, 
            orderBy: {{field: CREATED_AT, direction: DESC}}{after_clause}
        ) {{
          pageInfo {{
            hasNextPage
            endCursor
          }}
          nodes {{
            number
            title
            body
            state
            createdAt
            closedAt
            url
            comments {{
              totalCount
            }}
            labels(first: 10) {{
              nodes {{
                name
              }}
            }}
            assignees(first: 5) {{
              nodes {{
                login
              }}
            }}
            timelineItems(itemTypes: [ASSIGNED_EVENT], first: 1) {{
              nodes {{
                ... on AssignedEvent {{
                  createdAt
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """

  return {"query": query}


async def scrape_repo_state(  # noqa: PLR0915
  client: httpx.AsyncClient,
  owner: str,
  name: str,
  state: str,
  limit: int | None = None,
) -> list[GitHubIssue]:
  """Scrape issues of a specific state from a repo."""
  repo_full = f"{owner}/{name}"
  issues_list: list[GitHubIssue] = []
  cursor: str | None = None
  page = 1

  desc = f"{repo_full} ({state.lower()})"
  pbar = tqdm(unit="issue", desc=desc, leave=False)

  while True:
    if limit and len(issues_list) >= limit:
      break

    query = build_query(owner, name, state, cursor)
    response = await client.post(GRAPHQL_URL, json=query)

    # Rate limit handling
    if response.status_code == 403:
      pbar.write("⚠️  Rate limited. Waiting 60s...")
      await asyncio.sleep(60)
      continue

    if response.status_code != 200:
      pbar.write(f"❌ Error {response.status_code}")
      break

    data = response.json()

    if "errors" in data:
      pbar.write(f"❌ GraphQL errors: {data['errors']}")
      break

    repo_data = data["data"]["repository"]
    issues_data = repo_data["issues"]
    nodes = issues_data["nodes"]
    page_info = issues_data["pageInfo"]

    for item in nodes:
      if limit and len(issues_list) >= limit:
        break

      # Get assignee
      assignees = item.get("assignees", {}).get("nodes", [])
      assignee = assignees[0]["login"] if assignees else None

      # Get assignment timestamp
      timeline_items = item.get("timelineItems", {}).get("nodes", [])
      assigned_at = timeline_items[0].get("createdAt") if timeline_items else None

      # Get labels
      labels_nodes = item.get("labels", {}).get("nodes", [])
      labels_str = ",".join([label["name"] for label in labels_nodes])

      # Determine issue type
      if state == "CLOSED":
        issue_type = "closed"
      elif assignee:
        issue_type = "open_assigned"
      else:
        issue_type = "open_unassigned"

      issue = GitHubIssue(
        id=f"{owner}_{name}-{item['number']}",
        repo=repo_full,
        title=item["title"],
        body=item.get("body"),
        labels=labels_str,
        assignee=assignee,
        state=item["state"].lower(),
        issue_type=issue_type,
        created_at=item["createdAt"],
        assigned_at=assigned_at,
        closed_at=item.get("closedAt"),
        comments=item["comments"]["totalCount"],
        html_url=item["url"],
      )

      issues_list.append(issue)
      pbar.update(1)

    pbar.set_postfix({"page": page, "total": len(issues_list)})

    if not page_info["hasNextPage"]:
      break

    cursor = page_info["endCursor"]
    page += 1
    await asyncio.sleep(0.3)

  pbar.close()
  return issues_list


async def main() -> None:
  """Scrape all repos and save to single JSON file."""
  print("🚀 Starting comprehensive issue scraping...\n")

  all_issues: list[GitHubIssue] = []

  async with httpx.AsyncClient(headers=HEADERS, timeout=60.0) as client:
    for owner, name in REPOS:
      repo_full = f"{owner}/{name}"
      print(f"\n📦 Processing {repo_full}...")

      # Fetch closed issues
      closed = await scrape_repo_state(client, owner, name, "CLOSED", None)

      # Fetch open issues
      open_all = await scrape_repo_state(client, owner, name, "OPEN", None)

      # Combine
      all_issues.extend(closed)
      all_issues.extend(open_all)

      print(f"  ✅ Total from {repo_full}: {len(closed) + len(open_all)}")

  if not all_issues:
    print("\n❌ No issues found")
    return

  # Convert to DataFrame
  df = pd.DataFrame([issue.model_dump() for issue in all_issues])

  # Save to single JSON file
  data_dir = Paths.data_root / "github_issues"
  data_dir.mkdir(parents=True, exist_ok=True)
  output = data_dir / "all_tickets.json"
  df.to_json(output, orient="records", indent=2)

  print("\n" + "=" * 80)
  print("📊 FINAL STATISTICS")
  print("=" * 80)

  total = len(df)
  type1 = len(df[df["issue_type"] == "closed"])
  type2 = len(df[df["issue_type"] == "open_assigned"])
  type3 = len(df[df["issue_type"] == "open_unassigned"])

  print(f"\nTotal issues: {total}")
  print(f"  Type 1 - Closed (ML training): {type1}")
  print(f"  Type 2 - Open + Assigned (in-progress): {type2}")
  print(f"  Type 3 - Open + Unassigned (backlog): {type3}")

  assigned_at_count = df["assigned_at"].notna().sum()
  print(
    f"\nAssignment timestamps captured: {assigned_at_count}/{total} "
    f"({assigned_at_count / total * 100:.1f}%)"
  )

  print(f"\n💾 Saved to: {output}")
  print("\n🎉 Scraping complete!")


if __name__ == "__main__":
  asyncio.run(main())
