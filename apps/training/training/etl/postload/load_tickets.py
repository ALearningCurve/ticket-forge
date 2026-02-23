"""Run scrape -> transform -> Postgres load for GitHub tickets."""

import argparse
import asyncio
import os
from typing import Iterable

import psycopg2
from psycopg2.extras import Json
from training.etl.ingest.scrape_github_issues_improved import scrape_all_issues
from training.etl.transform.run_transform import transform_records

EMBEDDING_DIM = 384


def _vector_to_pg(value: object) -> str:
  """Convert an embedding to pgvector text format `[x,y,...]`."""
  if value is None:
    msg = "embedding is missing"
    raise ValueError(msg)

  if hasattr(value, "tolist"):
    value = value.tolist()

  if not isinstance(value, list):
    msg = f"embedding must be a list, got: {type(value).__name__}"
    raise TypeError(msg)

  if len(value) != EMBEDDING_DIM:
    msg = f"embedding must have {EMBEDDING_DIM} values, got {len(value)}"
    raise ValueError(msg)

  return "[" + ",".join(map(str, value)) + "]"


def _labels_to_json(value: object) -> list[str]:
  """Normalize labels into a JSONB-friendly list of strings."""
  if value is None:
    return []

  if isinstance(value, list):
    return [str(v).strip() for v in value if str(v).strip()]

  if isinstance(value, str):
    if not value.strip():
      return []
    return [part.strip() for part in value.split(",") if part.strip()]

  return [str(value)]


def _map_status(issue_type: object, state: object) -> str:
  """Map source issue state to ticket_status enum."""
  issue_type_s = str(issue_type or "").strip().lower()
  state_s = str(state or "").strip().lower()

  if issue_type_s == "closed" or state_s == "closed":
    return "closed"
  if issue_type_s == "open_assigned":
    return "in-progress"
  return "open"


def upsert_tickets(
  tickets: Iterable[dict],
  dsn: str | None = None,
) -> int:
  """Upsert transformed tickets into the `tickets` table."""
  resolved_dsn = dsn or os.environ.get("DATABASE_URL")
  if not resolved_dsn:
    msg = "No Postgres DSN provided. Pass `dsn` or set DATABASE_URL."
    raise RuntimeError(msg)

  sql = """
  INSERT INTO tickets (
    ticket_id,
    title,
    description,
    ticket_vector,
    labels,
    status,
    resolution_time_actual,
    created_at,
    updated_at
  )
  VALUES (
    %s,
    %s,
    %s,
    %s::vector,
    %s::jsonb,
    %s::ticket_status,
    CASE WHEN %s IS NULL THEN NULL ELSE make_interval(hours => %s) END,
    COALESCE(%s::timestamptz, now()),
    now()
  )
  ON CONFLICT (ticket_id)
  DO UPDATE SET
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    ticket_vector = EXCLUDED.ticket_vector,
    labels = EXCLUDED.labels,
    status = EXCLUDED.status,
    resolution_time_actual = EXCLUDED.resolution_time_actual,
    created_at = EXCLUDED.created_at,
    updated_at = now()
  """

  processed = 0
  conn = psycopg2.connect(resolved_dsn)

  try:
    with conn, conn.cursor() as cur:
      for ticket in tickets:
        ticket_id = str(ticket.get("id", "")).strip()
        if not ticket_id:
          continue

        title = str(ticket.get("title") or "")
        description = str(ticket.get("normalized_text") or ticket.get("body") or "")
        vector_text = _vector_to_pg(ticket.get("embedding"))
        labels_json = Json(_labels_to_json(ticket.get("labels")))
        status = _map_status(ticket.get("issue_type"), ticket.get("state"))

        hours = ticket.get("completion_hours_business")
        if hours is not None:
          try:
            hours = float(hours)
          except (TypeError, ValueError):
            hours = None

        created_at = ticket.get("created_at")

        cur.execute(
          sql,
          (
            ticket_id,
            title,
            description,
            vector_text,
            labels_json,
            status,
            hours,
            hours,
            created_at,
          ),
        )
        processed += 1
  except Exception:
    conn.rollback()
    raise
  finally:
    conn.close()

  return processed


async def run_pipeline(
  dsn: str | None = None,
  limit_per_state: int | None = None,
) -> int:
  """Execute scrape -> transform -> upsert pipeline end-to-end."""
  print("Step 1/3: Scraping GitHub issues...")
  raw_records = await scrape_all_issues(limit_per_state=limit_per_state)
  print(f"Scraped {len(raw_records)} records")

  print("Step 2/3: Transforming records...")
  transformed = transform_records(raw_records)
  print(f"Transformed {len(transformed)} records")

  print("Step 3/3: Upserting into Postgres...")
  loaded = upsert_tickets(transformed, dsn=dsn)
  print(f"Upserted {loaded} ticket(s) into Postgres")
  return loaded


def main() -> None:
  """CLI runner for end-to-end ticket pipeline."""
  parser = argparse.ArgumentParser(
    description="Run GraphQL scrape, transform, and Postgres load"
  )
  parser.add_argument(
    "--dsn",
    default=None,
    help="Postgres DSN (defaults to DATABASE_URL env var)",
  )
  parser.add_argument(
    "--limit-per-state",
    type=int,
    default=None,
    help="Optional cap per repo/state for quick testing",
  )
  args = parser.parse_args()

  asyncio.run(run_pipeline(dsn=args.dsn, limit_per_state=args.limit_per_state))


if __name__ == "__main__":
  main()
