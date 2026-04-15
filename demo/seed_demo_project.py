#!/usr/bin/env python3
"""Seed sample tickets into a demo project with demo users.

This is the primary Issue #122 seeding path:
- replace (recreate) demo users referenced by sample data,
- create/reset one demo project board,
- add relevant users as project members,
- seed sample tickets into that project board.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import psycopg2
from dotenv import load_dotenv
from ml_core.embeddings import get_embedding_service
from ml_core.keywords import get_keyword_extractor
from ml_core.profiles import ProfileUpdater
from ml_core.retrieval.hybrid_retrieval import vector_to_pgvector_text
from pipelines.etl.dsn import resolve_postgres_dsn
from shared.logging import get_logger
from web_backend.security.hashing import hash_password

logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH)

DEMO_EMAIL_DOMAIN = "demo.ticketforge.local"
CANONICAL_COLUMNS = ("Backlog", "To Do", "In Progress", "In Review", "Done")
COL_DONE = "done"
COL_PROGRESS = "in progress"
COL_BACKLOG = "backlog"
COL_TODO = "to do"
REQUIRED_TF_ENV = ("TF_VAR_project_id", "TF_VAR_state_bucket")


@dataclass(frozen=True)
class SeedResult:
  """Aggregate result payload for one run."""

  project_slug: str
  project_id: str
  users_recreated: int
  users_retained: int
  tickets_seeded: int
  tickets_assigned: int
  tickets_unassigned: int
  users_credentials: list[dict[str, str]]


@dataclass(frozen=True)
class _UserProfileSeed:
  """Derived profile payload for one github username."""

  github_username: str
  full_name: str
  resume_base_vector_text: str
  profile_vector_text: str
  keywords_text: str
  tickets_closed_count: int


def _write_credentials_file(
  *,
  path: Path,
  project_slug: str,
  project_id: str,
  credentials: list[dict[str, str]],
) -> None:
  """Persist username/password mapping to disk."""
  payload = {
    "generated_at": datetime.now(tz=UTC).isoformat(),
    "project_slug": project_slug,
    "project_id": project_id,
    "credentials": credentials,
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_proxy_dsn(tf_outputs_text: str) -> str:
  """Extract Proxy DSN from `just tf-outputs` output."""
  pattern = r"Proxy DSN[^\n]*:\n(postgresql://[^\s]+)"
  match = re.search(pattern, tf_outputs_text, flags=re.MULTILINE)
  if match is None:
    msg = "Could not parse Proxy DSN from `just tf-outputs` output"
    raise RuntimeError(msg)
  return match.group(1).strip()


def _build_subprocess_env() -> dict[str, str]:
  """Build subprocess env from shell/.env and validate terraform vars."""
  env = dict(os.environ)
  missing = [name for name in REQUIRED_TF_ENV if not env.get(name)]
  if missing:
    msg = "Missing required environment variables (set in .env): " + ", ".join(missing)
    raise RuntimeError(msg)
  return env


def _run_command(
  command: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
  """Run a command and raise with stderr/stdout details on failure."""
  result = subprocess.run(  # noqa: S603
    command,
    check=False,
    capture_output=True,
    text=True,
    env=env,
    cwd=str(REPO_ROOT),
  )
  if result.returncode != 0:
    stderr = result.stderr.strip() or result.stdout.strip()
    msg = f"Command failed: {' '.join(command)} :: {stderr}"
    raise RuntimeError(msg)
  return result


def _read_tf_output_raw(name: str, env: dict[str, str]) -> str:
  """Read one terraform output value as raw text."""
  result = _run_command(["just", "tf", "output", "-raw", name], env)
  value = result.stdout.strip()
  if not value:
    msg = f"Terraform output `{name}` is empty"
    raise RuntimeError(msg)
  return value


def _resolve_seed_dsn(explicit_dsn: str | None, env: dict[str, str]) -> str:
  """Resolve DSN from explicit arg or terraform outputs."""
  if explicit_dsn:
    logger.info("Using explicit DSN provided via --dsn.")
    return resolve_postgres_dsn(explicit_dsn)

  logger.info("Resolving DSN from terraform outputs (`just tf-outputs`).")
  _run_command(["just", "tf-init"], env)
  tf_outputs = _run_command(["just", "tf-outputs"], env).stdout
  return resolve_postgres_dsn(_extract_proxy_dsn(tf_outputs))


def _dsn_host_port(dsn: str) -> tuple[str, int]:
  """Extract host/port from Postgres DSN."""
  parsed = urlparse(dsn)
  host = parsed.hostname or "127.0.0.1"
  port = int(parsed.port or 5432)
  return host, port


def _is_port_open(host: str, port: int) -> bool:
  """Return True when TCP port is reachable."""
  with socket.socket() as sock:
    sock.settimeout(1.0)
    try:
      sock.connect((host, port))
    except OSError:
      return False
    else:
      return True


def _start_cloud_sql_proxy(
  *,
  env: dict[str, str],
  local_port: int,
  timeout_seconds: float,
) -> subprocess.Popen[str]:
  """Start Cloud SQL proxy and wait until localhost port is reachable."""
  proxy_env = dict(env)
  if not proxy_env.get("AIRFLOW_VM_NAME"):
    proxy_env["AIRFLOW_VM_NAME"] = _read_tf_output_raw("airflow_vm_instance_name", env)
  if not proxy_env.get("TF_VAR_airflow_zone"):
    proxy_env["TF_VAR_airflow_zone"] = _read_tf_output_raw("airflow_vm_zone", env)

  process = subprocess.Popen(  # noqa: S603
    ["just", "gcp-proxy", "cloud-sql", str(local_port)],
    cwd=str(REPO_ROOT),
    env=proxy_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
  )

  deadline = time.time() + max(timeout_seconds, 1.0)
  while time.time() < deadline:
    if _is_port_open("127.0.0.1", local_port):
      logger.info("Cloud SQL proxy is ready on 127.0.0.1:%s", local_port)
      return process
    if process.poll() is not None:
      break
    time.sleep(1.0)

  output_tail = ""
  if process.stdout is not None:
    output_tail = process.stdout.read()[-4000:]
  if process.poll() is None:
    process.terminate()
  msg = (
    f"Cloud SQL proxy failed to start on port {local_port}. Last output: {output_tail}"
  )
  raise RuntimeError(msg)


def _load_records(path: Path) -> list[dict[str, Any]]:
  """Load a JSON list of ticket records."""
  if not path.exists():
    msg = f"Input file not found: {path}"
    raise RuntimeError(msg)
  raw_data = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(raw_data, list):
    msg = f"Input must be a list, got: {type(raw_data).__name__}"
    raise TypeError(msg)
  return [item for item in raw_data if isinstance(item, dict)]


def _collect_usernames(records: list[dict[str, Any]]) -> list[str]:
  """Collect unique assignee usernames from records."""
  usernames = {
    str(record.get("assignee") or "").strip().lower()
    for record in records
    if str(record.get("assignee") or "").strip()
  }
  return sorted(usernames)


def _build_user_profile_seeds(
  *,
  usernames: list[str],
  records: list[dict[str, Any]],
) -> dict[str, _UserProfileSeed]:
  """Build non-stub profiles by replaying closed assigned tickets chronologically."""
  user_records: dict[str, list[dict[str, Any]]] = {
    username: [] for username in usernames
  }
  for record in records:
    username = str(record.get("assignee") or "").strip().lower()
    if username in user_records:
      user_records[username].append(record)

  embedding_service = get_embedding_service(model_name="all-MiniLM-L6-v2")
  keyword_extractor = get_keyword_extractor()
  alpha = ProfileUpdater().alpha
  profile_seeds: dict[str, _UserProfileSeed] = {}

  def _record_text(record: dict[str, Any]) -> str:
    """Build embedding text for one ticket record."""
    title = str(record.get("title") or "").strip()
    body = str(record.get("body") or "").strip()
    return f"{title}\n\n{body}".strip() or title or "Demo ticket context"

  def _record_time(record: dict[str, Any]) -> datetime:
    """Resolve chronological event time for ticket replay."""
    closed_at = _parse_timestamp(record.get("closed_at"))
    created_at = _parse_timestamp(record.get("created_at"))
    if str(record.get("state") or "").strip().lower() == "closed":
      return closed_at
    return created_at

  for username in usernames:
    assigned_records = sorted(user_records[username], key=_record_time)
    if assigned_records:
      bootstrap_text = _record_text(assigned_records[0])
    else:
      bootstrap_text = f"{username} demo project contributor"

    bootstrap_vector = embedding_service.embed_text(bootstrap_text).astype(np.float32)
    profile_vector = bootstrap_vector.copy()
    closed_keyword_tokens: list[str] = []
    closed_count = 0
    for record in assigned_records:
      state = str(record.get("state") or "").strip().lower()
      if state != "closed":
        continue
      ticket_text = _record_text(record)
      ticket_vector = embedding_service.embed_text(ticket_text).astype(np.float32)
      profile_vector = alpha * profile_vector + (1.0 - alpha) * ticket_vector
      closed_keyword_tokens.extend(keyword_extractor.extract(ticket_text, top_n=20))
      closed_count += 1

    if not closed_keyword_tokens:
      closed_keyword_tokens = keyword_extractor.extract(bootstrap_text, top_n=20)

    full_name = " ".join(_username_to_names(username)).strip()
    profile_seeds[username] = _UserProfileSeed(
      github_username=username,
      full_name=full_name,
      resume_base_vector_text=vector_to_pgvector_text(
        list(map(float, bootstrap_vector.tolist()))
      ),
      profile_vector_text=vector_to_pgvector_text(
        list(map(float, profile_vector.tolist()))
      ),
      keywords_text=" ".join(closed_keyword_tokens),
      tickets_closed_count=closed_count,
    )
  return profile_seeds


def _username_to_names(username: str) -> tuple[str, str]:
  """Derive first and last names from username."""
  normalized = "".join(ch if ch.isalnum() else " " for ch in username).strip()
  parts = [part for part in normalized.split() if part]
  if not parts:
    return "Demo", "User"
  first_name = parts[0].capitalize()
  last_name = " ".join(part.capitalize() for part in parts[1:]) or "User"
  return first_name, last_name


def _email_for_username(username: str) -> str:
  """Build deterministic synthetic email for username."""
  local = "".join(ch if ch.isalnum() else "." for ch in username.lower())
  local = ".".join(part for part in local.split(".") if part) or "demo.user"
  if len(local) > 48:
    digest = hashlib.sha256(username.encode("utf-8")).hexdigest()[:8]
    local = f"{local[:39]}.{digest}"
  return f"{local}@{DEMO_EMAIL_DOMAIN}"


def _password_for_username(username: str) -> str:
  """Build deterministic demo password per username."""
  digest = hashlib.sha256(username.encode("utf-8")).hexdigest()[:6]
  return f"TfDemo@{digest}!"


def _ticket_key_for_source_id(source_id: str) -> str:
  """Build deterministic <=20 char ticket key from source id."""
  digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8].upper()
  return f"DEMO-{digest}"


def _parse_timestamp(value: object) -> datetime:
  """Parse ISO timestamp into UTC datetime with fallback to now."""
  if not isinstance(value, str) or not value.strip():
    return datetime.now(tz=UTC)
  text = value.strip().replace("Z", "+00:00")
  try:
    parsed = datetime.fromisoformat(text)
  except ValueError:
    return datetime.now(tz=UTC)
  if parsed.tzinfo is None:
    return parsed.replace(tzinfo=UTC)
  return parsed.astimezone(UTC)


def _parse_labels(raw_labels: object) -> list[str]:
  """Normalize labels to list of non-empty strings."""
  if isinstance(raw_labels, str):
    return [label.strip() for label in raw_labels.split(",") if label.strip()]
  if isinstance(raw_labels, list):
    return [str(label).strip() for label in raw_labels if str(label).strip()]
  return []


def _infer_priority(labels: list[str]) -> str:
  """Infer board priority from labels."""
  normalized = {label.lower() for label in labels}
  if "critical" in normalized:
    return "critical"
  if "high" in normalized:
    return "high"
  if "low" in normalized:
    return "low"
  return "medium"


def _infer_type(labels: list[str]) -> str:
  """Infer board ticket type from labels."""
  normalized = {label.lower() for label in labels}
  if any("bug" in label for label in normalized):
    return "bug"
  if any(label in {"feature", "enhancement", "story"} for label in normalized):
    return "story"
  return "task"


def _ticket_status_for_record(state: str) -> str:
  """Map a seed record state to the normalized tickets status enum."""
  normalized_state = state.strip().lower()
  if normalized_state == "closed":
    return "closed"
  if normalized_state in {"in-progress", "in progress"}:
    return "in-progress"
  return "open"


def _column_for_record(
  columns: dict[str, str], state: str, assignee: str | None
) -> str:
  """Choose target board column for one ticket record."""
  normalized_state = state.strip().lower()
  if normalized_state == "closed" and COL_DONE in columns:
    return columns[COL_DONE]
  if assignee and COL_PROGRESS in columns:
    return columns[COL_PROGRESS]
  if COL_BACKLOG in columns:
    return columns[COL_BACKLOG]
  if COL_TODO in columns:
    return columns[COL_TODO]
  first_column = next(iter(columns.values()), None)
  if first_column is None:
    msg = "No board columns available"
    raise RuntimeError(msg)
  return first_column


def _delete_project_by_slug(cur: psycopg2.extensions.cursor, slug: str) -> None:
  """Delete target project if it exists (cascade board artifacts)."""
  cur.execute("DELETE FROM projects WHERE slug = %s", (slug,))


def _member_id_for_username(
  cur: psycopg2.extensions.cursor, username: str
) -> int | None:
  """Resolve users.member_id by github_username."""
  cur.execute(
    """
    SELECT member_id
    FROM users
    WHERE lower(github_username) = lower(%s)
    LIMIT 1
    """,
    (username,),
  )
  row = cur.fetchone()
  if row is None or row[0] is None:
    return None
  return int(row[0])


def _upsert_non_stub_user_profiles(
  cur: psycopg2.extensions.cursor,
  *,
  profile_seeds: dict[str, _UserProfileSeed],
) -> dict[str, int]:
  """Upsert `users` rows with non-stub vectors and skill keywords."""
  member_ids: dict[str, int] = {}
  for username, seed in profile_seeds.items():
    cur.execute(
      """
      INSERT INTO users (
        github_username,
        full_name,
        resume_base_vector,
        profile_vector,
        skill_keywords,
        tickets_closed_count,
        created_at,
        updated_at
      )
      VALUES (
        %s,
        %s,
        %s::vector,
        %s::vector,
        to_tsvector('english', %s),
        %s,
        now(),
        now()
      )
      ON CONFLICT (github_username)
      DO UPDATE SET
        full_name = EXCLUDED.full_name,
        resume_base_vector = EXCLUDED.resume_base_vector,
        profile_vector = EXCLUDED.profile_vector,
        skill_keywords = EXCLUDED.skill_keywords,
        tickets_closed_count = EXCLUDED.tickets_closed_count,
        updated_at = now()
      RETURNING member_id
      """,
      (
        seed.github_username,
        seed.full_name,
        seed.resume_base_vector_text,
        seed.profile_vector_text,
        seed.keywords_text,
        seed.tickets_closed_count,
      ),
    )
    row = cur.fetchone()
    if row is None:
      msg = f"Expected member_id for user profile upsert: {username}"
      raise RuntimeError(msg)
    member_ids[username] = int(row[0])
  return member_ids


def _delete_auth_user(cur: psycopg2.extensions.cursor, user_id: str) -> None:
  """Delete one auth user after removing dependent rows."""
  cur.execute("DELETE FROM refresh_tokens WHERE user_id::text = %s", (user_id,))
  cur.execute("DELETE FROM project_members WHERE user_id::text = %s", (user_id,))
  cur.execute(
    "UPDATE project_tickets SET assignee_id = NULL WHERE assignee_id::text = %s",
    (user_id,),
  )

  cur.execute("SELECT COUNT(*) FROM projects WHERE created_by::text = %s", (user_id,))
  projects_row = cur.fetchone()
  projects_count = int(projects_row[0]) if projects_row is not None else 0
  if projects_count > 0:
    msg = f"Cannot replace user {user_id}: referenced by projects.created_by"
    raise RuntimeError(msg)

  cur.execute(
    "SELECT COUNT(*) FROM project_tickets WHERE created_by::text = %s",
    (user_id,),
  )
  created_by_row = cur.fetchone()
  created_by_count = int(created_by_row[0]) if created_by_row is not None else 0
  if created_by_count > 0:
    msg = f"Cannot replace user {user_id}: referenced by project_tickets.created_by"
    raise RuntimeError(msg)

  cur.execute("DELETE FROM auth_users WHERE id::text = %s", (user_id,))


def _ensure_demo_users(
  cur: psycopg2.extensions.cursor,
  *,
  usernames: list[str],
  owner_username: str,
  replace_owner_user: bool,
) -> tuple[dict[str, str], list[dict[str, str]], int, int]:
  """Replace/recreate relevant users and return auth user map + credentials."""
  owner_normalized = owner_username.lower()
  user_id_map: dict[str, str] = {}
  credentials: list[dict[str, str]] = []
  recreated = 0
  retained = 0

  for username in usernames:
    cur.execute(
      """
      SELECT id::text
      FROM auth_users
      WHERE lower(username) = lower(%s)
      ORDER BY created_at ASC
      """,
      (username,),
    )
    existing_ids = [str(row[0]) for row in cur.fetchall()]

    should_replace = replace_owner_user or username != owner_normalized
    if existing_ids and should_replace:
      for existing_id in existing_ids:
        _delete_auth_user(cur, existing_id)
      recreated += 1
    elif existing_ids:
      retained += 1

    cur.execute(
      (
        "SELECT id::text, username FROM auth_users "
        "WHERE lower(username)=lower(%s) LIMIT 1"
      ),
      (username,),
    )
    row = cur.fetchone()
    if row is not None:
      auth_user_id = str(row[0])
      normalized_stored = str(row[1]).strip().lower()
      if normalized_stored != username:
        cur.execute("SELECT 1 FROM auth_users WHERE username = %s", (username,))
        if cur.fetchone() is None:
          cur.execute(
            "UPDATE auth_users SET username=%s, updated_at=now() WHERE id::text=%s",
            (username, auth_user_id),
          )
      user_id_map[username] = auth_user_id
      credentials.append({"username": username, "password": "<unchanged>"})
      continue

    first_name, last_name = _username_to_names(username)
    email = _email_for_username(username)
    cur.execute("SELECT 1 FROM auth_users WHERE lower(email) = lower(%s)", (email,))
    if cur.fetchone() is not None:
      digest = hashlib.sha256(username.encode("utf-8")).hexdigest()[:6]
      email = f"{email.split('@', maxsplit=1)[0]}.{digest}@{DEMO_EMAIL_DOMAIN}"

    plain_password = _password_for_username(username)
    member_id = _member_id_for_username(cur, username)
    auth_user_id = str(uuid.uuid4())
    cur.execute(
      """
      INSERT INTO auth_users (
        id, username, first_name, last_name, email,
        password_hash, member_id, is_active, created_at, updated_at
      )
      VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, now(), now())
      """,
      (
        auth_user_id,
        username,
        first_name,
        last_name,
        email,
        hash_password(plain_password),
        member_id,
      ),
    )
    user_id_map[username] = auth_user_id
    credentials.append({"username": username, "password": plain_password})
    recreated += 1

  return user_id_map, credentials, recreated, retained


def _owner_id(cur: psycopg2.extensions.cursor, owner_username: str) -> str:
  """Resolve owner auth user id."""
  cur.execute(
    "SELECT id::text FROM auth_users WHERE lower(username)=lower(%s) LIMIT 1",
    (owner_username,),
  )
  row = cur.fetchone()
  if row is None:
    msg = f"Owner user not found: {owner_username}"
    raise RuntimeError(msg)
  return str(row[0])


def _create_project(
  cur: psycopg2.extensions.cursor,
  *,
  slug: str,
  name: str,
  owner_id: str,
) -> str:
  """Create a fresh project row and return project id."""
  cur.execute(
    """
    INSERT INTO projects (
      id, name, slug, description, created_by,
      default_ticket_size, weekly_points_per_member, size_points_map
    )
    VALUES (%s, %s, %s, %s, %s, 'M', 10, %s::jsonb)
    RETURNING id::text
    """,
    (
      str(uuid.uuid4()),
      name,
      slug,
      "Demo project seeded from Issue #122 sample ticket payload.",
      owner_id,
      json.dumps({"S": 1, "M": 2, "L": 3, "XL": 5}),
    ),
  )
  row = cur.fetchone()
  if row is None:
    msg = "Failed to create project"
    raise RuntimeError(msg)
  return str(row[0])


def _create_columns(cur: psycopg2.extensions.cursor, project_id: str) -> dict[str, str]:
  """Create canonical columns and return lowercase name->id map."""
  mapping: dict[str, str] = {}
  for index, name in enumerate(CANONICAL_COLUMNS):
    column_id = str(uuid.uuid4())
    cur.execute(
      """
      INSERT INTO project_board_columns (id, project_id, name, position, created_at)
      VALUES (%s, %s, %s, %s, now())
      """,
      (column_id, project_id, name, index),
    )
    mapping[name.lower()] = column_id
  return mapping


def _add_members(
  cur: psycopg2.extensions.cursor,
  *,
  project_id: str,
  owner_id: str,
  user_ids: list[str],
) -> None:
  """Add owner and users as project members."""
  cur.execute(
    """
    INSERT INTO project_members (id, project_id, user_id, role, joined_at)
    VALUES (%s, %s, %s, 'owner', now())
    ON CONFLICT (project_id, user_id) DO NOTHING
    """,
    (str(uuid.uuid4()), project_id, owner_id),
  )
  for user_id in user_ids:
    if user_id == owner_id:
      continue
    cur.execute(
      """
      INSERT INTO project_members (id, project_id, user_id, role, joined_at)
      VALUES (%s, %s, %s, 'member', now())
      ON CONFLICT (project_id, user_id) DO NOTHING
      """,
      (str(uuid.uuid4()), project_id, user_id),
    )


def _seed_project_tickets(  # noqa: PLR0913
  cur: psycopg2.extensions.cursor,
  *,
  records: list[dict[str, Any]],
  project_id: str,
  owner_id: str,
  columns: dict[str, str],
  user_map: dict[str, str],
) -> tuple[int, int, int]:
  """Insert all sample records into project_tickets."""
  positions: dict[str, int] = dict.fromkeys(columns.values(), 0)
  total = 0
  assigned = 0
  embedding_service = get_embedding_service(model_name="all-MiniLM-L6-v2")

  for record in records:
    source_id = str(record.get("id") or "").strip()
    if not source_id:
      continue
    assignee_username = str(record.get("assignee") or "").strip().lower()
    assignee_id = user_map.get(assignee_username)
    if assignee_id is not None:
      assigned += 1
    state = str(record.get("state") or "open")
    column_id = _column_for_record(columns, state, assignee_username or None)
    position = positions[column_id]
    positions[column_id] = position + 1
    labels = _parse_labels(record.get("labels"))
    title = str(record.get("title") or source_id)[:2000]
    description = str(record.get("body") or "")
    ticket_text = f"{title}\n\n{description}".strip() or title or "Demo ticket context"
    ticket_vector = embedding_service.embed_text(ticket_text).astype(np.float32)
    ticket_status = _ticket_status_for_record(state)
    created_at = _parse_timestamp(record.get("created_at"))
    closed_at = _parse_timestamp(record.get("closed_at"))
    resolution_time_actual = (
      closed_at - created_at if ticket_status == "closed" else None
    )
    size_bucket = (
      str(record.get("size_bucket") or record.get("size") or "").strip().upper()
    )
    size_updated_at = datetime.now(tz=UTC) if size_bucket else None
    size_source = "manual" if size_bucket else None

    cur.execute(
      """
      INSERT INTO tickets (
        ticket_id, title, description, ticket_vector, labels, status,
        resolution_time_actual, project_id, created_at, updated_at
      )
      VALUES (
        %s, %s, %s, %s::vector, %s::jsonb, %s,
        %s, %s, %s, now()
      )
      ON CONFLICT (ticket_id)
      DO UPDATE SET
        title = EXCLUDED.title,
        description = EXCLUDED.description,
        ticket_vector = EXCLUDED.ticket_vector,
        labels = EXCLUDED.labels,
        status = EXCLUDED.status,
        resolution_time_actual = EXCLUDED.resolution_time_actual,
        project_id = EXCLUDED.project_id,
        updated_at = now()
      """,
      (
        _ticket_key_for_source_id(source_id),
        title,
        description,
        vector_to_pgvector_text(list(map(float, ticket_vector.tolist()))),
        json.dumps(labels),
        ticket_status,
        resolution_time_actual,
        project_id,
        created_at,
      ),
    )

    cur.execute(
      """
      INSERT INTO project_tickets (
        id, project_id, column_id, assignee_id, created_by,
        ticket_key, title, description, priority, type, labels,
        size_bucket, size_source, size_confidence, size_updated_at,
        due_date, position, created_at, updated_at
      )
      VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s::jsonb,
        %s, %s, NULL, %s,
        NULL, %s, %s, now()
      )
      """,
      (
        str(uuid.uuid4()),
        project_id,
        column_id,
        assignee_id,
        owner_id,
        _ticket_key_for_source_id(source_id),
        title,
        description,
        _infer_priority(labels),
        _infer_type(labels),
        json.dumps(labels),
        size_bucket,
        size_source,
        size_updated_at,
        position,
        created_at,
      ),
    )
    total += 1

  return total, assigned, max(total - assigned, 0)


def seed_demo_project(  # noqa: PLR0913
  *,
  dsn: str,
  tickets_input_path: Path,
  users_input_path: Path,
  project_slug: str,
  project_name: str,
  owner_username: str,
  replace_owner_user: bool,
) -> SeedResult:
  """Execute full project-first seeding flow."""
  ticket_records = _load_records(tickets_input_path)
  user_records = _load_records(users_input_path)
  usernames = _collect_usernames(user_records)
  owner_normalized = owner_username.strip().lower()
  if owner_normalized not in usernames:
    usernames.append(owner_normalized)
  usernames = sorted(set(usernames))

  with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
    _delete_project_by_slug(cur, project_slug)

    profile_seeds = _build_user_profile_seeds(
      usernames=usernames,
      records=ticket_records,
    )
    _upsert_non_stub_user_profiles(cur, profile_seeds=profile_seeds)
    user_map, credentials, recreated_count, retained_count = _ensure_demo_users(
      cur,
      usernames=usernames,
      owner_username=owner_normalized,
      replace_owner_user=replace_owner_user,
    )
    owner_id = user_map.get(owner_normalized) or _owner_id(cur, owner_normalized)
    project_id = _create_project(
      cur,
      slug=project_slug,
      name=project_name,
      owner_id=owner_id,
    )
    columns = _create_columns(cur, project_id)
    _add_members(
      cur,
      project_id=project_id,
      owner_id=owner_id,
      user_ids=sorted(set(user_map.values())),
    )
    seeded, assigned, unassigned = _seed_project_tickets(
      cur,
      records=ticket_records,
      project_id=project_id,
      owner_id=owner_id,
      columns=columns,
      user_map=user_map,
    )
    conn.commit()

  return SeedResult(
    project_slug=project_slug,
    project_id=project_id,
    users_recreated=recreated_count,
    users_retained=retained_count,
    tickets_seeded=seeded,
    tickets_assigned=assigned,
    tickets_unassigned=unassigned,
    users_credentials=credentials,
  )


def _build_argument_parser() -> argparse.ArgumentParser:
  """Build CLI parser."""
  parser = argparse.ArgumentParser(
    description="Seed sample ticket data into one demo project."
  )
  parser.add_argument(
    "--input",
    default="demo/data/sample_tickets.json",
    help="Ticket payload JSON to seed into project_tickets.",
  )
  parser.add_argument(
    "--users-input",
    default="demo/data/sample_tickets.json",
    help="Payload JSON used to derive relevant demo users.",
  )
  parser.add_argument(
    "--project-slug",
    default="seeded-demo-board",
    help="Target demo project slug.",
  )
  parser.add_argument(
    "--project-name",
    default="Seeded Demo Board",
    help="Target demo project name.",
  )
  parser.add_argument(
    "--owner-username",
    default="darshanrk18",
    help="Project owner username.",
  )
  parser.add_argument(
    "--retain-owner-user",
    action="store_false",
    dest="replace_owner_user",
    help="Retain owner account instead of recreating it.",
  )
  parser.set_defaults(replace_owner_user=True)
  parser.add_argument(
    "--dsn",
    default=None,
    help="Optional explicit Postgres DSN override.",
  )
  parser.add_argument(
    "--no-auto-proxy",
    action="store_true",
    help="Disable automatic Cloud SQL proxy startup for localhost DSNs.",
  )
  parser.add_argument(
    "--proxy-timeout-seconds",
    type=float,
    default=90.0,
    help="Maximum wait time for automatic Cloud SQL proxy startup.",
  )
  parser.add_argument(
    "--keep-proxy",
    action="store_true",
    help="Keep auto-started proxy process running after script completion.",
  )
  parser.add_argument(
    "--credentials-out",
    default="demo/data/demo_user_credentials.json",
    help="Output path for username/password mapping JSON.",
  )
  return parser


def main() -> None:
  """Run project seeding entrypoint."""
  args = _build_argument_parser().parse_args()
  command_env = _build_subprocess_env()
  proxy_process: subprocess.Popen[str] | None = None
  try:
    dsn = _resolve_seed_dsn(args.dsn, command_env)
    host, port = _dsn_host_port(dsn)
    if not args.no_auto_proxy and host in {"127.0.0.1", "localhost"}:
      if _is_port_open(host, port):
        logger.info("Detected existing proxy on %s:%s", host, port)
      else:
        logger.info("Starting Cloud SQL proxy automatically for %s:%s", host, port)
        proxy_process = _start_cloud_sql_proxy(
          env=command_env,
          local_port=port,
          timeout_seconds=max(args.proxy_timeout_seconds, 1.0),
        )

    result = seed_demo_project(
      dsn=dsn,
      tickets_input_path=Path(args.input),
      users_input_path=Path(args.users_input),
      project_slug=args.project_slug,
      project_name=args.project_name,
      owner_username=args.owner_username,
      replace_owner_user=args.replace_owner_user,
    )
    logger.info(
      "Seed complete: project=%s (%s) users_recreated=%s users_retained=%s "
      "tickets_seeded=%s assigned=%s unassigned=%s",
      result.project_slug,
      result.project_id,
      result.users_recreated,
      result.users_retained,
      result.tickets_seeded,
      result.tickets_assigned,
      result.tickets_unassigned,
    )
    credentials_path = Path(args.credentials_out)
    _write_credentials_file(
      path=credentials_path,
      project_slug=result.project_slug,
      project_id=result.project_id,
      credentials=result.users_credentials,
    )
    logger.info(
      "Stored username/password mapping at: %s",
      credentials_path,
    )
  finally:
    if (
      proxy_process is not None and not args.keep_proxy and proxy_process.poll() is None
    ):
      proxy_process.terminate()
      logger.info("Stopped auto-managed Cloud SQL proxy process.")


if __name__ == "__main__":
  main()
