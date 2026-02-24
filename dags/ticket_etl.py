"""Airflow DAG that runs ticket ETL with anomaly detection and bias mitigation."""
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from email_callbacks import send_dag_status_email
from shared.configuration import Paths

# Make workspace packages importable from Airflow's DAG context.
REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (
  REPO_ROOT / "apps" / "training",
  REPO_ROOT / "libs" / "ml-core",
  REPO_ROOT / "libs" / "shared",
):
  path_str = str(path)
  if path_str not in sys.path:
    sys.path.append(path_str)

# Training module imports are deferred to individual task functions

DAG_ID = "ticket_etl"


def _require_database_url() -> str:
  """Return DATABASE_URL from env or raise a task failure."""
  dsn = os.environ.get("DATABASE_URL")
  if not dsn:
    msg = "DATABASE_URL is required for ticket_etl DAG."
    raise AirflowFailException(msg)
  return dsn


def validate_runtime_config(**context: Any) -> dict[str, Any]:
  """Read dag_run.conf and normalize ticket ETL runtime config."""
  dag_run = context.get("dag_run")
  conf = dag_run.conf if dag_run and dag_run.conf else {}

  raw_limit = conf.get("limit_per_state")
  limit_per_state: int | None = None
  if raw_limit is not None:
    try:
      limit_per_state = int(raw_limit)
    except (TypeError, ValueError) as exc:
      msg = "limit_per_state must be an integer when provided"
      raise AirflowFailException(msg) from exc

  # Generate timestamped output directory
  run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
  output_dir = Paths.data_root / f"github_issues-{run_timestamp}"
  output_dir.mkdir(parents=True, exist_ok=True)

  runtime = {
    "dsn": _require_database_url(),
    "limit_per_state": limit_per_state,
    "output_dir": str(output_dir),
    "run_timestamp": run_timestamp,
  }

  # Push to XCom
  context["task_instance"].xcom_push(key="runtime", value=runtime)
  return runtime


def scrape_github_issues(**context: Any) -> dict[str, Any]:
  """Scrape GitHub issues with optional limit_per_state."""
  from training.etl.ingest.scrape_github_issues_improved import scrape_all_issues

  runtime = context["task_instance"].xcom_pull(
    task_ids="validate_runtime_config", key="runtime"
  )

  limit_per_state = runtime.get("limit_per_state")
  print(f"Scraping GitHub issues (limit_per_state={limit_per_state})...")

  raw_records = asyncio.run(scrape_all_issues(limit_per_state=limit_per_state))
  print(f"Scraped {len(raw_records)} records")

  # Store in XCom for next task
  context["task_instance"].xcom_push(key="raw_records", value=raw_records)
  return {"records_scraped": len(raw_records)}


def run_transform(**context: Any) -> dict[str, Any]:
  """Transform raw records into ticket features."""
  from training.etl.transform.run_transform import transform_records

  raw_records = context["task_instance"].xcom_pull(
    task_ids="scrape_github_issues", key="raw_records"
  )
  runtime = context["task_instance"].xcom_pull(
    task_ids="validate_runtime_config", key="runtime"
  )

  output_dir = Path(runtime["output_dir"])

  print(f"Transforming {len(raw_records)} records...")
  transformed = transform_records(raw_records)
  print(f"Transformed {len(transformed)} records")

  # Save to file for use by anomaly detection and bias modules
  transform_path = output_dir / "tickets_transformed_improved.jsonl"

  with open(transform_path, "w", encoding="utf-8") as f:
    for record in transformed:
      f.write(json.dumps(record) + "\n")

  print(f"Saved transformed data to {transform_path}")

  # Store in XCom for database load
  context["task_instance"].xcom_push(key="transformed_records", value=transformed)
  context["task_instance"].xcom_push(key="transform_path", value=str(transform_path))
  return {"records_transformed": len(transformed)}


def run_anomaly_check(**context: Any) -> dict[str, Any]:
  """Run anomaly detection on transformed data and output detailed results."""
  from training.analysis.run_anomaly_check import run_anomaly_check as analyze_anomaly

  transform_path = context["task_instance"].xcom_pull(
    task_ids="run_transform", key="transform_path"
  )
  results = analyze_anomaly(
    data_path=transform_path,
    outlier_threshold=3.0,
    enable_alerts=False,
  )

  anomaly_report = results["anomaly_report"]
  anomaly_text = results["text_report"]
  print(anomaly_text)

  # If anomalies detected, raise an error to fail the task
  if anomaly_report["total_anomalies"] > 5:
    msg = f"Anomalies detected: {anomaly_report['total_anomalies']}"
    raise AirflowFailException(msg)

  context["task_instance"].xcom_push(key="anomaly_report", value=anomaly_report)
  context["task_instance"].xcom_push(key="anomaly_email_text", value=anomaly_text)
  return {"anomalies_detected": anomaly_report["has_anomalies"]}


def run_bias_detection(**context: Any) -> dict[str, Any]:
  """Run bias detection analysis on timestamped data."""
  from training.analysis.detect_bias import run_bias_detection as analyze_bias

  print("Starting bias detection analysis...")

  transform_path = context["task_instance"].xcom_pull(
    task_ids="run_transform", key="transform_path"
  )

  # Call the analysis module function
  bias_detection_report = analyze_bias(transform_path)

  context["task_instance"].xcom_push(
    key="bias_detection_report", value=bias_detection_report
  )
  print("Bias detection analysis complete!")
  return {"bias_detection_done": True}


def run_bias_mitigation(**context: Any) -> dict[str, Any]:
  """Run bias mitigation (sample weights mode) on timestamped data."""
  from training.analysis.run_bias_mitigation import run_bias_mitigation_weights

  print("Starting bias mitigation...")

  transform_path = context["task_instance"].xcom_pull(
    task_ids="run_transform", key="transform_path"
  )
  runtime = context["task_instance"].xcom_pull(
    task_ids="validate_runtime_config", key="runtime"
  )

  output_dir = runtime["output_dir"]

  # Call the analysis module function
  mitigation_results = run_bias_mitigation_weights(
    data_path=transform_path, output_dir=output_dir
  )

  context["task_instance"].xcom_push(
    key="bias_mitigation_results", value=mitigation_results
  )
  context["task_instance"].xcom_push(
    key="weights_path", value=mitigation_results["weights_path"]
  )

  print("Bias mitigation complete!")
  return {"bias_mitigation_done": True}


def prepare_bias_report(**context: Any) -> dict[str, Any]:
  """Combine bias detection and mitigation results into a detailed report."""
  from training.analysis.detect_bias import generate_bias_report_text

  bias_detection_report = context["task_instance"].xcom_pull(
    task_ids="run_bias_detection", key="bias_detection_report"
  )
  bias_mitigation_results = context["task_instance"].xcom_pull(
    task_ids="run_bias_mitigation", key="bias_mitigation_results"
  )

  # Combine both reports for email
  combined_report = {**bias_detection_report}
  combined_report["weights_by_group"] = bias_mitigation_results.get(
    "weights_by_group", {}
  )

  # Generate detailed text report
  report_text = generate_bias_report_text(
    bias_detection_report, bias_mitigation_results
  )
  print("\n" + report_text + "\n")

  combined_report["text_report"] = report_text

  context["task_instance"].xcom_push(key="combined_bias_report", value=combined_report)
  context["task_instance"].xcom_push(key="bias_email_text", value=report_text)
  return combined_report


def save_dataset_and_weights(**context: Any) -> dict[str, Any]:
  """Save transformed dataset (compressed) and bias mitigation weights."""
  print("Saving dataset and weights...")

  transform_path = context["task_instance"].xcom_pull(
    task_ids="run_transform", key="transform_path"
  )
  weights_path = context["task_instance"].xcom_pull(
    task_ids="run_bias_mitigation", key="weights_path"
  )

  output_dir = Path(transform_path).parent

  # Compress the dataset
  compressed_path = output_dir / "tickets_transformed_improved.jsonl.gz"
  print(f"Compressing dataset to {compressed_path}...")

  with open(transform_path, "rb") as f_in:
    with gzip.open(compressed_path, "wb") as f_out:
      f_out.writelines(f_in)

  print(f"Saved compressed dataset to {compressed_path}")

  # Verify bias weights exist
  if Path(weights_path).exists():
    print(f"Bias mitigation weights already saved at {weights_path}")
  else:
    msg = f"Bias mitigation weights not found at {weights_path}"
    raise AirflowFailException(msg)

  return {
    "dataset_saved": str(compressed_path),
    "weights_saved": str(weights_path),
  }


def load_tickets_to_db(**context: Any) -> dict[str, int]:
  """Load transformed tickets and assignments into Postgres."""
  from training.etl.postload.load_tickets import (
    upsert_assignments,
    upsert_tickets,
  )

  runtime = context["task_instance"].xcom_pull(
    task_ids="validate_runtime_config", key="runtime"
  )
  transformed = context["task_instance"].xcom_pull(
    task_ids="run_transform", key="transformed_records"
  )

  dsn = str(runtime["dsn"])

  print("Step 1/2: Upserting tickets...")
  loaded_tickets = upsert_tickets(transformed, dsn=dsn)
  print(f"Upserted {loaded_tickets} ticket(s) into Postgres")

  print("Step 2/2: Upserting assignments...")
  assigned_count, missing_user_count = upsert_assignments(transformed, dsn=dsn)
  print(f"Upserted {assigned_count} assignment row(s)")
  if missing_user_count:
    print(f"Skipped {missing_user_count} assignment(s): assignee not found in users")

  return {"tickets_loaded": loaded_tickets, "assignments_upserted": assigned_count}


with DAG(
  dag_id=DAG_ID,
  schedule="@monthly",
  start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
  catchup=False,
  default_args={
    "owner": "ticketforge",
    "retries": 0,
  },
  max_active_runs=1,
  tags=["etl", "airflow", "tickets"],
) as dag:
  # ===== Configuration =====
  validate_task = PythonOperator(
    task_id="validate_runtime_config",
    python_callable=validate_runtime_config,
    provide_context=True,
  )

  # ===== Scrape & Transform =====
  scrape_task = PythonOperator(
    task_id="scrape_github_issues",
    python_callable=scrape_github_issues,
    provide_context=True,
  )

  transform_task = PythonOperator(
    task_id="run_transform",
    python_callable=run_transform,
    provide_context=True,
  )

  # ===== Anomaly Detection =====
  anomaly_task = PythonOperator(
    task_id="run_anomaly_check",
    python_callable=run_anomaly_check,
    provide_context=True,
  )

  # ===== Bias Detection & Mitigation (parallel) =====
  bias_detect_task = PythonOperator(
    task_id="run_bias_detection",
    python_callable=run_bias_detection,
    provide_context=True,
  )

  bias_mitigate_task = PythonOperator(
    task_id="run_bias_mitigation",
    python_callable=run_bias_mitigation,
    provide_context=True,
  )

  # ===== Save Dataset & Weights =====
  save_task = PythonOperator(
    task_id="save_dataset_and_weights",
    python_callable=save_dataset_and_weights,
    provide_context=True,
  )

  # ===== Prepare Bias Report =====
  prepare_report_task = PythonOperator(
    task_id="prepare_bias_report",
    python_callable=prepare_bias_report,
    provide_context=True,
  )

  # ===== Load to Database =====
  load_db_task = PythonOperator(
    task_id="load_tickets_to_db",
    python_callable=load_tickets_to_db,
    provide_context=True,
  )

  # ===== Finalization =====
  def send_email_with_report(**context: Any) -> None:
    """Send email with bias report."""
    anomaly_text = context["task_instance"].xcom_pull(
      task_ids="run_anomaly_check", key="anomaly_email_text"
    )
    bias_text = context["task_instance"].xcom_pull(
      task_ids="prepare_bias_report", key="bias_email_text"
    )

    additional_parts = [text for text in [anomaly_text, bias_text] if text]
    additional_text = "\n\n".join(additional_parts) if additional_parts else None

    send_dag_status_email(additional_text=additional_text, **context)

  send_email_task = PythonOperator(
    task_id="send_status_email",
    python_callable=send_email_with_report,
    provide_context=True,
    trigger_rule=TriggerRule.ALL_DONE,
  )

  # ===== Task Dependencies =====
  # Config -> Scrape -> Transform -> Anomaly Detection
  validate_task >> scrape_task >> transform_task >> anomaly_task

  # Anomaly Detection -> Bias Detection & Mitigation (parallel)
  anomaly_task >> [bias_detect_task, bias_mitigate_task]

  # Bias tasks -> Save Dataset & Weights AND Prepare Report (parallel)
  [bias_detect_task, bias_mitigate_task] >> save_task
  [bias_detect_task, bias_mitigate_task] >> prepare_report_task

  # Anomaly Detection -> Load to DB (independent of bias path)
  anomaly_task >> load_db_task

  # All paths converge: save_task, load_db_task, and prepare_report_task before email
  [save_task, load_db_task, prepare_report_task] >> send_email_task
