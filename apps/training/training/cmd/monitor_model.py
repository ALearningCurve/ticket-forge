"""Run drift monitoring against the latest cloud-published training dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.cloud import storage
from shared.configuration import Paths
from shared.logging import get_logger
from training.analysis.drift_detection import (
  compare_profile_reports,
  load_drift_thresholds,
  write_drift_report,
)
from training.analysis.run_data_profiling import run_data_profiling
from training.cloud_storage_loader import resolve_cloud_dataset

logger = get_logger(__name__)

_DEFAULT_BASELINE_OBJECT = "monitoring/latest_data_profile_report.json"
_DEFAULT_LATEST_REPORT_OBJECT = "monitoring/latest_drift_report.json"
_DEFAULT_REPORTS_PREFIX = "monitoring/reports"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run dataset drift monitoring.")
  parser.add_argument("--runid", required=True, help="Monitoring run identifier")
  parser.add_argument("--bucket-uri", default=None, help="gs:// bucket URI override")
  parser.add_argument("--trigger", default="schedule", help="Trigger source")
  parser.add_argument(
    "--trigger-reason",
    default="scheduled-monitoring",
    help="Human-readable reason for this monitoring run",
  )
  parser.add_argument(
    "--baseline-object",
    default=_DEFAULT_BASELINE_OBJECT,
    help="Bucket object path for the latest baseline profile JSON",
  )
  parser.add_argument(
    "--reports-prefix",
    default=_DEFAULT_REPORTS_PREFIX,
    help="Bucket prefix under which drift reports are persisted",
  )
  return parser.parse_args()


def _find_downloaded_dataset_file(local_dir: Path) -> Path:
  """Return the downloaded dataset artifact path."""
  for filename in (
    "tickets_transformed_improved.jsonl",
    "tickets_transformed_improved.jsonl.gz",
  ):
    candidate = local_dir / filename
    if candidate.exists():
      return candidate
  msg = f"No transformed dataset found in {local_dir}"
  raise FileNotFoundError(msg)


def _read_json_blob(bucket_name: str, object_name: str) -> dict[str, Any] | None:
  """Read a JSON object from Cloud Storage when it exists."""
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(object_name)
  if not blob.exists(client):
    return None

  raw = blob.download_as_text()
  parsed = json.loads(raw)
  if not isinstance(parsed, dict):
    msg = f"Expected JSON object in gs://{bucket_name}/{object_name}"
    raise TypeError(msg)
  return parsed


def _write_json_blob(
  bucket_name: str,
  object_name: str,
  payload: dict[str, Any],
) -> str:
  """Write a JSON object to Cloud Storage and return the gs:// URI."""
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(object_name)
  blob.upload_from_string(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    content_type="application/json",
  )
  return f"gs://{bucket_name}/{object_name}"


def _blob_uri(bucket_name: str, object_name: str) -> str:
  """Return the gs:// URI for a bucket object path."""
  return f"gs://{bucket_name}/{object_name}"


def main() -> int:
  """Run the monitoring workflow and persist its artifacts."""
  args = _parse_args()
  run_dir = Paths.models_root / args.runid
  run_dir.mkdir(parents=True, exist_ok=True)

  dataset_ref = resolve_cloud_dataset(args.bucket_uri)
  dataset_path = _find_downloaded_dataset_file(dataset_ref.local_directory)
  current_profile = run_data_profiling(data_path=dataset_path, output_dir=run_dir)
  baseline_profile = _read_json_blob(dataset_ref.bucket_name, args.baseline_object)

  thresholds = load_drift_thresholds()
  if baseline_profile is None:
    report: dict[str, Any] = {
      "generated_at": datetime.now(tz=UTC).isoformat(),
      "drift_detected": False,
      "breaches": [],
      "baseline_initialized": True,
      "thresholds": thresholds.to_dict(),
      "baseline_dataset": None,
      "current_dataset": current_profile.get("dataset"),
      "row_count": {
        "baseline": None,
        "current": current_profile.get("row_count"),
        "delta_ratio": 0.0,
        "drifted": False,
      },
      "numeric_drift": {},
      "categorical_drift": {},
      "validation_drift": {
        "baseline_failed_expectations": 0,
        "current_failed_expectations": (
          current_profile.get("ge_validation", {}) or {}
        ).get("failed_expectations", 0),
        "failed_expectations_delta": 0,
        "drifted": False,
      },
    }
  else:
    report = compare_profile_reports(
      baseline_profile=baseline_profile,
      current_profile=current_profile,
      thresholds=thresholds,
    )
    report["baseline_initialized"] = False

  report.update(
    {
      "run_id": args.runid,
      "trigger": args.trigger,
      "trigger_reason": args.trigger_reason,
      "dataset_id": dataset_ref.dataset_id,
      "dataset_uri": dataset_ref.dataset_uri,
      "dataset_version": dataset_ref.dataset_version,
      "bucket_name": dataset_ref.bucket_name,
      "retrain_recommended": bool(report.get("drift_detected", False)),
    }
  )

  current_profile_object = (
    f"{args.reports_prefix}/{args.runid}/data_profile_report.json"
  )
  drift_report_object = f"{args.reports_prefix}/{args.runid}/drift_report.json"
  report["current_profile_uri"] = _blob_uri(
    dataset_ref.bucket_name,
    current_profile_object,
  )
  report["drift_report_uri"] = _blob_uri(dataset_ref.bucket_name, drift_report_object)
  report["latest_profile_uri"] = _blob_uri(
    dataset_ref.bucket_name,
    args.baseline_object,
  )
  report["latest_drift_report_uri"] = _blob_uri(
    dataset_ref.bucket_name,
    _DEFAULT_LATEST_REPORT_OBJECT,
  )

  _write_json_blob(
    dataset_ref.bucket_name,
    current_profile_object,
    current_profile,
  )
  _write_json_blob(
    dataset_ref.bucket_name,
    args.baseline_object,
    current_profile,
  )
  _write_json_blob(
    dataset_ref.bucket_name,
    drift_report_object,
    report,
  )
  _write_json_blob(
    dataset_ref.bucket_name,
    _DEFAULT_LATEST_REPORT_OBJECT,
    report,
  )

  report_path = write_drift_report(run_dir / "drift_report.json", report)
  logger.info("Drift report written to %s", report_path)
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
