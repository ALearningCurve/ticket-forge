"""Data profiling script using Great Expectations and custom skew detection."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ml_core.anomaly.ge_validator import GreatExpectationsValidator
from ml_core.anomaly.validator import SchemaValidator
from shared.configuration import Paths

# Suppress noisy GE logs
logging.getLogger("great_expectations").setLevel(logging.ERROR)
logging.getLogger("great_expectations.data_context").setLevel(logging.ERROR)
logging.getLogger("great_expectations.data_context.types.base").setLevel(logging.ERROR)


class NumpyEncoder(json.JSONEncoder):
  """Custom JSON encoder to handle numpy types."""

  def default(self, o: object) -> object:
    """Encode numpy types to native Python types."""
    if isinstance(o, np.integer):
      return int(o)
    if isinstance(o, np.floating):
      return float(o)
    if isinstance(o, np.bool_):
      return bool(o)
    if isinstance(o, np.ndarray):
      return o.tolist()
    return super().default(o)


SAMPLE_PATH = Paths.data_root / "github_issues" / "sample_tickets_transformed.jsonl"
FULL_PATH = Paths.data_root / "github_issues" / "tickets_transformed_improved.jsonl"
SCHEMA_OUTPUT = Paths.data_root / "github_issues" / "ticket_schema.json"
PROFILE_OUTPUT = Paths.data_root / "github_issues" / "data_profile_report.json"

NUMERIC_COLS = [
  "completion_hours_business",
  "seniority_enum",
  "historical_avg_completion_hours",
]
CATEGORICAL_COLS = ["repo", "issue_type", "state"]


def load_jsonl(path: Path) -> pd.DataFrame:
  """Load a JSONL file into a DataFrame."""
  tickets = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      if line.strip():
        tickets.append(json.loads(line))
  return pd.DataFrame(tickets)


def detect_skew(sample_df: pd.DataFrame, full_df: pd.DataFrame) -> dict:
  """Compare distributions between sample and full dataset.

  Args:
      sample_df: Sampled dataset
      full_df: Full dataset

  Returns:
      Dictionary of skew results per column
  """
  skew_results = {}

  for col in NUMERIC_COLS:
    if col not in sample_df.columns or col not in full_df.columns:
      continue

    sample_mean = float(sample_df[col].dropna().mean())  # type: ignore[arg-type]
    full_mean = float(full_df[col].dropna().mean())  # type: ignore[arg-type]
    sample_std = float(sample_df[col].dropna().std())  # type: ignore[arg-type]
    full_std = float(full_df[col].dropna().std())  # type: ignore[arg-type]

    mean_diff_pct = abs(sample_mean - full_mean) / full_mean * 100 if full_mean else 0
    std_diff_pct = abs(sample_std - full_std) / full_std * 100 if full_std else 0

    skew_results[col] = {
      "sample_mean": round(sample_mean, 4),
      "full_mean": round(full_mean, 4),
      "mean_diff_pct": round(mean_diff_pct, 2),
      "sample_std": round(sample_std, 4),
      "full_std": round(full_std, 4),
      "std_diff_pct": round(std_diff_pct, 2),
      "skewed": mean_diff_pct > 10,
    }

  for col in CATEGORICAL_COLS:
    if col not in sample_df.columns or col not in full_df.columns:
      continue

    sample_dist = sample_df[col].value_counts(normalize=True).to_dict()
    full_dist = full_df[col].value_counts(normalize=True).to_dict()

    max_diff = 0.0
    for key in full_dist:
      sample_val = sample_dist.get(key, 0.0)
      diff = abs(sample_val - full_dist[key])
      max_diff = max(max_diff, diff)

    skew_results[col] = {
      "sample_distribution": {k: round(v, 4) for k, v in sample_dist.items()},
      "full_distribution": {k: round(v, 4) for k, v in full_dist.items()},
      "max_distribution_diff": round(max_diff, 4),
      "skewed": max_diff > 0.1,
    }

  return skew_results


def main() -> None:
  """Run data profiling on sample tickets."""
  print("Loading sample dataset...")
  sample_df = load_jsonl(SAMPLE_PATH)
  print("Loaded", len(sample_df), "sample tickets")

  full_df = None
  if FULL_PATH.exists():
    print("Loading full dataset for skew detection...")
    full_df = load_jsonl(FULL_PATH)
    print("Loaded", len(full_df), "full tickets")

  # 1. Schema statistics using SchemaValidator
  print("\nGenerating statistics...")
  validator = SchemaValidator({})
  stats = validator.generate_statistics(sample_df)
  schema = validator.generate_schema_from_data(sample_df)
  print("Rows:", stats["row_count"])
  print("Columns:", stats["column_count"])

  # 2. Great Expectations validation
  print("\nRunning Great Expectations validation...")
  ge_validator = GreatExpectationsValidator()
  ge_validator.create_expectations(sample_df)
  validation_results = ge_validator.validate_data(sample_df)
  print("Validation passed:", validation_results["success"])
  print("Total expectations:", validation_results["total_expectations"])
  print("Failed expectations:", validation_results["failed_expectations"])

  # 3. Save GE schema
  ge_validator.save_schema(str(SCHEMA_OUTPUT))

  # 4. Skew detection
  skew_results = {}
  if full_df is not None:
    print("\nDetecting skew between sample and full dataset...")
    skew_results = detect_skew(sample_df, full_df)
    skewed_cols = [col for col, res in skew_results.items() if res["skewed"]]
    if skewed_cols:
      print("Skewed columns:", skewed_cols)
    else:
      print("No significant skew detected")

  # 5. Save full profile report
  profile = {
    "dataset": str(SAMPLE_PATH),
    "row_count": stats["row_count"],
    "column_count": stats["column_count"],
    "schema": {col: t.__name__ for col, t in schema.items()},
    "numeric_stats": stats["numeric_stats"],
    "categorical_stats": stats["categorical_stats"],
    "ge_validation": validation_results,
    "skew_vs_full_dataset": skew_results,
  }

  with open(PROFILE_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2, cls=NumpyEncoder)

  print("\nProfile report saved to", PROFILE_OUTPUT)
  print("Done!")


if __name__ == "__main__":
  main()
