"""Run bias detection analysis on ticket data."""

import json

import pandas as pd
from ml_core.bias import DataSlicer  # type: ignore[attr-defined]
from shared.configuration import Paths


def run_bias_analysis() -> None:
  """Analyze bias in ticket data distribution."""
  print("Starting bias detection analysis...\n")

  data_path = Paths.data_root / "github_issues" / "tickets_transformed_improved.jsonl"

  print("Loading data from", data_path)
  tickets = []
  with open(data_path, encoding="utf-8") as f:
    for line in f:
      if line.strip():
        tickets.append(json.loads(line))

  df = pd.DataFrame(tickets)
  print("Loaded", len(df), "tickets\n")

  print("DATA DISTRIBUTION ANALYSIS")
  print("=" * 80)

  slicer = DataSlicer(df)

  print("\nBY REPOSITORY:")
  for repo, slice_df in slicer.slice_by_repo().items():
    avg = slice_df["completion_hours_business"].mean()
    print("  ", repo, ":", len(slice_df), "tickets, avg hours:", round(avg, 2))

  print("\nBY SENIORITY:")
  for seniority, slice_df in slicer.slice_by_seniority().items():
    avg = slice_df["completion_hours_business"].mean()
    print("  ", seniority, ":", len(slice_df), "tickets, avg hours:", round(avg, 2))

  print("\nBY COMPLETION TIME:")
  for bucket, slice_df in slicer.slice_by_completion_time().items():
    print("  ", bucket, ":", len(slice_df), "tickets")

  print("\nBY LABEL:")
  for label in ["bug", "enhancement", "crash"]:
    for name, slice_df in slicer.slice_by_label(label).items():
      print("  ", name, ":", len(slice_df), "tickets")

  print("\n" + "=" * 80)
  print("Analysis complete!")


if __name__ == "__main__":
  run_bias_analysis()
