"""Run bias detection analysis on ticket prediction data."""

import json

import pandas as pd
from shared.configuration import Paths
from training.bias import DataSlicer


def run_bias_analysis() -> None:
  """Analyze bias in ticket assignment predictions."""
  print("Starting bias detection analysis...\n")

  # Load transformed tickets
  data_path = Paths.data_root / "github_issues" / "tickets_transformed_improved.jsonl"

  print(f"Loading data from {data_path}...")
  tickets = []
  with open(data_path, encoding="utf-8") as f:
    for line in f:
      if line.strip():
        tickets.append(json.loads(line))

  df = pd.DataFrame(tickets)
  print(f"Loaded {len(df)} tickets\n")

  # For now, we'll analyze the data distribution
  # Later, when you have model predictions, we'll compare predicted vs actual

  print("DATA DISTRIBUTION ANALYSIS")

  # Initialize slicer
  slicer = DataSlicer(df)

  # By repository
  print("\nBY REPOSITORY:")
  for repo, slice_df in slicer.slice_by_repo().items():
    avg = slice_df["completion_hours_business"].mean()
    print("  ", repo, ":", len(slice_df), "tickets, avg hours:", round(avg, 2))

  # By seniority
  print("\nBY SENIORITY:")
  for seniority, slice_df in slicer.slice_by_seniority().items():
    avg = slice_df["completion_hours_business"].mean()
    print("  ", seniority, ":", len(slice_df), "tickets, avg hours:", round(avg, 2))

  # By completion time
  print("\nBY COMPLETION TIME:")
  for bucket, slice_df in slicer.slice_by_completion_time().items():
    print("  ", bucket, ":", len(slice_df), "tickets")

  # By label
  print("\nBY LABEL:")
  for label in ["bug", "enhancement", "crash"]:
    for name, slice_df in slicer.slice_by_label(label).items():
      print("  ", name, ":", len(slice_df), "tickets")

  print("Analysis complete!")


if __name__ == "__main__":
  run_bias_analysis()
