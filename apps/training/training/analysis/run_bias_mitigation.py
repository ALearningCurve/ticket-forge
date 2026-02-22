"""Demonstrate bias mitigation on ticket data."""

import json

import pandas as pd
from ml_core.bias import BiasMitigator  # type: ignore[attr-defined]
from shared.configuration import Paths


def main() -> None:
  """Run bias mitigation analysis."""
  print("Starting bias mitigation demo...\n")

  data_path = Paths.data_root / "github_issues" / "tickets_transformed_improved.jsonl"
  print("Loading data...")

  tickets = []
  with open(data_path, encoding="utf-8") as f:
    for line in f:
      if line.strip():
        tickets.append(json.loads(line))

  df = pd.DataFrame(tickets)
  print("Loaded", len(df), "tickets\n")

  print("ORIGINAL DISTRIBUTION BY REPO:")
  print(df["repo"].value_counts())
  print()

  print("Applying resampling mitigation...")
  mitigator = BiasMitigator()
  balanced_df = mitigator.resample_underrepresented(df, "repo")

  print("\nBALANCED DISTRIBUTION BY REPO:")
  print(balanced_df["repo"].value_counts())
  print()

  print("Computing sample weights for training...")
  weights = mitigator.compute_sample_weights(df, "repo")

  print("\nSAMPLE WEIGHTS BY REPO:")
  for repo in df["repo"].unique():
    repo_mask = df["repo"] == repo
    if repo_mask.any():
      repo_weight = float(weights[repo_mask].iloc[0])
      print("  ", repo, ": weight =", round(repo_weight, 4))

  output_path = Paths.data_root / "github_issues" / "tickets_balanced.jsonl"
  print("\nSaving balanced dataset to", output_path)

  with open(output_path, "w", encoding="utf-8") as f:
    for row in balanced_df.to_dict(orient="records"):
      f.write(json.dumps(row) + "\n")

  print("Saved", len(balanced_df), "tickets")
  print("\nBias mitigation complete!")


if __name__ == "__main__":
  main()
