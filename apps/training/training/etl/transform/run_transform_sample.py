"""Transform pipeline for sampled tickets."""

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from training.etl.transform.embed import embed_text
from training.etl.transform.engineer_features import enrich_engineer_features
from training.etl.transform.keyword_extraction import extract_keywords
from training.etl.transform.normalize_text import normalize_ticket_text
from training.etl.transform.temporal_features import compute_business_completion_hours

INPUT_PATH = Path("data/github_issues/sample_tickets.json")
OUTPUT_PATH = Path("data/github_issues/sample_tickets_transformed.jsonl")


def load_records(path: Path) -> list[dict]:
  """Load records from JSON file."""
  with open(path, encoding="utf-8") as f:
    data = json.load(f)

  if isinstance(data, list):
    return data

  if isinstance(data, dict) and "tickets" in data:
    return data["tickets"]

  return [data]


def main() -> None:
  """Run the transformation pipeline on sampled tickets."""
  device = "GPU" if torch.cuda.is_available() else "CPU"
  print("Using device:", device)

  records = load_records(INPUT_PATH)
  df = pd.DataFrame(records)
  print("Loaded", len(df), "tickets")

  df["title"] = df.get("title", "")
  df["body"] = df.get("body", "")

  if "assignee" not in df.columns:
    df["assignee"] = pd.NA
  else:
    df["assignee"] = df["assignee"].fillna(pd.NA)

  if "seniority" not in df.columns:
    df["seniority"] = "mid"

  print("Normalizing text...")
  tqdm.pandas(desc="Normalizing")
  df["normalized_text"] = df.progress_apply(
    lambda r: normalize_ticket_text(r["title"], r["body"]),
    axis=1,
  )

  print("Computing temporal features...")
  tqdm.pandas(desc="Temporal features")
  df["completion_hours_business"] = df.progress_apply(
    lambda r: compute_business_completion_hours(
      r.get("created_at"),
      r.get("assigned_at"),
      r.get("closed_at"),
    ),
    axis=1,
  )

  print("Enriching engineer features...")
  df = enrich_engineer_features(df)

  print("Extracting keywords...")
  df["keywords"] = extract_keywords(df["normalized_text"].tolist())

  print("Generating embeddings...")
  df["embedding"] = embed_text(df["normalized_text"].tolist())
  df["embedding_model"] = "all-MiniLM-L6-v2"

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  print("Saving to", OUTPUT_PATH)
  with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in df.to_dict(orient="records"):
      f.write(json.dumps(row) + "\n")

  print("Saved", len(df), "records")


if __name__ == "__main__":
  main()
