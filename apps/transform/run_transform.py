import json
import gzip
from pathlib import Path

import pandas as pd

from transform.normalize_text import normalize_ticket_text
from transform.temporal_features import compute_business_completion_hours
from transform.engineer_features import enrich_engineer_features
from transform.keyword_extraction import extract_keywords
from transform.embed import embed_text


# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/github_issues/tickets_final.json.gz")
OUTPUT_PATH = Path("data/github_issues/tickets_transformed.jsonl")


# -----------------------------
# Data loader (robust)
# -----------------------------
def load_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        content = f.read().strip()

        # JSON array
        if content.startswith("["):
            return json.loads(content)

        # JSON object
        if content.startswith("{"):
            obj = json.loads(content)
            if "tickets" in obj:
                return obj["tickets"]
            return [obj]

        # JSONL
        return [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    records = load_records(INPUT_PATH)
    df = pd.DataFrame(records)

    print(f"Loaded {len(df)} tickets")

    # Ensure required fields exist
    df["title"] = df.get("title", "")
    df["body"] = df.get("body", "")
    df["createdAt"] = df.get("createdAt")
    df["closedAt"] = df.get("closedAt")

    if "assignee" not in df.columns:
        df["assignee"] = "unknown"

    if "seniority" not in df.columns:
        df["seniority"] = "mid"

    # -----------------------------
    # Text normalization
    # -----------------------------
    print("Normalizing text...")
    df["normalized_text"] = df.apply(
        lambda r: normalize_ticket_text(r["title"], r["body"]),
        axis=1,
    )

    # -----------------------------
    # Temporal features
    # -----------------------------
    print("Computing temporal features...")
    df["completion_hours_business"] = df.apply(
        lambda r: compute_business_completion_hours(
            r["createdAt"], r["closedAt"]
        ),
        axis=1,
    )

    # -----------------------------
    # Engineer features
    # -----------------------------
    print("Enriching engineer features...")
    df = enrich_engineer_features(df)

    # -----------------------------
    # Keyword extraction
    # -----------------------------
    print("Extracting keywords...")
    df["keywords"] = extract_keywords(df["normalized_text"].tolist())

    # -----------------------------
    # Embeddings (stub)
    # -----------------------------
    print("Generating embeddings...")
    df["embedding"] = embed_text(df["normalized_text"].tolist())
    df["embedding_model"] = "stub-v1"

    # -----------------------------
    # Write output (JSONL)
    # -----------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")

    print(f"✓ Saved {len(df)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
