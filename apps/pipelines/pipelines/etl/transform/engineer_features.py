import pandas as pd

SENIORITY_MAP = {
  "intern": 0,
  "junior": 1,
  "mid": 2,
  "senior": 3,
  "staff": 4,
  "principal": 5,
}


def enrich_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  """Add engineer-level features to the ticket DataFrame.

  Sorts df by created_at in-place before computing historical features
  to ensure chronological order (required for leak-free expanding mean).

  Adds:
    - seniority_enum: integer encoding of seniority level (0=intern to
      5=principal). Defaults to 2 (mid) if seniority column is missing
      or value is unrecognised.
    - historical_avg_completion_hours: per-assignee expanding mean of
      completion_hours_business, shifted by 1 to avoid data leakage
      (only uses tickets completed before the current one).
      Set to None if assignee, completion_hours_business, or created_at
      columns are missing.
  """
  # Seniority handling
  if "seniority" in df.columns:
    df["seniority_enum"] = (
      df["seniority"].astype(str).str.lower().map(SENIORITY_MAP).fillna(2).astype(int)
    )
  else:
    df["seniority_enum"] = 2  # default = mid

  # Historical completion speed
  if (
    "assignee" in df.columns
    and "completion_hours_business" in df.columns
    and "created_at" in df.columns
  ):
    df = df.sort_values("created_at")
    df["historical_avg_completion_hours"] = df.groupby("assignee")[
      "completion_hours_business"
    ].transform(lambda x: x.expanding().mean().shift(1))
  else:
    df["historical_avg_completion_hours"] = None

  return df
