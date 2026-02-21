"""Data slicing for bias analysis."""

from typing import Any

import pandas as pd


class DataSlicer:
  """Slice data by different features for bias detection."""

  def __init__(self, data: pd.DataFrame) -> None:
    """Initialize data slicer.

    Args:
        data: DataFrame with ticket data
    """
    self.data = data

  def slice_by_repo(self) -> dict[str, pd.DataFrame]:
    """Slice data by repository."""
    return {str(repo): group for repo, group in self.data.groupby("repo")}

  def slice_by_seniority(self) -> dict[str, pd.DataFrame]:
    """Slice data by engineer seniority level."""
    return {
      str(seniority): group for seniority, group in self.data.groupby("seniority")
    }

  def slice_by_label(self, label: str) -> dict[str, Any]:
    """Slice data by presence of a specific label.

    Args:
        label: Label to check for

    Returns:
        Dict with has_label and no_label DataFrames
    """
    has_label = self.data[self.data["labels"].str.contains(label, na=False)]
    no_label = self.data[~self.data["labels"].str.contains(label, na=False)]

    return {f"has_{label}": has_label, f"no_{label}": no_label}

  def slice_by_completion_time(self) -> dict[str, pd.DataFrame]:
    """Slice by ticket completion time buckets."""
    data = self.data.copy()

    def categorize_time(hours: float | None) -> str:
      if hours is None or pd.isna(hours):
        return "unknown"
      if hours < 5:
        return "fast"
      if hours < 24:
        return "medium"
      return "slow"

    data["time_bucket"] = data["completion_hours_business"].apply(categorize_time)

    return {str(bucket): group for bucket, group in data.groupby("time_bucket")}

  def slice_by_keywords(self, keyword: str) -> dict[str, Any]:
    """Slice data by presence of a keyword.

    Args:
        keyword: Keyword to check for

    Returns:
        Dict with has_keyword and no_keyword DataFrames
    """

    def has_keyword(keywords_list: list[str]) -> bool:
      if not isinstance(keywords_list, list):
        return False
      return keyword.lower() in [k.lower() for k in keywords_list]

    has_kw = self.data[self.data["keywords"].apply(has_keyword)]
    no_kw = self.data[~self.data["keywords"].apply(has_keyword)]

    return {f"has_{keyword}": has_kw, f"no_{keyword}": no_kw}

  def get_all_slices(self) -> dict[str, dict[str, pd.DataFrame]]:
    """Get all predefined slices."""
    return {
      "by_repo": self.slice_by_repo(),
      "by_seniority": self.slice_by_seniority(),
      "by_bug_label": self.slice_by_label("bug"),
      "by_completion_time": self.slice_by_completion_time(),
    }
