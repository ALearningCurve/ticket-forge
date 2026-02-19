"""Bias mitigation techniques."""

import pandas as pd
from sklearn.utils import resample


class BiasMitigator:
  """Mitigate bias in datasets through resampling and reweighting."""

  @staticmethod
  def resample_underrepresented(
    data: pd.DataFrame, group_col: str, target_size: int | None = None
  ) -> pd.DataFrame:
    """Resample underrepresented groups to balance dataset.

    Args:
        data: Input DataFrame
        group_col: Column to balance by
        target_size: Target size for each group

    Returns:
        Resampled DataFrame with balanced groups
    """
    groups = data.groupby(group_col)

    if target_size is None:
      max_size = groups.size().max()
      target_size = int(max_size) if not pd.isna(max_size) else 0  # type: ignore[return-value]

    resampled_groups = []

    for _name, group in groups:
      if len(group) < target_size:
        resampled = resample(
          group, n_samples=target_size, replace=True, random_state=42
        )
      else:
        resampled = group

      resampled_groups.append(resampled)

    return pd.concat(resampled_groups, ignore_index=True)

  @staticmethod
  def compute_sample_weights(data: pd.DataFrame, group_col: str) -> pd.Series:
    """Compute sample weights to balance groups."""
    group_counts = data[group_col].value_counts()
    total = len(data)

    result = data[group_col].map(
      lambda x: total / (len(group_counts) * group_counts[x])
    )
    return pd.Series(result)  # Explicit cast to Series

  @staticmethod
  def apply_fairness_threshold(
    predictions: pd.Series,
    sensitive_feature: pd.Series,
    threshold: float = 0.5,
  ) -> pd.Series:
    """Apply threshold adjustment for fairness.

    Args:
        predictions: Model predictions
        sensitive_feature: Sensitive attribute
        threshold: Decision threshold

    Returns:
        Adjusted predictions
    """
    adjusted = predictions.copy()

    for group in sensitive_feature.unique():
      group_mask = sensitive_feature == group
      group_preds = predictions[group_mask]

      group_mean = group_preds.mean()
      global_mean = predictions.mean()

      if group_mean > global_mean:
        adjustment = (group_mean - global_mean) * 0.5
        adjusted[group_mask] = adjusted[group_mask] - adjustment

    return adjusted
