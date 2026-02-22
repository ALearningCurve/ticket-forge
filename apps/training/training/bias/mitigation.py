"""Bias mitigation techniques using Fairlearn.

Supports mitigation for:
- Regressor models (continuous predictions)
- Recommendation models (ranking/scoring predictions)
"""

import pandas as pd
from fairlearn.metrics import MetricFrame
from fairlearn.reductions import (
  BoundedGroupLoss,
  ExponentiatedGradient,
  SquareLoss,
)
from sklearn.base import BaseEstimator
from sklearn.utils import resample


class BiasMitigator:
  """Mitigate bias using Fairlearn for regressor and recommendation models."""

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
    """Compute sample weights to balance groups for fair training.

    Uses inverse frequency weighting so underrepresented groups
    receive higher importance during training.

    Args:
        data: Input DataFrame
        group_col: Column to balance by

    Returns:
        Series of sample weights
    """
    group_counts = data[group_col].value_counts()
    total = len(data)

    result = data[group_col].map(
      lambda x: total / (len(group_counts) * group_counts[x])
    )
    return pd.Series(result)

  @staticmethod
  def train_regressor_with_fairness(  # noqa: PLR0913
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    sensitive_features: pd.Series,
    upper_bound: float = 0.1,
    min_val: float = 0.0,
    max_val: float = 1.0,
  ) -> ExponentiatedGradient:
    """Train a regressor with Fairlearn's fairness constraints.

    Uses ExponentiatedGradient with SquareLoss (appropriate for regression)
    to train a model that satisfies bounded group loss constraints.

    Args:
        estimator: Base sklearn regressor to use
        x: Feature matrix
        y: Target values (continuous)
        sensitive_features: Group membership for each sample
        upper_bound: Maximum allowed loss difference between groups
        min_val: Minimum value for SquareLoss clipping
        max_val: Maximum value for SquareLoss clipping

    Returns:
        Fitted ExponentiatedGradient model
    """
    constraint = BoundedGroupLoss(SquareLoss(min_val, max_val), upper_bound=upper_bound)

    mitigator = ExponentiatedGradient(
      estimator=estimator,
      constraints=constraint,
    )
    mitigator.fit(x, y, sensitive_features=sensitive_features)
    return mitigator

  @staticmethod
  def train_recommendation_with_fairness(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    sensitive_features: pd.Series,
    upper_bound: float = 0.1,
  ) -> ExponentiatedGradient:
    """Train a recommendation model with Fairlearn's fairness constraints.

    Uses ExponentiatedGradient with SquareLoss to ensure fair ranking scores
    across groups.

    Args:
        estimator: Base sklearn estimator to use for scoring
        x: Feature matrix
        y: Target relevance scores
        sensitive_features: Group membership for each sample
        upper_bound: Maximum allowed score difference between groups

    Returns:
        Fitted ExponentiatedGradient model
    """
    # For recommendation, we use SquareLoss on the predicted scores
    constraint = BoundedGroupLoss(SquareLoss(0.0, 1.0), upper_bound=upper_bound)

    mitigator = ExponentiatedGradient(
      estimator=estimator,
      constraints=constraint,
    )
    mitigator.fit(x, y, sensitive_features=sensitive_features)
    return mitigator

  @staticmethod
  def adjust_predictions_for_fairness(
    predictions: pd.Series,
    sensitive_features: pd.Series,
    method: str = "equalize_mean",
  ) -> pd.Series:
    """Adjust predictions to reduce disparity across groups.

    For both regressor and recommendation models, this adjusts predictions
    to reduce the gap between groups.

    Args:
        predictions: Model predictions (continuous scores)
        sensitive_features: Group membership for each sample
        method: Adjustment method ("equalize_mean" or "scale_variance")

    Returns:
        Adjusted predictions
    """
    adjusted = predictions.copy()
    global_mean = predictions.mean()
    global_std = predictions.std()

    for group in sensitive_features.unique():
      group_mask = sensitive_features == group
      group_preds = predictions[group_mask]
      group_mean = group_preds.mean()

      if method == "equalize_mean":
        # Shift group predictions toward global mean
        adjustment = (group_mean - global_mean) * 0.5
        adjusted.loc[group_mask] = adjusted.loc[group_mask] - adjustment

      elif method == "scale_variance":
        # Scale group predictions to match global variance
        group_std = group_preds.std()
        if group_std > 0:
          centered = group_preds - group_mean
          scaled = centered * (global_std / group_std)
          adjusted.loc[group_mask] = scaled + global_mean

    return adjusted

  @staticmethod
  def compute_group_statistics(
    predictions: pd.Series,
    sensitive_features: pd.Series,
  ) -> dict[str, dict[str, float]]:
    """Compute prediction statistics per group.

    Useful for understanding disparity before/after mitigation.

    Args:
        predictions: Model predictions
        sensitive_features: Group membership for each sample

    Returns:
        Dictionary of group statistics
    """
    stats = {}
    for group in sensitive_features.unique():
      group_preds = predictions.loc[sensitive_features == group]
      stats[str(group)] = {
        "mean": round(float(group_preds.mean()), 4),
        "std": round(float(group_preds.std()), 4),
        "min": round(float(group_preds.min()), 4),
        "max": round(float(group_preds.max()), 4),
        "count": len(group_preds),
      }
    return stats

  @staticmethod
  def get_fairlearn_metrics_summary(
    metric_frame: MetricFrame,
    metric_name: str = "mae",
  ) -> dict[str, float]:
    """Extract summary statistics from a Fairlearn MetricFrame.

    Args:
        metric_frame: Fairlearn MetricFrame object
        metric_name: Name of metric to summarize

    Returns:
        Dictionary with difference and ratio metrics
    """
    diff = pd.Series(metric_frame.difference(method="between_groups"))
    ratio = pd.Series(metric_frame.ratio(method="between_groups"))
    group_min = pd.Series(metric_frame.group_min())
    group_max = pd.Series(metric_frame.group_max())
    return {
      f"{metric_name}_difference": float(diff.loc[metric_name]),
      f"{metric_name}_ratio": float(ratio.loc[metric_name]),
      f"group_min_{metric_name}": float(group_min.loc[metric_name]),
      f"group_max_{metric_name}": float(group_max.loc[metric_name]),
    }
