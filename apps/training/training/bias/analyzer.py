"""Bias detection and analysis using Fairlearn.

Supports bias detection for:
- Regressor models (continuous predictions like completion time)
- Recommendation models (ranking/scoring predictions)
"""

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
  """Calculate RMSE for use with MetricFrame."""
  return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _count(y_true: pd.Series, y_pred: pd.Series) -> int:
  """Count samples for use with MetricFrame."""
  return len(y_true)


def _ndcg_at_k(y_true: pd.Series, y_pred: pd.Series, k: int = 10) -> float:
  """Calculate NDCG@k for recommendation evaluation.

  Args:
      y_true: True relevance scores
      y_pred: Predicted scores
      k: Number of top items to consider

  Returns:
      NDCG score between 0 and 1
  """
  if len(y_true) == 0:
    return 0.0

  # Get top-k indices by predicted score
  top_k_indices = np.argsort(y_pred)[::-1][:k]
  dcg = sum(
    y_true.iloc[idx] / np.log2(rank + 2)
    for rank, idx in enumerate(top_k_indices)  # type: ignore[union-attr]
  )

  # Ideal DCG
  ideal_order = np.argsort(y_true)[::-1][:k]
  idcg = sum(
    y_true.iloc[idx] / np.log2(rank + 2)
    for rank, idx in enumerate(ideal_order)  # type: ignore[union-attr]
  )

  return float(dcg / idcg) if idcg > 0 else 0.0


def _mean_reciprocal_rank(y_true: pd.Series, y_pred: pd.Series) -> float:
  """Calculate Mean Reciprocal Rank for recommendation evaluation.

  Args:
      y_true: True relevance (1 for relevant, 0 for not)
      y_pred: Predicted scores

  Returns:
      MRR score between 0 and 1
  """
  if len(y_true) == 0 or y_true.sum() == 0:
    return 0.0

  # Sort by predicted score descending
  sorted_indices = np.argsort(y_pred)[::-1]
  sorted_true = y_true.iloc[sorted_indices]  # type: ignore[union-attr]

  # Find first relevant item
  relevant_positions = np.where(sorted_true > 0)[0]
  if len(relevant_positions) == 0:
    return 0.0

  return float(1.0 / (relevant_positions[0] + 1))


# Metric dictionaries for different model types
REGRESSION_METRICS = {
  "mae": mean_absolute_error,
  "rmse": _rmse,
  "r2": r2_score,
  "count": _count,
}

RECOMMENDATION_METRICS = {
  "ndcg": _ndcg_at_k,
  "mrr": _mean_reciprocal_rank,
  "mae": mean_absolute_error,  # Score prediction error
  "count": _count,
}


class BiasAnalyzer:
  """Detect and analyze bias using Fairlearn for regressor and recommendation models."""

  def __init__(self, threshold: float = 0.1, model_type: str = "regressor") -> None:
    """Initialize bias analyzer.

    Args:
        threshold: Performance difference threshold to flag as biased
        model_type: Type of model ("regressor" or "recommendation")
    """
    self.threshold = threshold
    self.model_type = model_type
    self._metrics = (
      REGRESSION_METRICS if model_type == "regressor" else RECOMMENDATION_METRICS
    )
    self._primary_metric = "mae" if model_type == "regressor" else "ndcg"

  def analyze_regression_metrics(
    self, y_true: pd.Series, y_pred: pd.Series
  ) -> dict[str, float | None]:
    """Calculate regression metrics for a slice.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mask = y_true.notna() & y_pred.notna()
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
      return {"mae": None, "rmse": None, "r2": None, "count": 0}

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_clean, y_pred_clean)

    return {
      "mae": round(float(mae), 4),
      "rmse": round(float(rmse), 4),
      "r2": round(float(r2), 4),
      "count": len(y_true_clean),
    }

  def analyze_recommendation_metrics(
    self, y_true: pd.Series, y_pred: pd.Series
  ) -> dict[str, float | None]:
    """Calculate recommendation metrics for a slice.

    Args:
        y_true: True relevance scores
        y_pred: Predicted scores

    Returns:
        Dictionary of metrics
    """
    mask = y_true.notna() & y_pred.notna()
    y_true_clean = y_true.loc[mask]
    y_pred_clean = y_pred.loc[mask]

    if len(y_true_clean) == 0:
      return {"ndcg": None, "mrr": None, "mae": None, "count": 0}

    return {
      "ndcg": round(_ndcg_at_k(y_true_clean, y_pred_clean), 4),
      "mrr": round(_mean_reciprocal_rank(y_true_clean, y_pred_clean), 4),
      "mae": round(float(mean_absolute_error(y_true_clean, y_pred_clean)), 4),
      "count": len(y_true_clean),
    }

  def analyze_with_metricframe(
    self,
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
  ) -> MetricFrame:
    """Analyze bias using Fairlearn's MetricFrame.

    Args:
        y_true: True values
        y_pred: Predicted values
        sensitive_features: Group membership for each sample

    Returns:
        Fairlearn MetricFrame with group-wise metrics
    """
    mask = y_true.notna() & y_pred.notna()

    return MetricFrame(
      metrics=self._metrics,
      y_true=y_true[mask],
      y_pred=y_pred[mask],
      sensitive_features=sensitive_features[mask],
    )

  def detect_bias_fairlearn(
    self,
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
  ) -> dict[str, Any]:
    """Detect bias using Fairlearn MetricFrame.

    Args:
        y_true: True values
        y_pred: Predicted values
        sensitive_features: Group membership for each sample

    Returns:
        Analysis results with Fairlearn metrics and bias detection
    """
    metric_frame = self.analyze_with_metricframe(y_true, y_pred, sensitive_features)
    metric = self._primary_metric

    # Get group metrics as dict
    by_group = metric_frame.by_group.to_dict()

    # Calculate disparity metrics using Fairlearn
    _diff: Any = metric_frame.difference(method="between_groups")[metric]
    _ratio: Any = metric_frame.ratio(method="between_groups")[metric]
    metric_diff: float = float(_diff)
    metric_ratio: float = float(_ratio)

    # Get min/max groups
    metric_by_group: pd.Series = pd.Series(metric_frame.by_group[metric])

    # For regression (MAE), lower is better; for recommendation (NDCG), higher is better
    if self.model_type == "regressor":
      best_group = metric_by_group.idxmin()
      worst_group = metric_by_group.idxmax()
      best_val: float = float(metric_by_group.at[best_group])
      relative_gap = float(metric_diff / best_val) if best_val > 0 else 0.0
    else:  # recommendation - higher NDCG is better
      best_group = metric_by_group.idxmax()
      worst_group = metric_by_group.idxmin()
      best_val = float(metric_by_group.at[best_group])
      relative_gap = float(metric_diff / best_val) if best_val > 0 else 0.0

    bias_detected = relative_gap > self.threshold

    return {
      "model_type": self.model_type,
      "primary_metric": metric,
      "metric_frame": metric_frame,
      "metrics_by_group": by_group,
      "best_group": {
        "name": str(best_group),
        metric: round(float(metric_by_group.at[best_group]), 4),
      },
      "worst_group": {
        "name": str(worst_group),
        metric: round(float(metric_by_group.at[worst_group]), 4),
      },
      f"{metric}_difference": round(metric_diff, 4),
      f"{metric}_ratio": round(metric_ratio, 4),
      "relative_gap": round(relative_gap, 4),
      "bias_detected": bias_detected,
      "threshold": self.threshold,
    }

  def compare_slices(
    self, slices: dict[str, pd.DataFrame], y_true_col: str, y_pred_col: str
  ) -> dict[str, Any]:
    """Compare model performance across slices.

    Args:
        slices: Dictionary of slice_name to DataFrame
        y_true_col: Column name with true values
        y_pred_col: Column name with predictions

    Returns:
        Analysis results with bias detection
    """
    metrics_by_slice = {}
    analyze_fn = (
      self.analyze_regression_metrics
      if self.model_type == "regressor"
      else self.analyze_recommendation_metrics
    )
    metric = self._primary_metric

    for slice_name, slice_df in slices.items():
      if y_true_col not in slice_df or y_pred_col not in slice_df:
        continue

      metrics = analyze_fn(
        pd.Series(slice_df[y_true_col]), pd.Series(slice_df[y_pred_col])
      )
      metrics_by_slice[slice_name] = metrics

    valid_slices = {
      name: m for name, m in metrics_by_slice.items() if m[metric] is not None
    }

    if not valid_slices:
      return {
        "metrics_by_slice": metrics_by_slice,
        "bias_detected": False,
        "message": "No valid slices for comparison",
      }

    # For regression (MAE), lower is better; for recommendation (NDCG), higher is better
    if self.model_type == "regressor":
      best_slice = min(valid_slices.items(), key=lambda x: x[1][metric])  # type: ignore[arg-type]
      worst_slice = max(valid_slices.items(), key=lambda x: x[1][metric])  # type: ignore[arg-type]
    else:
      best_slice = max(valid_slices.items(), key=lambda x: x[1][metric])  # type: ignore[arg-type]
      worst_slice = min(valid_slices.items(), key=lambda x: x[1][metric])  # type: ignore[arg-type]

    metric_gap = abs(worst_slice[1][metric] - best_slice[1][metric])  # type: ignore[operator]
    relative_gap = (
      metric_gap / best_slice[1][metric] if best_slice[1][metric] > 0 else 0  # type: ignore[operator]
    )

    bias_detected = relative_gap > self.threshold

    return {
      "model_type": self.model_type,
      "primary_metric": metric,
      "metrics_by_slice": metrics_by_slice,
      "best_slice": {"name": best_slice[0], metric: best_slice[1][metric]},
      "worst_slice": {"name": worst_slice[0], metric: worst_slice[1][metric]},
      f"{metric}_gap": round(metric_gap, 4),
      "relative_gap": round(relative_gap, 4),
      "bias_detected": bias_detected,
      "threshold": self.threshold,
    }

  def detect_bias_multiple_dimensions(
    self,
    all_slices: dict[str, dict[str, pd.DataFrame]],
    y_true_col: str,
    y_pred_col: str,
  ) -> dict[str, Any]:
    """Detect bias across multiple slicing dimensions.

    Args:
        all_slices: Nested dict of dimension to slice_name to DataFrame
        y_true_col: Column with true values
        y_pred_col: Column with predictions

    Returns:
        Comprehensive bias analysis across all dimensions
    """
    results = {}

    for dimension, slices in all_slices.items():
      results[dimension] = self.compare_slices(slices, y_true_col, y_pred_col)

    biased_dimensions = [
      dim for dim, res in results.items() if res.get("bias_detected", False)
    ]

    summary = {
      "model_type": self.model_type,
      "total_dimensions_checked": len(all_slices),
      "biased_dimensions": biased_dimensions,
      "bias_count": len(biased_dimensions),
      "overall_bias_detected": len(biased_dimensions) > 0,
    }

    return {"summary": summary, "detailed_results": results}
