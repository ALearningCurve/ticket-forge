"""Bias detection and analysis."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BiasAnalyzer:
  """Detect and analyze bias in model predictions across data slices."""

  def __init__(self, threshold: float = 0.1) -> None:
    """Initialize bias analyzer.

    Args:
        threshold: Performance difference threshold to flag as biased
    """
    self.threshold = threshold

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
      "mae": round(mae, 4),
      "rmse": round(rmse, 4),
      "r2": round(r2, 4),
      "count": len(y_true_clean),
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

    for slice_name, slice_df in slices.items():
      if y_true_col not in slice_df or y_pred_col not in slice_df:
        continue

      metrics = self.analyze_regression_metrics(
        pd.Series(slice_df[y_true_col]), pd.Series(slice_df[y_pred_col])
      )
      metrics_by_slice[slice_name] = metrics

    valid_slices = {
      name: m for name, m in metrics_by_slice.items() if m["mae"] is not None
    }

    if not valid_slices:
      return {
        "metrics_by_slice": metrics_by_slice,
        "bias_detected": False,
        "message": "No valid slices for comparison",
      }

    best_slice = min(valid_slices.items(), key=lambda x: x[1]["mae"])
    worst_slice = max(valid_slices.items(), key=lambda x: x[1]["mae"])

    mae_gap = worst_slice[1]["mae"] - best_slice[1]["mae"]
    relative_gap = mae_gap / best_slice[1]["mae"] if best_slice[1]["mae"] > 0 else 0

    bias_detected = relative_gap > self.threshold

    return {
      "metrics_by_slice": metrics_by_slice,
      "best_slice": {"name": best_slice[0], "mae": best_slice[1]["mae"]},
      "worst_slice": {"name": worst_slice[0], "mae": worst_slice[1]["mae"]},
      "mae_gap": round(mae_gap, 4),
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
      "total_dimensions_checked": len(all_slices),
      "biased_dimensions": biased_dimensions,
      "bias_count": len(biased_dimensions),
      "overall_bias_detected": len(biased_dimensions) > 0,
    }

    return {"summary": summary, "detailed_results": results}
