from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
from shared import get_logger
from shared.cache import JsonSaver, fs_cache
from shared.configuration import Paths
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from training.bias import BiasAnalyzer, BiasMitigator, BiasReport
from training.dataset import Dataset, X_t, Y_t

logger = get_logger(__name__)

# Default sensitive features to check for bias
DEFAULT_SENSITIVE_FEATURES = ["repo", "seniority"]


def load_fit_dump(
  fit_grid: Callable[
    [X_t, Y_t, PredefinedSplit, Y_t | None],
    RandomizedSearchCV,
  ],
  run_id: str,
  model_name: str,
) -> None:
  """Fits model by loading the appropriate dataset, doing cv, saving model,
  and saving eval metrics.

  Args:
      fit_grid: function to do hyperparam search, now accepting n_gram as well
      run_id: UUID of the training run
      model_name: identified of the type of model (i.e. logistic)
  """

  # Define the cached fit function
  @fs_cache(Paths.models_root / run_id / f"{model_name}.pkl")
  def _run_search() -> RandomizedSearchCV:
    # 1. Get the combined data, PredefinedSplit, and per-sample weights
    x_comp, y_comp, cv_split, weights = Dataset.as_sklearn_cv_split_with_weights()

    # 2. Pass them into the fit_grid logic
    return fit_grid(x_comp, y_comp, cv_split, weights)

  # Execute (either loads from disk or runs the fit)
  res = _run_search()

  # Display results and evaluate on test set
  pretty_print_gridsearch(res, run_id, model_name)


def get_test_accuracy(
  grid: RandomizedSearchCV,
  run_id: str,
  model_name: str,
) -> None:
  """Computes the test accuracy for the best model in the grid search.

  Args:
      grid: the sklearn hyperparam grid
      run_id: UUID of the training run
      model_name: identified of the type of model (i.e. logistic)
  """

  @fs_cache(Paths.models_root / run_id / f"eval_{model_name}.json", saver=JsonSaver())
  def compute_metrics() -> dict[str, float]:
    test_dataset = Dataset(split="test")

    x = test_dataset.load_x()  # noqa: N806
    y = test_dataset.load_y()

    y_pred = grid.predict(x)
    mse = mean_squared_error(y, y_pred)

    return {
      "mae": mean_absolute_error(y, y_pred),
      "mse": mse,
      "rmse": np.sqrt(mse),
      "r2": r2_score(y, y_pred),
    }

  metrics = compute_metrics()

  logger.info("Test metrics: %s", metrics)


def evaluate_bias(
  grid: RandomizedSearchCV,
  run_id: str,
  model_name: str,
  sensitive_feature: str = "repo",
) -> dict | None:
  """Run bias analysis on model predictions, and auto-remediate if bias detected.

  Evaluates the model on slices of the test set defined by sensitive_feature.
  If bias is detected (relative gap > threshold), applies post-processing
  prediction adjustment via BiasMitigator.adjust_predictions_for_fairness()
  and re-evaluates. Both before and after metrics are saved to the bias report
  for documentation.

  Args:
      grid: Fitted grid search with best model
      run_id: UUID of the training run
      model_name: Identifier of the model type
      sensitive_feature: Feature to use for bias analysis (e.g. "repo", "seniority")

  Returns:
      Bias analysis results (post-remediation if bias was detected), or None
      if sensitive feature not available
  """
  model_type = "regressor"
  threshold = 0.4
  test_dataset = Dataset(split="test")

  x = test_dataset.load_x()
  y = test_dataset.load_y()
  y_pred = grid.predict(x)

  # Try to get sensitive features from test data metadata
  try:
    test_meta = test_dataset.load_metadata()
    if sensitive_feature not in test_meta.columns:
      logger.warning(
        "Bias analysis skipped: %r not in test metadata", sensitive_feature
      )
      return None
    sensitive_features = pd.Series(test_meta[sensitive_feature].to_numpy())
  except (AttributeError, FileNotFoundError):
    logger.warning("Bias analysis skipped: metadata not available")
    return None

  analyzer = BiasAnalyzer(threshold=threshold, model_type=model_type)
  y_true_series = pd.Series(y)
  y_pred_series = pd.Series(y_pred)

  # --- Initial bias detection ---
  analysis_before = analyzer.detect_bias_fairlearn(
    y_true=y_true_series,
    y_pred=y_pred_series,
    sensitive_features=sensitive_features,
  )

  primary_metric = analysis_before["primary_metric"]
  logger.info(
    "Bias Analysis (%s): Best=%s (%s=%.4f), Worst=%s (%s=%.4f), Gap=%.1f%%",
    sensitive_feature,
    analysis_before["best_group"]["name"],
    primary_metric,
    analysis_before["best_group"][primary_metric],
    analysis_before["worst_group"]["name"],
    primary_metric,
    analysis_before["worst_group"][primary_metric],
    analysis_before["relative_gap"] * 100,
  )

  # --- Auto-remediation if bias detected ---
  remediation_applied = False
  analysis_after = None

  if analysis_before["bias_detected"]:
    logger.warning(
      "Bias detected for %r (gap=%.1f%% > threshold=%.1f%%). "
      "Applying prediction adjustment remediation...",
      sensitive_feature,
      analysis_before["relative_gap"] * 100,
      threshold * 100,
    )

    # Apply post-processing prediction adjustment
    y_pred_adjusted = BiasMitigator.adjust_predictions_for_fairness(
      predictions=y_pred_series,
      sensitive_features=sensitive_features,
      method="equalize_mean",
    )

    # Re-evaluate after remediation
    analysis_after = analyzer.detect_bias_fairlearn(
      y_true=y_true_series,
      y_pred=y_pred_adjusted,
      sensitive_features=sensitive_features,
    )
    remediation_applied = True

    post_metric = analysis_after.get("primary_metric", primary_metric)
    logger.info(
      "Post-remediation Bias (%s): Best=%s (%s=%.4f), Worst=%s (%s=%.4f), Gap=%.1f%%",
      sensitive_feature,
      analysis_after["best_group"]["name"],
      post_metric,
      analysis_after["best_group"][post_metric],
      analysis_after["worst_group"]["name"],
      post_metric,
      analysis_after["worst_group"][post_metric],
      analysis_after["relative_gap"] * 100,
    )

    gap_improvement = (
      analysis_before["relative_gap"] - analysis_after["relative_gap"]
    ) * 100
    logger.info(
      "Remediation reduced bias gap by %.1f percentage points for %r",
      gap_improvement,
      sensitive_feature,
    )
  else:
    logger.info("No significant bias detected for: %s", sensitive_feature)

  # --- Save report (includes before + after if remediation was applied) ---
  final_analysis = analysis_after if remediation_applied else analysis_before
  report_path = (
    Paths.models_root / run_id / f"bias_{model_name}_{sensitive_feature}.txt"
  )
  report_data = {
    "summary": {
      "model_type": model_type,
      "total_dimensions_checked": 1,
      "biased_dimensions": (
        [sensitive_feature] if final_analysis["bias_detected"] else []
      ),
      "bias_count": 1 if final_analysis["bias_detected"] else 0,
      "overall_bias_detected": final_analysis["bias_detected"],
      "bias_detected_before_remediation": analysis_before["bias_detected"],
      "remediation_applied": remediation_applied,
      "remediation_method": "equalize_mean" if remediation_applied else None,
    },
    "detailed_results": {
      sensitive_feature: final_analysis,
    },
    "remediation_details": {
      sensitive_feature: {
        "before_remediation": analysis_before,
        **({"after_remediation": analysis_after} if remediation_applied else {}),
      }
    },
  }
  BiasReport.save_report(report_data, str(report_path))

  return final_analysis


def pretty_print_gridsearch(
  grid: RandomizedSearchCV,
  run_id: str,
  model_name: str,
) -> None:
  """Given gridsearch cv, creates pretty tabular view.

  Args:
      grid: the gridcv whose results we should display.
      run_id: UUID of the training run
      model_name: identified of the type of model (i.e. logistic)
  """
  df = (
    pl.DataFrame(grid.cv_results_, strict=False)[
      [
        "mean_fit_time",
        "mean_score_time",
        "params",
        "mean_test_score",
        "rank_test_score",
      ]
    ]
    .with_columns(
      pl.col("mean_test_score").round(2),
      *[
        pl.duration(seconds=pl.col(col), time_unit="ms").alias(col)
        for col in ["mean_fit_time", "mean_score_time"]
      ],
      pl.col("params").struct.json_encode(),
    )
    .sort(pl.col("rank_test_score"))
  )
  logger.info("Hyper-parameter search results:")
  with pl.Config(tbl_hide_dataframe_shape=True):
    logger.info("\n%s", df)
    total_time = df["mean_fit_time"].sum() + df["mean_score_time"].sum()
    logger.info("Total training time: %s", total_time)
    get_test_accuracy(grid, run_id, model_name)

    # Run bias analysis on test set for all sensitive features
    for feature in DEFAULT_SENSITIVE_FEATURES:
      evaluate_bias(grid, run_id, model_name, sensitive_feature=feature)
