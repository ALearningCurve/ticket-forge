"""Model sensitivity analysis: hyperparameter sensitivity and SHAP feature importance.

Generates two sets of plots per model run:
  1. Hyperparameter sensitivity — how each hyperparameter affects CV score,
     loaded from cv_results_{model}.json saved by the training harness.
  2. SHAP feature importance — mean absolute SHAP values across test samples,
     loaded from the model pickle and evaluated against the test dataset.

Outputs are saved as PNG files under the run directory:
  models/{run_id}/hyperparam_sensitivity_{model}.png
  models/{run_id}/shap_importance_{model}.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shared.configuration import TRAIN_USE_DUMMY_DATA, Paths
from shared.logging import get_logger
from training.dataset import Dataset

logger = get_logger(__name__)

# Number of top features to display in SHAP bar chart.
_TOP_N_FEATURES = 20
# Number of SHAP background samples for TreeExplainer (KernelExplainer fallback).
_SHAP_BACKGROUND_SAMPLES = 100
# Embedding dimension label prefix.
_EMBEDDING_PREFIX = "emb_"


def _feature_names(n_features: int) -> list[str]:
  """Generate feature names for 384-dim embedding inputs.

  Args:
      n_features: Total number of features (embedding dimensions).

  Returns:
      List of feature name strings like ['emb_000', 'emb_001', ...].
  """
  width = len(str(n_features - 1))
  return [f"{_EMBEDDING_PREFIX}{str(i).zfill(width)}" for i in range(n_features)]


# ---------------------------------------------------------------------------
# Hyperparameter sensitivity
# ---------------------------------------------------------------------------


def plot_hyperparam_sensitivity(
  cv_results_path: Path,
  output_path: Path,
  model_name: str,
) -> None:
  """Plot how each hyperparameter affects CV score.

  Reads cv_results_{model}.json, groups results by each hyperparameter value,
  computes mean test score per value, and saves a bar chart per hyperparameter.

  Args:
      cv_results_path: Path to cv_results_{model}.json.
      output_path:     Path to save the output PNG.
      model_name:      Model name for the plot title.
  """
  if not cv_results_path.exists():
    logger.warning(
      "cv_results not found at %s — skipping hyperparam plot",
      cv_results_path,
    )
    return

  with open(cv_results_path) as f:
    cv_results: dict = json.load(f)

  df = pd.DataFrame(cv_results)
  # mean_test_score from sklearn is negative MSE — convert to positive MSE
  df["score"] = -df["mean_test_score"]

  # Extract individual hyperparameter columns (param_ prefix added by sklearn)
  param_cols = [c for c in df.columns if c.startswith("param_")]
  if not param_cols:
    logger.warning("No param_ columns found in cv_results for %s", model_name)
    return

  n_params = len(param_cols)
  fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5), squeeze=False)
  fig.suptitle(f"Hyperparameter Sensitivity — {model_name}", fontsize=14)

  for ax, col in zip(axes[0], param_cols, strict=False):
    param_name = col.replace("param_", "")
    # Bin continuous params into deciles for readability
    try:
      series = pd.to_numeric(df[col], errors="coerce")
      is_numeric = series.notna().all()
    except Exception:
      is_numeric = False

    if is_numeric:
      try:
        df["_bin"] = pd.qcut(
          series,
          q=min(10, series.nunique()),
          duplicates="drop",
        )
        grouped = df.groupby("_bin", observed=True)["score"].mean().reset_index()
        labels = [str(b) for b in grouped["_bin"]]
        values = grouped["score"].tolist()
      except Exception:
        grouped = df.groupby(col)["score"].mean().reset_index()
        labels = [str(v) for v in grouped[col]]
        values = grouped["score"].tolist()
    else:
      grouped = df.groupby(col)["score"].mean().reset_index()
      labels = [str(v) for v in grouped[col]]
      values = grouped["score"].tolist()

    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.9, len(labels)))  # type: ignore[arg-type]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean CV MSE (lower = better)")
    ax.set_title(param_name)
    ax.grid(axis="y", alpha=0.3)

  plt.tight_layout()
  plt.savefig(output_path, dpi=100, bbox_inches="tight")
  plt.close()
  logger.info("Hyperparameter sensitivity plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------


def plot_shap_importance(
  model_path: Path,
  output_path: Path,
  model_name: str,
  top_n: int = _TOP_N_FEATURES,
) -> None:
  """Compute and plot SHAP mean absolute feature importances.

  Uses TreeExplainer for tree-based models (RandomForest, XGBoost) and
  falls back to LinearExplainer for linear models. KernelExplainer is used
  as a last resort but is slow — a warning is logged in that case.

  Args:
      model_path:  Path to the model pickle file.
      output_path: Path to save the output PNG.
      model_name:  Model name for the plot title.
      top_n:       Number of top features to display.
  """
  try:
    import shap  # type: ignore[import]
  except ImportError:
    logger.warning(
      "shap not installed — skipping SHAP plot for %s. Run: pip install shap",
      model_name,
    )
    return

  if not model_path.exists():
    logger.warning("Model pickle not found at %s — skipping SHAP plot", model_path)
    return

  try:
    grid = joblib.load(model_path)
    model = grid.best_estimator_
  except Exception as e:
    logger.warning(
      "Could not load model from %s: %s — skipping SHAP plot",
      model_path,
      e,
    )
    return

  # Load test data
  test_ds = Dataset(split="test", subset_size=_SHAP_BACKGROUND_SAMPLES)
  x_test = test_ds.load_x()
  feat_names = _feature_names(x_test.shape[1])

  logger.info(
    "Computing SHAP values for %s on %d samples...",
    model_name,
    len(x_test),
  )

  try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
  except Exception:
    try:
      explainer = shap.LinearExplainer(model, x_test)
      shap_values = explainer.shap_values(x_test)
    except Exception:
      logger.warning(
        "TreeExplainer and LinearExplainer failed for %s"
        " — using KernelExplainer (slow)",
        model_name,
      )
      background = shap.sample(x_test, min(50, len(x_test)))
      explainer = shap.KernelExplainer(model.predict, background)
      shap_values = explainer.shap_values(x_test)

  # Mean absolute SHAP value per feature
  shap_array = np.array(shap_values)
  # For multiclass: shap_values is (n_classes, n_samples, n_features)
  # For binary/regression: shap_values is (n_samples, n_features)
  if shap_array.ndim == 3:
    mean_abs_shap = np.abs(shap_array).mean(axis=(0, 1))
  else:
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
  indices = np.argsort(mean_abs_shap)[::-1][:top_n]
  top_names = [feat_names[i] for i in indices]
  top_vals = mean_abs_shap[indices]

  fig, ax = plt.subplots(figsize=(10, 6))
  colors = plt.get_cmap("RdYlGn_r")(np.linspace(0.1, 0.9, len(top_names)))  # type: ignore[arg-type]
  ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1])
  ax.set_yticks(range(len(top_names)))
  ax.set_yticklabels(top_names[::-1], fontsize=9)
  ax.set_xlabel("Mean |SHAP value|")
  ax.set_title(f"Top {top_n} Feature Importances (SHAP) — {model_name}")
  ax.grid(axis="x", alpha=0.3)

  plt.tight_layout()
  plt.savefig(output_path, dpi=100, bbox_inches="tight")
  plt.close()
  logger.info("SHAP importance plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_sensitivity_analysis(run_id: str) -> None:
  """Run full sensitivity analysis for all models in a training run.

  For each model found in the run directory:
    - Plots hyperparameter sensitivity from cv_results_{model}.json
    - Plots SHAP feature importance from {model}.pkl

  Args:
      run_id: Training run identifier (subdirectory under models_root).
  """
  if TRAIN_USE_DUMMY_DATA:
    logger.info("TRAIN_USE_DUMMY_DATA=True — skipping sensitivity analysis")
    return

  run_dir = Paths.models_root / run_id
  if not run_dir.exists():
    logger.warning("Run directory %s not found", run_dir)
    return

  model_pkls = list(run_dir.glob("*.pkl"))
  if not model_pkls:
    logger.warning("No model pickles found in %s", run_dir)
    return

  for pkl_path in model_pkls:
    model_name = pkl_path.stem  # e.g. "random_forest"

    logger.info("Running sensitivity analysis for %s", model_name)

    # Hyperparameter sensitivity
    cv_results_path = run_dir / f"cv_results_{model_name}.json"
    hyperparam_plot = run_dir / f"hyperparam_sensitivity_{model_name}.png"
    plot_hyperparam_sensitivity(cv_results_path, hyperparam_plot, model_name)

    # SHAP feature importance
    shap_plot = run_dir / f"shap_importance_{model_name}.png"
    plot_shap_importance(pkl_path, shap_plot, model_name)


def save_cv_results(run_id: str) -> None:
  """Extract and save cv_results_ from all model pickles in a run directory.

  Called from cmd/train.py after training completes. Saves one
  cv_results_{model}.json per pickle found in the run directory.
  Skips models where the pickle is missing or unloadable.

  Args:
      run_id: Training run identifier (subdirectory under models_root).
  """
  import json

  run_dir = Paths.models_root / run_id
  for pkl_path in run_dir.glob("*.pkl"):
    model_name = pkl_path.stem
    cv_path = run_dir / f"cv_results_{model_name}.json"
    if cv_path.exists():
      logger.info("cv_results already saved for %s", model_name)
      continue
    try:
      grid = joblib.load(pkl_path)
      serializable: dict[str, list] = {}
      for key, val in grid.cv_results_.items():
        if hasattr(val, "tolist"):
          serializable[key] = val.tolist()
        else:
          serializable[key] = list(val)
      with open(cv_path, "w") as f:
        json.dump(serializable, f)
      logger.info("cv_results saved to %s", cv_path)
    except Exception as e:
      logger.warning("Could not extract cv_results from %s: %s", pkl_path, e)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Run sensitivity analysis for a training run."
  )
  parser.add_argument(
    "--runid",
    "-r",
    required=True,
    help="Run ID (subdirectory under models/)",
  )
  args = parser.parse_args()
  run_sensitivity_analysis(args.runid)
