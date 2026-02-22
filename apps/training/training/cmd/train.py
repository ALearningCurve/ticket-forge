import argparse
import datetime
import importlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from shared.configuration import Paths
from shared.logging import get_logger

logger = get_logger(__name__)
models = {"forest", "linear", "svm", "xgboost"}
models_with_sample_weight = models.difference(set(["svm"]))


def _parse_arguments() -> tuple[set[str], str]:
  """Parse command line arguments.

  Returns:
      Tuple of (models_to_train, run_id)
  """
  parser = argparse.ArgumentParser(
    description=f"utility to train the models. scripts executed with {Paths.repo_root=}"
  )

  parser.add_argument(
    "--models",
    "-m",
    nargs="*",
    type=str,
    help="the models to train, defaults to those which support sample weights",
    choices=models,
    default=models_with_sample_weight,
  )
  parser.add_argument(
    "--runid",
    "-r",
    type=str,
    help="run identifier, defaults to current timestamp",
    default=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
  )

  args = parser.parse_args()
  return args.models, args.runid


def _train_models(models_list: set[str], run_id: str) -> None:
  """Train all specified models.

  Args:
      models_list: Set of model names to train
      run_id: Run identifier for saving outputs
  """
  for model in models_list:
    start = time.perf_counter()
    success = False
    logger.info(f"---------- TRAINING {model} ----------")

    try:
      mod = importlib.import_module(f"training.trainers.train_{model}", package="src")
      mod.main(run_id)
      success = True

    except Exception:
      logger.exception(f"Error training model {model}")

    train_time = datetime.timedelta(seconds=time.perf_counter() - start)
    msg = "SUCCEEDED" if success else "FAILED"
    logger.info(f"---------- TRAINING {model} {msg} in ({str(train_time)}) ----------")


def _load_metrics(run_dir: Path) -> tuple[dict[str, dict[str, float]], list]:
  """Load metrics from all evaluation files in the run directory.

  Args:
      run_dir: Directory containing eval_*.json files

  Returns:
      Tuple of (metrics_data, best_models) where best_models is sorted by R2 score
  """
  metrics_data = {}
  best_models = []

  for eval_file in run_dir.glob("eval_*.json"):
    model_name = eval_file.stem.replace("eval_", "")
    with open(eval_file) as f:
      metrics_data[model_name] = json.load(f)

      # Track best model by R2 score (highest is best)
      if "r2" in metrics_data[model_name]:
        best_models.append(
          (model_name, metrics_data[model_name]["r2"], metrics_data[model_name])
        )

  return metrics_data, best_models


def _save_best_model_info(best_models: list, run_dir: Path) -> None:
  """Save information about the best model to best.txt.

  Args:
      best_models: List of (model_name, r2_score, metrics) tuples
      run_dir: Directory to save best.txt to
  """
  if not best_models:
    return

  best_models.sort(key=lambda x: x[1], reverse=True)
  best_model_name = best_models[0][0]
  best_model_score = best_models[0][1]

  best_file = run_dir / "best.txt"
  with open(best_file, "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"R2 Score: {best_model_score:.4f}\n")
    f.write("\nAll Metrics:\n")
    for key, value in best_models[0][2].items():
      f.write(f"{key}: {value:.4f}\n")

  logger.info(f"{'=' * 50}")
  logger.info(f"Best model: {best_model_name} (R2: {best_model_score:.4f})")
  logger.info(f"Results saved to {best_file}")
  logger.info(f"{'=' * 50}")


def main() -> None:
  """Trains models according to user params."""
  # Parse arguments
  models_list, run_id = _parse_arguments()

  # Create output directory for this run
  run_dir = Paths.models_root / run_id
  run_dir.mkdir(parents=True, exist_ok=True)

  # Train models
  _train_models(models_list, run_id)

  # Load metrics and identify best model
  metrics_data, best_models = _load_metrics(run_dir)

  # Plot metrics if we have data
  if metrics_data:
    _plot_metrics(metrics_data, run_dir)

  # Save best model info
  _save_best_model_info(best_models, run_dir)


def _plot_metrics(metrics_data: dict[str, dict[str, float]], run_dir: Path) -> None:
  """Plot model metrics and save as performance.png.

  Args:
      metrics_data: Dictionary of model names to their metrics
      run_dir: Directory to save the plot to
  """
  model_names = list(metrics_data.keys())
  metric_keys = ["mae", "mse", "rmse", "r2"]

  # Create subplots for different metrics
  fig, axes = plt.subplots(2, 2, figsize=(12, 10))
  fig.suptitle("Model Performance Comparison", fontsize=16)

  for idx, metric in enumerate(metric_keys):
    ax = axes[idx // 2, idx % 2]
    values = [metrics_data[model].get(metric, 0) for model in model_names]

    # Use different colors for each metric
    colors = plt.get_cmap("viridis")(  # type: ignore
      [i / len(model_names) for i in range(len(model_names))]
    )
    ax.bar(model_names, values, color=colors)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by Model")
    ax.grid(axis="y", alpha=0.3)

    # Rotate x labels for readability
    ax.tick_params(axis="x", rotation=45)

  plt.tight_layout()

  # Save plot
  plot_file = run_dir / "performance.png"
  plt.savefig(plot_file, dpi=100, bbox_inches="tight")
  logger.info(f"Performance plot saved to {plot_file}")
  plt.close()


if __name__ == "__main__":
  main()
