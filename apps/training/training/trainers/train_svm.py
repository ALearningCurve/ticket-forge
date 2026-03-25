# %%

# uncomment when running in notebook mode
# import sys

# sys.path.append("..")

from typing import Any

from scipy.stats import loguniform
from sklearn.base import BaseEstimator
from sklearn.svm import SVR
from training.trainers.base import BaseTrainer
from training.trainers.utils.harness import load_fit_debug, load_fit_dump


class SVMTrainer(BaseTrainer):
  """Trainer for Support Vector Regression."""

  def get_model(self) -> BaseEstimator:
    """Create an unfitted SVR with default parameters.

    Returns:
        SVR instance (note: does not support sample_weight in fit()).
    """
    return SVR()

  def get_search_params(self) -> list[dict[str, Any]]:
    """Return hyperparameter search space for SVM.

    Returns:
        List containing a single param distribution dict.
    """
    return [
      {
        "C": loguniform(1e-3, 1e3),
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "kernel": ["rbf", "poly", "sigmoid"],
        "epsilon": [0.01, 0.1, 0.2, 0.5],
      },
    ]

  def get_search_config(self) -> dict[str, Any]:
    """Override search config for SVM.

    Returns:
        Config dict with n_iter=50, n_jobs=-1, parallel CV evaluation.
    """
    config = super().get_search_config()
    config["n_iter"] = 50
    config["n_jobs"] = -1
    return config


# %%
def main(run_id: str, debug: bool = False) -> None:
  """Trains svm models on all the feature datasets.

  Args:
    run_id: Training run identifier.
    debug: If True, skip hyperparameter tuning.
  """
  trainer = SVMTrainer()
  if debug:
    load_fit_debug(trainer.fit_simple, run_id, "svm")
  else:
    load_fit_dump(trainer.fit_grid, run_id, "svm")


if __name__ == "__main__":
  main("TESTING")
