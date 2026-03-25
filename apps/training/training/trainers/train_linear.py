# %%

# uncomment when running in notebook mode
# import sys
# sys.path.append("..")

from typing import Any

from scipy.stats import loguniform
from shared.configuration import RANDOM_SEED
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDRegressor
from training.trainers.base import BaseTrainer
from training.trainers.utils.harness import load_fit_debug, load_fit_dump


# %%
class LinearTrainer(BaseTrainer):
  """Trainer for SGDRegressor (linear models)."""

  def get_model(self) -> BaseEstimator:
    """Create an unfitted SGDRegressor with default parameters.

    Returns:
        SGDRegressor instance.
    """
    return SGDRegressor(random_state=RANDOM_SEED, max_iter=4000)

  def get_search_params(self) -> list[dict[str, Any]]:
    """Return hyperparameter search space for linear models.

    Returns:
        List containing a single param distribution dict.
    """
    return [
      {
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": loguniform(1e-5, 1e5),
      }
    ]

  def get_search_config(self) -> dict[str, Any]:
    """Override search config for linear models.

    Returns:
        Config dict with n_iter=20, n_jobs=-1 (parallel loss evaluation).
    """
    config = super().get_search_config()
    config["n_iter"] = 20
    config["n_jobs"] = -1
    return config


def main(run_id: str, debug: bool = False) -> None:
  """Trains linear models on all the feature datasets.

  Args:
    run_id: Training run identifier.
    debug: If True, skip hyperparameter tuning.
  """
  trainer = LinearTrainer()
  if debug:
    load_fit_debug(trainer.fit_simple, run_id, "linear")
  else:
    load_fit_dump(trainer.fit_grid, run_id, "linear")


if __name__ == "__main__":
  main("TESTING")
# %%
