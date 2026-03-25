# %%
# uncomment when running in notebook mode
# import sys
# sys.path.append("..")

from typing import Any

from scipy.stats import uniform
from shared.configuration import RANDOM_SEED
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from training.trainers.base import BaseTrainer
from training.trainers.utils.harness import load_fit_debug, load_fit_dump


# %%
class ForestTrainer(BaseTrainer):
  """Trainer for RandomForestRegressor."""

  def get_model(self) -> BaseEstimator:
    """Create an unfitted RandomForestRegressor with default parameters.

    Returns:
        RandomForestRegressor instance.
    """
    return RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)

  def get_search_params(self) -> list[dict[str, Any]]:
    """Return hyperparameter search space for random forest.

    Returns:
        List containing a single param distribution dict.
    """
    return [
      {
        "max_depth": range(5, 30),
        "max_samples": uniform(0.2, 0.6),
        "min_samples_split": range(2, 10),
        "n_estimators": range(10, 100),
      }
    ]

  def get_search_config(self) -> dict[str, Any]:
    """Override search config for forest models.

    Returns:
        Config dict with n_iter=20, n_jobs=-1.
    """
    config = super().get_search_config()
    config["n_iter"] = 20
    config["n_jobs"] = -1
    return config


def main(run_id: str, debug: bool = False) -> None:
  """Trains forest models on all the feature datasets.

  Args:
    run_id: Training run identifier.
    debug: If True, skip hyperparameter tuning.
  """
  trainer = ForestTrainer()
  if debug:
    load_fit_debug(trainer.fit_simple, run_id, "forest")
  else:
    load_fit_dump(trainer.fit_grid, run_id, "forest")


if __name__ == "__main__":
  main("TESTING")
