from typing import Any

import xgboost as xgb
from shared.configuration import RANDOM_SEED
from sklearn.base import BaseEstimator
from training.trainers.base import BaseTrainer
from training.trainers.utils.harness import load_fit_debug, load_fit_dump

# %%

XGB_VERBOSE = 1  # 3=debug, 2=info, 1=warning
xgb.set_config(verbosity=XGB_VERBOSE)


class XGBoostTrainer(BaseTrainer):
  """Trainer for XGBoost regression."""

  def get_model(self) -> BaseEstimator:
    """Create an unfitted XGBRegressor with default parameters.

    Returns:
        XGBRegressor instance.
    """
    return xgb.XGBRegressor(
      random_state=RANDOM_SEED,
      device="cpu",
      tree_method="hist",
      max_bin=63,
      n_jobs=-1,
    )

  def get_search_params(self) -> list[dict[str, Any]]:
    """Return hyperparameter search space for XGBoost.

    References:
    - https://github.com/szilard/benchm-ml?tab=readme-ov-file#boosting-gradient-boosted-treesgradient-boosting-machines
    - https://xgboost.readthedocs.io/en/stable/parameter.html

    Defaults: max_depth=6, learning_rate=.3, min_child_weight=1, n_estimators=100,
    gamma=0, subsample=1, colsample_bytree=1

    Returns:
        List containing a single param distribution dict.
    """
    return [
      {
        "max_depth": [1, 3, 4, 5, 6, 7],
        "learning_rate": [
          0.01,
          0.03,
          0.1,
          0.3,
          0.5,
        ],
        "min_child_weight": [1, 5, 10],
        "n_estimators": [10, 30, 50],
        "gamma": [0, 0.1, 1],
        "subsample": [0.1, 0.2, 0.3],
        "colsample_bytree": [0.2, 0.3, 0.4],
      }
    ]

  def get_search_config(self) -> dict[str, Any]:
    """Override search config for XGBoost.

    Uses n_jobs=1 since XGBoost handles parallelism internally via n_jobs in
    the model creation. Parallel cv_split may cause excessive resource usage.

    Returns:
        Config dict with n_iter=20, n_jobs=1.
    """
    config = super().get_search_config()
    config["n_iter"] = 20
    config["n_jobs"] = 1  # XGBoost handles parallelism internally
    return config


def main(run_id: str, debug: bool = False) -> None:
  """Trains xgboost models on all the feature datasets.

  Args:
    run_id: Training run identifier.
    debug: If True, skip hyperparameter tuning.
  """
  trainer = XGBoostTrainer()
  if debug:
    load_fit_debug(trainer.fit_simple, run_id, "xgboost")
  else:
    load_fit_dump(trainer.fit_grid, run_id, "xgboost")


if __name__ == "__main__":
  main("TESTING")

# %%
