"""LightGBM trainer for ticket complexity classification."""

import lightgbm as lgb
from shared.configuration import RANDOM_SEED
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from training.trainers.utils.harness import X_t, Y_t, load_fit_dump


def fit_grid(
  x: X_t,
  y: Y_t,
  cv_split: PredefinedSplit,
  sample_weight: Y_t | None = None,
) -> RandomizedSearchCV:
  """Performs grid search and returns the result.

  Args:
    x: x data to use for training
    y: true labels of dataset
    cv_split: the predefined split to use
    sample_weight: per-sample weights for bias-aware training, or None

  Returns:
      result of the grid search.
  """
  param_grid = [
    {
      "n_estimators": [100, 200, 300, 500],
      "max_depth": [4, 6, 8, 10, -1],  # -1 = unlimited
      "num_leaves": [31, 63, 127, 255],  # key LightGBM param
      "learning_rate": [0.01, 0.05, 0.1, 0.3],
      "subsample": [0.6, 0.8, 1.0],
      "colsample_bytree": [0.6, 0.8, 1.0],
      "min_child_samples": [10, 20, 50],  # min data per leaf
      "reg_alpha": [0, 0.1, 1.0],  # L1 regularization
      "reg_lambda": [0, 0.1, 1.0],  # L2 regularization
    }
  ]

  model = lgb.LGBMClassifier(
    random_state=RANDOM_SEED,
    class_weight="balanced",
    objective="multiclass",
    n_jobs=-1,
    verbose=-1,  # suppress LightGBM output
  )

  grid = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    cv=cv_split,
    scoring="f1_macro",
    refit=True,
    n_iter=30,
    random_state=RANDOM_SEED,
    error_score="raise",
    verbose=2,
  )

  return grid.fit(x, y, sample_weight=sample_weight)


def main(run_id: str) -> None:
  """Trains LightGBM model."""
  load_fit_dump(fit_grid, run_id, "lgbm")


if __name__ == "__main__":
  main("TESTING")
