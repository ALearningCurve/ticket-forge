from typing import Callable

import numpy as np
import polars as pl
from ml_core import Dataset, X_t, Y_t
from shared.cache import JsonSaver, fs_cache
from shared.configuration import Paths
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV


def load_fit_dump(
  fit_grid: Callable[
    [X_t, Y_t, PredefinedSplit],
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
    # 1. Get the combined data and the PredefinedSplit
    x_comp, y_comp, cv_split = Dataset.as_sklearn_cv_split()

    # 2. Pass them into the fit_grid logic
    return fit_grid(x_comp, y_comp, cv_split)

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

  print("test metrics")
  print(metrics)


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
  print("Hyper-parameter search results:")
  with pl.Config(tbl_hide_dataframe_shape=True):
    print(df)
    total_time = df["mean_fit_time"].sum() + df["mean_score_time"].sum()
    print(f"total training time = {total_time}")
    get_test_accuracy(grid, run_id, model_name)

    print()
