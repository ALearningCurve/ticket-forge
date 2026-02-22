"""Dataset utilities for machine learning pipelines."""

import json
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel
from shared.configuration import Paths, Splits_t
from sklearn.datasets import make_regression
from sklearn.model_selection import PredefinedSplit

X_t = npt.NDArray[Any]
Y_t = npt.NDArray[np.floating]


class Dataset(BaseModel):
  """Represents training dataset for ticket time prediction.

  Currently a stub since there is no data provided yet
  """

  split: Splits_t
  subset_size: int | None = None

  def load_x(self) -> X_t:
    """Loads the x dataset.

    Returns:
        the x features
    """
    # Generate a synthetic regression dataset
    dataset = make_regression(  # pyright: ignore[reportConstantRedefinition]
      n_samples=100,
      n_features=1,
      noise=20,
      random_state=42,
    )  # type: ignore
    x = dataset[0]  # type: ignore
    if self.subset_size is not None:
      return x[: self.subset_size]  # type: ignore
    return x  # type: ignore

  def load_y(self) -> Y_t:
    """Loads the y vector.

    Returns:
        y features
    """
    # Generate a synthetic regression dataset
    dataset = make_regression(  # pyright: ignore[reportConstantRedefinition]
      n_samples=100,
      n_features=1,
      noise=20,
      random_state=42,
    )  # type: ignore
    y = dataset[1]
    if self.subset_size is not None:
      return y[: self.subset_size]  # type: ignore
    return y

  def load_metadata(self) -> pd.DataFrame:
    """Load metadata containing sensitive features for bias analysis.

    Returns:
        DataFrame with columns like 'repo', 'seniority', 'labels'
        that can be used as sensitive features for bias detection.
    """
    # TODO: Load from actual data source when available, e.g.:
    # return pd.read_parquet(Paths.data_root / self.split / "metadata.parquet")

    # For now, generate synthetic metadata matching the dataset size
    n_samples = 100
    if self.subset_size is not None:
      n_samples = self.subset_size

    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
      {
        "repo": rng.choice(["terraform", "ansible", "prometheus"], size=n_samples),
        "seniority": rng.choice(["junior", "mid", "senior"], size=n_samples),
        "labels": rng.choice(
          ["bug", "enhancement", "feature", "bug,critical"], size=n_samples
        ),
        "completion_hours_business": rng.uniform(1, 100, size=n_samples),
      }
    )

  def load_sample_weights(self) -> Y_t:
    """Load per-sample weights for bias-aware training.

    Attempts to load weights from sample_weights.json (produced by
    run_bias_mitigation.py --mode weight). Falls back to synthetic
    inverse-frequency weights derived from the metadata 'repo' column.

    Returns:
        Array of per-sample weights, same length as load_y().
    """
    group_col = "repo"
    meta = self.load_metadata()
    weights_path = Paths.data_root / "github_issues" / "sample_weights.json"
    try:
      with open(weights_path) as f:
        data: dict[str, Any] = json.load(f)
      saved_col: str = data.get("group_col", group_col)
      weights_by_group: dict[str, float] = data.get("weights_by_group", {})
      if saved_col in meta.columns and weights_by_group:
        w = meta[saved_col].map(weights_by_group).fillna(1.0)  # type: ignore[arg-type]
        return w.to_numpy(dtype=np.float64)  # type: ignore[return-value]
    except (FileNotFoundError, json.JSONDecodeError):
      pass

    # Synthetic fallback: inverse-frequency weights from metadata repo column
    group_counts = meta[group_col].value_counts()
    total = len(meta)
    n_groups = len(group_counts)
    w = meta[group_col].map(lambda g: total / (n_groups * group_counts[g]))
    return w.to_numpy(dtype=np.float64)  # type: ignore[return-value]

  @staticmethod
  def as_sklearn_cv_split(
    subset_size: int | None = None,
  ) -> tuple[X_t, Y_t, PredefinedSplit]:
    """Creates predefined sklearn cross validation split with
    fixed training and validation partitions.

    Returns:
        The predefined split along with the combined x and y data.
        This is returned as a tuple [x,y,cv_split]
    """
    train = Dataset(split="train", subset_size=subset_size)
    validation = Dataset(split="validation", subset_size=subset_size)

    x_train = train.load_x()
    y_train = train.load_y()
    x_val = validation.load_x()
    y_val = validation.load_y()

    # Concatenate features and targets
    # Using np.vstack for 2D features (samples, features)
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.hstack([y_train, y_val])

    # Create the test_fold indicator array:
    # -1 for all training samples
    #  0 for all validation samples
    test_fold = np.concatenate(
      [np.full(x_train.shape[0], -1), np.full(x_val.shape[0], 0)]
    )

    cv_split = PredefinedSplit(test_fold)

    return x_combined, y_combined, cv_split

  @staticmethod
  def as_sklearn_cv_split_with_weights(
    subset_size: int | None = None,
  ) -> tuple[X_t, Y_t, PredefinedSplit, Y_t]:
    """Like as_sklearn_cv_split but also returns per-sample weights.

    Returns:
        Tuple of (x_combined, y_combined, cv_split, weights_combined)
    """
    train = Dataset(split="train", subset_size=subset_size)
    validation = Dataset(split="validation", subset_size=subset_size)

    x_train, y_train = train.load_x(), train.load_y()
    w_train = train.load_sample_weights()
    x_val, y_val = validation.load_x(), validation.load_y()
    w_val = validation.load_sample_weights()

    x_combined = np.vstack([x_train, x_val])
    y_combined = np.hstack([y_train, y_val])
    w_combined = np.hstack([w_train, w_val])

    test_fold = np.concatenate(
      [np.full(x_train.shape[0], -1), np.full(x_val.shape[0], 0)]
    )
    cv_split = PredefinedSplit(test_fold)

    return x_combined, y_combined, cv_split, w_combined
