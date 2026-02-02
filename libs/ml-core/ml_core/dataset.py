"""Dataset utilities for machine learning pipelines."""

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from shared.configuration import Splits_t
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
