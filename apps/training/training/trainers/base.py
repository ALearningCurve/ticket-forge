"""Base trainer abstraction for model-agnostic training patterns.

Each concrete trainer (train_linear, train_xgboost, etc.) should implement:
1. get_model() — Creates and returns the base estimator with defaults
2. get_search_params() — Returns the dict of hyperparameters and their search space
3. get_search_config() — Returns n_iter, n_jobs, and other RandomizedSearchCV config

This abstraction reduces duplication and makes it easy to add new models.
"""

from abc import ABC, abstractmethod
from typing import Any

from shared.configuration import RANDOM_SEED
from sklearn.base import BaseEstimator
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from training.trainers.utils.harness import X_t, Y_t


class BaseTrainer(ABC):
  """Abstract base class for model trainers.

  Subclasses must implement get_model() and get_search_params().
  """

  @abstractmethod
  def get_model(self) -> BaseEstimator:
    """Return a new unfitted model instance with default parameters.

    Returns:
        An sklearn estimator (must have fit() and predict() methods).
    """

  @abstractmethod
  def get_search_params(self) -> list[dict[str, Any]]:
    """Return hyperparameter search space for RandomizedSearchCV.

    Returns:
        List of param distribution dicts for RandomizedSearchCV.
        Typically a single dict: [{"param1": [...], "param2": [...], ...}]
    """

  def get_search_config(self) -> dict[str, Any]:
    """Return configuration for RandomizedSearchCV.

    Override this method in subclasses if using non-default search config.

    Returns:
        Dict with keys: n_iter, n_jobs, random_state, error_score, verbose
    """
    return {
      "n_iter": 20,
      "n_jobs": -1,
      "random_state": RANDOM_SEED,
      "error_score": "raise",
      "verbose": 3,
    }

  def fit_grid(
    self,
    x: X_t,
    y: Y_t,
    cv_split: PredefinedSplit,
    sample_weight: Y_t | None = None,
  ) -> RandomizedSearchCV:
    """Perform hyperparameter search using the configured model and params.

    Args:
        x: Feature matrix.
        y: Target values.
        cv_split: Cross-validation split.
        sample_weight: Optional per-sample weights.

    Returns:
        Fitted RandomizedSearchCV object.
    """
    model = self.get_model()
    param_grid = self.get_search_params()
    search_config = self.get_search_config()

    grid = RandomizedSearchCV(
      estimator=model,
      param_distributions=param_grid,
      cv=cv_split,
      scoring="neg_mean_squared_error",
      refit=True,
      **search_config,
    )

    return grid.fit(x, y, sample_weight=sample_weight)

  def fit_simple(
    self,
    x: X_t,
    y: Y_t,
    sample_weight: Y_t | None = None,
  ) -> BaseEstimator:
    """Train model with default parameters (debug mode).

    Args:
        x: Feature matrix.
        y: Target values.
        sample_weight: Optional per-sample weights.

    Returns:
        Fitted model.
    """
    model = self.get_model()
    return model.fit(x, y, sample_weight=sample_weight)
