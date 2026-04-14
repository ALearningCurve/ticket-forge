"""Tests for the training command helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV


def _make_grid_pickle(directory: Path, model_name: str = "forest") -> Path:
  """Train a tiny grid and pickle it for command tests."""
  x = np.random.default_rng(0).random((20, 4))
  y = np.random.default_rng(0).random(20)
  grid = RandomizedSearchCV(
    LinearRegression(),
    param_distributions={"fit_intercept": [True, False]},
    n_iter=2,
    cv=2,
    scoring="neg_mean_squared_error",
    random_state=0,
  )
  grid.fit(x, y)
  path = directory / f"{model_name}.pkl"
  joblib.dump(grid, path)
  return path


class TestLogBestEstimator:
  """Tests for logging the best estimator during training."""

  def test_logs_best_estimator_to_mlflow(self, tmp_path: Path) -> None:
    """The helper logs the fitted best estimator as an MLflow model."""
    from training.cmd.train import _log_best_estimator

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_grid_pickle(run_dir, "forest")

    with (
      patch("training.cmd.train.mlflow.sklearn.log_model") as mock_log_model,
      patch("training.cmd.train.mlflow.log_params") as mock_log_params,
    ):
      _log_best_estimator(run_dir, "forest")

    mock_log_model.assert_called_once()
    assert mock_log_model.call_args.kwargs["name"] == "best_estimator"
    assert mock_log_model.call_args.kwargs["registered_model_name"] is None
    mock_log_params.assert_called_once()

  def test_raises_when_pickle_missing(self, tmp_path: Path) -> None:
    """The helper fails fast if the fitted pickle is absent."""
    from training.cmd.train import _log_best_estimator

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Model pickle not found"):
      _log_best_estimator(run_dir, "forest")

  class TestModelArtifactStems:
    """Tests for model artifact stem resolution."""

    def test_includes_random_forest_alias_for_forest(self) -> None:
      """Forest model uses random_forest artifact filenames from trainer output."""
      from training.cmd.train import _model_artifact_stems

      assert _model_artifact_stems("forest") == ("random_forest", "forest")

    def test_defaults_to_model_name_without_alias(self) -> None:
      """Models without aliases resolve to their own artifact stem."""
      from training.cmd.train import _model_artifact_stems

      assert _model_artifact_stems("svm") == ("svm",)
