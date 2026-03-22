"""Tests for model sensitivity analysis (hyperparam + SHAP)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cv_results(tmp_path: Path, model_name: str) -> Path:
  """Write a minimal cv_results JSON file and return its path."""
  cv = {
    "param_max_depth": [3, 5, 7, 3, 5],
    "param_n_estimators": [10, 50, 100, 10, 50],
    "mean_test_score": [-10.0, -8.0, -6.0, -9.0, -7.0],
    "std_test_score": [0.5, 0.4, 0.3, 0.6, 0.5],
    "rank_test_score": [5, 3, 1, 4, 2],
  }
  path = tmp_path / f"cv_results_{model_name}.json"
  path.write_text(json.dumps(cv))
  return path


def _make_model_pickle(tmp_path: Path, model_name: str) -> Path:
  """Write a real sklearn GridSearch pickle and return its path."""
  x = np.random.default_rng(42).random((20, 5))
  y = np.random.default_rng(42).random(20)
  grid = RandomizedSearchCV(
    LinearRegression(),
    param_distributions={"fit_intercept": [True, False]},
    n_iter=2,
    cv=2,
    scoring="neg_mean_squared_error",
    random_state=42,
  )
  grid.fit(x, y)
  path = tmp_path / f"{model_name}.pkl"
  joblib.dump(grid, path)
  return path


# ---------------------------------------------------------------------------
# _feature_names
# ---------------------------------------------------------------------------


class TestFeatureNames:
  def test_returns_correct_count(self) -> None:
    from training.analysis.run_sensitivity_analysis import _feature_names

    names = _feature_names(384)
    assert len(names) == 384

  def test_names_have_correct_prefix(self) -> None:
    from training.analysis.run_sensitivity_analysis import _feature_names

    names = _feature_names(10)
    assert all(n.startswith("emb_") for n in names)

  def test_names_are_zero_padded(self) -> None:
    from training.analysis.run_sensitivity_analysis import _feature_names

    # 100 features → width=2, so emb_00 .. emb_99
    names = _feature_names(100)
    assert names[0] == "emb_00"
    assert names[99] == "emb_99"

    # 1000 features → width=3, so emb_000 .. emb_999
    names_1000 = _feature_names(1000)
    assert names_1000[0] == "emb_000"
    assert names_1000[384] == "emb_384"


# ---------------------------------------------------------------------------
# plot_hyperparam_sensitivity
# ---------------------------------------------------------------------------


class TestPlotHyperparamSensitivity:
  def test_skips_when_cv_results_missing(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import plot_hyperparam_sensitivity

    missing = tmp_path / "cv_results_nonexistent.json"
    output = tmp_path / "out.png"
    # Should not raise
    plot_hyperparam_sensitivity(missing, output, "test_model")
    assert not output.exists()

  def test_generates_plot_file(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import plot_hyperparam_sensitivity

    cv_path = _make_cv_results(tmp_path, "random_forest")
    output = tmp_path / "hyperparam.png"
    plot_hyperparam_sensitivity(cv_path, output, "random_forest")
    assert output.exists()

  def test_skips_when_no_param_columns(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import plot_hyperparam_sensitivity

    # cv_results with no param_ columns
    cv = {"mean_test_score": [-10.0, -8.0], "rank_test_score": [2, 1]}
    cv_path = tmp_path / "cv_results_linear.json"
    cv_path.write_text(json.dumps(cv))
    output = tmp_path / "out.png"
    plot_hyperparam_sensitivity(cv_path, output, "linear")
    assert not output.exists()


# ---------------------------------------------------------------------------
# plot_shap_importance
# ---------------------------------------------------------------------------


class TestPlotShapImportance:
  def test_skips_when_pickle_missing(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import plot_shap_importance

    missing = tmp_path / "nonexistent.pkl"
    output = tmp_path / "shap.png"
    plot_shap_importance(missing, output, "test_model")
    assert not output.exists()

  def test_skips_when_shap_not_installed(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import plot_shap_importance

    pkl_path = _make_model_pickle(tmp_path, "random_forest")
    output = tmp_path / "shap.png"

    with patch.dict("sys.modules", {"shap": None}):
      plot_shap_importance(pkl_path, output, "random_forest")
    assert not output.exists()

  def test_skips_when_pickle_is_corrupted(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import plot_shap_importance

    bad_pkl = tmp_path / "bad.pkl"
    bad_pkl.write_bytes(b"not a valid pickle")
    output = tmp_path / "shap.png"
    plot_shap_importance(bad_pkl, output, "bad_model")
    assert not output.exists()


# ---------------------------------------------------------------------------
# save_cv_results
# ---------------------------------------------------------------------------


class TestSaveCvResults:
  def test_saves_cv_results_json(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import save_cv_results

    run_id = "test_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    _make_model_pickle(run_dir, "random_forest")

    with patch("training.analysis.run_sensitivity_analysis.Paths") as mp:
      mp.models_root = tmp_path
      save_cv_results(run_id)

    cv_path = run_dir / "cv_results_random_forest.json"
    assert cv_path.exists()
    data = json.loads(cv_path.read_text())
    assert "param_fit_intercept" in data
    assert "mean_test_score" in data

  def test_skips_if_already_exists(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import save_cv_results

    run_id = "test_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    _make_model_pickle(run_dir, "random_forest")

    # Pre-create the file
    existing = run_dir / "cv_results_random_forest.json"
    existing.write_text('{"existing": true}')

    with patch("training.analysis.run_sensitivity_analysis.Paths") as mp:
      mp.models_root = tmp_path
      save_cv_results(run_id)

    # Should not overwrite
    data = json.loads(existing.read_text())
    assert data == {"existing": True}

  def test_skips_corrupted_pickle_gracefully(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import save_cv_results

    run_id = "test_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    bad_pkl = run_dir / "bad_model.pkl"
    bad_pkl.write_bytes(b"not a pickle")

    with patch("training.analysis.run_sensitivity_analysis.Paths") as mp:
      mp.models_root = tmp_path
      # Should not raise
      save_cv_results(run_id)

    assert not (run_dir / "cv_results_bad_model.json").exists()


# ---------------------------------------------------------------------------
# run_sensitivity_analysis
# ---------------------------------------------------------------------------


class TestRunSensitivityAnalysis:
  def test_skips_when_dummy_data_true(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import run_sensitivity_analysis

    with patch("training.analysis.run_sensitivity_analysis.TRAIN_USE_DUMMY_DATA", True):
      # Should return early without error
      run_sensitivity_analysis("any_run_id")

  def test_skips_when_run_dir_missing(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import run_sensitivity_analysis

    _dummy = patch(
      "training.analysis.run_sensitivity_analysis.TRAIN_USE_DUMMY_DATA",
      False,
    )
    with patch("training.analysis.run_sensitivity_analysis.Paths") as mp, _dummy:
      mp.models_root = tmp_path
      # Should not raise even if run dir doesn't exist
      run_sensitivity_analysis("nonexistent_run")

  def test_skips_when_no_pickles(self, tmp_path: Path) -> None:
    from training.analysis.run_sensitivity_analysis import run_sensitivity_analysis

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    _dummy = patch(
      "training.analysis.run_sensitivity_analysis.TRAIN_USE_DUMMY_DATA",
      False,
    )
    with patch("training.analysis.run_sensitivity_analysis.Paths") as mp, _dummy:
      mp.models_root = tmp_path
      run_sensitivity_analysis("empty_run")
