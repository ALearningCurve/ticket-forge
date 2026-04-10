"""Tests for bias evaluation and confidence metrics in the training harness."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from training.bias import BiasReport
from training.trainers.utils.harness import evaluate_bias


def _make_grid(y_pred: list[float]) -> MagicMock:
  """Create a mock RandomizedSearchCV that returns fixed predictions."""
  grid = MagicMock()
  grid.predict.return_value = np.array(y_pred)
  return grid


def _make_dataset(
  y: list[float],
  sensitive_feature: str,
  feature_values: list[str],
) -> MagicMock:
  """Create a mock Dataset returning fixed y and metadata."""
  ds = MagicMock()
  ds.load_x.return_value = np.zeros((len(y), 384))
  ds.load_y.return_value = np.array(y)
  ds.load_metadata.return_value = pd.DataFrame(
    {
      sensitive_feature: feature_values,
      "repo": feature_values,
      "seniority": ["mid"] * len(y),
      "labels": ["bug"] * len(y),
      "completion_hours_business": y,
    }
  )
  return ds


class TestEvaluateBiasNoBias:
  """Tests for evaluate_bias when no bias is detected."""

  def test_returns_analysis_when_no_bias(self, tmp_path) -> None:
    """Test that analysis is returned when no bias is detected."""
    y = [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    y_pred = [11.0, 21.0, 31.0, 11.0, 21.0, 31.0]
    groups = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]

    grid = _make_grid(y_pred)
    mock_ds = _make_dataset(y, "repo", groups)

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
      patch("training.trainers.utils.harness.BiasReport.save_report"),
    ):
      mock_paths.models_root = tmp_path
      result = evaluate_bias(grid, "run-001", "linear", sensitive_feature="repo")

    assert result is not None
    assert result["bias_detected"] is False
    assert result["primary_metric"] == "ndcg"

  def test_report_structure_when_no_bias(self, tmp_path) -> None:
    """Test that report has correct flat structure when no bias detected."""
    y = [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    y_pred = [11.0, 21.0, 31.0, 11.0, 21.0, 31.0]
    groups = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]

    grid = _make_grid(y_pred)
    mock_ds = _make_dataset(y, "repo", groups)
    saved_reports = []

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
      patch(
        "training.trainers.utils.harness.BiasReport.save_report",
        side_effect=lambda data, path: saved_reports.append(data),
      ),
    ):
      mock_paths.models_root = tmp_path
      evaluate_bias(grid, "run-001", "linear", sensitive_feature="repo")

    report = saved_reports[0]
    assert report["summary"]["overall_bias_detected"] is False
    assert report["summary"]["bias_count"] == 0
    assert "bias_detected" in report["detailed_results"]["repo"]

  def test_report_renders_as_text(self, tmp_path) -> None:
    """Test that BiasReport.generate_text_report() works with report structure."""
    y = [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    y_pred = [11.0, 21.0, 31.0, 11.0, 21.0, 31.0]
    groups = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]

    grid = _make_grid(y_pred)
    mock_ds = _make_dataset(y, "repo", groups)
    saved_reports: list[dict] = []

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
      patch(
        "training.trainers.utils.harness.BiasReport.save_report",
        side_effect=lambda data, path: saved_reports.append(data),
      ),
    ):
      mock_paths.models_root = tmp_path
      evaluate_bias(grid, "run-001", "linear", sensitive_feature="repo")

    report = saved_reports[0]
    text_report = BiasReport.generate_text_report(report)
    assert isinstance(text_report, str)
    assert len(text_report) > 0


class TestEvaluateBiasDetected:
  """Tests for evaluate_bias when bias is detected."""

  def test_returns_analysis_when_bias_detected(self, tmp_path) -> None:
    """Test that analysis is returned when bias is detected."""
    y = [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    y_pred = [11.0, 21.0, 31.0, 41.0, 51.0, 50.0, 10.0, 1.0, 5.0, 15.0]
    groups = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]

    grid = _make_grid(y_pred)
    mock_ds = _make_dataset(y, "repo", groups)

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
      patch("training.trainers.utils.harness.BiasReport.save_report"),
    ):
      mock_paths.models_root = tmp_path
      result = evaluate_bias(grid, "run-001", "linear", sensitive_feature="repo")

    assert result is not None
    assert "bias_detected" in result

  def test_report_summary_flags_bias_correctly(self, tmp_path) -> None:
    """Test report summary correctly identifies biased dimension."""
    y = [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    y_pred = [11.0, 21.0, 31.0, 41.0, 51.0, 50.0, 10.0, 1.0, 5.0, 15.0]
    groups = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]

    grid = _make_grid(y_pred)
    mock_ds = _make_dataset(y, "repo", groups)
    saved_reports = []

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
      patch("training.trainers.utils.harness.DEFAULT_BIAS_THRESHOLD", 0.15),
      patch(
        "training.trainers.utils.harness.BiasReport.save_report",
        side_effect=lambda data, path: saved_reports.append(data),
      ),
    ):
      mock_paths.models_root = tmp_path
      evaluate_bias(grid, "run-001", "linear", sensitive_feature="repo")

    summary = saved_reports[0]["summary"]
    assert summary["overall_bias_detected"] is True
    assert "repo" in summary["biased_dimensions"]
    assert summary["bias_count"] == 1

  def test_report_detailed_results_is_flat(self, tmp_path) -> None:
    """Test detailed_results contains flat analysis dict not nested schema."""
    y = [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    y_pred = [11.0, 21.0, 31.0, 41.0, 51.0, 50.0, 10.0, 1.0, 5.0, 15.0]
    groups = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]

    grid = _make_grid(y_pred)
    mock_ds = _make_dataset(y, "repo", groups)
    saved_reports = []

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
      patch(
        "training.trainers.utils.harness.BiasReport.save_report",
        side_effect=lambda data, path: saved_reports.append(data),
      ),
    ):
      mock_paths.models_root = tmp_path
      evaluate_bias(grid, "run-001", "linear", sensitive_feature="repo")

    repo_result = saved_reports[0]["detailed_results"]["repo"]
    assert "bias_detected" in repo_result
    assert "best_group" in repo_result
    assert "worst_group" in repo_result


class TestEvaluateBiasEdgeCases:
  """Tests for evaluate_bias edge cases."""

  def test_returns_none_when_sensitive_feature_missing(self, tmp_path) -> None:
    """Test returns None when sensitive feature not in metadata."""
    y = [10.0, 20.0, 30.0]
    y_pred = [11.0, 21.0, 31.0]

    grid = _make_grid(y_pred)
    mock_ds = MagicMock()
    mock_ds.load_x.return_value = np.zeros((3, 384))
    mock_ds.load_y.return_value = np.array(y)
    mock_ds.load_metadata.return_value = pd.DataFrame({"repo": ["A", "A", "B"]})

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
    ):
      mock_paths.models_root = tmp_path
      result = evaluate_bias(
        grid, "run-001", "linear", sensitive_feature="nonexistent_feature"
      )

    assert result is None

  def test_returns_none_when_metadata_unavailable(self, tmp_path) -> None:
    """Test returns None when metadata raises FileNotFoundError."""
    y = [10.0, 20.0, 30.0]
    y_pred = [11.0, 21.0, 31.0]

    grid = _make_grid(y_pred)
    mock_ds = MagicMock()
    mock_ds.load_x.return_value = np.zeros((3, 384))
    mock_ds.load_y.return_value = np.array(y)
    mock_ds.load_metadata.side_effect = FileNotFoundError("no data")

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
    ):
      mock_paths.models_root = tmp_path
      result = evaluate_bias(
        grid,
        "run-001",
        "linear",
        sensitive_feature="repo",
        threshold=0.15,
      )

    assert result is None


class TestGetTestAccuracy:
  """Tests for confidence score metrics in get_test_accuracy."""

  def test_confidence_metrics_present(self, tmp_path: Path) -> None:
    """Test that mean_confidence and low_confidence_rate are in eval output."""
    import json

    from training.trainers.utils.harness import get_test_accuracy

    y = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    proba = np.eye(4)[[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]]  # perfect confidence

    grid = MagicMock()
    grid.predict.return_value = np.array(y)
    grid.predict_proba.return_value = proba

    mock_ds = MagicMock()
    mock_ds.load_x.return_value = np.zeros((10, 384))
    mock_ds.load_y.return_value = np.array(y)

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
    ):
      mock_paths.models_root = tmp_path
      get_test_accuracy(grid, "run-001", "xgboost")

    eval_path = tmp_path / "run-001" / "eval_xgboost.json"
    assert eval_path.exists()
    metrics = json.loads(eval_path.read_text())
    assert "mean_confidence" in metrics
    assert "low_confidence_rate" in metrics
    assert metrics["mean_confidence"] == 1.0
    assert metrics["low_confidence_rate"] == 0.0

  def test_confidence_metrics_low_confidence(self, tmp_path: Path) -> None:
    """Test low_confidence_rate is non-zero when model is uncertain."""
    import json

    from training.trainers.utils.harness import get_test_accuracy

    y = [0, 1, 2, 3]
    # uniform probabilities = maximum uncertainty
    proba = np.full((4, 4), 0.25)

    grid = MagicMock()
    grid.predict.return_value = np.array(y)
    grid.predict_proba.return_value = proba

    mock_ds = MagicMock()
    mock_ds.load_x.return_value = np.zeros((4, 384))
    mock_ds.load_y.return_value = np.array(y)

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
    ):
      mock_paths.models_root = tmp_path
      get_test_accuracy(grid, "run-001", "xgboost")

    metrics = json.loads((tmp_path / "run-001" / "eval_xgboost.json").read_text())
    assert metrics["mean_confidence"] == 0.25
    assert metrics["low_confidence_rate"] == 1.0  # all predictions < 50% confident

  def test_confidence_metrics_absent_when_no_predict_proba(
    self, tmp_path: Path
  ) -> None:
    """Test mean_confidence is 0 when model has no predict_proba."""
    import json

    from training.trainers.utils.harness import get_test_accuracy

    y = [0, 1, 2, 3]

    grid = MagicMock(spec=["predict"])  # no predict_proba
    grid.predict.return_value = np.array(y)

    mock_ds = MagicMock()
    mock_ds.load_x.return_value = np.zeros((4, 384))
    mock_ds.load_y.return_value = np.array(y)

    with (
      patch("training.trainers.utils.harness.Dataset", return_value=mock_ds),
      patch("training.trainers.utils.harness.Paths") as mock_paths,
    ):
      mock_paths.models_root = tmp_path
      get_test_accuracy(grid, "run-001", "svm")

    metrics = json.loads((tmp_path / "run-001" / "eval_svm.json").read_text())
    assert metrics["mean_confidence"] == 0.0
    assert metrics["low_confidence_rate"] == 0.0
