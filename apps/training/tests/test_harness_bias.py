"""Tests for bias evaluation and remediation in training harness."""

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


class TestEvaluateBiasNoRemediationNeeded:
  """Tests for evaluate_bias when no bias is detected."""

  def test_returns_analysis_when_no_bias(self, tmp_path) -> None:
    """Test that analysis is returned when no bias is detected."""
    n = 6
    y = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    # Both groups A and B have MAE=1 — no bias
    y_pred = [11.0, 21.0, 31.0, 11.0, 21.0, 31.0]
    groups = ["A", "A", "A", "B", "B", "B"]

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
    assert result["primary_metric"] == "mae"

  def test_report_has_no_remediation_fields_when_no_bias(self, tmp_path) -> None:
    """Test that report does not include remediation_details when no bias."""
    y = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    y_pred = [11.0, 21.0, 31.0, 11.0, 21.0, 31.0]
    groups = ["A", "A", "A", "B", "B", "B"]

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
    assert report["summary"]["remediation_applied"] is False
    assert report["summary"]["remediation_method"] is None
    assert "after_remediation" not in report["remediation_details"]["repo"]


class TestBiasReportTextGeneration:
  """Tests that BiasReport.generate_text_report() handles the new schema."""

  def test_text_report_renders_without_error_when_no_remediation(
    self, tmp_path
  ) -> None:
    """Report text should render correctly when no remediation was applied."""
    y = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    y_pred = [11.0, 21.0, 31.0, 11.0, 21.0, 31.0]
    groups = ["A", "A", "A", "B", "B", "B"]

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
    # Should not raise — detailed_results is flat so generate_text_report works
    text_report = BiasReport.generate_text_report(report)
    assert isinstance(text_report, str)
    assert len(text_report) > 0

  def test_text_report_renders_without_error_when_remediation_applied(
    self, tmp_path
  ) -> None:
    """Report text should render correctly when remediation was applied."""
    y = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    y_pred = [11.0, 21.0, 31.0, 60.0, 70.0, 80.0]
    groups = ["A", "A", "A", "B", "B", "B"]

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
    assert report["summary"]["remediation_applied"] is True
    # detailed_results contains the flat final_analysis so the report renders
    assert "bias_detected" in report["detailed_results"]["repo"]
    text_report = BiasReport.generate_text_report(report)
    assert isinstance(text_report, str)
    assert len(text_report) > 0

  """Tests for evaluate_bias when bias is detected and remediation is applied."""

  def test_returns_post_remediation_analysis(self, tmp_path) -> None:
    """Test that post-remediation analysis is returned when bias detected."""
    # Group A: MAE=1 (good), Group B: MAE=50 (bad) — clear bias
    y = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    y_pred = [11.0, 21.0, 31.0, 60.0, 70.0, 80.0]
    groups = ["A", "A", "A", "B", "B", "B"]

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
    # Result should be post-remediation analysis
    assert "bias_detected" in result

  def test_report_includes_before_and_after_remediation(self, tmp_path) -> None:
    """Test report includes both before and after in remediation_details."""
    y = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    y_pred = [11.0, 21.0, 31.0, 60.0, 70.0, 80.0]
    groups = ["A", "A", "A", "B", "B", "B"]

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
    assert report["summary"]["remediation_applied"] is True
    assert report["summary"]["remediation_method"] == "equalize_mean"
    assert "before_remediation" in report["remediation_details"]["repo"]
    assert "after_remediation" in report["remediation_details"]["repo"]
    # detailed_results should contain the final (post-remediation) analysis
    assert "bias_detected" in report["detailed_results"]["repo"]

  def test_report_summary_flags_bias_correctly(self, tmp_path) -> None:
    """Test report summary correctly identifies biased dimension."""
    y = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    y_pred = [11.0, 21.0, 31.0, 60.0, 70.0, 80.0]
    groups = ["A", "A", "A", "B", "B", "B"]

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

    summary = saved_reports[0]["summary"]
    assert summary["overall_bias_detected"] is True
    assert "repo" in summary["biased_dimensions"]
    assert summary["bias_count"] == 1


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
    # Metadata does not contain "nonexistent_feature"
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
      result = evaluate_bias(grid, "run-001", "linear", sensitive_feature="repo")

    assert result is None
