"""Tests for bias detection."""

import pandas as pd
import pytest
from ml_core.bias import BiasAnalyzer, BiasReport, DataSlicer


class TestDataSlicer:
  """Test cases for DataSlicer."""

  @pytest.fixture
  def sample_data(self) -> pd.DataFrame:
    """Create sample ticket data."""
    return pd.DataFrame(
      {
        "id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "repo": [
          "terraform",
          "terraform",
          "ansible",
          "ansible",
          "prometheus",
          "prometheus",
        ],
        "seniority": ["junior", "mid", "mid", "senior", "senior", "junior"],
        "labels": [
          "bug,crash",
          "enhancement",
          "bug",
          "new",
          "enhancement,critical",
          "bug",
        ],
        "keywords": [
          ["aws", "docker"],
          ["python"],
          ["kubernetes"],
          ["go"],
          ["grafana"],
          ["terraform"],
        ],
        "completion_hours_business": [2.5, 15.0, 8.0, 48.0, 3.0, None],
      }
    )

  def test_slice_by_repo(self, sample_data: pd.DataFrame) -> None:
    """Test slicing by repository."""
    slicer = DataSlicer(sample_data)
    slices = slicer.slice_by_repo()

    assert len(slices) == 3
    assert "terraform" in slices
    assert len(slices["terraform"]) == 2

  def test_slice_by_seniority(self, sample_data: pd.DataFrame) -> None:
    """Test slicing by seniority level."""
    slicer = DataSlicer(sample_data)
    slices = slicer.slice_by_seniority()

    assert "junior" in slices
    assert "mid" in slices
    assert "senior" in slices
    assert len(slices["junior"]) == 2

  def test_slice_by_label(self, sample_data: pd.DataFrame) -> None:
    """Test slicing by label presence."""
    slicer = DataSlicer(sample_data)
    slices = slicer.slice_by_label("bug")

    assert "has_bug" in slices
    assert "no_bug" in slices
    assert len(slices["has_bug"]) == 3
    assert len(slices["no_bug"]) == 3

  def test_slice_by_completion_time(self, sample_data: pd.DataFrame) -> None:
    """Test slicing by completion time buckets."""
    slicer = DataSlicer(sample_data)
    slices = slicer.slice_by_completion_time()

    assert "fast" in slices
    assert "medium" in slices
    assert "slow" in slices


class TestBiasAnalyzer:
  """Test cases for BiasAnalyzer."""

  @pytest.fixture
  def analyzer(self) -> BiasAnalyzer:
    """Create a bias analyzer."""
    return BiasAnalyzer(threshold=0.1)

  def test_analyze_regression_metrics(self, analyzer: BiasAnalyzer) -> None:
    """Test calculating regression metrics."""
    y_true = pd.Series([10, 20, 30, 40, 50])
    y_pred = pd.Series([12, 18, 32, 38, 52])

    metrics = analyzer.analyze_regression_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "count" in metrics
    assert metrics["count"] == 5

  def test_analyze_with_missing_values(self, analyzer: BiasAnalyzer) -> None:
    """Test handling of missing values."""
    y_true = pd.Series([10, None, 30, None, 50])
    y_pred = pd.Series([12, 18, None, 38, 52])

    metrics = analyzer.analyze_regression_metrics(y_true, y_pred)

    assert metrics["count"] == 2

  def test_compare_slices_no_bias(self, analyzer: BiasAnalyzer) -> None:
    """Test slice comparison when no bias exists."""
    # Both slices have similar MAE (~1.0)
    slice1 = pd.DataFrame({"y_true": [10, 20, 30], "y_pred": [11, 21, 31]})
    slice2 = pd.DataFrame({"y_true": [10, 20, 30], "y_pred": [11, 21, 29]})

    slices = {"slice1": slice1, "slice2": slice2}
    result = analyzer.compare_slices(slices, "y_true", "y_pred")

    assert "bias_detected" in result
    # MAE difference should be small, no bias
    assert result["bias_detected"] is False

  def test_compare_slices_with_bias(self, analyzer: BiasAnalyzer) -> None:
    """Test slice comparison when bias exists."""
    # slice1: good predictions (MAE=1)
    slice1 = pd.DataFrame({"y_true": [10, 20, 30], "y_pred": [11, 21, 31]})
    # slice2: bad predictions (MAE=10)
    slice2 = pd.DataFrame({"y_true": [10, 20, 30], "y_pred": [20, 30, 40]})

    slices = {"good_slice": slice1, "bad_slice": slice2}
    result = analyzer.compare_slices(slices, "y_true", "y_pred")

    assert result["bias_detected"] is True
    assert result["best_slice"]["name"] == "good_slice"
    assert result["worst_slice"]["name"] == "bad_slice"
    # Relative gap should be (10-1)/1 = 9.0 = 900% > 10% threshold
    assert result["relative_gap"] > analyzer.threshold

  def test_detect_bias_multiple_dimensions(self, analyzer: BiasAnalyzer) -> None:
    """Test bias detection across multiple dimensions."""
    df = pd.DataFrame(
      {
        "repo": ["A", "A", "B", "B"],
        "seniority": ["junior", "senior", "junior", "senior"],
        "y_true": [10, 20, 10, 20],
        "y_pred": [10, 20, 15, 25],
      }
    )

    all_slices = {
      "by_repo": {
        "A": df[df["repo"] == "A"],
        "B": df[df["repo"] == "B"],
      },
      "by_seniority": {
        "junior": df[df["seniority"] == "junior"],
        "senior": df[df["seniority"] == "senior"],
      },
    }

    result = analyzer.detect_bias_multiple_dimensions(all_slices, "y_true", "y_pred")  # type: ignore[arg-type]
    assert "summary" in result
    assert "detailed_results" in result
    assert result["summary"]["total_dimensions_checked"] == 2


class TestBiasReport:
  """Test cases for BiasReport."""

  def test_generate_text_report(self) -> None:
    """Test generating text report."""
    analysis = {
      "summary": {
        "total_dimensions_checked": 2,
        "biased_dimensions": ["by_repo"],
        "bias_count": 1,
        "overall_bias_detected": True,
      },
      "detailed_results": {
        "by_repo": {
          "bias_detected": True,
          "best_slice": {"name": "terraform", "mae": 2.5},
          "worst_slice": {"name": "ansible", "mae": 5.0},
          "mae_gap": 2.5,
          "relative_gap": 1.0,
          "threshold": 0.1,
          "metrics_by_slice": {
            "terraform": {
              "mae": 2.5,
              "rmse": 3.0,
              "r2": 0.9,
              "count": 100,
            },
            "ansible": {
              "mae": 5.0,
              "rmse": 6.0,
              "r2": 0.7,
              "count": 100,
            },
          },
        },
      },
    }

    report = BiasReport.generate_text_report(analysis)

    assert "BIAS DETECTION REPORT" in report
    assert "YES" in report
    assert "by_repo" in report
    assert "terraform" in report
