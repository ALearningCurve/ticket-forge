"""Tests for bias detection."""

import pandas as pd
import pytest
from training.bias import BiasAnalyzer, BiasMitigator, BiasReport, DataSlicer


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


class TestBiasAnalyzerRegressor:
  """Test cases for BiasAnalyzer with regressor model type."""

  @pytest.fixture
  def analyzer(self) -> BiasAnalyzer:
    """Create a regressor bias analyzer."""
    return BiasAnalyzer(threshold=0.1, model_type="regressor")

  def test_model_type_is_regressor(self, analyzer: BiasAnalyzer) -> None:
    """Test analyzer is configured for regressor."""
    assert analyzer.model_type == "regressor"
    assert analyzer._primary_metric == "mae"

  def test_analyze_with_metricframe(self, analyzer: BiasAnalyzer) -> None:
    """Test Fairlearn MetricFrame analysis for regressor."""
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    y_pred = pd.Series([12.0, 18.0, 32.0, 38.0, 52.0, 58.0])
    sensitive_features = pd.Series(["A", "A", "A", "B", "B", "B"])

    metric_frame = analyzer.analyze_with_metricframe(y_true, y_pred, sensitive_features)

    assert hasattr(metric_frame, "by_group")
    assert hasattr(metric_frame, "overall")
    assert "mae" in metric_frame.by_group.columns
    assert "rmse" in metric_frame.by_group.columns
    assert "r2" in metric_frame.by_group.columns

  def test_detect_bias_fairlearn_no_bias(self, analyzer: BiasAnalyzer) -> None:
    """Test Fairlearn bias detection with similar performance."""
    y_true = pd.Series([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
    y_pred = pd.Series([11.0, 21.0, 31.0, 11.0, 21.0, 31.0])
    sensitive_features = pd.Series(["A", "A", "A", "B", "B", "B"])

    result = analyzer.detect_bias_fairlearn(y_true, y_pred, sensitive_features)

    assert result["model_type"] == "regressor"
    assert result["primary_metric"] == "mae"
    assert "metric_frame" in result
    assert "bias_detected" in result
    assert result["bias_detected"] is False

  def test_detect_bias_fairlearn_with_bias(self, analyzer: BiasAnalyzer) -> None:
    """Test Fairlearn bias detection with disparate performance."""
    # Group A: good predictions (MAE=1)
    # Group B: bad predictions (MAE=10)
    y_true = pd.Series([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
    y_pred = pd.Series([11.0, 21.0, 31.0, 20.0, 30.0, 40.0])
    sensitive_features = pd.Series(["A", "A", "A", "B", "B", "B"])

    result = analyzer.detect_bias_fairlearn(y_true, y_pred, sensitive_features)

    assert result["bias_detected"] is True
    assert result["best_group"]["name"] == "A"
    assert result["worst_group"]["name"] == "B"
    assert result["mae_difference"] > 0
    assert result["relative_gap"] > analyzer.threshold


class TestBiasAnalyzerRecommendation:
  """Test cases for BiasAnalyzer with recommendation model type."""

  @pytest.fixture
  def analyzer(self) -> BiasAnalyzer:
    """Create a recommendation bias analyzer."""
    return BiasAnalyzer(threshold=0.1, model_type="recommendation")

  def test_model_type_is_recommendation(self, analyzer: BiasAnalyzer) -> None:
    """Test analyzer is configured for recommendation."""
    assert analyzer.model_type == "recommendation"
    assert analyzer._primary_metric == "ndcg"

  def test_analyze_recommendation_metrics(self, analyzer: BiasAnalyzer) -> None:
    """Test calculating recommendation metrics."""
    y_true = pd.Series([3.0, 2.0, 1.0, 0.0, 0.0])
    y_pred = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])  # Good ranking

    metrics = analyzer.analyze_recommendation_metrics(y_true, y_pred)

    assert "ndcg" in metrics
    assert "mrr" in metrics
    assert "mae" in metrics
    assert "count" in metrics
    assert metrics["count"] == 5
    assert metrics["ndcg"] is not None
    assert metrics["ndcg"] > 0  # Should have good NDCG for correct ranking

  def test_compare_slices_recommendation(self, analyzer: BiasAnalyzer) -> None:
    """Test slice comparison for recommendation model."""
    # slice1: good ranking
    slice1 = pd.DataFrame(
      {
        "y_true": [3.0, 2.0, 1.0],
        "y_pred": [3.0, 2.0, 1.0],
      }
    )
    # slice2: bad ranking
    slice2 = pd.DataFrame(
      {
        "y_true": [3.0, 2.0, 1.0],
        "y_pred": [1.0, 2.0, 3.0],  # Reversed ranking
      }
    )

    slices = {"good_slice": slice1, "bad_slice": slice2}
    result = analyzer.compare_slices(slices, "y_true", "y_pred")

    assert result["model_type"] == "recommendation"
    assert result["primary_metric"] == "ndcg"
    assert "bias_detected" in result
    # Good slice should have higher NDCG
    assert result["best_slice"]["name"] == "good_slice"


class TestBiasMitigator:
  """Test cases for BiasMitigator."""

  @pytest.fixture
  def sample_data(self) -> pd.DataFrame:
    """Create sample imbalanced data."""
    return pd.DataFrame(
      {
        "id": range(10),
        "repo": ["A", "A", "A", "A", "A", "A", "A", "B", "B", "C"],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
      }
    )

  def test_resample_underrepresented(self, sample_data: pd.DataFrame) -> None:
    """Test resampling underrepresented groups."""
    balanced = BiasMitigator.resample_underrepresented(sample_data, "repo")

    # Each group should have 7 samples (size of largest group A)
    counts = balanced["repo"].value_counts()
    assert counts["A"] == 7
    assert counts["B"] == 7
    assert counts["C"] == 7
    assert len(balanced) == 21

  def test_resample_with_custom_target(self, sample_data: pd.DataFrame) -> None:
    """Test resampling with custom target size."""
    balanced = BiasMitigator.resample_underrepresented(
      sample_data, "repo", target_size=5
    )

    # Each group should have 5 samples (or original if larger)
    counts = balanced["repo"].value_counts()
    assert counts["A"] == 7  # Original size preserved (larger than target)
    assert counts["B"] == 5
    assert counts["C"] == 5

  def test_compute_sample_weights(self, sample_data: pd.DataFrame) -> None:
    """Test sample weight computation for balanced training."""
    weights = BiasMitigator.compute_sample_weights(sample_data, "repo")

    # Underrepresented groups should have higher weights
    weight_a = weights.loc[sample_data["repo"] == "A"].iloc[0]
    weight_b = weights.loc[sample_data["repo"] == "B"].iloc[0]
    weight_c = weights.loc[sample_data["repo"] == "C"].iloc[0]

    # C (1 sample) should have highest weight, A (7 samples) lowest
    assert weight_c > weight_b > weight_a
    assert len(weights) == len(sample_data)

  def test_adjust_predictions_for_fairness(self) -> None:
    """Test prediction adjustment for fairness."""
    predictions = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    sensitive_features = pd.Series(["A", "A", "A", "B", "B", "B"])

    adjusted = BiasMitigator.adjust_predictions_for_fairness(
      predictions, sensitive_features, method="equalize_mean"
    )

    # Group B has higher mean, should be adjusted down
    group_b_original_mean = predictions[sensitive_features == "B"].mean()
    group_b_adjusted_mean = adjusted[sensitive_features == "B"].mean()

    assert group_b_adjusted_mean < group_b_original_mean
    assert len(adjusted) == len(predictions)

  def test_compute_group_statistics(self) -> None:
    """Test computing group statistics."""
    predictions = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    sensitive_features = pd.Series(["A", "A", "A", "B", "B", "B"])

    stats = BiasMitigator.compute_group_statistics(predictions, sensitive_features)

    assert "A" in stats
    assert "B" in stats
    assert "mean" in stats["A"]
    assert "std" in stats["A"]
    assert "count" in stats["A"]
    assert stats["A"]["count"] == 3
    assert stats["B"]["count"] == 3

  def test_get_fairlearn_metrics_summary(self) -> None:
    """Test extracting metrics from Fairlearn MetricFrame."""
    analyzer = BiasAnalyzer(model_type="regressor")
    y_true = pd.Series([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
    y_pred = pd.Series([11.0, 21.0, 31.0, 20.0, 30.0, 40.0])
    sensitive_features = pd.Series(["A", "A", "A", "B", "B", "B"])

    metric_frame = analyzer.analyze_with_metricframe(y_true, y_pred, sensitive_features)
    summary = BiasMitigator.get_fairlearn_metrics_summary(metric_frame, "mae")

    assert "mae_difference" in summary
    assert "mae_ratio" in summary
    assert "group_min_mae" in summary
    assert "group_max_mae" in summary
    assert summary["mae_difference"] > 0
