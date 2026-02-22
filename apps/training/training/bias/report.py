"""Bias detection reporting."""

from typing import Any

from shared import get_logger

logger = get_logger(__name__)


class BiasReport:
  """Generate human-readable bias detection reports."""

  @staticmethod
  def _format_summary(analysis: dict[str, Any]) -> list[str]:
    """Format summary section of report."""
    lines = []
    summary = analysis["summary"]

    lines.append("\nSUMMARY\n")
    model_type = summary.get("model_type", "regressor")
    lines.append(f"Model type: {model_type}")
    lines.append(f"Dimensions checked: {summary['total_dimensions_checked']}")
    bias_text = "YES" if summary["overall_bias_detected"] else "NO"
    lines.append("Bias detected: " + bias_text)

    dims = (
      ", ".join(summary["biased_dimensions"])
      if summary["biased_dimensions"]
      else "None"
    )
    lines.append("Biased dimensions: " + dims)
    return lines

  @staticmethod
  def _format_dimension_result(
    dimension: str, result: dict[str, Any], primary_metric: str
  ) -> list[str]:
    """Format results for a single dimension."""
    lines = []
    lines.append(f"\n{dimension.upper()}")
    lines.append("-" * 80)

    if not result.get("bias_detected"):
      lines.append("No significant bias detected")
    else:
      lines.append("BIAS DETECTED")

      best = result.get("best_slice", result.get("best_group", {}))
      worst = result.get("worst_slice", result.get("worst_group", {}))

      lines.append(f"\nBest performing slice: {best.get('name', 'N/A')}")
      if primary_metric in best:
        lines.append(f"  {primary_metric.upper()}: {best[primary_metric]:.4f}")

      lines.append(f"\nWorst performing slice: {worst.get('name', 'N/A')}")
      if primary_metric in worst:
        lines.append(f"  {primary_metric.upper()}: {worst[primary_metric]:.4f}")

      gap_key = f"{primary_metric}_gap"
      if gap_key in result:
        lines.append(f"\nPerformance gap: {result[gap_key]:.4f}")
      lines.append(
        f"Relative gap: {result['relative_gap'] * 100:.1f}% "
        f"(threshold: {result['threshold'] * 100:.1f}%)"
      )

    # Show metrics for all slices
    metrics_key = (
      "metrics_by_slice" if "metrics_by_slice" in result else "metrics_by_group"
    )
    if metrics_key in result:
      lines.append("\nMetrics by slice:")
      for slice_name, metrics in result[metrics_key].items():
        if isinstance(metrics, dict) and metrics.get(primary_metric) is not None:
          line = BiasReport._format_metric_row(slice_name, metrics, primary_metric)
          lines.append(line)

    return lines

  @staticmethod
  def _format_metric_row(
    slice_name: str, metrics: dict[str, Any], primary_metric: str
  ) -> str:
    """Format a single metric row."""
    metric_str = f"{primary_metric.upper()}: {metrics[primary_metric]:.4f}"
    if "rmse" in metrics and metrics["rmse"] is not None:
      metric_str += f" | RMSE: {metrics['rmse']:.4f}"
    if "r2" in metrics and metrics["r2"] is not None:
      metric_str += f" | R²: {metrics['r2']:.4f}"
    if "ndcg" in metrics and metrics["ndcg"] is not None:
      metric_str += f" | NDCG: {metrics['ndcg']:.4f}"
    if "mrr" in metrics and metrics["mrr"] is not None:
      metric_str += f" | MRR: {metrics['mrr']:.4f}"
    metric_str += f" | n={metrics.get('count', 'N/A')}"
    return f"  {slice_name:30} | {metric_str}"

  @staticmethod
  def generate_text_report(analysis: dict[str, Any]) -> str:
    """Generate a text report from bias analysis.

    Args:
        analysis: Analysis results from BiasAnalyzer

    Returns:
        Formatted text report
    """
    lines = ["BIAS DETECTION REPORT"]

    # Add summary
    lines.extend(BiasReport._format_summary(analysis))

    # Detailed results per dimension
    lines.append("\nDETAILED ANALYSIS")

    summary = analysis["summary"]
    model_type = summary.get("model_type", "regressor")
    primary_metric = "mae" if model_type == "regressor" else "ndcg"

    details = analysis["detailed_results"]
    for dimension, result in details.items():
      lines.extend(
        BiasReport._format_dimension_result(dimension, result, primary_metric)
      )

    lines.append("\nEND OF REPORT")

    return "\n".join(lines)

  @staticmethod
  def save_report(analysis: dict[str, Any], output_path: str) -> None:
    """Save bias report to file.

    Args:
        analysis: Analysis results
        output_path: Path to save report
    """
    report = BiasReport.generate_text_report(analysis)

    with open(output_path, "w", encoding="utf-8") as f:
      f.write(report)

    logger.info("Bias report saved to %s", output_path)
