"""Bias detection reporting."""

from typing import Any


class BiasReport:
  """Generate human-readable bias detection reports."""

  @staticmethod
  def generate_text_report(analysis: dict[str, Any]) -> str:
    """Generate a text report from bias analysis.

    Args:
        analysis: Analysis results from BiasAnalyzer

    Returns:
        Formatted text report
    """
    lines = []
    lines.append("BIAS DETECTION REPORT")

    summary = analysis["summary"]
    details = analysis["detailed_results"]

    # Overall summary
    lines.append("\nSUMMARY\n")
    lines.append(f"Dimensions checked: {summary['total_dimensions_checked']}")
    bias_text = "YES" if summary["overall_bias_detected"] else "NO"
    lines.append("Bias detected: " + bias_text)

    dims = ", ".join(summary["biased_dimensions"])
    lines.append("Biased dimensions: " + dims)

    # Detailed results per dimension

    lines.append("DETAILED ANALYSIS")

    for dimension, result in details.items():
      lines.append(f"\n{dimension.upper()}")
      lines.append("-" * 80)

      if not result.get("bias_detected"):
        lines.append("No significant bias detected")
      else:
        lines.append("BIAS DETECTED")

        best = result["best_slice"]
        worst = result["worst_slice"]

        lines.append(f"\nBest performing slice: {best['name']}")
        lines.append(f"  MAE: {best['mae']:.4f}")

        lines.append(f"\nWorst performing slice: {worst['name']}")
        lines.append(f"  MAE: {worst['mae']:.4f}")

        lines.append(f"\nPerformance gap: {result['mae_gap']:.4f}")
        lines.append(
          f"Relative gap: {result['relative_gap'] * 100:.1f}% "
          f"(threshold: {result['threshold'] * 100:.1f}%)"
        )

      # Show metrics for all slices
      lines.append("\nMetrics by slice:")
      for slice_name, metrics in result["metrics_by_slice"].items():
        if metrics["mae"] is not None:
          lines.append(
            f"  {slice_name:30} | MAE: {metrics['mae']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | "
            f"R²: {metrics['r2']:.4f} | "
            f"n={metrics['count']}"
          )

    lines.append("END OF REPORT")

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

    print(f"Bias report saved to {output_path}")
