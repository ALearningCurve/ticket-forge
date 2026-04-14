"""Validation gate checks for model CI/CD."""

from __future__ import annotations

from typing import Any

from training.analysis.gate_config import GateConfig


def _relative_degradation(
  candidate: float,
  baseline: float,
) -> float:
  """Compute relative degradation for a higher-is-better metric."""
  if baseline == 0:
    return 0.0
  return (baseline - candidate) / abs(baseline)


def evaluate_validation_gate(
  metrics: dict[str, float],
  baseline_metrics: dict[str, float] | None,
  config: GateConfig,
) -> dict[str, Any]:
  """Evaluate candidate metrics against absolute and baseline-relative thresholds.

  Args:
      metrics: Candidate metrics dictionary.
      baseline_metrics: Production baseline metrics if available.
      config: Loaded gate configuration.

  Returns:
      Validation gate decision payload.
  """
  fail_reasons: list[str] = []
  metric_deltas: dict[str, float] = {}

  accuracy = float(metrics.get("accuracy", 0.0))
  macro_f1 = float(metrics.get("macro_f1", 0.0))
  if accuracy < config.min_accuracy:
    fail_reasons.append(
      f"accuracy-below-threshold:{accuracy:.4f}<{config.min_accuracy:.4f}"
    )
  if macro_f1 < config.min_macro_f1:
    fail_reasons.append(
      f"macro_f1-below-threshold:{macro_f1:.4f}<{config.min_macro_f1:.4f}"
    )

  if not baseline_metrics:
    return {
      "passed": len(fail_reasons) == 0,
      "metrics": metrics,
      "baseline_metrics": None,
      "metric_deltas": metric_deltas,
      "thresholds": {
        "min_accuracy": config.min_accuracy,
        "min_macro_f1": config.min_macro_f1,
        "max_accuracy_relative_drop": config.max_accuracy_relative_drop,
        "max_macro_f1_relative_drop": config.max_macro_f1_relative_drop,
      },
      "fail_reasons": fail_reasons,
      "note": "no-production-baseline",
    }

  baseline_accuracy = float(baseline_metrics.get("accuracy", 0.0))
  baseline_macro_f1 = float(baseline_metrics.get("macro_f1", 0.0))

  accuracy_drop = _relative_degradation(accuracy, baseline_accuracy)
  metric_deltas["accuracy"] = accuracy_drop
  if accuracy_drop > config.max_accuracy_relative_drop:
    fail_reasons.append(
      "accuracy-relative-drop-exceeds-threshold:"
      f"{accuracy_drop:.4f}>{config.max_accuracy_relative_drop:.4f}"
    )
  macro_f1_drop = _relative_degradation(macro_f1, baseline_macro_f1)
  metric_deltas["macro_f1"] = macro_f1_drop
  if macro_f1_drop > config.max_macro_f1_relative_drop:
    fail_reasons.append(
      "macro_f1-relative-drop-exceeds-threshold:"
      f"{macro_f1_drop:.4f}>{config.max_macro_f1_relative_drop:.4f}"
    )

  return {
    "passed": len(fail_reasons) == 0,
    "metrics": metrics,
    "baseline_metrics": baseline_metrics,
    "metric_deltas": metric_deltas,
    "thresholds": {
      "min_accuracy": config.min_accuracy,
      "min_macro_f1": config.min_macro_f1,
      "max_accuracy_relative_drop": config.max_accuracy_relative_drop,
      "max_macro_f1_relative_drop": config.max_macro_f1_relative_drop,
    },
    "fail_reasons": fail_reasons,
  }
