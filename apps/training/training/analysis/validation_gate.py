"""Validation gate checks for model CI/CD."""

from __future__ import annotations

from typing import Any

from training.analysis.gate_config import GateConfig


def evaluate_validation_gate(
  metrics: dict[str, float], config: GateConfig
) -> dict[str, Any]:
  """Evaluate candidate metrics against validation thresholds.

  Args:
      metrics: Candidate metrics dictionary.
      config: Loaded gate configuration.

  Returns:
      Validation gate decision payload.
  """
  fail_reasons: list[str] = []

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

  return {
    "passed": len(fail_reasons) == 0,
    "metrics": metrics,
    "thresholds": {
      "min_accuracy": config.min_accuracy,
      "min_macro_f1": config.min_macro_f1,
    },
    "fail_reasons": fail_reasons,
  }
