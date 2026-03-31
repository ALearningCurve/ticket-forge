"""Tests for regression guardrail threshold behavior."""

from __future__ import annotations

from training.analysis.regression_guardrail import evaluate_regression_guardrail


def test_guardrail_passes_when_within_threshold() -> None:
  """Guardrail passes when candidate degradation is below threshold."""
  result = evaluate_regression_guardrail(
    candidate_metrics={"accuracy": 0.78, "macro_f1": 0.75},
    baseline_metrics={"accuracy": 0.77, "macro_f1": 0.74},
    max_allowed_degradation=0.10,
  )
  assert result["passed"] is True


def test_guardrail_blocks_when_over_threshold() -> None:
  """Guardrail blocks when accuracy or macro_f1 degradation exceeds threshold."""
  result = evaluate_regression_guardrail(
    candidate_metrics={"accuracy": 0.50, "macro_f1": 0.45},
    baseline_metrics={"accuracy": 0.78, "macro_f1": 0.75},
    max_allowed_degradation=0.10,
  )
  assert result["passed"] is False
  assert result["fail_reasons"]


def test_guardrail_passes_without_baseline() -> None:
  """Guardrail allows first deployment when no baseline exists."""
  result = evaluate_regression_guardrail(
    candidate_metrics={"accuracy": 0.78, "macro_f1": 0.75},
    baseline_metrics=None,
    max_allowed_degradation=0.10,
  )
  assert result["passed"] is True
  assert result.get("note") == "no-production-baseline"
