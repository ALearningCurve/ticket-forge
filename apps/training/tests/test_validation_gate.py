"""Tests for validation gate thresholds."""

from __future__ import annotations

from training.analysis.gate_config import GateConfig
from training.analysis.validation_gate import evaluate_validation_gate


def test_validation_gate_passes_on_good_metrics() -> None:
  """Validation gate passes when absolute and relative checks both succeed."""
  config = GateConfig(
    min_accuracy=0.35,
    min_macro_f1=0.35,
    max_accuracy_relative_drop=0.10,
    max_macro_f1_relative_drop=0.10,
  )
  result = evaluate_validation_gate(
    {"accuracy": 0.78, "macro_f1": 0.75},
    {"accuracy": 0.80, "macro_f1": 0.78},
    config,
  )
  assert result["passed"] is True
  assert result["fail_reasons"] == []


def test_validation_gate_fails_on_bad_metrics() -> None:
  """Validation gate fails when absolute or relative checks regress."""
  config = GateConfig(
    min_accuracy=0.35,
    min_macro_f1=0.35,
    max_accuracy_relative_drop=0.10,
    max_macro_f1_relative_drop=0.10,
  )
  result = evaluate_validation_gate(
    {"accuracy": 0.68, "macro_f1": 0.60},
    {"accuracy": 0.80, "macro_f1": 0.78},
    config,
  )
  assert result["passed"] is False
  assert len(result["fail_reasons"]) == 2


def test_validation_gate_fails_when_relative_drop_is_too_high() -> None:
  """Validation gate fails when the candidate is worse than the baseline."""
  config = GateConfig(
    min_accuracy=0.35,
    min_macro_f1=0.35,
    max_accuracy_relative_drop=0.05,
    max_macro_f1_relative_drop=0.05,
  )
  result = evaluate_validation_gate(
    {"accuracy": 0.75, "macro_f1": 0.70},
    {"accuracy": 0.80, "macro_f1": 0.78},
    config,
  )
  assert result["passed"] is False
  assert len(result["fail_reasons"]) == 2
