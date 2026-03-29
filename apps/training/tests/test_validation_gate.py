"""Tests for validation gate thresholds."""

from __future__ import annotations

from training.analysis.gate_config import GateConfig
from training.analysis.validation_gate import evaluate_validation_gate


def test_validation_gate_passes_on_good_metrics() -> None:
  """Validation gate passes when accuracy and macro_f1 satisfy thresholds."""
  config = GateConfig(min_accuracy=0.70, min_macro_f1=0.65)
  result = evaluate_validation_gate({"accuracy": 0.78, "macro_f1": 0.75}, config)
  assert result["passed"] is True
  assert result["fail_reasons"] == []


def test_validation_gate_fails_on_bad_metrics() -> None:
  """Validation gate fails when either metric violates thresholds."""
  config = GateConfig(min_accuracy=0.70, min_macro_f1=0.65)
  result = evaluate_validation_gate({"accuracy": 0.55, "macro_f1": 0.50}, config)
  assert result["passed"] is False
  assert len(result["fail_reasons"]) == 2
