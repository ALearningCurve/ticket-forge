"""Configuration helpers for model CI/CD gates.

Loads gate thresholds and behavior toggles from environment variables with
safe defaults for local and CI execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from shared.configuration import getenv_or


@dataclass(slots=True)
class GateConfig:
  """Gate threshold configuration for model CI/CD decisions.

  Attributes:
      min_accuracy: Minimum allowed absolute accuracy.
      min_macro_f1: Minimum allowed absolute macro F1.
      max_accuracy_relative_drop: Maximum allowed accuracy drop vs baseline.
      max_macro_f1_relative_drop: Maximum allowed macro F1 drop vs baseline.
      max_bias_relative_gap: Maximum allowed relative bias gap.
      max_regression_degradation: Maximum allowed degradation vs production.
      bias_slices: Data slices checked by bias analysis.
  """

  min_accuracy: float = 0.35
  min_macro_f1: float = 0.35
  max_accuracy_relative_drop: float = 0.10
  max_macro_f1_relative_drop: float = 0.10
  max_bias_relative_gap: float = 0.40
  max_regression_degradation: float = 0.10
  bias_slices: tuple[str, ...] = ("repo", "seniority")

  def to_dict(self) -> dict[str, float | tuple[str, ...]]:
    """Return a dictionary representation of the configuration."""
    return asdict(self)


def load_gate_config() -> GateConfig:
  """Load gate configuration from environment variables.

  Returns:
      Parsed GateConfig with defaults when environment variables are absent.
  """
  slices = getenv_or("MODEL_CICD_BIAS_SLICES", "repo,seniority") or "repo,seniority"
  parsed_slices = tuple(s.strip() for s in slices.split(",") if s.strip())

  def getf(key: str, default: str) -> float:
    """Gets and parses from environment, falling back to default if not set."""
    return float(getenv_or(key, default) or default)

  return GateConfig(
    min_accuracy=getf("MODEL_CICD_MIN_ACCURACY", "0.35"),
    min_macro_f1=getf("MODEL_CICD_MIN_MACRO_F1", "0.35"),
    max_accuracy_relative_drop=getf("MODEL_CICD_MAX_ACCURACY_RELATIVE_DROP", "0.10"),
    max_macro_f1_relative_drop=getf("MODEL_CICD_MAX_MACRO_F1_RELATIVE_DROP", "0.10"),
    max_bias_relative_gap=getf("MODEL_CICD_MAX_BIAS_RELATIVE_GAP", "0.70"),
    max_regression_degradation=getf("MODEL_CICD_MAX_REGRESSION_DEGRADATION", "0.10"),
    bias_slices=parsed_slices or ("repo", "seniority"),
  )
