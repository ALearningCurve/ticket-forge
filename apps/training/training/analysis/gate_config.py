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
      min_r2: Minimum required R2 for validation gate.
      max_mae: Maximum allowed MAE for validation gate.
      max_bias_relative_gap: Maximum allowed relative bias gap.
      max_regression_degradation: Maximum allowed degradation vs production.
      bias_slices: Data slices checked by bias analysis.
  """

  min_r2: float = 0.60
  max_mae: float = 20.0
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

  return GateConfig(
    min_r2=float(getenv_or("MODEL_CICD_MIN_R2", "0.60") or "0.60"),
    max_mae=float(getenv_or("MODEL_CICD_MAX_MAE", "20.0") or "20.0"),
    max_bias_relative_gap=float(
      getenv_or("MODEL_CICD_MAX_BIAS_RELATIVE_GAP", "0.40") or "0.40"
    ),
    max_regression_degradation=float(
      getenv_or("MODEL_CICD_MAX_REGRESSION_DEGRADATION", "0.10") or "0.10"
    ),
    bias_slices=parsed_slices or ("repo", "seniority"),
  )
