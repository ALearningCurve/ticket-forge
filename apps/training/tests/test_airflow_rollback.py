"""Tests for rollback helper script contract."""

from pathlib import Path


def test_rollback_script_rolls_back_airflow_image_variable() -> None:
  """Rollback script applies terraform with airflow_image override."""
  script = Path("scripts/ci/airflow_rollback.sh").read_text(encoding="utf-8")

  assert "terraform -chdir=terraform apply" in script
  assert "airflow_image=" in script
