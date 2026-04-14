"""Tests for serving-monitoring helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from training.cmd.monitor_model import (
  _fetch_serving_records,
  _resolve_backend_url,
  _resolve_bucket_name,
)


def test_resolve_bucket_name_from_explicit_uri() -> None:
  """Explicit bucket URIs should resolve to the bucket name."""
  assert _resolve_bucket_name("gs://ticketforge-monitoring-prod/reports") == (
    "ticketforge-monitoring-prod"
  )


def test_resolve_backend_url_prefers_explicit_value() -> None:
  """An explicit backend URL should bypass env and gcloud lookup."""
  with patch("training.cmd.monitor_model.subprocess.run") as mock_run:
    resolved = _resolve_backend_url("https://backend.example.run.app/")

  assert resolved == "https://backend.example.run.app"
  mock_run.assert_not_called()


def test_resolve_backend_url_falls_back_to_gcloud() -> None:
  """Backend URL should resolve via gcloud when args/env are absent."""
  completed = MagicMock()
  completed.returncode = 0
  completed.stdout = "https://ticketforge-backend.run.app\n"
  completed.stderr = ""

  with patch.dict("os.environ", {}, clear=False):
    with patch(
      "training.cmd.monitor_model.subprocess.run",
      return_value=completed,
    ) as mock_run:
      resolved = _resolve_backend_url(None)

  assert resolved == "https://ticketforge-backend.run.app"
  assert mock_run.call_count == 1


def test_resolve_bucket_name_requires_gs_uri() -> None:
  """Invalid monitoring bucket values should fail fast."""
  with pytest.raises(ValueError, match="Invalid bucket URI"):
    _resolve_bucket_name("ticketforge-monitoring-prod")


def test_fetch_serving_records_returns_records_on_success() -> None:
  """Successful fetch should return parsed records."""
  mock_response = MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = [{"predicted_bucket": "M", "confidence": 0.85}]
  mock_response.raise_for_status = MagicMock()

  with patch("training.cmd.monitor_model.httpx.get", return_value=mock_response):
    records = _fetch_serving_records("https://backend.example.run.app", limit=10)

  assert len(records) == 1
  assert records[0]["predicted_bucket"] == "M"


def test_fetch_serving_records_retries_on_timeout() -> None:
  """Timeout on first attempt should retry and succeed on second."""
  mock_response = MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = [{"id": 1}]
  mock_response.raise_for_status = MagicMock()

  with (
    patch(
      "training.cmd.monitor_model.httpx.get",
      side_effect=[httpx.ReadTimeout("timed out"), mock_response],
    ),
    patch("time.sleep"),
  ):
    records = _fetch_serving_records("https://backend.example.run.app", limit=5)

  assert len(records) == 1


def test_fetch_serving_records_raises_after_all_retries_fail() -> None:
  """All retries exhausted should raise so the workflow fails visibly."""
  with (
    patch(
      "training.cmd.monitor_model.httpx.get",
      side_effect=httpx.ReadTimeout("timed out"),
    ),
    patch("time.sleep"),
    pytest.raises(RuntimeError, match="All 3 attempts"),
  ):
    _fetch_serving_records("https://backend.example.run.app", limit=5)


def test_fetch_serving_records_raises_on_connect_error() -> None:
  """Persistent connection errors should raise after retries."""
  with (
    patch(
      "training.cmd.monitor_model.httpx.get",
      side_effect=httpx.ConnectError("connection refused"),
    ),
    patch("time.sleep"),
    pytest.raises(RuntimeError, match="All 3 attempts"),
  ):
    _fetch_serving_records("https://backend.example.run.app", limit=5)
