"""Tests for project ticket sizing behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

VALID_SIGNUP = {
    "username": "sizeuser",
    "first_name": "Size",
    "last_name": "Tester",
    "email": "size@example.com",
    "password": "SecurePass1",
}


async def _auth_headers(client) -> dict[str, str]:
    """Create a user and return bearer auth headers for API calls."""
    response = await client.post("/api/v1/auth/signup", json=VALID_SIGNUP)
    assert response.status_code == 201
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def _create_project(client, headers: dict[str, str]) -> dict:
    """Create a project using the API and return its JSON payload."""
    response = await client.post(
        "/api/v1/projects",
        headers=headers,
        json={"name": "Sizing Demo", "board_columns": [], "member_ids": []},
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.asyncio
async def test_create_ticket_auto_predicts_size(client) -> None:
    """New tickets should be auto-sized when no manual bucket is provided."""
    headers = await _auth_headers(client)
    project = await _create_project(client, headers)
    column_id = project["board_columns"][0]["id"]

    fake_prediction = SimpleNamespace(predicted_bucket="L", confidence=0.91)
    with patch(
        "web_backend.services.tickets.predict_ticket_size",
        new=AsyncMock(return_value=fake_prediction),
    ):
        response = await client.post(
            f"/api/v1/projects/{project['slug']}/tickets",
            headers=headers,
            json={"title": "Ship production sizing flow", "column_id": column_id},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["size_bucket"] == "L"
    assert payload["size_source"] == "predicted"
    assert payload["size_confidence"] == 0.91


@pytest.mark.asyncio
async def test_manual_ticket_size_skips_reprediction_on_update(client) -> None:
    """Manual ticket sizes should remain authoritative across edits."""
    headers = await _auth_headers(client)
    project = await _create_project(client, headers)
    column_id = project["board_columns"][0]["id"]

    create_response = await client.post(
        f"/api/v1/projects/{project['slug']}/tickets",
        headers=headers,
        json={
            "title": "Manual size ticket",
            "column_id": column_id,
            "size_bucket": "M",
        },
    )
    assert create_response.status_code == 201

    with patch(
        "web_backend.services.tickets.predict_ticket_size",
        new=AsyncMock(),
    ) as mock_predict:
        update_response = await client.patch(
            f"/api/v1/projects/{project['slug']}/tickets/"
            f"{create_response.json()['ticket_key']}",
            headers=headers,
            json={"title": "Manual size ticket updated"},
        )

    assert update_response.status_code == 200
    payload = update_response.json()
    assert payload["size_bucket"] == "M"
    assert payload["size_source"] == "manual"
    assert payload["size_confidence"] is None
    mock_predict.assert_not_called()


@pytest.mark.asyncio
async def test_clearing_manual_size_recomputes_prediction(client) -> None:
    """Switching back to auto mode should recompute and persist an AI size."""
    headers = await _auth_headers(client)
    project = await _create_project(client, headers)
    column_id = project["board_columns"][0]["id"]

    create_response = await client.post(
        f"/api/v1/projects/{project['slug']}/tickets",
        headers=headers,
        json={
            "title": "Manual size ticket",
            "column_id": column_id,
            "size_bucket": "S",
        },
    )
    assert create_response.status_code == 201
    ticket_key = create_response.json()["ticket_key"]

    fake_prediction = SimpleNamespace(predicted_bucket="XL", confidence=0.66)
    with patch(
        "web_backend.services.tickets.predict_ticket_size",
        new=AsyncMock(return_value=fake_prediction),
    ):
        response = await client.patch(
            f"/api/v1/projects/{project['slug']}/tickets/{ticket_key}",
            headers=headers,
            json={"size_bucket": None},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["size_bucket"] == "XL"
    assert payload["size_source"] == "predicted"
    assert payload["size_confidence"] == 0.66


@pytest.mark.asyncio
async def test_classify_missing_endpoint_backfills_unsized_tickets(client) -> None:
    """Batch sizing endpoint should backfill legacy unsized project tickets."""
    headers = await _auth_headers(client)
    project = await _create_project(client, headers)
    column_id = project["board_columns"][0]["id"]

    with patch(
        "web_backend.services.tickets.predict_ticket_size",
        new=AsyncMock(side_effect=RuntimeError("temporary inference outage")),
    ):
        create_response = await client.post(
            f"/api/v1/projects/{project['slug']}/tickets",
            headers=headers,
            json={"title": "Legacy unsized ticket", "column_id": column_id},
        )

    assert create_response.status_code == 201
    assert create_response.json()["size_bucket"] is None

    fake_prediction = SimpleNamespace(predicted_bucket="M", confidence=0.58)
    with patch(
        "web_backend.services.tickets.predict_ticket_size",
        new=AsyncMock(return_value=fake_prediction),
    ):
        response = await client.post(
            f"/api/v1/projects/{project['slug']}/tickets/classify-missing",
            headers=headers,
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated_count"] == 1
    assert payload["tickets"][0]["size_bucket"] == "M"
    assert payload["tickets"][0]["size_source"] == "predicted"
