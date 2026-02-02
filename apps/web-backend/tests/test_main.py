"""Tests for main application."""

from fastapi.testclient import TestClient
from web_backend.main import app

client = TestClient(app)


def test_read_root() -> None:
  """Test the root endpoint. Dummy test to be removed."""
  response = client.get("/")
  assert response.status_code == 200
  assert response.json() == {"message": "Hello from TicketForge!"}
