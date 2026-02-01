"""Main FastAPI application."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TicketForge Web Backend")


class ReadResponse(BaseModel):
  """Response from "/" route."""

  message: str


@app.get("/")
def read_root() -> ReadResponse:
  """Root endpoint."""
  return ReadResponse(message="Hello from TicketForge!")
