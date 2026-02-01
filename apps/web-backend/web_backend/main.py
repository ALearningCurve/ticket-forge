"""Main FastAPI application."""

from fastapi import FastAPI

app = FastAPI(title="TicketForge Web Backend")


@app.get("/")
def read_root():
  """Root endpoint."""
  return {"message": "Hello from TicketForge!"}
