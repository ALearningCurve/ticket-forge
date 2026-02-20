"""Main FastAPI application."""

from fastapi import FastAPI
from pydantic import BaseModel

from web_backend.routes.coldstart import router as coldstart_router

app = FastAPI(title="TicketForge Web Backend")

# -- Routes ---------------------------------------------------------------- #

app.include_router(coldstart_router, prefix="/api/v1")


class ReadResponse(BaseModel):
    """Response from "/" route."""

    message: str


@app.get("/")
def read_root() -> ReadResponse:
    """Root endpoint."""
    return ReadResponse(message="Hello from TicketForge!")