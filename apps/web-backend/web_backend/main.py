"""Main FastAPI application."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from web_backend.routes.coldstart import router as coldstart_router

load_dotenv()

print("DATABASE_URL:", os.environ.get("DATABASE_URL"))

app = FastAPI(title="TicketForge Web Backend")

app.include_router(coldstart_router, prefix="/api/v1")


class ReadResponse(BaseModel):
    """Response model for the root endpoint."""

    message: str


@app.get("/")
def read_root() -> ReadResponse:
    """Return a simple greeting from the TicketForge backend."""
    return ReadResponse(message="Hello from TicketForge!")