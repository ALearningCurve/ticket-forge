"""Main FastAPI application."""
from dotenv import load_dotenv
load_dotenv()

import os
print("DATABASE_URL:", os.environ.get("DATABASE_URL"))

from fastapi import FastAPI
from pydantic import BaseModel

from web_backend.routes.coldstart import router as coldstart_router

app = FastAPI(title="TicketForge Web Backend")

app.include_router(coldstart_router, prefix="/api/v1")


class ReadResponse(BaseModel):
    message: str


@app.get("/")
def read_root() -> ReadResponse:
    return ReadResponse(message="Hello from TicketForge!")