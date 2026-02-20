# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

# Define which app we are building (defaults to web-backend)
ARG APP_NAME=web-backend

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# 1. Copy Workspace Metadata
# We copy all pyprojects because uv needs the full tree to resolve the workspace
COPY pyproject.toml uv.lock ./
COPY apps/ apps/
COPY libs/ libs/

# Remove source code but keep pyproject.toml files to cache dependency install
# This trick ensures we only re-install if dependencies change, not code.
RUN find apps -type f ! -name 'pyproject.toml' -delete && \
    find libs -type f ! -name 'pyproject.toml' -delete

# 2. Install dependencies (Third-party)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package ${APP_NAME}

# 3. Copy actual source code back in
COPY apps/${APP_NAME} ./apps/${APP_NAME}
COPY libs/ ./libs/

# 4. Final Sync (Install local packages)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package ${APP_NAME}


# --- Stage 2: Runtime ---
FROM python:3.12-slim AS runtime

ARG APP_NAME=web-backend
# Set an env var so the app knows its own name if needed
ENV APP_NAME=${APP_NAME}

WORKDIR /app

# Copy venv and code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/apps/${APP_NAME} ./apps/${APP_NAME}
COPY --from=builder /app/libs ./libs

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python"]
