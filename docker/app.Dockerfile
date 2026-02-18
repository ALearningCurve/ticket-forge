FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps for psycopg2 and basic imaging/ocr support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev git curl tesseract-ocr poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for the coldstart job
COPY docker/pyproject.toml /app/pyproject.toml
# Install dependencies listed in docker/pyproject.toml without requiring Poetry
RUN python - <<'PY'
import tomllib, subprocess, sys
data = tomllib.loads(open('pyproject.toml','rb').read().decode('utf-8'))
deps = data.get('project', {}).get('dependencies', [])
if not deps:
    sys.exit(0)
cmd = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + deps
subprocess.check_call(cmd)
PY

# Copy the repository (adjust if you prefer a smaller context)
COPY . /app

# Default command: run coldstart against the compose postgres service
CMD ["python", "apps/training/training/etl/ingest/resume/coldstart.py", "--pg", "--dsn", "postgresql://ticketforge:ticketforge@postgres:5432/ticketforge"]
