#!/bin/bash
set -e

# Wait for database to be ready with retries
MAX_RETRIES=30
RETRY_COUNT=0

echo "Waiting for database to be ready..."
while ! airflow db check 2>/dev/null; do
  RETRY_COUNT=$((RETRY_COUNT + 1))
  if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "ERROR: Database failed to become ready after $MAX_RETRIES attempts"
    exit 1
  fi
  echo "Database not ready yet, retrying... ($RETRY_COUNT/$MAX_RETRIES)"
  sleep 2
done

echo "Database is ready, initializing Airflow..."

# Initialize database (only runs migrations if needed)
airflow db migrate

# Create default connections
airflow connections create-default-connections

# Create admin user (ignore if already exists)
airflow users create \
  --username airflow \
  --password airflow \
  --firstname Air \
  --lastname Flow \
  --role Admin \
  --email airflow@example.com 2>/dev/null || true

echo "Airflow initialization complete"

# Start Airflow scheduler in background and webserver in foreground
airflow scheduler &
exec airflow webserver
