#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <airflow-url>"
  exit 2
fi

AIRFLOW_URL="$1"
HEALTH_URL="${AIRFLOW_URL%/}/health"
DAGS_URL="${AIRFLOW_URL%/}/api/v1/dags"

attempt=0
max_attempts=3
sleep_seconds=5

while (( attempt < max_attempts )); do
  attempt=$((attempt + 1))
  echo "Smoketest attempt ${attempt}/${max_attempts}: ${HEALTH_URL}"

  if curl --fail --silent --show-error "$HEALTH_URL" >/dev/null; then
    echo "Health endpoint is reachable"

    if curl --fail --silent --show-error "$DAGS_URL" | grep -q '"dags"'; then
      echo "DAG API reachable and payload looks valid"
      exit 0
    fi

    echo "DAG API check failed"
  else
    echo "Health endpoint check failed"
  fi

  sleep "$sleep_seconds"
done

echo "Smoketest failed after ${max_attempts} attempts"
exit 1
