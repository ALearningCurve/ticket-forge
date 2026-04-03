#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <db-host> <db-user> <db-name>"
  exit 2
fi

DB_HOST="$1"
DB_USER="$2"
DB_NAME="$3"

export PGPASSWORD="${PGPASSWORD:-}"

if [[ -z "$PGPASSWORD" ]]; then
  echo "PGPASSWORD env var must be set"
  exit 2
fi

before_count=$(psql "host=${DB_HOST} user=${DB_USER} dbname=${DB_NAME}" -Atc "select count(*) from dag_run;")
echo "dag_run rows before restart check: ${before_count}"

after_count=$(psql "host=${DB_HOST} user=${DB_USER} dbname=${DB_NAME}" -Atc "select count(*) from dag_run;")
echo "dag_run rows after restart check: ${after_count}"

if [[ "$after_count" -lt "$before_count" ]]; then
  echo "Persistence check failed: dag_run count regressed"
  exit 1
fi

echo "Persistence check passed"
