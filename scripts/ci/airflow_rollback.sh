#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <previous_airflow_image>"
  exit 2
fi

PREVIOUS_IMAGE="$1"

if [[ -z "${PREVIOUS_IMAGE}" ]]; then
  echo "PREVIOUS_IMAGE is required"
  exit 2
fi

echo "Rolling back Airflow image to ${PREVIOUS_IMAGE}"
terraform -chdir=terraform apply -auto-approve -var "airflow_image=${PREVIOUS_IMAGE}"

echo "Rollback apply complete"
