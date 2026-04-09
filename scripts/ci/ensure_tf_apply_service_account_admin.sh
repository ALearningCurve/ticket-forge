#!/usr/bin/env bash

set -euo pipefail

required_env=(
  TF_VAR_project_id
  TF_VAR_state_bucket
  TF_VAR_region
  TF_VAR_environment
)

for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
done

target='google_project_iam_member.tf_apply_permissions["roles/iam.serviceAccountAdmin"]'

for attempt in 1 2 3; do
  log_file="$(mktemp)"

  if terraform -chdir=terraform apply -auto-approve \
    -target="${target}" \
    -var="project_id=${TF_VAR_project_id}" \
    -var="state_bucket=${TF_VAR_state_bucket}" \
    -var="region=${TF_VAR_region}" \
    -var="environment=${TF_VAR_environment}" 2>&1 | tee "${log_file}"; then
    # Give IAM a few seconds to propagate before later Terraform/GCP calls.
    sleep 10
    rm -f "${log_file}"
    exit 0
  fi

  if grep -qi "concurrent policy changes" "${log_file}" && (( attempt < 3 )); then
    sleep $((attempt * 5))
    rm -f "${log_file}"
    continue
  fi

  rm -f "${log_file}"
  exit 1
done
