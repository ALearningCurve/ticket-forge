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

mlflow_service_name="${TF_VAR_mlflow_service_name:-mlflow-tracking}"
mlflow_repo="${TF_VAR_mlflow_artifact_registry_repository:-mlflow-repo}"
mlflow_tag="${TF_VAR_mlflow_image_tag:-v3.10.0}"
mlflow_image="${TF_VAR_region}-docker.pkg.dev/${TF_VAR_project_id}/${mlflow_repo}/mlflow-gcp:${mlflow_tag}"

append_output() {
  local key="$1"
  local value="$2"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf '%s=%s\n' "${key}" "${value}" >> "${GITHUB_OUTPUT}"
  fi
}

service_status_json() {
  gcloud run services describe "${mlflow_service_name}" \
    --project="${TF_VAR_project_id}" \
    --region="${TF_VAR_region}" \
    --format=json 2>/dev/null || true
}

service_url() {
  local service_json="$1"
  SERVICE_JSON="${service_json}" python3 - <<'PY'
import json
import os

raw = os.environ.get("SERVICE_JSON", "").strip()
if not raw:
  raise SystemExit(0)
service = json.loads(raw)
print((service.get("status") or {}).get("url", ""))
PY
}

service_ready() {
  local service_json="$1"
  SERVICE_JSON="${service_json}" python3 - <<'PY'
import json
import os

raw = os.environ.get("SERVICE_JSON", "").strip()
if not raw:
  raise SystemExit(1)
service = json.loads(raw)
conditions = (service.get("status") or {}).get("conditions") or []
ready = next((item for item in conditions if item.get("type") == "Ready"), None)
raise SystemExit(0 if ready and ready.get("status") == "True" else 1)
PY
}

service_has_public_invoker() {
  local policy_json
  policy_json="$(gcloud run services get-iam-policy "${mlflow_service_name}" \
    --project="${TF_VAR_project_id}" \
    --region="${TF_VAR_region}" \
    --format=json 2>/dev/null || echo '{}')"

  POLICY_JSON="${policy_json}" python3 - <<'PY'
import json
import os

policy = json.loads(os.environ.get("POLICY_JSON", "{}") or "{}")
bindings = policy.get("bindings") or []
for binding in bindings:
    if binding.get("role") != "roles/run.invoker":
        continue
    if "allUsers" in (binding.get("members") or []):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

ensure_mlflow_repo() {
  gcloud services enable artifactregistry.googleapis.com --project="${TF_VAR_project_id}" >/dev/null

  if gcloud artifacts repositories describe "${mlflow_repo}" \
    --project="${TF_VAR_project_id}" \
    --location="${TF_VAR_region}" >/dev/null 2>&1; then
    return 0
  fi

  gcloud artifacts repositories create "${mlflow_repo}" \
    --project="${TF_VAR_project_id}" \
    --location="${TF_VAR_region}" \
    --repository-format=docker \
    --description="MLflow server images"
}

build_mlflow_image() {
  gcloud auth configure-docker "${TF_VAR_region}-docker.pkg.dev" --quiet >/dev/null

  printf '%s\n' \
    "FROM ghcr.io/mlflow/mlflow:${mlflow_tag}-full" \
    "RUN pip install --no-cache-dir google-cloud-storage Flask-WTF" \
    | docker buildx build \
      --platform linux/amd64 \
      --provenance=false \
      --push \
      -t "${mlflow_image}" \
      -f- .
}

terraform_apply_mlflow() {
  local log_file
  log_file="$(mktemp)"

  if terraform -chdir=terraform apply -auto-approve \
    -lock-timeout=180s \
    -target=google_cloud_run_v2_service.mlflow \
    -target=google_cloud_run_v2_service_iam_member.public_invoker \
    -var="project_id=${TF_VAR_project_id}" \
    -var="state_bucket=${TF_VAR_state_bucket}" \
    -var="region=${TF_VAR_region}" \
    -var="environment=${TF_VAR_environment}" \
    -var="mlflow_server_image=${mlflow_image}" 2>&1 | tee "${log_file}"; then
    rm -f "${log_file}"
    return 0
  fi

  cat "${log_file}" >&2
  rm -f "${log_file}"
  return 1
}

service_exists() {
  gcloud run services describe "${mlflow_service_name}" \
    --project="${TF_VAR_project_id}" \
    --region="${TF_VAR_region}" >/dev/null 2>&1
}

for attempt in 1 2 3; do
  service_json="$(service_status_json)"
  service_is_ready="false"
  if [[ -n "${service_json}" ]] && service_ready "${service_json}"; then
    service_is_ready="true"
  fi

  if [[ "${service_is_ready}" == "true" ]] && service_has_public_invoker; then
    url="$(service_url "${service_json}")"
    if [[ -n "${url}" ]]; then
      append_output "tracking_uri" "${url}"
      append_output "service_name" "${mlflow_service_name}"
      append_output "image" "${mlflow_image}"
      exit 0
    fi
  fi

  ensure_mlflow_repo

  if [[ "${service_is_ready}" != "true" ]]; then
    build_mlflow_image

    if service_exists; then
      gcloud run services update "${mlflow_service_name}" \
        --project="${TF_VAR_project_id}" \
        --region="${TF_VAR_region}" \
        --image="${mlflow_image}" \
        --quiet
    fi
  fi

  terraform_apply_mlflow

  sleep $((attempt * 10))
done

final_service_json="$(service_status_json)"
if [[ -n "${final_service_json}" ]] && service_ready "${final_service_json}" && service_has_public_invoker; then
  final_url="$(service_url "${final_service_json}")"
  if [[ -n "${final_url}" ]]; then
    append_output "tracking_uri" "${final_url}"
    append_output "service_name" "${mlflow_service_name}"
    append_output "image" "${mlflow_image}"
    exit 0
  fi
fi

echo "MLflow tracking service is not ready after recovery attempts" >&2
exit 1
