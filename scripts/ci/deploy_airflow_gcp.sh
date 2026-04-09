#!/usr/bin/env bash
set -euo pipefail

location_candidates=()

add_secret_version_if_changed() {
  local secret_id="$1"
  local secret_value="$2"

  local desired_b64
  desired_b64="$(printf '%s' "${secret_value}" | base64 | tr -d '\n')"

  local latest_b64=""
  if latest_b64="$(gcloud secrets versions access latest --project="${TF_VAR_project_id}" --secret="${secret_id}" 2>/dev/null | base64 | tr -d '\n')"; then
    if [[ "${latest_b64}" == "${desired_b64}" ]]; then
      echo "Secret ${secret_id} unchanged; skipping new secret version"
      return 0
    fi
  fi

  printf '%s' "${secret_value}" | gcloud secrets versions add "${secret_id}" --project="${TF_VAR_project_id}" --data-file=-
}

resolve_live_revision_name() {
  local service_name="$1"
  local service_json

  if ! service_json="$(gcloud run services describe "${service_name}" --project="${TF_VAR_project_id}" --region="${TF_VAR_region}" --format=json 2>/dev/null)"; then
    return 0
  fi

  printf '%s' "${service_json}" | jq -r '((.status.traffic // []) | map(select((.percent // 0) > 0))[0].revisionName) // .status.latestReadyRevisionName // empty'
}

resolve_revision_image() {
  local revision_name="$1"
  gcloud run revisions describe "${revision_name}" --project="${TF_VAR_project_id}" --region="${TF_VAR_region}" --format=json | jq -r '.spec.containers[0].image // empty'
}

resolve_revision_env() {
  local revision_name="$1"
  local env_name="$2"
  gcloud run revisions describe "${revision_name}" --project="${TF_VAR_project_id}" --region="${TF_VAR_region}" --format=json | jq -r --arg env_name "${env_name}" '.spec.containers[0].env[]? | select(.name == $env_name) | .value // empty'
}

derive_region_from_zone() {
  local zone="$1"
  if [[ "${zone}" != *-* ]]; then
    return 1
  fi

  printf '%s\n' "${zone%-*}"
}

parse_location_candidate() {
  local raw_candidate="$1"
  local trimmed_candidate
  local candidate_region=""
  local candidate_zone=""

  trimmed_candidate="$(printf '%s' "${raw_candidate}" | tr -d '[:space:]')"
  if [[ -z "${trimmed_candidate}" ]]; then
    return 1
  fi

  if [[ "${trimmed_candidate}" == */* ]]; then
    candidate_region="${trimmed_candidate%%/*}"
    candidate_zone="${trimmed_candidate##*/}"
  elif [[ "${trimmed_candidate}" == *:* ]]; then
    candidate_region="${trimmed_candidate%%:*}"
    candidate_zone="${trimmed_candidate##*:}"
  else
    candidate_zone="${trimmed_candidate}"
    candidate_region="$(derive_region_from_zone "${candidate_zone}")" || {
      echo "ERROR: Could not infer region from candidate zone '${trimmed_candidate}'" >&2
      return 2
    }
  fi

  if [[ -z "${candidate_region}" || -z "${candidate_zone}" ]]; then
    echo "ERROR: Invalid Airflow location candidate '${raw_candidate}'" >&2
    return 2
  fi

  printf '%s\t%s\n' "${candidate_region}" "${candidate_zone}"
}

location_candidate_seen() {
  local desired_candidate="$1"
  local existing_candidate

  for existing_candidate in "${location_candidates[@]-}"; do
    if [[ "${existing_candidate}" == "${desired_candidate}" ]]; then
      return 0
    fi
  done

  return 1
}

add_location_candidate() {
  local candidate_region="$1"
  local candidate_zone="$2"
  local encoded_candidate="${candidate_region}/${candidate_zone}"

  if ! location_candidate_seen "${encoded_candidate}"; then
    location_candidates+=("${encoded_candidate}")
  fi
}

build_location_candidates() {
  local parsed_candidate=""
  local candidate_region=""
  local candidate_zone=""
  local raw_candidate=""

  location_candidates=()
  add_location_candidate "${primary_airflow_region}" "${primary_airflow_zone}"

  if [[ -z "${AIRFLOW_LOCATION_CANDIDATES:-}" ]]; then
    return 0
  fi

  while IFS= read -r raw_candidate; do
    [[ -z "${raw_candidate}" ]] && continue

    parsed_candidate="$(parse_location_candidate "${raw_candidate}")"
    candidate_region="${parsed_candidate%%$'\t'*}"
    candidate_zone="${parsed_candidate#*$'\t'}"

    add_location_candidate "${candidate_region}" "${candidate_zone}"
  done < <(printf '%s\n' "${AIRFLOW_LOCATION_CANDIDATES}" | tr ',' '\n')
}

is_capacity_exhausted_log() {
  local log_path="$1"

  grep -Eq \
    "does not have enough resources available to fulfill the request|currently unavailable in the .* zone|ZONE_RESOURCE_POOL_EXHAUSTED|Try a different zone" \
    "${log_path}"
}

resolve_live_serving_values() {
  local backend_service_name="${TF_VAR_web_backend_service_name:-ticketforge-backend}"
  local frontend_service_name="${TF_VAR_web_frontend_service_name:-ticketforge-frontend}"
  local backend_live_revision=""
  local frontend_live_revision=""

  backend_live_revision="$(resolve_live_revision_name "${backend_service_name}")"
  frontend_live_revision="$(resolve_live_revision_name "${frontend_service_name}")"

  web_backend_image="${TF_VAR_web_backend_image:-}"
  if [[ -z "${web_backend_image}" && -n "${backend_live_revision}" ]]; then
    web_backend_image="$(resolve_revision_image "${backend_live_revision}")"
  fi

  web_backend_cors_origins="${TF_VAR_web_backend_cors_origins:-}"
  if [[ -z "${web_backend_cors_origins}" && -n "${backend_live_revision}" ]]; then
    web_backend_cors_origins="$(resolve_revision_env "${backend_live_revision}" "CORS_ORIGINS")"
  fi

  web_frontend_image="${TF_VAR_web_frontend_image:-}"
  if [[ -z "${web_frontend_image}" && -n "${frontend_live_revision}" ]]; then
    web_frontend_image="$(resolve_revision_image "${frontend_live_revision}")"
  fi

  web_frontend_api_url="${TF_VAR_web_frontend_api_url:-}"
  if [[ -z "${web_frontend_api_url}" && -n "${frontend_live_revision}" ]]; then
    web_frontend_api_url="$(resolve_revision_env "${frontend_live_revision}" "NEXT_PUBLIC_API_URL")"
  fi

  if [[ -z "${web_backend_image}" || -z "${web_backend_cors_origins}" || -z "${web_frontend_image}" || -z "${web_frontend_api_url}" ]]; then
    echo "ERROR: Could not resolve live serving values required to protect backend/frontend during Airflow deploy."
    echo "backend_image=${web_backend_image:-<missing>}"
    echo "backend_cors=${web_backend_cors_origins:-<missing>}"
    echo "frontend_image=${web_frontend_image:-<missing>}"
    echo "frontend_api_url=${web_frontend_api_url:-<missing>}"
    exit 1
  fi
}

run_secret_bootstrap_apply() {
  local target_region="$1"
  local target_zone="$2"

  terraform -chdir=terraform apply -auto-approve \
    -target="google_secret_manager_secret.airflow_runtime" \
    -var="project_id=${TF_VAR_project_id}" \
    -var="state_bucket=${TF_VAR_state_bucket}" \
    -var="region=${TF_VAR_region}" \
    -var="airflow_region=${target_region}" \
    -var="zone=${target_zone}" \
    -var="airflow_zone=${target_zone}" \
    -var="environment=prod"
}

run_full_apply() {
  local target_region="$1"
  local target_zone="$2"
  local apply_log="$3"

  terraform -chdir=terraform apply -auto-approve \
    -var="project_id=${TF_VAR_project_id}" \
    -var="state_bucket=${TF_VAR_state_bucket}" \
    -var="region=${TF_VAR_region}" \
    -var="airflow_region=${target_region}" \
    -var="zone=${target_zone}" \
    -var="airflow_zone=${target_zone}" \
    -var="environment=prod" \
    -var="web_backend_image=${web_backend_image}" \
    -var="web_backend_cors_origins=${web_backend_cors_origins}" \
    -var="web_frontend_image=${web_frontend_image}" \
    -var="web_frontend_api_url=${web_frontend_api_url}" \
    -var="airflow_repo_ref=${repo_ref}" > >(tee "${apply_log}") 2>&1
}

main() {
  local github_token_secret_id=""
  local gmail_username_secret_id=""
  local gmail_password_secret_id=""
  local github_token_value=""
  local import_region=""
  local import_zone=""
  local apply_log=""
  local candidate_index=0
  local candidate_count=0
  local current_candidate=""
  local current_region=""
  local current_zone=""
  local status=0

  : "${TF_VAR_project_id:?TF_VAR_project_id must be set in environment}"
  : "${TF_VAR_state_bucket:?TF_VAR_state_bucket must be set in environment}"
  : "${TF_VAR_region:?TF_VAR_region must be set in environment}"

  github_token_value="${AIRFLOW_GITHUB_TOKEN:-${GITHUB_TOKEN:-}}"
  : "${github_token_value:?AIRFLOW_GITHUB_TOKEN (or legacy GITHUB_TOKEN) must be set in environment}"
  : "${GMAIL_APP_USERNAME:?GMAIL_APP_USERNAME must be set in environment}"
  : "${GMAIL_APP_PASSWORD:?GMAIL_APP_PASSWORD must be set in environment}"

  github_token_secret_id="${TF_VAR_airflow_github_token_secret_id:-airflow-github-token-prod}"
  gmail_username_secret_id="${TF_VAR_airflow_gmail_app_username_secret_id:-airflow-gmail-app-username-prod}"
  gmail_password_secret_id="${TF_VAR_airflow_gmail_app_password_secret_id:-airflow-gmail-app-password-prod}"

  repo_ref="${AIRFLOW_REPO_REF:-$(git rev-parse HEAD)}"
  primary_airflow_region="${TF_VAR_airflow_region:-${TF_VAR_region}}"
  primary_airflow_zone="${TF_VAR_airflow_zone:-${TF_VAR_zone:-${primary_airflow_region}-c}}"

  if [[ "${repo_ref}" =~ ^[0-9a-f]{40}$ ]]; then
    if ! git ls-remote origin | awk '{print $1}' | grep -Fxq "${repo_ref}"; then
      echo "ERROR: ${repo_ref} is not available on origin."
      echo "Push this commit first, or set AIRFLOW_REPO_REF to a branch/tag/sha that exists on GitHub."
      exit 1
    fi
  fi

  build_location_candidates

  echo "Deploying Airflow from repo ref ${repo_ref}"
  echo "Airflow location candidates:"
  for current_candidate in "${location_candidates[@]}"; do
    echo "  - ${current_candidate}"
  done

  resolve_live_serving_values

  import_region="${primary_airflow_region}"
  import_zone="${primary_airflow_zone}"

  if gcloud secrets describe "${github_token_secret_id}" --project="${TF_VAR_project_id}" >/dev/null 2>&1; then
    terraform -chdir=terraform import -var="project_id=${TF_VAR_project_id}" -var="state_bucket=${TF_VAR_state_bucket}" -var="region=${TF_VAR_region}" -var="airflow_region=${import_region}" -var="zone=${import_zone}" -var="airflow_zone=${import_zone}" -var="environment=prod" 'google_secret_manager_secret.airflow_runtime["github_token"]' "projects/${TF_VAR_project_id}/secrets/${github_token_secret_id}" >/dev/null 2>&1 || true
  fi
  if gcloud secrets describe "${gmail_username_secret_id}" --project="${TF_VAR_project_id}" >/dev/null 2>&1; then
    terraform -chdir=terraform import -var="project_id=${TF_VAR_project_id}" -var="state_bucket=${TF_VAR_state_bucket}" -var="region=${TF_VAR_region}" -var="airflow_region=${import_region}" -var="zone=${import_zone}" -var="airflow_zone=${import_zone}" -var="environment=prod" 'google_secret_manager_secret.airflow_runtime["gmail_app_username"]' "projects/${TF_VAR_project_id}/secrets/${gmail_username_secret_id}" >/dev/null 2>&1 || true
  fi
  if gcloud secrets describe "${gmail_password_secret_id}" --project="${TF_VAR_project_id}" >/dev/null 2>&1; then
    terraform -chdir=terraform import -var="project_id=${TF_VAR_project_id}" -var="state_bucket=${TF_VAR_state_bucket}" -var="region=${TF_VAR_region}" -var="airflow_region=${import_region}" -var="zone=${import_zone}" -var="airflow_zone=${import_zone}" -var="environment=prod" 'google_secret_manager_secret.airflow_runtime["gmail_app_password"]' "projects/${TF_VAR_project_id}/secrets/${gmail_password_secret_id}" >/dev/null 2>&1 || true
  fi

  run_secret_bootstrap_apply "${import_region}" "${import_zone}"

  add_secret_version_if_changed "${github_token_secret_id}" "${github_token_value}"
  add_secret_version_if_changed "${gmail_username_secret_id}" "${GMAIL_APP_USERNAME}"
  add_secret_version_if_changed "${gmail_password_secret_id}" "${GMAIL_APP_PASSWORD}"

  candidate_count="${#location_candidates[@]}"
  candidate_index=0

  for current_candidate in "${location_candidates[@]}"; do
    candidate_index=$((candidate_index + 1))
    current_region="${current_candidate%%/*}"
    current_zone="${current_candidate##*/}"
    apply_log="$(mktemp -t airflow-terraform-apply.XXXXXX.log)"

    echo "Attempting Airflow deploy in ${current_region}/${current_zone} (${candidate_index}/${candidate_count})"

    if run_full_apply "${current_region}" "${current_zone}" "${apply_log}"; then
      rm -f "${apply_log}"
      return 0
    fi

    status=$?

    if is_capacity_exhausted_log "${apply_log}" && (( candidate_index < candidate_count )); then
      echo "Capacity exhausted in ${current_region}/${current_zone}; trying next candidate"
      rm -f "${apply_log}"
      continue
    fi

    if is_capacity_exhausted_log "${apply_log}"; then
      echo "Capacity exhausted in ${current_region}/${current_zone} and no candidates remain"
    else
      echo "Airflow deploy failed in ${current_region}/${current_zone} for a non-capacity reason"
    fi

    echo "Terraform apply log: ${apply_log}"
    exit "${status}"
  done
}

if [[ "${BASH_SOURCE[0]-}" == "${0}" ]]; then
  main "$@"
fi
