#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <airflow-url>"
  exit 2
fi

AIRFLOW_URL="$1"
HEALTH_URL="${AIRFLOW_URL%/}/health"
DAGS_URL="${AIRFLOW_URL%/}/api/v1/dags"

AIRFLOW_SMOKETEST_USERNAME="${AIRFLOW_SMOKETEST_USERNAME:-}"
AIRFLOW_SMOKETEST_PASSWORD="${AIRFLOW_SMOKETEST_PASSWORD:-}"

DEFAULT_AIRFLOW_IAP_INSTANCE="airflow-vm-prod"
DEFAULT_AIRFLOW_IAP_ZONE="${TF_VAR_airflow_zone:-${TF_VAR_zone:-us-east1-c}}"
AIRFLOW_IAP_INSTANCE="${AIRFLOW_IAP_INSTANCE:-}"
AIRFLOW_IAP_ZONE="${AIRFLOW_IAP_ZONE:-}"
AIRFLOW_IAP_LOCAL_PORT="${AIRFLOW_IAP_LOCAL_PORT:-18080}"

AIRFLOW_SMOKETEST_MAX_ATTEMPTS="${AIRFLOW_SMOKETEST_MAX_ATTEMPTS:-24}"
AIRFLOW_SMOKETEST_SLEEP_SECONDS="${AIRFLOW_SMOKETEST_SLEEP_SECONDS:-10}"
AIRFLOW_IAP_START_MAX_ATTEMPTS="${AIRFLOW_IAP_START_MAX_ATTEMPTS:-30}"
AIRFLOW_IAP_RETRY_SLEEP_SECONDS="${AIRFLOW_IAP_RETRY_SLEEP_SECONDS:-10}"
AIRFLOW_IAP_READY_WAIT_ATTEMPTS="${AIRFLOW_IAP_READY_WAIT_ATTEMPTS:-20}"

tunnel_pid=""

cleanup() {
  if [[ -n "$tunnel_pid" ]] && kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    kill "$tunnel_pid" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

resolve_iap_target_from_terraform() {
  local terraform_instance=""
  local terraform_zone=""

  if command -v terraform >/dev/null 2>&1 && [[ -d terraform ]]; then
    terraform_instance="$(terraform -chdir=terraform output -raw airflow_vm_instance_name 2>/dev/null || true)"
    terraform_zone="$(terraform -chdir=terraform output -raw airflow_vm_zone 2>/dev/null || true)"
  fi

  if [[ -n "${terraform_instance}" ]]; then
    AIRFLOW_IAP_INSTANCE="${terraform_instance}"
  fi

  if [[ -n "${terraform_zone}" ]]; then
    AIRFLOW_IAP_ZONE="${terraform_zone}"
  fi

  AIRFLOW_IAP_INSTANCE="${AIRFLOW_IAP_INSTANCE:-${DEFAULT_AIRFLOW_IAP_INSTANCE}}"
  AIRFLOW_IAP_ZONE="${AIRFLOW_IAP_ZONE:-${DEFAULT_AIRFLOW_IAP_ZONE}}"
}

set_local_urls() {
  AIRFLOW_URL="http://127.0.0.1:${AIRFLOW_IAP_LOCAL_PORT}"
  HEALTH_URL="${AIRFLOW_URL%/}/health"
  DAGS_URL="${AIRFLOW_URL%/}/api/v1/dags"
}

wait_for_tunnel_ready() {
  local i=0
  while (( i < AIRFLOW_IAP_READY_WAIT_ATTEMPTS )); do
    i=$((i + 1))

    if curl --silent --max-time 2 "http://127.0.0.1:${AIRFLOW_IAP_LOCAL_PORT}/health" >/dev/null 2>&1; then
      set_local_urls
      return 0
    fi

    if [[ -n "$tunnel_pid" ]] && ! kill -0 "$tunnel_pid" >/dev/null 2>&1; then
      return 1
    fi

    sleep 1
  done

  return 1
}

stop_tunnel_if_running() {
  if [[ -n "$tunnel_pid" ]] && kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    kill "$tunnel_pid" >/dev/null 2>&1 || true
  fi
  tunnel_pid=""
}

start_direct_iap_tunnel() {
  gcloud compute start-iap-tunnel "$AIRFLOW_IAP_INSTANCE" 8080 \
    --zone "$AIRFLOW_IAP_ZONE" \
    --local-host-port="127.0.0.1:${AIRFLOW_IAP_LOCAL_PORT}" \
    >/tmp/airflow_iap_tunnel.log 2>&1 &
  tunnel_pid=$!

  if wait_for_tunnel_ready; then
    echo "IAP tunnel ready; using ${AIRFLOW_URL}"
    return 0
  fi

  stop_tunnel_if_running

  if [[ -f /tmp/airflow_iap_tunnel.log ]]; then
    tail -n 20 /tmp/airflow_iap_tunnel.log || true
  fi

  return 1
}

start_ssh_iap_tunnel() {
  gcloud compute ssh "$AIRFLOW_IAP_INSTANCE" \
    --zone "$AIRFLOW_IAP_ZONE" \
    --tunnel-through-iap \
    --quiet \
    -- -N -L "127.0.0.1:${AIRFLOW_IAP_LOCAL_PORT}:127.0.0.1:8080" \
    >/tmp/airflow_iap_ssh_tunnel.log 2>&1 &
  tunnel_pid=$!

  if wait_for_tunnel_ready; then
    echo "IAP SSH tunnel ready; using ${AIRFLOW_URL}"
    return 0
  fi

  stop_tunnel_if_running

  if [[ -f /tmp/airflow_iap_ssh_tunnel.log ]]; then
    tail -n 20 /tmp/airflow_iap_ssh_tunnel.log || true
  fi

  return 1
}

host_from_url() {
  local url="$1"
  local without_scheme="${url#*://}"
  local host_port="${without_scheme%%/*}"
  echo "${host_port%%:*}"
}

is_private_ipv4() {
  local host="$1"
  [[ "$host" =~ ^10\. ]] || [[ "$host" =~ ^192\.168\. ]] || [[ "$host" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]]
}

start_iap_tunnel_if_needed() {
  local host
  host="$(host_from_url "$AIRFLOW_URL")"

  if ! is_private_ipv4 "$host"; then
    return 0
  fi

  if ! command -v gcloud >/dev/null 2>&1; then
    echo "Target ${host} is private, but gcloud is not available for IAP tunneling"
    return 1
  fi

  local start_attempt=0
  while (( start_attempt < AIRFLOW_IAP_START_MAX_ATTEMPTS )); do
    start_attempt=$((start_attempt + 1))
    echo "Detected private Airflow URL (${host}); starting IAP tunnel via ${AIRFLOW_IAP_INSTANCE}/${AIRFLOW_IAP_ZONE} (attempt ${start_attempt}/${AIRFLOW_IAP_START_MAX_ATTEMPTS})"

    if start_direct_iap_tunnel; then
      return 0
    fi

    echo "Direct IAP port tunnel failed; trying SSH-over-IAP tunnel fallback"
    if start_ssh_iap_tunnel; then
      return 0
    fi

    if (( start_attempt < AIRFLOW_IAP_START_MAX_ATTEMPTS )); then
      echo "Retrying IAP tunnel in ${AIRFLOW_IAP_RETRY_SLEEP_SECONDS}s..."
      sleep "$AIRFLOW_IAP_RETRY_SLEEP_SECONDS"
    fi
  done

  echo "Timed out waiting for IAP tunnel to become ready"
  return 1
}

attempt=0
max_attempts="$AIRFLOW_SMOKETEST_MAX_ATTEMPTS"
sleep_seconds="$AIRFLOW_SMOKETEST_SLEEP_SECONDS"

resolve_iap_target_from_terraform
start_iap_tunnel_if_needed

while (( attempt < max_attempts )); do
  attempt=$((attempt + 1))
  echo "Smoketest attempt ${attempt}/${max_attempts}: ${HEALTH_URL}"

  if curl --fail --silent --show-error "$HEALTH_URL" >/dev/null; then
    echo "Health endpoint is reachable"

    if [[ -n "$AIRFLOW_SMOKETEST_USERNAME" ]] && [[ -n "$AIRFLOW_SMOKETEST_PASSWORD" ]]; then
      http_code="$(curl --silent --show-error --output /tmp/airflow_dags_response.json --write-out '%{http_code}' --user "${AIRFLOW_SMOKETEST_USERNAME}:${AIRFLOW_SMOKETEST_PASSWORD}" "$DAGS_URL" || true)"
    else
      http_code="$(curl --silent --show-error --output /tmp/airflow_dags_response.json --write-out '%{http_code}' "$DAGS_URL" || true)"
    fi

    if [[ "$http_code" == "200" ]] && grep -q '"dags"' /tmp/airflow_dags_response.json; then
      echo "DAG API reachable and payload looks valid"
      exit 0
    fi

    if [[ "$http_code" == "401" ]] && [[ -z "$AIRFLOW_SMOKETEST_USERNAME" ]]; then
      echo "DAG API reachable but requires authentication (401)"
      exit 0
    fi

    echo "DAG API check failed (HTTP ${http_code})"
  else
    echo "Health endpoint check failed"
  fi

  sleep "$sleep_seconds"
done

echo "Smoketest failed after ${max_attempts} attempts"
exit 1
