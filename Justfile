set dotenv-load := true

# Configure repository and install dependencies
default: install-deps

# install all 3rd party packages
[group('lang-agnostic')]
install-deps:
    uv sync --all-packages
    npm i
    uv run pre-commit install
    uv tool install 'dvc[gs]'

# Runs python tests. Any args are forwarded to pytest.
[group('python')]
[positional-arguments]
pytest *args='':
    uv run pytest "$@"

# Runs python linting. Specify the directories/files to lint as positional args.
[group('python')]
[positional-arguments]
pylint *args=".":
    uv run ruff check --fix "$@"
    uv run ruff format "$@"
    # uv run pyright "$@"

# Run all python checks on particular files and directories
[group('python')]
pycheck *args=".":
    just pylint "$@"
    just pytest "$@"

# Run pre-commit hooks
[group('lang-agnostic')]
[positional-arguments]
precommit *args='run':
    uv run pre-commit "$@"

# runs all checks on the repo from repo-root
[group('lang-agnostic')]
check:
    just pycheck .

    @if [ -d "terraform/.terraform" ]; then \
        just tf-check; \
    else \
        echo "tf not initialized "; \
    fi

# runs the training script
[group('data-pipeline')]
[positional-arguments]
train *args='':
    uv run python -m training.cmd.train {{ args }}

# runs the CI training script
[group('data-pipeline')]
[positional-arguments]
train-with-gates *args='':
    uv run python -m training.cmd.train_with_gates {{ args }}

# initializes terraform
[group('ops')]
tf-init:
    terraform -chdir=terraform init -backend-config="bucket=${TF_VAR_state_bucket}"

# format terraform
[group('ops')]
tf-lint:
    terraform -chdir=terraform fmt -recursive

# assert good linting
[group('ops')]
tf-check:
    terraform -chdir=terraform fmt -check -recursive
    terraform -chdir=terraform validate

# runs terraform plan
[group('ops')]
[positional-arguments]
tf-plan *args='':
    terraform -chdir=terraform plan {{ args }}

# runs terraform apply
[group('ops')]
[positional-arguments]
tf-apply *args='':
    terraform -chdir=terraform apply {{ args }}

# runs arbitrary terraform command
[group('ops')]
[positional-arguments]
tf *args='':
    terraform -chdir=terraform {{ args }}

# get workload identity federation provider ID for GitHub Actions integration
[group('ops')]
gcp-get-wif-provider:
    @gcloud iam workload-identity-pools providers describe github-provider \
        --project=${TF_VAR_project_id} --location="global" --workload-identity-pool=github-actions-pool \
        --format="value(name)"

# gets the GitHub repository ID for a given repo (defaults to this repo)
[group('ops')]
get-repo-id repo='alearningcurve/ticket-forge':
    @gh api -H "Accept: application/vnd.github+json" repos/{{ repo }} | jq .id

# Build Cloud Run-oriented images locally (matches CI docker-build-apps job)
[group('ops')]
docker-build-apps:
    #!/usr/bin/env bash
    set -euo pipefail
    docker build -f docker/base.Dockerfile --target cloudrun-web-backend --build-arg APP_NAME=web-backend -t ticketforge-api:local .
    docker build -f docker/inference.Dockerfile -t ticketforge-inference:local .
    docker build -f docker/frontend.Dockerfile --build-arg NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 -t ticketforge-web:local .

[group('ops')]
tf-build-push-mlflow-image repo='mlflow-repo' tag='v3.10.0':
    @: "${TF_VAR_project_id:?TF_VAR_project_id must be set in .env}"
    @: "${TF_VAR_region:?TF_VAR_region must be set in .env}"
    gcloud services enable artifactregistry.googleapis.com --project="${TF_VAR_project_id}"
    @if ! gcloud artifacts repositories describe "{{ repo }}" --location="${TF_VAR_region}" --project="${TF_VAR_project_id}" >/dev/null 2>&1; then \
        gcloud artifacts repositories create "{{ repo }}" --repository-format=docker --location="${TF_VAR_region}" --description="MLflow server images" --project="${TF_VAR_project_id}"; \
    fi
    gcloud auth configure-docker "${TF_VAR_region}-docker.pkg.dev"
    printf '%s\n' "FROM ghcr.io/mlflow/mlflow:{{ tag }}-full" "RUN pip install --no-cache-dir google-cloud-storage Flask-WTF" | docker build -t "${TF_VAR_region}-docker.pkg.dev/${TF_VAR_project_id}/{{ repo }}/mlflow-gcp:{{ tag }}" -f- .
    docker push "${TF_VAR_region}-docker.pkg.dev/${TF_VAR_project_id}/{{ repo }}/mlflow-gcp:{{ tag }}"

# starts airflow docker environment
[group('airflow')]
airflow-up:
    chmod +777 ./data ./models
    docker compose up -d --build postgres pgadmin airflow

# runs airflow smoketest script against provided URL (local or deployed)
[group('ops')]
gcp-airflow-smoketest url:
    bash scripts/ci/airflow_smoketest.sh {{ url }}

# runs backend smoketest script against provided URL
[group('ops')]
gcp-backend-smoketest url expected_model_version='':
    bash scripts/ci/backend_smoketest.sh {{ url }} {{ expected_model_version }}

# runs frontend smoketest script against provided URL
[group('ops')]
gcp-frontend-smoketest url:
    bash scripts/ci/frontend_smoketest.sh {{ url }}

# dispatches the Serving Deploy workflow and optionally watches it to completion
[group('ops')]
gcp-serving-deploy deploy_backend='true' deploy_frontend='true' serving_model_version='' ref='main' watch='true':
    DEPLOY_BACKEND={{ deploy_backend }} \
    DEPLOY_FRONTEND={{ deploy_frontend }} \
    SERVING_MODEL_VERSION="{{ serving_model_version }}" \
    DEPLOY_REF="{{ ref }}" \
    WATCH_SERVING_DEPLOY={{ watch }} \
    bash scripts/ci/deploy_serving.sh

# validates required workflows, scripts, and reports before submission
[group('ops')]
verify-submission-ready:
    bash scripts/ci/verify_submission_ready.sh

# applies ticketforge Postgres schema init scripts through Cloud SQL Auth Proxy
[group('ops')]
gcp-ticketforge-schema-init:
    bash scripts/ci/apply_ticketforge_schema_init.sh

# triggers an Airflow DAG run through the REST API (supports private URL via IAP tunnel)
[group('ops')]
[positional-arguments]
gcp-airflow-trigger url dag_id conf_json='{}' run_id='':
    #!/usr/bin/env bash
    set -euo pipefail

    if [[ -n "{{ run_id }}" ]]; then
        bash scripts/ci/airflow_trigger_dag.sh "{{ url }}" "{{ dag_id }}" '{{ conf_json }}' "{{ run_id }}"
    else
        bash scripts/ci/airflow_trigger_dag.sh "{{ url }}" "{{ dag_id }}" '{{ conf_json }}'
    fi


# creates/updates Airflow deployment on GCP with Terraform
[group('ops')]
gcp-airflow-deploy:
    bash scripts/ci/deploy_airflow_gcp.sh

# opens proxy tunnel to GCP resources (e.g. Airflow webserver, Cloud SQL instances)
[group('ops')]
gcp-proxy target local_port='18080':
        #!/usr/bin/env bash
        set -euo pipefail

        : "${TF_VAR_project_id:?TF_VAR_project_id must be set in environment}"

        case "{{ target }}" in
            airflow)
                zone="$(terraform -chdir=terraform output -raw airflow_vm_zone 2>/dev/null || true)"
                instance="$(terraform -chdir=terraform output -raw airflow_vm_instance_name 2>/dev/null || true)"
                zone="${zone:-${TF_VAR_airflow_zone:-${TF_VAR_zone:-us-east1-c}}}"
                instance="${instance:-${AIRFLOW_VM_NAME:-airflow-vm-prod}}"
                echo "Opening Airflow IAP tunnel: 127.0.0.1:{{ local_port }} -> ${instance}:8080 (${zone})"
                exec gcloud compute start-iap-tunnel "${instance}" 8080 \
                    --zone="${zone}" \
                    --local-host-port="127.0.0.1:{{ local_port }}"
                ;;
            cloud-sql|cloudsql|sql)
                zone="$(terraform -chdir=terraform output -raw airflow_vm_zone 2>/dev/null || true)"
                instance="$(terraform -chdir=terraform output -raw airflow_vm_instance_name 2>/dev/null || true)"
                zone="${zone:-${TF_VAR_airflow_zone:-${TF_VAR_zone:-us-east1-c}}}"
                instance="${instance:-${AIRFLOW_VM_NAME:-airflow-vm-prod}}"
                cloud_sql_private_ip="$(terraform -chdir=terraform output -raw cloud_sql_private_ip 2>/dev/null || true)"
                if [[ -z "${cloud_sql_private_ip}" ]]; then
                    echo "Could not resolve cloud_sql_private_ip from Terraform outputs."
                    echo "Run: just tf output cloud_sql_private_ip"
                    exit 1
                fi
                echo "Opening Cloud SQL IAP relay: 127.0.0.1:{{ local_port }} -> ${cloud_sql_private_ip}:5432 via ${instance} (${zone})"
                exec gcloud compute ssh \
                    --project="${TF_VAR_project_id}" \
                    --zone="${zone}" \
                    --tunnel-through-iap \
                    "${instance}" \
                    -- -N -L "127.0.0.1:{{ local_port }}:${cloud_sql_private_ip}:5432"
                ;;
            *)
                echo "Usage: just gcp-proxy <airflow|cloud-sql> [local_port]"
                exit 2
                ;;
        esac

# output connection credentials for deployed services
[group('ops')]
tf-outputs:
    #!/usr/bin/env bash
    set -euo pipefail

    : "${TF_VAR_project_id:?TF_VAR_project_id must be set in environment}"

    env_name="${TF_VAR_environment:-prod}"
    airflow_admin_username="${TF_VAR_airflow_admin_username:-airflow}"
    airflow_db_user="${TF_VAR_airflow_db_user:-airflow}"
    mlflow_db_user="${TF_VAR_mlflow_db_user:-mlflow}"
    mlflow_service_name="${TF_VAR_mlflow_service_name:-mlflow-tracking}"

    airflow_admin_secret_id="airflow-admin-password-${env_name}"
    airflow_db_secret_id="airflow-db-password-${env_name}"
    ticketforge_db_secret_id="ticketforge-db-password-${env_name}"
    mlflow_admin_secret_id="${mlflow_service_name}-admin-password"
    mlflow_db_secret_id="${mlflow_service_name}-db-password"
    github_token_secret_id="${TF_VAR_airflow_github_token_secret_id:-airflow-github-token-prod}"
    gmail_username_secret_id="${TF_VAR_airflow_gmail_app_username_secret_id:-airflow-gmail-app-username-prod}"
    gmail_password_secret_id="${TF_VAR_airflow_gmail_app_password_secret_id:-airflow-gmail-app-password-prod}"

    get_secret() {
        local secret_id="$1"
        gcloud secrets versions access latest \
            --project="${TF_VAR_project_id}" \
            --secret="${secret_id}" 2>/dev/null || echo "<missing or inaccessible>"
    }

    tf_output_or() {
        local output_name="$1"
        local default_value="$2"

        local value=""
        if value="$(just tf output -raw "${output_name}" 2>/dev/null)"; then
            if [[ -n "${value}" ]]; then
                echo "${value}"
                return 0
            fi
        fi

        echo "${default_value}"
    }

    urlencode() {
        local raw="$1"
        python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$raw"
    }

    airflow_url="$(tf_output_or airflow_webserver_url '<unavailable>')"
    mlflow_url="$(tf_output_or mlflow_tracking_uri '<unavailable>')"
    cloud_sql_private_ip="$(tf_output_or cloud_sql_private_ip '<unavailable>')"
    cloud_sql_connection_name="$(tf_output_or cloud_sql_instance_connection_name '<unavailable>')"
    airflow_db_name="$(tf_output_or cloud_sql_database_name "${TF_VAR_airflow_db_name:-airflow}")"
    ticketforge_db_name="$(tf_output_or cloud_sql_ticketforge_database_name "${TF_VAR_ticketforge_db_name:-ticketforge}")"
    ticketforge_db_user="$(tf_output_or cloud_sql_ticketforge_database_user "${TF_VAR_ticketforge_db_user:-ticketforge}")"
    mlflow_db_name="$(tf_output_or cloud_sql_mlflow_database_name "${TF_VAR_mlflow_db_name:-mlflow}")"
    ticketforge_db_password="$(get_secret "${ticketforge_db_secret_id}")"

    ticketforge_db_user_uri="$(urlencode "${ticketforge_db_user}")"
    ticketforge_db_password_uri="$(urlencode "${ticketforge_db_password}")"
    ticketforge_db_name_uri="$(urlencode "${ticketforge_db_name}")"

    ticketforge_proxy_dsn="postgresql://${ticketforge_db_user_uri}:${ticketforge_db_password_uri}@127.0.0.1:5432/${ticketforge_db_name_uri}"
    ticketforge_private_dsn="postgresql://${ticketforge_db_user_uri}:${ticketforge_db_password_uri}@${cloud_sql_private_ip}:5432/${ticketforge_db_name_uri}"
    ticketforge_socket_dsn="postgresql://${ticketforge_db_user_uri}:${ticketforge_db_password_uri}@/${ticketforge_db_name_uri}?host=/cloudsql/${cloud_sql_connection_name}"

    echo "=== Airflow ==="
    echo "URL:      ${airflow_url}"
    echo "Username: ${airflow_admin_username}"
    echo "Password: $(get_secret "${airflow_admin_secret_id}")"
    echo

    echo "=== MLflow ==="
    echo "URL:      ${mlflow_url}"
    echo "Username: admin"
    echo "Password: $(get_secret "${mlflow_admin_secret_id}")"
    echo

    echo "=== Cloud SQL Users ==="
    echo "Airflow DB (${airflow_db_name}) user (${airflow_db_user}):     $(get_secret "${airflow_db_secret_id}")"
    echo "Ticketforge DB (${ticketforge_db_name}) user (${ticketforge_db_user}): $(get_secret "${ticketforge_db_secret_id}")"
    echo "MLflow DB (${mlflow_db_name}) user (${mlflow_db_user}):      $(get_secret "${mlflow_db_secret_id}")"
    echo

    echo "=== Runtime Integrations ==="
    echo "GMAIL_APP_USERNAME: $(get_secret "${gmail_username_secret_id}")"
    echo "GMAIL_APP_PASSWORD: $(get_secret "${gmail_password_secret_id}")"
    echo "GITHUB_TOKEN:       $(get_secret "${github_token_secret_id}")"

    echo
    echo "=== Ticketforge Connection Strings ==="
    echo "Proxy DSN (run: just gcp-proxy cloud-sql 5432):"
    echo "${ticketforge_proxy_dsn}"
    echo
    echo "Private IP DSN (requires network path to Cloud SQL private IP):"
    echo "${ticketforge_private_dsn}"
    echo
    echo "Cloud SQL Socket DSN (for GCP runtimes):"
    echo "${ticketforge_socket_dsn}"
