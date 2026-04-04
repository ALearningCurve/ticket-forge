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
    #!/usr/bin/env bash
    set -euo pipefail

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

    # Fail fast when deploying with a detached SHA that is not visible on origin.
    if [[ "${repo_ref}" =~ ^[0-9a-f]{40}$ ]]; then
        if ! git ls-remote origin | awk '{print $1}' | grep -Fxq "${repo_ref}"; then
            echo "ERROR: ${repo_ref} is not available on origin."
            echo "Push this commit first, or set AIRFLOW_REPO_REF to a branch/tag/sha that exists on GitHub."
            exit 1
        fi
    fi

    echo "Deploying Airflow from repo ref ${repo_ref}"

    # Import already-existing secrets so Terraform can manage them without 409 conflicts.
    if gcloud secrets describe "${github_token_secret_id}" --project="${TF_VAR_project_id}" >/dev/null 2>&1; then
        terraform -chdir=terraform import -var="project_id=${TF_VAR_project_id}" -var="state_bucket=${TF_VAR_state_bucket}" -var="region=${TF_VAR_region}" -var="zone=us-east1-b" -var="environment=prod" 'google_secret_manager_secret.airflow_runtime["github_token"]' "projects/${TF_VAR_project_id}/secrets/${github_token_secret_id}" >/dev/null 2>&1 || true
    fi
    if gcloud secrets describe "${gmail_username_secret_id}" --project="${TF_VAR_project_id}" >/dev/null 2>&1; then
        terraform -chdir=terraform import -var="project_id=${TF_VAR_project_id}" -var="state_bucket=${TF_VAR_state_bucket}" -var="region=${TF_VAR_region}" -var="zone=us-east1-b" -var="environment=prod" 'google_secret_manager_secret.airflow_runtime["gmail_app_username"]' "projects/${TF_VAR_project_id}/secrets/${gmail_username_secret_id}" >/dev/null 2>&1 || true
    fi
    if gcloud secrets describe "${gmail_password_secret_id}" --project="${TF_VAR_project_id}" >/dev/null 2>&1; then
        terraform -chdir=terraform import -var="project_id=${TF_VAR_project_id}" -var="state_bucket=${TF_VAR_state_bucket}" -var="region=${TF_VAR_region}" -var="zone=us-east1-b" -var="environment=prod" 'google_secret_manager_secret.airflow_runtime["gmail_app_password"]' "projects/${TF_VAR_project_id}/secrets/${gmail_password_secret_id}" >/dev/null 2>&1 || true
    fi

    # Ensure Terraform-managed runtime secrets exist before adding versions.
    terraform -chdir=terraform apply -auto-approve \
        -target="google_secret_manager_secret.airflow_runtime" \
        -var="project_id=${TF_VAR_project_id}" \
        -var="state_bucket=${TF_VAR_state_bucket}" \
        -var="region=${TF_VAR_region}" \
        -var="zone=us-east1-b" \
        -var="environment=prod"

    printf '%s' "${github_token_value}" | gcloud secrets versions add "${github_token_secret_id}" --project="${TF_VAR_project_id}" --data-file=-
    printf '%s' "${GMAIL_APP_USERNAME}" | gcloud secrets versions add "${gmail_username_secret_id}" --project="${TF_VAR_project_id}" --data-file=-
    printf '%s' "${GMAIL_APP_PASSWORD}" | gcloud secrets versions add "${gmail_password_secret_id}" --project="${TF_VAR_project_id}" --data-file=-

    terraform -chdir=terraform apply -auto-approve \
      -var="project_id=${TF_VAR_project_id}" \
      -var="state_bucket=${TF_VAR_state_bucket}" \
      -var="region=${TF_VAR_region}" \
      -var="zone=us-east1-b" \
      -var="environment=prod" \
      -var="airflow_repo_ref=${repo_ref}"

# opens proxy tunnel to GCP resources (e.g. Airflow webserver, Cloud SQL instances)
[group('ops')]
gcp-proxy target local_port='18080':
        #!/usr/bin/env bash
        set -euo pipefail

        : "${TF_VAR_project_id:?TF_VAR_project_id must be set in environment}"
        region="${TF_VAR_region:-us-east1}"

        case "{{ target }}" in
            airflow)
                zone="${TF_VAR_zone:-us-east1-b}"
                instance="${AIRFLOW_VM_NAME:-airflow-vm-prod}"
                echo "Opening Airflow IAP tunnel: 127.0.0.1:{{ local_port }} -> ${instance}:8080 (${zone})"
                exec gcloud compute start-iap-tunnel "${instance}" 8080 \
                    --zone="${zone}" \
                    --local-host-port="127.0.0.1:{{ local_port }}"
                ;;
            cloud-sql|cloudsql|sql)
                sql_instance="${TF_VAR_shared_cloud_sql_instance_name:-mlflow-tracking-sql}"
                sql_target="${TF_VAR_project_id}:${region}:${sql_instance}"
                if ! command -v cloud-sql-proxy >/dev/null 2>&1; then
                    echo "cloud-sql-proxy is required for Cloud SQL tunneling."
                    echo "Install: https://cloud.google.com/sql/docs/postgres/connect-auth-proxy#install"
                    exit 1
                fi
                echo "Opening Cloud SQL proxy: 127.0.0.1:{{ local_port }} -> ${sql_target}"
                exec cloud-sql-proxy --private-ip "${sql_target}" --port "{{ local_port }}"
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
    ticketforge_db_user="${TF_VAR_ticketforge_db_user:-ticketforge}"
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

    airflow_url="$(terraform -chdir=terraform output -raw airflow_webserver_url 2>/dev/null || echo '<unavailable>')"
    mlflow_url="$(terraform -chdir=terraform output -raw mlflow_tracking_uri 2>/dev/null || echo '<unavailable>')"

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
    echo "Airflow DB user (${airflow_db_user}):     $(get_secret "${airflow_db_secret_id}")"
    echo "Ticketforge DB user (${ticketforge_db_user}): $(get_secret "${ticketforge_db_secret_id}")"
    echo "MLflow DB user (${mlflow_db_user}):      $(get_secret "${mlflow_db_secret_id}")"
    echo

    echo "=== Runtime Integrations ==="
    echo "GMAIL_APP_USERNAME: $(get_secret "${gmail_username_secret_id}")"
    echo "GMAIL_APP_PASSWORD: $(get_secret "${gmail_password_secret_id}")"
    echo "GITHUB_TOKEN:       $(get_secret "${github_token_secret_id}")"
