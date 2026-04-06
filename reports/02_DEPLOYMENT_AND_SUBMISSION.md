# Deployment, Monitoring, and Submission Readiness

This document groups the operational deliverables for issues `#104` to `#107`.

## 1. Deployment Automation and Release Management

Primary workflows:

- `.github/workflows/ci.yml`
  - blocks deployment until repository, model, backend, frontend, CodeQL, and
    Terraform checks succeed.
- `.github/workflows/airflow-deploy.yml`
  - deploys the Airflow runtime after successful `CI/CD` runs on `main`
  - records deployment traceability in `reports/runtime/airflow_deployment_report.json`
  - uploads the deployment report as a workflow artifact
- `.github/workflows/model-cicd.yml`
  - runs deterministic gated retraining and promotion
  - stores model release lineage in:
    - `models/<run_id>/gate_report.json`
    - `models/<run_id>/run_manifest.json`
    - `models/<run_id>/operations_report.json`

Release traceability now covers:

- source commit / ref
- workflow URL
- dataset source and version
- promoted model version
- deployment target and deployed runtime revision

## 2. Reproducibility and Verification

Fresh-machine prerequisites:

- `uv`
- `node` and `npm`
- `just`
- `terraform`
- `gh`
- `gcloud`

Required configuration and workflow artifacts:

- `README.md`
- `terraform/README.md`
- `terraform/providers.tf`
- `terraform/variables.tf`
- `terraform/main.tf`
- `terraform/secrets.tf`
- `docker-compose.yml`
- `package.json`
- `apps/web-frontend/package.json`
- `apps/web-backend/pyproject.toml`
- `apps/training/README.md`
- `.github/workflows/ci.yml`
- `.github/workflows/model-cicd.yml`
- `.github/workflows/model-monitoring.yml`
- `.github/workflows/airflow-deploy.yml`
- `scripts/ci/airflow_smoketest.sh`
- `scripts/ci/airflow_trigger_dag.sh`
- `scripts/ci/verify_submission_ready.sh`

Core verification commands:

```bash
just install-deps
just verify-submission-ready
just gcp-airflow-deploy
just gcp-airflow-smoketest "<airflow-url>"
```

`just verify-submission-ready` emits `PASS`, `WARN`, and `FAIL` lines so a clean
submission check is easy to interpret on a fresh machine.

Local retraining verification:

```bash
just train-with-gates -- --runid local-verify --trigger workflow_dispatch --source-uri dvc --promote false
```

Cloud-backed monitoring verification:

```bash
uv run python -m training.cmd.monitor_model \
  --runid monitor-local \
  --trigger workflow_dispatch \
  --trigger-reason manual-check
```

## 3. Monitoring, Drift Detection, and Retraining

Primary workflow:

- `.github/workflows/model-monitoring.yml`
  - runs on a schedule and on manual dispatch
  - resolves the latest cloud-published dataset from the bucket `index.json`
  - generates a fresh `data_profile_report.json`
  - compares it against the previous monitoring baseline
  - persists:
    - `models/<run_id>/drift_report.json`
    - `models/<run_id>/operations_report.json`
    - `gs://.../monitoring/reports/<run_id>/...`
  - triggers `Model CI/CD` with `dataset_source=gcs` when drift thresholds are breached

Thresholds are versioned in code through:

- `apps/training/training/analysis/drift_detection.py`
- `apps/training/training/analysis/gate_config.py`

This keeps automated retraining aligned with the same candidate-vs-production
quality gates already used for model promotion decisions.

## 4. Notifications and Reporting

Unified operations report schema:

- `apps/training/training/analysis/ops_report.py`

Produced reports:

- deploy: `reports/runtime/airflow_deployment_report.json`
- monitoring: `models/<run_id>/operations_report.json`
- training / retraining: `models/<run_id>/operations_report.json`

Retained workflow artifacts:

- `airflow-deployment-report-<github_run_id>`
- `model-monitoring-<github_run_id>`
- `model-artifacts-<github_run_id>`
- `model-ops-artifacts-<github_run_id>`

Notification coverage:

- deploy outcome email from `.github/workflows/airflow-deploy.yml`
- monitoring / drift email from `.github/workflows/model-monitoring.yml`
- training / retraining outcome email from `.github/workflows/model-cicd.yml`

Each notification includes:

- workflow run URL
- trigger source / trigger reason
- dataset source and version when applicable
- model version when applicable
- deployment target / deployed revision when applicable
- failure reasons when present

## 5. Submission Checklist

- `just verify-submission-ready` passes
- `CI/CD` passes on the final branch / PR
- `Airflow Deploy` succeeds and uploads the deployment report artifact
- `Model Monitoring` succeeds and uploads drift + operations report artifacts
- `Model CI/CD` succeeds and uploads gate + manifest + retraining operations artifacts
- `reports/00_DATA_PIPELINE.md`, `reports/01_ML_PIPELINE.md`, and this report are committed
- board issues `#104` to `#107` can link directly to the workflows and report artifacts above
