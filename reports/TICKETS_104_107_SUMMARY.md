# Tickets #104-#107: Summary of Changes

**Author:** Vikas Neriyanuru
**Branch:** `codex/mlops-104-107`
**PR:** [#164 - feat(#104-#107): serving deploy, inference endpoint, monitoring fix](https://github.com/ALearningCurve/ticket-forge/pull/164)
**Merged:** 2026-04-14 12:54 UTC
**Earlier PR:** [#120 - Add MLOps monitoring, reporting, and submission checks](https://github.com/ALearningCurve/ticket-forge/pull/120) (merged 2026-04-08)

---

## Deployed URLs

| Service | URL |
|---------|-----|
| **Frontend (use this)** | https://ticketforge-frontend-r5ebf6yyyq-ue.a.run.app/ |
| **Backend API** | https://ticketforge-backend-r5ebf6yyyq-ue.a.run.app/ |
| **Backend Health** | https://ticketforge-backend-r5ebf6yyyq-ue.a.run.app/health |
| **API Docs (Swagger)** | https://ticketforge-backend-r5ebf6yyyq-ue.a.run.app/docs |

**Important:** Do NOT use `ticketforge-web-*` or `ticketforge-api-*` URLs. Those are from a separate `app_serving.tf` deployment that does not have `NEXT_PUBLIC_API_URL` configured, so the frontend falls back to `127.0.0.1:8000` and API calls fail.

The correct services come from `serving.tf`:
- `ticketforge-frontend` (var `web_frontend_service_name`)
- `ticketforge-backend` (var `web_backend_service_name`)

### How to find the deployed URL in the future

1. **From Terraform:** `cd terraform && terraform output web_frontend_service_url`
2. **From GCP Console:** Cloud Run > Services > `ticketforge-frontend` > URL at the top
3. **From CLI:** `gcloud run services describe ticketforge-frontend --project=<PROJECT_ID> --region=<REGION> --format='value(status.url)'`
4. **From GitHub Actions:** Check the latest successful [Serving Deploy](https://github.com/ALearningCurve/ticket-forge/actions/workflows/serving-deploy.yml) run > `deploy-frontend` step output

---

## Ticket Breakdown

### #104 - Deployment Automation and Release Management

**What it required:**
- GitHub Actions deploy workflow for backend + frontend source code changes (on push to main)
- Build, push, deploy using Workload Identity Federation
- Deterministic model selection from MLflow
- Smoke tests for deployed services
- Deployment logging with commit SHA to serving revision traceability

**What was done:**
- Created `serving-deploy.yml` (569 lines) - full CI/CD pipeline that builds Docker images, pushes to Artifact Registry, applies Terraform, and smoke tests both backend and frontend Cloud Run services
- Created `terraform/serving.tf` - Cloud Run service definitions for `ticketforge-backend` and `ticketforge-frontend` with VPC access, secret injection, and public invoker IAM
- Created `terraform/outputs.tf`, `terraform/variables.tf`, `terraform/secrets.tf` additions for serving infrastructure
- Created `apps/web-backend/Dockerfile` and `apps/web-frontend/Dockerfile`
- Created `scripts/ci/backend_smoketest.sh` and `scripts/ci/frontend_smoketest.sh`
- Created `scripts/ci/deploy_serving.sh`

**Key files:**
- `.github/workflows/serving-deploy.yml`
- `terraform/serving.tf`
- `terraform/outputs.tf`
- `terraform/variables.tf`

**How to verify:**
- Latest serving deploy run: https://github.com/ALearningCurve/ticket-forge/actions/runs/24400051380 (success)
- `curl https://ticketforge-backend-r5ebf6yyyq-ue.a.run.app/health` should return `{"status":"ok"}`
- `curl https://ticketforge-frontend-r5ebf6yyyq-ue.a.run.app/` should return 200

---

### #105 - Reproducibility, Verification, and Submission Readiness

**What it required:**
- Fresh-machine replication guide
- Deployment verification script with pass/fail reporting
- Submission checklist mapping artifacts to files/workflows
- Environment configuration artifact list

**What was done:**
- Created `scripts/ci/verify_submission_ready.sh` - automated submission readiness checker
- Updated `reports/02_DEPLOYMENT_AND_SUBMISSION.md` with deployment verification snapshots
- Added `Justfile` targets: `verify-submission-ready`, `gcp-ticketforge-schema-init`
- Terraform README updated with infrastructure setup instructions

**Key files:**
- `scripts/ci/verify_submission_ready.sh`
- `reports/02_DEPLOYMENT_AND_SUBMISSION.md`
- `Justfile`

**How to verify:**
- Run `just verify-submission-ready` locally

---

### #106 - Monitoring, Drift Detection, and Retraining Lifecycle

**What it required:**
- Production performance metrics instrumentation
- Serving-time drift detection pipeline
- Versioned quality/drift thresholds with tests
- Automated retraining trigger
- Candidate-vs-production quality gates

**What was done:**
- Expanded `apps/training/training/cmd/monitor_model.py` (from ~100 lines to 365+ lines) with:
  - Production metric export via new `/api/v1/inference/monitoring/export` endpoint
  - Drift detection (PSI, chi-squared, KS tests)
  - Quality threshold checks
  - Automated retraining trigger via GitHub Actions dispatch
- Created `/api/v1/inference/monitoring/export` endpoint in backend for serving-time data export
- Created `apps/web-backend/web_backend/api/v1/inference.py` with prediction + monitoring endpoints
- Created `apps/web-backend/web_backend/services/inference.py` (436 lines) - MLflow model loading, prediction, logging
- Created `apps/web-backend/web_backend/models/inference.py` - InferenceLog ORM model
- Created `apps/web-backend/web_backend/schemas/inference.py` - request/response schemas
- **Later fix (d4a572e):** Added retry (3 attempts) + 90s timeout to monitoring fetch to handle Cloud Run cold starts. Raises `RuntimeError` on exhaustion instead of returning empty list (prevents drift baseline corruption)
- Added 4 tests for retry logic in `test_monitor_model.py`
- Added SQL migration trigger to `airflow-deploy.yml` so schema changes auto-deploy

**Key files:**
- `apps/training/training/cmd/monitor_model.py`
- `apps/training/tests/test_monitor_model.py`
- `apps/web-backend/web_backend/api/v1/inference.py`
- `apps/web-backend/web_backend/services/inference.py`
- `.github/workflows/model-monitoring.yml`
- `.github/workflows/airflow-deploy.yml` (added `scripts/postgres/init/**` path trigger)

**How to verify:**
- `curl https://ticketforge-backend-r5ebf6yyyq-ue.a.run.app/api/v1/inference/monitoring/export` should return `[]` (not 404)
- Model Monitoring workflow: https://github.com/ALearningCurve/ticket-forge/actions/workflows/model-monitoring.yml
- Note: monitoring still fails because the backend returns empty data (no inference logs yet) -- this is expected until real traffic flows

---

### #107 - Operational Notifications and Deployment/Retraining Reporting

**What it required:**
- Notifications for retraining trigger, retrain outcome, deploy outcome
- Links to workflow logs, model version, revision metadata
- Unified reporting format for success/failure

**What was done:**
- Updated `model-monitoring.yml` with `always()` steps that write ops report and send notification regardless of success/failure
- Model CI/CD workflow (`model-cicd.yml`) already had email notification; updated to include deployment trigger context
- Serving deploy workflow includes deployment status logging with commit SHA and revision metadata

**Key files:**
- `.github/workflows/model-monitoring.yml`
- `.github/workflows/model-cicd.yml`
- `.github/workflows/serving-deploy.yml`

**How to verify:**
- Check email notifications after any Model CI/CD or Model Monitoring run
- Latest Model CI/CD run (failed due to low accuracy threshold): https://github.com/ALearningCurve/ticket-forge/actions/runs/24401090724

---

## Additional Fixes in PR #164

| Change | Reason |
|--------|--------|
| Merge latest main into branch | Picked up dataset rebuild (#160), ticket sizing (#156), recommendations, auth hardening |
| Fix lint errors from merge (`92fec0d`) | Code style/formatting conflicts after merge |
| Fix CI/CD race condition | Changed `model-cicd.yml` from `workflow_run` trigger to `push` trigger + `wait-for-ci` polling gate |
| Frontend board/theme fixes | Minor fixes inherited from merge (board-column, board-view, theme-toggle, api.ts) |

---

## Current Known Issues

| Issue | Status | Details |
|-------|--------|---------|
| Model Monitoring workflow fails | Expected | Backend returns empty inference logs since no real traffic has flowed yet. The retry + timeout fix (d4a572e) handles the cold-start issue, but the monitoring still has no data to analyze |
| Model CI/CD accuracy gate fails | Known | accuracy 0.43 < 0.70 threshold, macro_f1 0.40 < 0.65 -- caused by dataset rebuild in PR #160. This is a data/model issue, not infra |
| `ticketforge-web` frontend broken | By design | The `app_serving.tf` services are a separate deployment stack. `ticketforge-web` has no `NEXT_PUBLIC_API_URL` env var, so it falls back to `localhost`. Use `ticketforge-frontend` instead |

---

## CI/CD Workflow Runs (2026-04-14)

| Time (UTC) | Workflow | Result | Link |
|------------|----------|--------|------|
| 04:56 | Serving Deploy | success | https://github.com/ALearningCurve/ticket-forge/actions/runs/24381686790 |
| 04:58 | TicketForge App Serving Deploy | success | https://github.com/ALearningCurve/ticket-forge/actions/runs/24381754808 |
| 07:18 | Model Monitoring (scheduled) | failure | https://github.com/ALearningCurve/ticket-forge/actions/runs/24386208925 |
| 12:54 | CI/CD (PR #164 merge) | success | https://github.com/ALearningCurve/ticket-forge/actions/runs/24400051343 |
| 12:54 | Serving Deploy (PR #164 merge) | success | https://github.com/ALearningCurve/ticket-forge/actions/runs/24400051380 |
| 12:56 | TicketForge App Serving Deploy | success | https://github.com/ALearningCurve/ticket-forge/actions/runs/24400169099 |
| 13:16 | Model CI/CD (manual) | failure | https://github.com/ALearningCurve/ticket-forge/actions/runs/24401090724 |
| 13:40 | Airflow Deploy (manual) | success | https://github.com/ALearningCurve/ticket-forge/actions/runs/24402268593 |

---

## Demo Credentials

Seeded by Darshan into the deployed Cloud SQL database on 2026-04-14:

| Username | Password |
|----------|----------|
| `vikas0804` | `TfDemo@30827b!` |
| `darshanrk18` | `TfDemo@44cbe8!` |
| `adityarajendrashanbhag` | `TfDemo@acd23c!` |
| `alearningcurve` | `TfDemo@07a326!` |
| `sampai28` | `TfDemo@e678ec!` |
| `skandhan-madhusudhana` | `TfDemo@12f481!` |

All verified working against `ticketforge-backend` as of 2026-04-14 ~2:00 PM EDT.
