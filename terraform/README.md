# Terraform

This folder manages a demo GCP storage bucket and uses a GCS backend for state.

## Prerequisites

- Terraform v1.14+ installed.
- A GCP project with billing enabled.
- Auth configured via one of:
  - `gcloud auth application-default login`
  - `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account key file (set in .env file)

## First-time setup

1. Set the following environment varibles in a .env file (hint: use `gcloud config list` to see project id):

```
TF_VAR_project_id=YOUR_PROJECT_ID
TF_VAR_region=us-east1
TF_VAR_state_bucket=tf-demo-bucket-unique-name
```

2. First-time bootstrapping if the state bucket does not exist:
- Phase 1 (Local): Comment out the backend `"gcs" {}` block in `main.tf`. Run terraform init (`just tf-init`) and terraform apply (`just tf-apply`).
- Phase 2 (Migration): Uncomment the backend `"gcs" {}` block in `main.tf`. Run terraform init again (`just tf-init`).
  - TF detects local state and a newly configured remote backend from first init. It will ask: "Do you want to copy existing state to the new backend?"; type yes and then delete local .tfstate file

## Common scripts (Just)

After the initial setup, you can run the following commands:

From repo root:

- Lint/format terraform files:
  - `just tf-lint`
- Assert correct formatting:
  - `just tf-check`
- Initialize and plan:
  - `just tf-plan`
- Initialize and apply:
  - `just tf-apply`
- Run arbitrary terraform commands:
  - `just tf` (i.e. `just tf apply`)
