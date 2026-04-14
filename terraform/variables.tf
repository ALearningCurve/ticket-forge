variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region for resources."
  type        = string
  default     = "us-east1"
}

variable "airflow_region" {
  description = "Optional dedicated GCP region for Airflow runtime resources. Defaults to region when unset."
  type        = string
  default     = null
  nullable    = true
}

variable "state_bucket" {
  description = "Name for the tfstate bucket (must be globally unique)."
  type        = string
}

variable "data_bucket" {
  description = "Name for the data bucket (must be globally unique)."
  type        = string
  default     = "ticketforge-dvc"
}

variable "repository" {
  description = "the github repository in format 'ORGANIZATION/REPO'"
  type        = string
  default     = "ALearningCurve/ticket-forge"
}

variable "repository_id" {
  description = "the github repository in id format"
  type        = string
  default     = "1142699076"
}

variable "mlflow_service_name" {
  description = "Cloud Run service name for MLflow tracking server."
  type        = string
  default     = "mlflow-tracking"
}

variable "shared_cloud_sql_instance_name" {
  description = "Cloud SQL instance name shared by MLflow, Airflow, and ticketforge databases (import existing instance before first apply when reusing one)."
  type        = string
  default     = "mlflow-tracking-sql"
}

variable "mlflow_server_image" {
  description = "Optional explicit container image for MLflow server. If null, Terraform uses Artifact Registry path <region>-docker.pkg.dev/<project>/mlflow-repo/mlflow-gcp:<tag>."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition = var.mlflow_server_image == null || can(
      regex(
        "^(((?:[a-z0-9-]+\\.)?gcr\\.io)|(?:[a-z0-9-]+-docker\\.pkg\\.dev)|docker\\.io)/",
        var.mlflow_server_image,
      )
    )
    error_message = "mlflow_server_image must be null or use one of: docker.io, [region.]gcr.io, or [region-]docker.pkg.dev."
  }
}

variable "mlflow_artifact_registry_repository" {
  description = "Artifact Registry Docker repository name for the MLflow image."
  type        = string
  default     = "mlflow-repo"
}

variable "mlflow_image_tag" {
  description = "MLflow image tag used when mlflow_server_image is null."
  type        = string
  default     = "v3.10.0"

  validation {
    condition = can(
      regex("^[A-Za-z0-9._-]+$", var.mlflow_image_tag)
    )
    error_message = "mlflow_image_tag may only contain letters, numbers, dot, underscore, and dash."
  }
}

variable "mlflow_tracking_uri_override" {
  description = "Optional explicit MLflow tracking URI for workflows that must deploy backend resources without planning changes to the MLflow service."
  type        = string
  default     = null
}

variable "mlflow_db_tier" {
  description = "Cloud SQL machine tier for MLflow backend store."
  type        = string
  default     = "db-g1-small"
}

variable "cloud_sql_max_connections" {
  description = "Optional Postgres max_connections override for the shared Cloud SQL instance. Leave null to use Cloud SQL tier defaults."
  type        = number
  default     = 500
  nullable    = true

  validation {
    condition = var.cloud_sql_max_connections == null || (
      var.cloud_sql_max_connections >= 20 && var.cloud_sql_max_connections <= 5000
    )
    error_message = "cloud_sql_max_connections must be null or between 20 and 5000."
  }
}

variable "mlflow_db_name" {
  description = "Cloud SQL database name used by MLflow backend store."
  type        = string
  default     = "mlflow"
}

variable "mlflow_db_user" {
  description = "Cloud SQL username used by MLflow backend store."
  type        = string
  default     = "mlflow"
}

variable "mlflow_additional_invokers" {
  description = "Extra Cloud Run invoker members for MLflow (e.g. user:you@example.com)."
  type        = list(string)
  default     = []
}

variable "mlflow_enable_proxy_multipart_upload" {
  description = "Enable MLflow proxied multipart upload for artifacts to avoid large single-request uploads through Cloud Run."
  type        = bool
  default     = true
}

variable "mlflow_multipart_upload_minimum_file_size" {
  description = "Minimum artifact size in bytes before MLflow uses multipart upload for proxied artifact writes."
  type        = number
  default     = 10485760 # 10 MiB

  validation {
    condition     = var.mlflow_multipart_upload_minimum_file_size >= 0
    error_message = "mlflow_multipart_upload_minimum_file_size must be >= 0."
  }
}

variable "mlflow_multipart_upload_chunk_size" {
  description = "Chunk size in bytes for MLflow multipart proxy uploads. Must be a multiple of 256 KiB for GCS."
  type        = number
  default     = 8388608 # 8 MiB

  validation {
    condition = (
      var.mlflow_multipart_upload_chunk_size >= 262144
      && floor(var.mlflow_multipart_upload_chunk_size / 262144) == var.mlflow_multipart_upload_chunk_size / 262144
    )
    error_message = "mlflow_multipart_upload_chunk_size must be >= 262144 and a multiple of 262144 (256 KiB)."
  }
}

variable "web_backend_service_name" {
  description = "Cloud Run service name for the production backend inference API."
  type        = string
  default     = "ticketforge-backend"
}

variable "web_frontend_service_name" {
  description = "Cloud Run service name for the production web frontend."
  type        = string
  default     = "ticketforge-frontend"
}

variable "web_backend_artifact_registry_repository" {
  description = "Artifact Registry repository for backend images."
  type        = string
  default     = "backend-repo"
}

variable "web_frontend_artifact_registry_repository" {
  description = "Artifact Registry repository for frontend images."
  type        = string
  default     = "frontend-repo"
}

variable "web_backend_image" {
  description = "Container image used by the backend Cloud Run service."
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "web_frontend_image" {
  description = "Container image used by the frontend Cloud Run service."
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "web_backend_cors_origins" {
  description = "Allowed browser origins for backend CORS and refresh cookie usage."
  type        = list(string)
  default     = ["http://localhost:3000"]
}

variable "web_frontend_api_url" {
  description = "Public backend API URL embedded into the frontend runtime."
  type        = string
  default     = ""
}

variable "web_backend_mlflow_model_name" {
  description = "Registered MLflow model served by the backend."
  type        = string
  default     = "ticket-forge-best"
}

variable "web_backend_mlflow_model_stage" {
  description = "MLflow stage used when no explicit serving version is pinned."
  type        = string
  default     = "Production"
}

variable "web_backend_serving_model_version" {
  description = "Optional explicit MLflow model version pinned into the deployed backend."
  type        = string
  default     = ""
}

variable "web_backend_jwt_secret_id" {
  description = "Secret Manager secret id for the backend JWT signing key."
  type        = string
  default     = "ticketforge-jwt-secret-prod"
}

variable "web_backend_jwt_secret_key" {
  description = "Optional explicit backend JWT secret. If null, Terraform generates one."
  type        = string
  default     = null
  sensitive   = true
  nullable    = true
}

variable "environment" {
  description = "Deployment environment for naming and labeling."
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["prod"], var.environment)
    error_message = "environment must be one of:prod." // Add staging sometime in the future?
  }
}

variable "zone" {
  description = "Default Compute Engine zone for the Airflow VM when airflow_zone is unset."
  type        = string
  default     = "us-east1-c"
}

variable "airflow_zone" {
  description = "Optional dedicated Compute Engine zone for the Airflow VM. Defaults to zone when unset."
  type        = string
  default     = null
  nullable    = true
}

variable "airflow_vm_machine_type" {
  description = "Machine type for the Airflow Compute Engine VM."
  type        = string
  default     = "e2-medium"
}

variable "airflow_vm_disk_size_gb" {
  description = "Boot disk size for the Airflow VM in GB."
  type        = number
  default     = 20

  validation {
    condition     = var.airflow_vm_disk_size_gb >= 20
    error_message = "airflow_vm_disk_size_gb must be >= 20."
  }
}

variable "airflow_repo_ref" {
  description = "Git ref (branch, tag, or commit SHA) checked out on the Airflow VM during deploy."
  type        = string
  default     = "main"
}

variable "airflow_version" {
  description = "Apache Airflow version installed natively on the Airflow VM."
  type        = string
  default     = "2.10.4"
}

variable "airflow_github_token_secret_id" {
  description = "Secret Manager secret id containing the GitHub token used by Airflow runtime."
  type        = string
  default     = "airflow-github-token-prod"
}

variable "airflow_gmail_app_username_secret_id" {
  description = "Secret Manager secret id containing the Gmail app username used by Airflow SMTP config."
  type        = string
  default     = "airflow-gmail-app-username-prod"
}

variable "airflow_gmail_app_password_secret_id" {
  description = "Secret Manager secret id containing the Gmail app password used by Airflow SMTP config."
  type        = string
  default     = "airflow-gmail-app-password-prod"
}

variable "airflow_webserver_secret_key_secret_id" {
  description = "Secret Manager secret id containing Airflow webserver secret key used for session/CSRF signing."
  type        = string
  default     = "airflow-webserver-secret-key-prod"
}

variable "airflow_webserver_secret_key" {
  description = "Optional explicit Airflow webserver secret key. If null, Terraform generates a strong key and stores it in Secret Manager."
  type        = string
  default     = null
  sensitive   = true
  nullable    = true
}

variable "airflow_admin_username" {
  description = "Default Airflow admin username."
  type        = string
  default     = "airflow"
}

variable "airflow_admin_password" {
  description = "Optional Airflow admin password. If null, fetched from Secret Manager."
  type        = string
  default     = null
  sensitive   = true
  nullable    = true
}

variable "airflow_db_name" {
  description = "Cloud SQL database name for Airflow metadata."
  type        = string
  default     = "airflow"
}

variable "airflow_db_user" {
  description = "Cloud SQL user for Airflow metadata database."
  type        = string
  default     = "airflow"
}

variable "airflow_db_password" {
  description = "Optional Cloud SQL password for airflow_db_user. If null, fetched from Secret Manager."
  type        = string
  default     = null
  sensitive   = true
  nullable    = true
}

variable "ticketforge_db_name" {
  description = "Cloud SQL database name for ticket-forge application tables."
  type        = string
  default     = "ticketforge"
}

variable "ticketforge_db_user" {
  description = "Cloud SQL user for ticket-forge application tables."
  type        = string
  default     = "ticketforge"
}

variable "ticketforge_db_password" {
  description = "Optional Cloud SQL password for ticketforge_db_user. If null, fetched from Secret Manager."
  type        = string
  default     = null
  sensitive   = true
  nullable    = true
}

variable "training_bucket_name" {
  description = "Cloud Storage bucket for training datasets and artifacts."
  type        = string
  default     = null
  nullable    = true
}

variable "enable_terraform_state_bucket" {
  description = "Whether to create the Terraform state bucket with this configuration."
  type        = bool
  default     = true
}

variable "enable_ticketforge_app_cloud_run" {
  description = <<-EOT
    Provision optional Cloud Run (v2) services for the TicketForge FastAPI API,
    lightweight inference HTTP process, and Next.js frontend. Requires three
    container image URIs via ticketforge_*_container_image variables.
  EOT
  type        = bool
  default     = false
}

variable "ticketforge_api_service_name" {
  description = "Cloud Run service name for the TicketForge FastAPI API."
  type        = string
  default     = "ticketforge-api"
}

variable "ticketforge_inference_service_name" {
  description = "Cloud Run service name for the TicketForge inference stub."
  type        = string
  default     = "ticketforge-inference"
}

variable "ticketforge_web_service_name" {
  description = "Cloud Run service name for the TicketForge Next.js UI."
  type        = string
  default     = "ticketforge-web"
}

variable "ticketforge_api_container_image" {
  description = "Container image URI for the API service (FastAPI / Uvicorn on port 8080)."
  type        = string
  default     = ""
}

variable "ticketforge_inference_container_image" {
  description = "Container image URI for the inference service (FastAPI stub on port 8080)."
  type        = string
  default     = ""
}

variable "ticketforge_web_container_image" {
  description = "Container image URI for the web service (Next.js on port 8080)."
  type        = string
  default     = ""
}
