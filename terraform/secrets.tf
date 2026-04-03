resource "random_password" "airflow_admin_password" {
  length  = 24
  special = false
}

resource "google_secret_manager_secret" "airflow_admin_password" {
  secret_id = "airflow-admin-password-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.airflow_services]
}

resource "google_secret_manager_secret_version" "airflow_admin_password" {
  secret      = google_secret_manager_secret.airflow_admin_password.id
  secret_data = coalesce(var.airflow_admin_password, random_password.airflow_admin_password.result)
}

resource "google_secret_manager_secret" "airflow_db_password" {
  secret_id = "airflow-db-password-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.airflow_services]
}

resource "google_secret_manager_secret_version" "airflow_db_password" {
  secret      = google_secret_manager_secret.airflow_db_password.id
  secret_data = coalesce(var.airflow_db_password, random_password.airflow_db_password.result)
}

resource "google_secret_manager_secret" "ticketforge_db_password" {
  secret_id = "ticketforge-db-password-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.airflow_services]
}

resource "google_secret_manager_secret_version" "ticketforge_db_password" {
  secret      = google_secret_manager_secret.ticketforge_db_password.id
  secret_data = coalesce(var.ticketforge_db_password, random_password.ticketforge_db_password.result)
}

data "google_secret_manager_secret_version" "airflow_admin_password" {
  secret  = google_secret_manager_secret.airflow_admin_password.secret_id
  version = "latest"
}

data "google_secret_manager_secret_version" "airflow_db_password" {
  secret  = google_secret_manager_secret.airflow_db_password.secret_id
  version = "latest"
}

data "google_secret_manager_secret_version" "ticketforge_db_password" {
  secret  = google_secret_manager_secret.ticketforge_db_password.secret_id
  version = "latest"
}

locals {
  airflow_admin_password        = data.google_secret_manager_secret_version.airflow_admin_password.secret_data
  airflow_db_password_value     = data.google_secret_manager_secret_version.airflow_db_password.secret_data
  ticketforge_db_password_value = data.google_secret_manager_secret_version.ticketforge_db_password.secret_data
  airflow_sqlalchemy_conn = format(
    "postgresql+psycopg2://%s:%s@%s/%s",
    var.airflow_db_user,
    local.airflow_db_password_value,
    google_sql_database_instance.mlflow.private_ip_address,
    var.airflow_db_name,
  )
  ticketforge_sqlalchemy_conn = format(
    "postgresql+psycopg2://%s:%s@%s/%s",
    var.ticketforge_db_user,
    local.ticketforge_db_password_value,
    google_sql_database_instance.mlflow.private_ip_address,
    var.ticketforge_db_name,
  )
}
