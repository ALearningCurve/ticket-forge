variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region for resources."
  type        = string
  default     = "us-east1"
}

variable "state_bucket" {
  description = "Name for the tfstate bucket (must be globally unique)."
  type        = string
}
