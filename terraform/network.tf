resource "google_compute_network" "airflow_vpc" {
  name                    = "airflow-vpc-${var.environment}"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "airflow_subnet" {
  name          = "airflow-subnet-${var.environment}"
  ip_cidr_range = "10.20.0.0/24"
  region        = var.region
  network       = google_compute_network.airflow_vpc.id
}

resource "google_compute_subnetwork" "airflow_vm_subnet" {
  count         = local.airflow_uses_dedicated_network ? 1 : 0
  name          = "airflow-vm-subnet-${var.environment}-${local.effective_airflow_region}"
  ip_cidr_range = "10.21.0.0/24"
  region        = local.effective_airflow_region
  network       = google_compute_network.airflow_vpc.id
}

resource "google_compute_global_address" "private_service_range" {
  name          = "airflow-private-range-${var.environment}"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.airflow_vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.airflow_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_service_range.name]

  depends_on = [google_project_service.airflow_services]
}

resource "google_compute_router" "airflow_router" {
  name    = "airflow-router-${var.environment}"
  region  = var.region
  network = google_compute_network.airflow_vpc.id

  bgp {
    asn = 64514
  }
}

resource "google_compute_router" "airflow_vm_router" {
  count   = local.airflow_uses_dedicated_network ? 1 : 0
  name    = "airflow-vm-router-${var.environment}-${local.effective_airflow_region}"
  region  = local.effective_airflow_region
  network = google_compute_network.airflow_vpc.id

  bgp {
    asn = 64514
  }
}

resource "google_compute_router_nat" "airflow_nat" {
  name                               = "airflow-nat-${var.environment}"
  router                             = google_compute_router.airflow_router.name
  region                             = google_compute_router.airflow_router.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

resource "google_compute_router_nat" "airflow_vm_nat" {
  count                              = local.airflow_uses_dedicated_network ? 1 : 0
  name                               = "airflow-vm-nat-${var.environment}-${local.effective_airflow_region}"
  router                             = google_compute_router.airflow_vm_router[0].name
  region                             = google_compute_router.airflow_vm_router[0].region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

resource "google_compute_firewall" "airflow_allow_web_internal" {
  name    = "airflow-allow-web-internal-${var.environment}"
  network = google_compute_network.airflow_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = local.airflow_internal_source_ranges
  target_tags   = ["airflow-web"]
}

resource "google_compute_firewall" "airflow_allow_web_iap" {
  name    = "airflow-allow-web-iap-${var.environment}"
  network = google_compute_network.airflow_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  # IAP TCP forwarding source range.
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["airflow-web"]
}

resource "google_compute_firewall" "airflow_allow_ssh" {
  name    = "airflow-allow-ssh-${var.environment}"
  network = google_compute_network.airflow_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # IAP TCP forwarding source range.
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["airflow"]
}
