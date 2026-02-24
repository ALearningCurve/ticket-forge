# Airflow Local Run (Repo Root Compose)

Airflow now runs from the **root** `docker-compose.yml`.

## Why

To avoid duplicate Postgres definitions and split ownership.

Single source of truth:

- `docker-compose.yml` (repo root)

## Prerequisites

- Docker Desktop / Docker Compose
- GitHub token for ticket scraping (see [setup](../../apps/training/README.md)) in `.env`
- Google mail application token (see [setup](../../apps/training/README.md)) in `.env`


## Start Services

From repo root:

```powershell
docker compose up -d postgres pgadmin airflow
```

or if you have just installed:

```sh
just airflow-up
```

UIs:

- Airflow: http://localhost:8080 (user: `airflow`, pass: `airflow`)
     - login with username and password of `airflow`
- pgAdmin: http://localhost:5050
     - see [connecting section](#connecting-to-pg-admin) to view on the gui

## DAGs

- `ticket_etl` (scrape -> transform -> tickets/assignments load)
- `resume_etl` (resume payload ingest)

Trigger ticket ETL:

```powershell
docker compose exec airflow airflow dags trigger ticket_etl
```

Trigger ticket ETL with test limit:

```powershell
docker compose exec airflow airflow dags trigger ticket_etl --conf '{"limit_per_state": 10}'
```

Trigger resume ingest:

```powershell
docker compose exec airflow airflow dags trigger resume_etl
```

Trigger resume ingest with payload:

```powershell
docker compose exec airflow airflow dags trigger resume_etl --conf '{"resumes":[{"filename":"john_doe.pdf","content_base64":"<base64_pdf>","github_username":"johndoe","full_name":"John Doe"}]}'
```

## Stop

```powershell
docker compose down
```

Remove volumes too:

```powershell
docker compose down -v
```


## Connecting to pg-admin

- Access pgAdmin: Open your browser to http://localhost:5050 (or the port mapped in your compose file) and log in with your configured credentials.
- Open Server Dialog: Click Add New Server on the dashboard=
- General Tab: Enter a recognizable Name for your connection (e.g., Docker DB).
- Connection Tab:

    Host name/address: Enter the exact Service Name of your Postgres container from the docker-compose.yml (e.g., postgres). Do not use localhost.
    Port: Use 5432 (the standard internal container port).
    Maintenance database: Enter postgres (or your custom POSTGRES_DB name).
    Username: Enter your ticketforge value.
    Password: Enter your root value.

- Save: Click Save. pgAdmin will now use the Docker internal network to resolve the service name and connect.
