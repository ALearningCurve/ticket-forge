# Demo Seeding

This directory contains the project-first demo seeding flow for TicketForge.

The main script is:

- `demo/seed_demo_project.py`

It performs an end-to-end setup for demo validation:

1. Resolves the DB DSN from Terraform outputs (or uses `--dsn` if provided).
2. Auto-starts Cloud SQL proxy when needed for localhost DSNs.
3. Recreates demo users derived from sample ticket assignees.
4. Stores username/password mapping to a local JSON file.
5. Builds non-stub `users.profile_vector` and `users.skill_keywords` by replaying closed assigned tickets.
6. Recreates the demo project board and seeds tickets into `project_tickets`.
7. Distributes synthetic open tickets across deterministic sprint windows using
   `due_date` bands and `sprint-*` labels.
8. Applies deterministic workload scatter for synthetic open tickets so member
   assignment counts are not perfectly uniform.
9. Caps seeded assignments with a deterministic per-member points budget so
   board capacity remains aligned with the default weekly threshold.

## Inputs

- `demo/data/sample_tickets.json` (default seeding input)
- `demo/data/real_tickets.json` (optional dataset)
- `demo/data/combined_tickets.json` (optional dataset)

Optional per-ticket sizing fields in input payloads:

- `ticket_size`
- `size_bucket`
- `size`

When present (and valid as `S|M|L|XL`), these are persisted as manual ticket sizes during seeding.

## Output

- `demo/data/demo_user_credentials.json` (ignored by git)

This file includes generated demo credentials and project metadata for the latest run.
All recreated demo users use the same shared password: `Demo123#`.

## Recommended Command

From repo root:

```bash
just demo-seed-project-data --project-slug seeded-demo-board --project-name "Seeded Demo Board" --owner-username darshanrk18 --users-input demo/data/sample_tickets.json --input demo/data/sample_tickets.json
```

## CLI Options

```bash
uv run python demo/seed_demo_project.py --help
```

Key flags:

- `--input`: ticket payload to seed into project board
- `--users-input`: payload used to derive relevant demo users
- `--retain-owner-user`: keep owner account instead of recreating it
- `--dsn`: explicit DSN override (skips terraform DSN lookup)
- `--no-auto-proxy`: disable automatic Cloud SQL proxy startup
- `--keep-proxy`: leave auto-started proxy running after completion
- `--credentials-out`: path to write credential mapping JSON

## Notes

- Re-running the script is intentional for demo refresh; it recreates the target project by slug.
- The default flow is optimized for demo verification of board data, ticket sizing, and recommendation behavior.
