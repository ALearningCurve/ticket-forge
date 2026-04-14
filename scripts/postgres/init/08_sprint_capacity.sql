-- Sprint capacity settings for TicketForge.
-- Runs after 07_ticket_size.sql.
-- Adds default ticket size, weekly point budget, and size-to-point mapping to projects.

-- Default ticket size for new tickets
ALTER TABLE projects
    ADD COLUMN IF NOT EXISTS default_ticket_size VARCHAR(5) DEFAULT 'M';

-- Weekly point budget per member
ALTER TABLE projects
    ADD COLUMN IF NOT EXISTS weekly_points_per_member INTEGER NOT NULL DEFAULT 10;

-- Points per ticket size (stored as JSON for flexibility)
-- Default: S=1, M=2, L=3, XL=5
ALTER TABLE projects
    ADD COLUMN IF NOT EXISTS size_points_map JSONB NOT NULL DEFAULT '{"S": 1, "M": 2, "L": 3, "XL": 5}';