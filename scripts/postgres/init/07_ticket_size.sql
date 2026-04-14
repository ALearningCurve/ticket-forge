-- Ticket size prediction columns for project_tickets.
-- Added by the inference/sizing feature.

ALTER TABLE project_tickets ADD COLUMN IF NOT EXISTS size_bucket VARCHAR(20);
ALTER TABLE project_tickets ADD COLUMN IF NOT EXISTS size_source VARCHAR(20);
ALTER TABLE project_tickets ADD COLUMN IF NOT EXISTS size_confidence FLOAT;
ALTER TABLE project_tickets ADD COLUMN IF NOT EXISTS size_updated_at TIMESTAMPTZ;