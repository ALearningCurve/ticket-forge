-- Example queries for the ticket-forge Postgres + PgVector schema.
-- Run after: docker compose up -d && psql -h localhost -U ticketforge -d ticketforge -f scripts/postgres/example_queries.sql
-- (Some examples are SELECT-only; others show INSERT patterns.)

-- ---------------------------------------------------------------------------
-- 1. Insert a user (engineer) with resume and profile vectors
-- ---------------------------------------------------------------------------
-- INSERT INTO users (
--   member_id,
--   full_name,
--   resume_base_vector,
--   profile_vector,
--   skill_keywords,
--   tickets_closed_count
-- ) VALUES (
--   gen_random_uuid(),
--   'Jane Doe',
--   '[0.1, -0.2, 0.3, ...]'::vector(384),  -- Resume embedding
--   '[0.1, -0.2, 0.3, ...]'::vector(384),  -- Initial profile = resume
--   to_tsvector('english', 'Kubernetes AWS Terraform Docker Python'),
--   0
-- );

-- ---------------------------------------------------------------------------
-- 2. Insert a ticket with vector and labels (JSONB)
-- ---------------------------------------------------------------------------
-- INSERT INTO tickets (
--   ticket_id,
--   title,
--   description,
--   ticket_vector,
--   labels,
--   status,
--   resolution_time_actual
-- ) VALUES (
--   'terraform-123',
--   'Bug: provider crash on EKS cluster',
--   'The Terraform provider crashes when deploying to EKS...',
--   '[0.1, -0.2, ...]'::vector(384),
--   '["bug", "priority:high", "terraform", "eks"]'::jsonb,
--   'closed',
--   INTERVAL '4 hours 30 minutes'
-- );

-- ---------------------------------------------------------------------------
-- 3. Create an assignment
-- ---------------------------------------------------------------------------
-- INSERT INTO assignments (ticket_id, engineer_id)
-- VALUES (
--   'terraform-123',
--   (SELECT member_id FROM users WHERE full_name = 'Jane Doe' LIMIT 1)
-- );

-- ---------------------------------------------------------------------------
-- 4. Semantic search: find users closest to a ticket vector (cosine similarity)
-- ---------------------------------------------------------------------------
-- Replace :ticket_vector with your query vector.
/*
WITH semantic_results AS (
  SELECT
    member_id,
    full_name,
    1 - (profile_vector <=> :ticket_vector::vector(384)) AS similarity,
    ROW_NUMBER() OVER (ORDER BY profile_vector <=> :ticket_vector::vector(384)) AS rank
  FROM users
  ORDER BY profile_vector <=> :ticket_vector::vector(384)
  LIMIT 10
)
SELECT * FROM semantic_results;
*/

-- Example with existing ticket (run when tickets exist):
SELECT
  u.member_id,
  u.full_name,
  1 - (u.profile_vector <=> t.ticket_vector) AS similarity
FROM users u
CROSS JOIN (SELECT ticket_vector FROM tickets LIMIT 1) t
ORDER BY u.profile_vector <=> t.ticket_vector
LIMIT 5;

-- ---------------------------------------------------------------------------
-- 5. Lexical search: find users by skill keywords (full-text search)
-- ---------------------------------------------------------------------------
-- Search for users with specific skills (e.g., "Kubernetes" AND "AWS").
SELECT
  member_id,
  full_name,
  ts_rank(skill_keywords, query) AS rank
FROM users,
     to_tsquery('english', 'Kubernetes & AWS') AS query
WHERE skill_keywords @@ query
ORDER BY rank DESC
LIMIT 5;

-- ---------------------------------------------------------------------------
-- 6. Hybrid search: Reciprocal Rank Fusion (RRF) combining semantic + lexical
-- ---------------------------------------------------------------------------
-- This combines vector similarity and keyword matching using RRF.
-- RRF formula: score = 1 / (k + rank) where k is typically 60.
/*
WITH semantic_results AS (
  SELECT
    member_id,
    full_name,
    ROW_NUMBER() OVER (ORDER BY profile_vector <=> :ticket_vector::vector(384)) AS rank
  FROM users
  ORDER BY profile_vector <=> :ticket_vector::vector(384)
  LIMIT 10
),
lexical_results AS (
  SELECT
    member_id,
    full_name,
    ROW_NUMBER() OVER (ORDER BY ts_rank(skill_keywords, :keyword_query) DESC) AS rank
  FROM users,
       to_tsquery('english', :keyword_query) AS query
  WHERE skill_keywords @@ query
  LIMIT 10
),
rrf_scores AS (
  SELECT
    COALESCE(s.member_id, l.member_id) AS member_id,
    COALESCE(s.full_name, l.full_name) AS full_name,
    (1.0 / (60 + COALESCE(s.rank, 100))) + (1.0 / (60 + COALESCE(l.rank, 100))) AS rrf_score
  FROM semantic_results s
  FULL OUTER JOIN lexical_results l ON s.member_id = l.member_id
)
SELECT member_id, full_name, rrf_score
FROM rrf_scores
ORDER BY rrf_score DESC
LIMIT 5;
*/

-- ---------------------------------------------------------------------------
-- 7. Update user profile vector (moving average with decay)
-- ---------------------------------------------------------------------------
-- When a ticket is closed, update the engineer's profile_vector.
-- This is a simplified example; the actual decay function would weight recent tickets more.
/*
WITH recent_tickets AS (
  SELECT ticket_vector
  FROM tickets t
  JOIN assignments a ON t.ticket_id = a.ticket_id
  WHERE a.engineer_id = :engineer_id
    AND t.status = 'closed'
    AND a.assigned_at > NOW() - INTERVAL '6 months'
  ORDER BY a.assigned_at DESC
  LIMIT 10
),
weighted_centroid AS (
  SELECT AVG(ticket_vector) AS new_profile_vector
  FROM recent_tickets
)
UPDATE users
SET
  profile_vector = (
    -- Weighted average: 70% current profile, 30% new tickets
    0.7 * profile_vector + 0.3 * (SELECT new_profile_vector FROM weighted_centroid)
  ),
  tickets_closed_count = tickets_closed_count + 1,
  updated_at = now()
WHERE member_id = :engineer_id;
*/

-- ---------------------------------------------------------------------------
-- 8. Update user skill keywords from closed tickets
-- ---------------------------------------------------------------------------
-- Extract keywords from ticket titles/descriptions and update skill_keywords.
/*
UPDATE users
SET
  skill_keywords = (
    SELECT to_tsvector('english', string_agg(title || ' ' || description, ' '))
    FROM tickets t
    JOIN assignments a ON t.ticket_id = a.ticket_id
    WHERE a.engineer_id = users.member_id
      AND t.status = 'closed'
      AND a.assigned_at > NOW() - INTERVAL '6 months'
  ),
  updated_at = now()
WHERE member_id = :engineer_id;
*/

-- ---------------------------------------------------------------------------
-- 9. Find tickets by label (JSONB query)
-- ---------------------------------------------------------------------------
SELECT
  ticket_id,
  title,
  labels,
  status
FROM tickets
WHERE labels @> '["bug"]'::jsonb
ORDER BY created_at DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- 10. Get assignment history for an engineer
-- ---------------------------------------------------------------------------
SELECT
  a.assigned_at,
  t.ticket_id,
  t.title,
  t.status,
  t.resolution_time_actual
FROM assignments a
JOIN tickets t ON a.ticket_id = t.ticket_id
WHERE a.engineer_id = (SELECT member_id FROM users LIMIT 1)
ORDER BY a.assigned_at DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- 11. Aggregate statistics: tickets closed per engineer
-- ---------------------------------------------------------------------------
SELECT
  u.member_id,
  u.full_name,
  u.tickets_closed_count,
  COUNT(a.assignment_id) AS actual_assignments,
  AVG(EXTRACT(EPOCH FROM t.resolution_time_actual) / 3600) AS avg_resolution_hours
FROM users u
LEFT JOIN assignments a ON u.member_id = a.engineer_id
LEFT JOIN tickets t ON a.ticket_id = t.ticket_id AND t.status = 'closed'
GROUP BY u.member_id, u.full_name, u.tickets_closed_count
ORDER BY tickets_closed_count DESC;

-- ---------------------------------------------------------------------------
-- 12. List all tables and row counts
-- ---------------------------------------------------------------------------
SELECT
  schemaname,
  relname AS table_name,
  n_live_tup AS row_estimate
FROM pg_stat_user_tables
ORDER BY schemaname, relname;
