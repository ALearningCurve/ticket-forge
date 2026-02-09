-- Enable pgvector for storing and querying embedding vectors.
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for full-text search (used with tsvector for hybrid search).
CREATE EXTENSION IF NOT EXISTS pg_trgm;
