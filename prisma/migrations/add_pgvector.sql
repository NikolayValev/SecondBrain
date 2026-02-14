-- Migration: Add pgvector support
-- Run: psql $DATABASE_URL -f prisma/migrations/add_pgvector.sql

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create embedding_vectors table alongside existing embeddings table
--    This stores native vector types for fast HNSW-indexed similarity search.
CREATE TABLE IF NOT EXISTS embedding_vectors (
    id          SERIAL PRIMARY KEY,
    chunk_id    INTEGER UNIQUE NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding   vector(768) NOT NULL,  -- change 768 to match your model dimensions
    model       TEXT NOT NULL,
    dimensions  INTEGER NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Create HNSW index for cosine similarity
CREATE INDEX IF NOT EXISTS idx_ev_embedding
    ON embedding_vectors
    USING hnsw (embedding vector_cosine_ops);

-- 4. (Optional) Backfill from existing embeddings table
--    Uncomment and adjust dimensions if you want to migrate existing data:
-- INSERT INTO embedding_vectors (chunk_id, embedding, model, dimensions, created_at)
-- SELECT chunk_id,
--        -- Unpack bytea → float4[] → vector  (requires custom function or application code)
--        model,
--        dimensions,
--        created_at
-- FROM embeddings
-- ON CONFLICT (chunk_id) DO NOTHING;
