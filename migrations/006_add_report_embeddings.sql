-- Migration 006: Add embeddings to reports table for RAG on reports
-- Required for The Oracle hybrid search (chunks + reports)

BEGIN;

-- Add embedding column for semantic search on reports
ALTER TABLE reports ADD COLUMN IF NOT EXISTS content_embedding vector(384);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_reports_embedding
    ON reports USING hnsw (content_embedding vector_cosine_ops);

-- Verify the column was added
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'reports' AND column_name = 'content_embedding'
    ) THEN
        RAISE NOTICE 'Migration 006: content_embedding column added successfully';
    ELSE
        RAISE EXCEPTION 'Migration 006: Failed to add content_embedding column';
    END IF;
END $$;

COMMIT;
