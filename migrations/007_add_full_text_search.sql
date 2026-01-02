-- Migration 007: Add Full-Text Search Support
-- Adds tsvector columns and GIN indexes for PostgreSQL Full-Text Search
-- Enables hybrid search (vector + keyword) for better retrieval

BEGIN;

-- Add tsvector columns for full-text search
ALTER TABLE articles ADD COLUMN IF NOT EXISTS full_text_tsv tsvector;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector;

-- Create custom multilingual text search configuration (English + Italian)
-- Using 'simple' as base to avoid aggressive stemming that removes proper nouns
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_ts_config WHERE cfgname = 'multilingual'
    ) THEN
        CREATE TEXT SEARCH CONFIGURATION multilingual (COPY = simple);
    END IF;
END
$$;

-- Populate tsvector columns for existing data
UPDATE articles
SET full_text_tsv = to_tsvector('multilingual', COALESCE(title, '') || ' ' || COALESCE(full_text, ''))
WHERE full_text_tsv IS NULL;

UPDATE chunks
SET content_tsv = to_tsvector('multilingual', COALESCE(content, ''))
WHERE content_tsv IS NULL;

-- Create GIN indexes for fast FTS (Generalized Inverted Index)
-- GIN indexes provide ~10-50ms query times on millions of documents
CREATE INDEX IF NOT EXISTS idx_articles_full_text_tsv ON articles USING GIN(full_text_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN(content_tsv);

-- Auto-update triggers for tsvector columns
-- Ensures tsvector stays in sync when articles/chunks are inserted or updated
CREATE OR REPLACE FUNCTION articles_tsv_update() RETURNS trigger AS $$
BEGIN
    NEW.full_text_tsv := to_tsvector('multilingual',
        COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.full_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('multilingual', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing triggers if they exist
DROP TRIGGER IF EXISTS articles_tsv_trigger ON articles;
DROP TRIGGER IF EXISTS chunks_tsv_trigger ON chunks;

-- Create triggers
CREATE TRIGGER articles_tsv_trigger
    BEFORE INSERT OR UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION articles_tsv_update();

CREATE TRIGGER chunks_tsv_trigger
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update();

COMMIT;

-- Verification queries (run separately to check migration success):
-- SELECT COUNT(*) FROM articles WHERE full_text_tsv IS NOT NULL;
-- SELECT COUNT(*) FROM chunks WHERE content_tsv IS NOT NULL;
-- \d articles -- Should show full_text_tsv column
-- \d chunks -- Should show content_tsv column
-- \di idx_articles_full_text_tsv -- Should show GIN index
-- \di idx_chunks_content_tsv -- Should show GIN index
