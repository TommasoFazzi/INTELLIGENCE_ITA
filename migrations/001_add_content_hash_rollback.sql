-- Rollback Migration: Remove content_hash column
-- Date: 2025-11-29
-- Purpose: Rollback Phase 2 deduplication changes if needed

-- Drop indexes first
DROP INDEX IF EXISTS idx_articles_content_hash;
DROP INDEX IF EXISTS idx_articles_published_date;

-- Remove column
ALTER TABLE articles
DROP COLUMN IF EXISTS content_hash;
