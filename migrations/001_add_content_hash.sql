-- Migration: Add content_hash column for content-based deduplication
-- Date: 2025-11-29
-- Purpose: Phase 2 deduplication - detect articles with identical content from different sources
--
-- This migration adds:
-- 1. content_hash column (MD5 hash of clean_text)
-- 2. Index on content_hash for fast lookups
-- 3. Index on published_date for time-based queries

-- Add content_hash column
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS content_hash VARCHAR(32);

-- Create index on content_hash for fast duplicate detection
CREATE INDEX IF NOT EXISTS idx_articles_content_hash
ON articles(content_hash);

-- Create index on published_date for time-windowed queries (7-day lookback)
CREATE INDEX IF NOT EXISTS idx_articles_published_date
ON articles(published_date);

-- Add comment to column for documentation
COMMENT ON COLUMN articles.content_hash IS
'MD5 hash of clean_text field for content-based deduplication. Computed during save_article().';

-- Migration verification query
-- Run this after migration to verify:
-- SELECT COUNT(*) as total,
--        COUNT(content_hash) as with_hash,
--        COUNT(*) - COUNT(content_hash) as without_hash
-- FROM articles;
