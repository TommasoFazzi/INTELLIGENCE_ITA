-- Rollback Migration 008: Remove Storylines

-- Drop views first
DROP VIEW IF EXISTS v_articles_with_storylines;
DROP VIEW IF EXISTS v_active_storylines;

-- Drop trigger and function
DROP TRIGGER IF EXISTS storylines_updated_at ON storylines;
DROP FUNCTION IF EXISTS update_storyline_timestamp();

-- Drop tables (junction table first due to FK)
DROP TABLE IF EXISTS article_storylines;
DROP TABLE IF EXISTS storylines;
