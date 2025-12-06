-- Rollback Migration 003: Remove entities table and geographic coordinates

-- Drop junction table first (foreign key constraints)
DROP TABLE IF EXISTS entity_mentions CASCADE;

-- Drop entities table
DROP TABLE IF EXISTS entities CASCADE;

-- Drop trigger function
DROP FUNCTION IF EXISTS update_entities_updated_at() CASCADE;

-- Log rollback completion
DO $$
BEGIN
    RAISE NOTICE '✓ Migration 003 rolled back: Removed entities table and geographic coordinates';
END $$;
