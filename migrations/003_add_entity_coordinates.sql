-- Migration 003: Create entities table and add geographic coordinates
-- Purpose: Extract entities from articles JSON into dedicated table for Intelligence Map

-- Create entities table
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- PERSON, ORG, GPE, LOC, FAC, etc.
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    mention_count INTEGER DEFAULT 1,
    
    -- Geographic coordinates
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    geo_status VARCHAR(20) DEFAULT 'PENDING',
    geocoded_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(name, entity_type)
);

-- Add comments for documentation
COMMENT ON TABLE entities IS 'Named entities extracted from articles for Intelligence Map visualization';
COMMENT ON COLUMN entities.name IS 'Entity name (e.g., "Taiwan", "IBM")';
COMMENT ON COLUMN entities.entity_type IS 'NER type: PERSON, ORG, GPE, LOC, FAC, etc.';
COMMENT ON COLUMN entities.mention_count IS 'Number of times entity appears across all articles';
COMMENT ON COLUMN entities.latitude IS 'Geographic latitude (-90 to 90)';
COMMENT ON COLUMN entities.longitude IS 'Geographic longitude (-180 to 180)';
COMMENT ON COLUMN entities.geo_status IS 'Geocoding status: PENDING, FOUND, NOT_FOUND, RETRY';
COMMENT ON COLUMN entities.geocoded_at IS 'Timestamp when entity was geocoded';

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_mention_count ON entities(mention_count DESC);
CREATE INDEX IF NOT EXISTS idx_entities_coordinates 
ON entities(latitude, longitude) 
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_geo_status ON entities(geo_status);

-- Create entity_mentions junction table (links entities to articles)
CREATE TABLE IF NOT EXISTS entity_mentions (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    mention_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(entity_id, article_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_article ON entity_mentions(article_id);

-- Update timestamp trigger for entities
CREATE OR REPLACE FUNCTION update_entities_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_entities_updated_at ON entities;
CREATE TRIGGER update_entities_updated_at
    BEFORE UPDATE ON entities
    FOR EACH ROW
    EXECUTE FUNCTION update_entities_updated_at();

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE '✓ Migration 003 completed: Created entities table with geographic coordinates';
END $$;
