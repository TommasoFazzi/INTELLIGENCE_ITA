-- Migration 012: Narrative Graph Schema
-- Extends storylines for narrative engine + adds storyline_edges for graph relationships
--
-- Prerequisite: Migration 008 (storylines + article_storylines tables must exist)

-- ============================================
-- ALTER STORYLINES - New Columns
-- ============================================

-- Summary vector for semantic search on narrative summaries
ALTER TABLE storylines
    ADD COLUMN IF NOT EXISTS summary_vector vector(384);

-- Narrative lifecycle status (coexists with legacy 'status' column)
-- emerging: new storyline, < 3 articles
-- active: confirmed narrative thread, growing
-- stabilized: mature narrative, low recent activity
-- archived: no longer relevant
ALTER TABLE storylines
    ADD COLUMN IF NOT EXISTS narrative_status VARCHAR(20) DEFAULT 'emerging';

-- Add CHECK constraint separately (IF NOT EXISTS not supported for constraints)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'storylines_narrative_status_check'
    ) THEN
        ALTER TABLE storylines
            ADD CONSTRAINT storylines_narrative_status_check
            CHECK (narrative_status IN ('emerging', 'active', 'stabilized', 'archived'));
    END IF;
END $$;

-- Timestamp for last graph edge computation
ALTER TABLE storylines
    ADD COLUMN IF NOT EXISTS last_graph_update TIMESTAMP;

-- ============================================
-- BACKWARD COMPATIBILITY TRIGGER
-- ============================================
-- Syncs legacy 'status' column from 'narrative_status':
--   emerging/active → ACTIVE
--   stabilized      → DORMANT
--   archived        → ARCHIVED

CREATE OR REPLACE FUNCTION sync_narrative_to_legacy_status()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.narrative_status IN ('emerging', 'active') THEN
        NEW.status = 'ACTIVE';
    ELSIF NEW.narrative_status = 'stabilized' THEN
        NEW.status = 'DORMANT';
    ELSIF NEW.narrative_status = 'archived' THEN
        NEW.status = 'ARCHIVED';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS sync_narrative_status ON storylines;
CREATE TRIGGER sync_narrative_status
    BEFORE INSERT OR UPDATE OF narrative_status ON storylines
    FOR EACH ROW
    EXECUTE FUNCTION sync_narrative_to_legacy_status();

-- ============================================
-- STORYLINE EDGES TABLE (Graph Relationships)
-- ============================================

CREATE TABLE IF NOT EXISTS storyline_edges (
    id SERIAL PRIMARY KEY,

    -- Graph endpoints
    source_story_id INTEGER NOT NULL REFERENCES storylines(id) ON DELETE CASCADE,
    target_story_id INTEGER NOT NULL REFERENCES storylines(id) ON DELETE CASCADE,

    -- Relationship metadata
    relation_type VARCHAR(30) DEFAULT 'relates_to',  -- relates_to, causes, escalates, contradicts
    weight FLOAT DEFAULT 0.0                         -- Connection strength (Jaccard index)
        CHECK (weight >= 0.0 AND weight <= 1.0),
    explanation TEXT,                                 -- LLM-generated explanation (optional, lazy)

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- No duplicate edges
    UNIQUE (source_story_id, target_story_id),

    -- No self-loops
    CHECK (source_story_id != target_story_id)
);

-- Auto-update timestamp on edge modification
CREATE OR REPLACE FUNCTION update_edge_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS storyline_edges_updated_at ON storyline_edges;
CREATE TRIGGER storyline_edges_updated_at
    BEFORE UPDATE ON storyline_edges
    FOR EACH ROW
    EXECUTE FUNCTION update_edge_timestamp();

-- ============================================
-- INDEXES
-- ============================================

-- HNSW index on summary_vector for fast semantic search
CREATE INDEX IF NOT EXISTS idx_storylines_summary_vector
    ON storylines USING hnsw (summary_vector vector_cosine_ops);

-- Narrative status lookups
CREATE INDEX IF NOT EXISTS idx_storylines_narrative_status
    ON storylines(narrative_status);

-- Edge lookups
CREATE INDEX IF NOT EXISTS idx_storyline_edges_source
    ON storyline_edges(source_story_id);

CREATE INDEX IF NOT EXISTS idx_storyline_edges_target
    ON storyline_edges(target_story_id);

CREATE INDEX IF NOT EXISTS idx_storyline_edges_weight
    ON storyline_edges(weight DESC)
    WHERE weight > 0.0;

-- ============================================
-- UPDATE VIEW
-- ============================================

-- Must DROP first because column list changed (CREATE OR REPLACE can't alter column names/order)
DROP VIEW IF EXISTS v_active_storylines;
CREATE VIEW v_active_storylines AS
SELECT
    s.id,
    s.title,
    s.summary,
    s.status,
    s.narrative_status,
    s.category,
    s.article_count,
    s.momentum_score,
    s.start_date,
    s.last_update,
    s.key_entities,
    s.last_graph_update,
    (s.summary_vector IS NOT NULL) AS has_summary_vector,
    EXTRACT(DAY FROM NOW() - s.start_date)::INTEGER AS days_active,
    EXTRACT(DAY FROM NOW() - s.last_update)::INTEGER AS days_since_update
FROM storylines s
WHERE s.narrative_status IN ('emerging', 'active')
ORDER BY s.momentum_score DESC, s.last_update DESC;

-- View for graph visualization: storylines + their edges
CREATE OR REPLACE VIEW v_storyline_graph AS
SELECT
    e.id AS edge_id,
    e.source_story_id,
    s1.title AS source_title,
    s1.narrative_status AS source_status,
    s1.momentum_score AS source_momentum,
    e.target_story_id,
    s2.title AS target_title,
    s2.narrative_status AS target_status,
    s2.momentum_score AS target_momentum,
    e.relation_type,
    e.weight,
    e.explanation
FROM storyline_edges e
JOIN storylines s1 ON e.source_story_id = s1.id
JOIN storylines s2 ON e.target_story_id = s2.id
WHERE s1.narrative_status IN ('emerging', 'active')
  AND s2.narrative_status IN ('emerging', 'active')
ORDER BY e.weight DESC;

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON COLUMN storylines.summary_vector IS 'Embedding of the narrative summary for semantic search';
COMMENT ON COLUMN storylines.narrative_status IS 'Lifecycle: emerging → active → stabilized → archived';
COMMENT ON COLUMN storylines.last_graph_update IS 'When graph edges were last recomputed for this storyline';
COMMENT ON TABLE storyline_edges IS 'Graph edges between storylines based on entity overlap and semantic similarity';
COMMENT ON COLUMN storyline_edges.weight IS 'Jaccard index of shared entities between storylines';
COMMENT ON COLUMN storyline_edges.relation_type IS 'Type: relates_to, causes, escalates, contradicts';
