-- Rollback Migration 012: Remove Narrative Graph Schema

-- Drop views
DROP VIEW IF EXISTS v_storyline_graph;

-- Restore original v_active_storylines (from migration 008)
CREATE OR REPLACE VIEW v_active_storylines AS
SELECT
    s.id,
    s.title,
    s.summary,
    s.status,
    s.category,
    s.article_count,
    s.momentum_score,
    s.start_date,
    s.last_update,
    s.key_entities,
    EXTRACT(DAY FROM NOW() - s.start_date)::INTEGER AS days_active,
    EXTRACT(DAY FROM NOW() - s.last_update)::INTEGER AS days_since_update
FROM storylines s
WHERE s.status = 'ACTIVE'
ORDER BY s.momentum_score DESC, s.last_update DESC;

-- Drop edge table and its trigger/function
DROP TRIGGER IF EXISTS storyline_edges_updated_at ON storyline_edges;
DROP FUNCTION IF EXISTS update_edge_timestamp();
DROP TABLE IF EXISTS storyline_edges;

-- Drop sync trigger and function
DROP TRIGGER IF EXISTS sync_narrative_status ON storylines;
DROP FUNCTION IF EXISTS sync_narrative_to_legacy_status();

-- Drop new indexes
DROP INDEX IF EXISTS idx_storylines_summary_vector;
DROP INDEX IF EXISTS idx_storylines_narrative_status;

-- Drop new columns from storylines
ALTER TABLE storylines DROP COLUMN IF EXISTS summary_vector;
ALTER TABLE storylines DROP COLUMN IF EXISTS narrative_status;
ALTER TABLE storylines DROP COLUMN IF EXISTS last_graph_update;

-- Drop constraint
ALTER TABLE storylines DROP CONSTRAINT IF EXISTS storylines_narrative_status_check;
