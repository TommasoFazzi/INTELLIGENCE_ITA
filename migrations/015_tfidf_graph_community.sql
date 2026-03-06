-- Migration 015: TF-IDF entity weights + community detection support
-- Run: psql $DATABASE_URL -f migrations/015_tfidf_graph_community.sql

-- 1. Community ID column on storylines
ALTER TABLE storylines ADD COLUMN IF NOT EXISTS community_id INTEGER DEFAULT NULL;

-- 2. Entity IDF materialized view
--    Counts how many storylines contain each entity and computes IDF score.
--    Higher IDF = rarer entity = stronger signal for Jaccard edges.
CREATE MATERIALIZED VIEW IF NOT EXISTS entity_idf AS
SELECT
    entity,
    COUNT(*)::integer AS doc_freq,
    LN(
        CAST(
            (SELECT COUNT(*) FROM storylines
             WHERE narrative_status IN ('emerging', 'active', 'stabilized'))
        AS float)
        / NULLIF(COUNT(*), 0)
    ) AS idf
FROM storylines,
     jsonb_array_elements_text(key_entities) AS entity
WHERE key_entities IS NOT NULL
  AND jsonb_typeof(key_entities) = 'array'
  AND narrative_status IN ('emerging', 'active', 'stabilized')
GROUP BY entity;

-- 3. Unique index required for REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_idf_unique ON entity_idf(entity);

-- 4. Index for community_id lookups (API color-by-community queries)
CREATE INDEX IF NOT EXISTS idx_storylines_community ON storylines(community_id)
    WHERE community_id IS NOT NULL;

-- 5. Update v_active_storylines to include community_id
CREATE OR REPLACE VIEW v_active_storylines AS
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
    s.community_id,
    (s.summary_vector IS NOT NULL) AS has_summary_vector,
    EXTRACT(DAY FROM NOW() - s.start_date)::INTEGER AS days_active,
    EXTRACT(DAY FROM NOW() - s.last_update)::INTEGER AS days_since_update
FROM storylines s
WHERE s.narrative_status IN ('emerging', 'active')
ORDER BY s.momentum_score DESC, s.last_update DESC;

-- Rollback:
-- DROP MATERIALIZED VIEW IF EXISTS entity_idf;
-- ALTER TABLE storylines DROP COLUMN IF EXISTS community_id;
-- Restore v_active_storylines without community_id (see migration 012)
