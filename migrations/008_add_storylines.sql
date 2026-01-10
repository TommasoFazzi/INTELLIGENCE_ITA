-- Migration 008: Add Storylines for Narrative Engine
-- Enables tracking of ongoing stories/narratives across multiple articles
-- Supports "delta reporting" - only report what's new in each story

-- ============================================
-- STORYLINES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS storylines (
    id SERIAL PRIMARY KEY,

    -- Identity
    title TEXT NOT NULL,                   -- "Taiwan Semiconductor Tensions"
    summary TEXT,                          -- Rolling summary, updated with each article

    -- Dual Vector Approach (prevents semantic drift)
    original_embedding VECTOR(384),        -- Immutable core semantic signature
    current_embedding VECTOR(384),         -- Evolves: 0.9*old + 0.1*new

    -- Key Actors (from NER)
    key_entities JSONB DEFAULT '[]',       -- ["Taiwan", "China", "TSMC", "USA"]

    -- Lifecycle Management
    status VARCHAR(20) DEFAULT 'ACTIVE'    -- ACTIVE, DORMANT, ARCHIVED
        CHECK (status IN ('ACTIVE', 'DORMANT', 'ARCHIVED')),
    start_date DATE,                       -- When story first appeared
    last_update TIMESTAMP,                 -- Last article added

    -- Metrics
    article_count INTEGER DEFAULT 0,       -- Total articles in this story
    momentum_score FLOAT DEFAULT 1.0,      -- Decay indicator (1.0 = hot, <0.3 = dormant)

    -- Classification
    category VARCHAR(50),                  -- intelligence, tech_economy, markets, etc.

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- ARTICLE-STORYLINE JUNCTION TABLE (M:N)
-- ============================================
-- An article can belong to multiple storylines
-- e.g., "US sanctions Chinese chips" belongs to both
--       "US-China Trade War" AND "Tech Decoupling" storylines

CREATE TABLE IF NOT EXISTS article_storylines (
    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    storyline_id INTEGER NOT NULL REFERENCES storylines(id) ON DELETE CASCADE,

    -- Relationship metadata
    relevance_score FLOAT DEFAULT 1.0      -- How central is this article (0-1)
        CHECK (relevance_score >= 0 AND relevance_score <= 1),
    is_origin BOOLEAN DEFAULT FALSE,       -- TRUE if this article started the storyline

    -- Timestamps
    added_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (article_id, storyline_id)
);

-- ============================================
-- INDEXES
-- ============================================

-- Storyline lookups
CREATE INDEX IF NOT EXISTS idx_storylines_status
    ON storylines(status);

CREATE INDEX IF NOT EXISTS idx_storylines_category
    ON storylines(category);

CREATE INDEX IF NOT EXISTS idx_storylines_last_update
    ON storylines(last_update DESC);

CREATE INDEX IF NOT EXISTS idx_storylines_momentum
    ON storylines(momentum_score DESC)
    WHERE status = 'ACTIVE';

-- Vector similarity search (requires pgvector ivfflat)
-- Note: Only create if enough rows exist, otherwise use exact search
-- CREATE INDEX idx_storylines_current_emb
--     ON storylines USING ivfflat (current_embedding vector_cosine_ops)
--     WITH (lists = 50);

-- For small datasets, use exact search (no index needed for <1000 rows)
-- pgvector will automatically use sequential scan

-- Junction table lookups
CREATE INDEX IF NOT EXISTS idx_article_storylines_article
    ON article_storylines(article_id);

CREATE INDEX IF NOT EXISTS idx_article_storylines_storyline
    ON article_storylines(storyline_id);

CREATE INDEX IF NOT EXISTS idx_article_storylines_relevance
    ON article_storylines(storyline_id, relevance_score DESC);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to update storyline's updated_at timestamp
CREATE OR REPLACE FUNCTION update_storyline_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for auto-updating timestamp
DROP TRIGGER IF EXISTS storylines_updated_at ON storylines;
CREATE TRIGGER storylines_updated_at
    BEFORE UPDATE ON storylines
    FOR EACH ROW
    EXECUTE FUNCTION update_storyline_timestamp();

-- ============================================
-- VIEWS FOR COMMON QUERIES
-- ============================================

-- Active storylines with article count (for dashboard)
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

-- Articles with their storylines (for report generation)
CREATE OR REPLACE VIEW v_articles_with_storylines AS
SELECT
    a.id AS article_id,
    a.title AS article_title,
    a.published_date,
    a.source,
    a.category,
    ARRAY_AGG(s.id) AS storyline_ids,
    ARRAY_AGG(s.title) AS storyline_titles
FROM articles a
LEFT JOIN article_storylines als ON a.id = als.article_id
LEFT JOIN storylines s ON als.storyline_id = s.id
GROUP BY a.id, a.title, a.published_date, a.source, a.category;

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON TABLE storylines IS 'Narrative threads that group related articles over time';
COMMENT ON TABLE article_storylines IS 'Many-to-many relationship between articles and storylines';
COMMENT ON COLUMN storylines.original_embedding IS 'Immutable semantic signature from first article';
COMMENT ON COLUMN storylines.current_embedding IS 'Evolving embedding: 0.9*old + 0.1*new with each article';
COMMENT ON COLUMN storylines.momentum_score IS 'Decay indicator: starts at 1.0, decays weekly if no new articles';
COMMENT ON COLUMN article_storylines.is_origin IS 'TRUE if this article initiated the storyline';
