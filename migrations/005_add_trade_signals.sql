-- ============================================================================
-- MIGRATION 005: Trade Signals with Macro Alignment
-- ============================================================================
-- Description: Normalized trade signals table with FK to reports and optional
--              FK to articles. Supports both report-level and article-level signals.
-- Author: Intelligence ITA Team
-- Date: 2025-12-25
-- Dependencies: 004_add_market_intelligence_schema.sql
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. Trade Signals Table (Normalized)
-- ============================================================================
-- Stores trade signals extracted from:
-- - Report-level: High-conviction signals from macro analysis (article_id = NULL)
-- - Article-level: Signals from individual articles with macro alignment check

CREATE TABLE IF NOT EXISTS trade_signals (
    id SERIAL PRIMARY KEY,

    -- Foreign keys (report required, article optional)
    report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    article_id INTEGER REFERENCES articles(id) ON DELETE SET NULL,

    -- Signal data
    ticker VARCHAR(10) NOT NULL,
    signal TEXT NOT NULL CHECK (signal IN ('BULLISH', 'BEARISH', 'NEUTRAL', 'WATCHLIST')),
    timeframe TEXT NOT NULL CHECK (timeframe IN ('SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM')),
    rationale TEXT NOT NULL,

    -- Confidence and alignment
    confidence DECIMAL(3, 2) CHECK (confidence >= 0 AND confidence <= 1),
    alignment_score DECIMAL(3, 2) CHECK (alignment_score >= 0 AND alignment_score <= 1),

    -- Source tracking
    signal_source TEXT NOT NULL DEFAULT 'article' CHECK (signal_source IN ('report', 'article')),

    -- Metadata
    category TEXT,  -- GEOPOLITICS, DEFENSE, ECONOMY, CYBER, ENERGY
    impact_score INTEGER CHECK (impact_score >= 0 AND impact_score <= 10),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Unique constraint for report-level signals (article_id IS NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_signals_report_unique
    ON trade_signals(report_id, ticker, signal, timeframe)
    WHERE article_id IS NULL;

-- Unique constraint for article-level signals (article_id IS NOT NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_signals_article_unique
    ON trade_signals(report_id, article_id, ticker, signal, timeframe)
    WHERE article_id IS NOT NULL;

-- ============================================================================
-- 2. Indexes for Performance
-- ============================================================================
-- Query patterns: by report, by ticker, by date range, by source type

CREATE INDEX IF NOT EXISTS idx_trade_signals_report ON trade_signals(report_id);
CREATE INDEX IF NOT EXISTS idx_trade_signals_article ON trade_signals(article_id) WHERE article_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trade_signals_ticker ON trade_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_trade_signals_created ON trade_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trade_signals_source ON trade_signals(signal_source);

-- Composite index for dashboard query: "Get all signals for a ticker in date range"
CREATE INDEX IF NOT EXISTS idx_trade_signals_ticker_date
    ON trade_signals(ticker, created_at DESC);

-- Composite index for report-level signals query
CREATE INDEX IF NOT EXISTS idx_trade_signals_report_source
    ON trade_signals(report_id, signal_source);

-- ============================================================================
-- 3. Comments
-- ============================================================================
COMMENT ON TABLE trade_signals IS 'Normalized trade signals extracted from macro reports and articles via macro-first pipeline';
COMMENT ON COLUMN trade_signals.report_id IS 'FK to reports table - all signals must belong to a report';
COMMENT ON COLUMN trade_signals.article_id IS 'FK to articles table - NULL for report-level signals';
COMMENT ON COLUMN trade_signals.signal_source IS 'Whether signal was extracted from full report or individual article';
COMMENT ON COLUMN trade_signals.alignment_score IS 'How well this signal aligns with the macro narrative (1.0 = perfect alignment)';
COMMENT ON COLUMN trade_signals.confidence IS 'LLM confidence in this specific signal (0.0-1.0)';
COMMENT ON COLUMN trade_signals.category IS 'Category of the signal: GEOPOLITICS, DEFENSE, ECONOMY, CYBER, ENERGY';
COMMENT ON COLUMN trade_signals.impact_score IS 'Impact severity score (0-10) if available';

-- ============================================================================
-- 4. Helper Functions
-- ============================================================================

-- Function to get all signals for a report (ordered by source and alignment)
CREATE OR REPLACE FUNCTION get_report_signals(p_report_id INTEGER)
RETURNS TABLE(
    id INTEGER,
    ticker VARCHAR,
    signal TEXT,
    timeframe TEXT,
    rationale TEXT,
    confidence DECIMAL,
    alignment_score DECIMAL,
    signal_source TEXT,
    article_title TEXT,
    category TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ts.id,
        ts.ticker,
        ts.signal,
        ts.timeframe,
        ts.rationale,
        ts.confidence,
        ts.alignment_score,
        ts.signal_source,
        a.title AS article_title,
        ts.category
    FROM trade_signals ts
    LEFT JOIN articles a ON ts.article_id = a.id
    WHERE ts.report_id = p_report_id
    ORDER BY
        ts.signal_source DESC,  -- 'report' before 'article'
        ts.alignment_score DESC NULLS LAST,
        ts.confidence DESC NULLS LAST;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_report_signals IS 'Returns all trade signals for a given report, ordered by source and alignment score';

-- Function to get signal summary by ticker (aggregated)
CREATE OR REPLACE FUNCTION get_ticker_signal_summary(
    p_ticker VARCHAR,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE(
    ticker VARCHAR,
    total_signals BIGINT,
    bullish_count BIGINT,
    bearish_count BIGINT,
    avg_confidence DECIMAL,
    avg_alignment DECIMAL,
    latest_signal TEXT,
    latest_date TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ts.ticker,
        COUNT(*) AS total_signals,
        COUNT(*) FILTER (WHERE ts.signal = 'BULLISH') AS bullish_count,
        COUNT(*) FILTER (WHERE ts.signal = 'BEARISH') AS bearish_count,
        AVG(ts.confidence)::DECIMAL(3,2) AS avg_confidence,
        AVG(ts.alignment_score)::DECIMAL(3,2) AS avg_alignment,
        (SELECT ts2.signal FROM trade_signals ts2
         WHERE ts2.ticker = p_ticker
         ORDER BY ts2.created_at DESC LIMIT 1) AS latest_signal,
        MAX(ts.created_at) AS latest_date
    FROM trade_signals ts
    WHERE ts.ticker = p_ticker
      AND ts.created_at > NOW() - (p_days || ' days')::INTERVAL
    GROUP BY ts.ticker;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_ticker_signal_summary IS 'Returns aggregated signal statistics for a ticker over N days';

-- ============================================================================
-- 5. Sample Queries for Validation (Commented out, for reference)
-- ============================================================================

-- Get all report-level signals for today:
-- SELECT * FROM trade_signals
-- WHERE signal_source = 'report'
--   AND created_at > CURRENT_DATE
-- ORDER BY confidence DESC;

-- Get bullish signals with high alignment:
-- SELECT ts.*, a.title
-- FROM trade_signals ts
-- LEFT JOIN articles a ON ts.article_id = a.id
-- WHERE ts.signal = 'BULLISH'
--   AND ts.alignment_score >= 0.8
-- ORDER BY ts.created_at DESC;

-- Get signal distribution by ticker (last 7 days):
-- SELECT ticker, signal, COUNT(*), AVG(confidence)
-- FROM trade_signals
-- WHERE created_at > NOW() - INTERVAL '7 days'
-- GROUP BY ticker, signal
-- ORDER BY COUNT(*) DESC;

-- ============================================================================
-- Migration Complete
-- ============================================================================

COMMIT;

-- Verification queries (run after migration)
-- \dt trade_signals
-- \d trade_signals
-- \df get_report_signals
-- \df get_ticker_signal_summary
