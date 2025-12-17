-- ============================================================================
-- MIGRATION 004: Market Data & AI Analysis Schema
-- ============================================================================
-- Description: Add tables for Yahoo Finance integration and AI-structured output
-- Author: Intelligence ITA Team
-- Date: 2025-12-15
-- Dependencies: 003_add_entities_table.sql
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. Ticker Mappings (Entity → Stock Symbol)
-- ============================================================================
-- Links entities (e.g., "Apple") to their stock tickers (e.g., "AAPL")
-- Supports multiple tickers per entity (e.g., dual listings)
CREATE TABLE IF NOT EXISTS ticker_mappings (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    exchange VARCHAR(20) DEFAULT 'US',              -- NYSE, NASDAQ, etc.
    confidence DECIMAL(3, 2) DEFAULT 1.00,          -- 1.00 = Manually verified
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Prevent duplicate mappings
    UNIQUE(entity_id, ticker)
);

-- Index for fast lookups during ingestion
CREATE INDEX IF NOT EXISTS idx_ticker_mappings_entity ON ticker_mappings(entity_id);
CREATE INDEX IF NOT EXISTS idx_ticker_mappings_ticker ON ticker_mappings(ticker);

COMMENT ON TABLE ticker_mappings IS 'Maps named entities to stock tickers for market data integration';
COMMENT ON COLUMN ticker_mappings.confidence IS 'Confidence score: 1.00 = manual, 0.80-0.99 = AI-mapped, <0.80 = fuzzy match';

-- ============================================================================
-- 2. Market Data (OHLCV Time Series)
-- ============================================================================
-- Stores daily market data for tracked tickers (Yahoo Finance source)
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,

    -- Standard OHLCV data
    open_price DECIMAL(12, 4),
    high_price DECIMAL(12, 4),
    low_price DECIMAL(12, 4),
    close_price DECIMAL(12, 4),
    volume BIGINT,

    -- Derived metrics (calculated in Python)
    volatility_7d DECIMAL(8, 4),                    -- 7-day rolling volatility
    relative_volume DECIMAL(8, 2),                  -- Volume vs 30-day average

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- One entry per ticker per day
    UNIQUE(ticker, date)
);

-- Indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date DESC);

COMMENT ON TABLE market_data IS 'Daily OHLCV market data from Yahoo Finance';
COMMENT ON COLUMN market_data.volatility_7d IS 'Calculated as std dev of daily returns over 7 days';
COMMENT ON COLUMN market_data.relative_volume IS 'Current volume / 30-day average volume';

-- ============================================================================
-- 3. Articles Table Extensions
-- ============================================================================
-- Add columns for AI-structured analysis and entity migration tracking

-- AI Analysis (Structured JSON output from Gemini)
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS ai_analysis JSONB;

-- Entity Migration Tracking (when entities were extracted from JSONB → entities table)
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS entities_migrated_at TIMESTAMP;

-- GIN index for fast JSON queries (e.g., "urgency > 8")
CREATE INDEX IF NOT EXISTS idx_articles_ai_analysis ON articles USING GIN(ai_analysis);

-- Index for migration tracking queries
CREATE INDEX IF NOT EXISTS idx_articles_entities_migrated ON articles(entities_migrated_at)
WHERE entities_migrated_at IS NOT NULL;

COMMENT ON COLUMN articles.ai_analysis IS 'Structured LLM output: sentiment, urgency, trade signals (JSON Schema validated)';
COMMENT ON COLUMN articles.entities_migrated_at IS 'Timestamp when entities were migrated from JSONB to entities table';

-- ============================================================================
-- 4. Helper Functions (Optional but useful)
-- ============================================================================

-- Function to get latest market data for a ticker
CREATE OR REPLACE FUNCTION get_latest_market_data(p_ticker VARCHAR)
RETURNS TABLE(
    ticker VARCHAR,
    date DATE,
    close_price DECIMAL,
    volume BIGINT,
    volatility_7d DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        md.ticker,
        md.date,
        md.close_price,
        md.volume,
        md.volatility_7d
    FROM market_data md
    WHERE md.ticker = p_ticker
    ORDER BY md.date DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_latest_market_data IS 'Returns most recent market data for a given ticker';

-- ============================================================================
-- 5. Sample Queries for Validation (Commented out, for reference)
-- ============================================================================
-- Uncomment these to test the schema after migration:

-- Get all tickers mapped to an entity:
-- SELECT tm.ticker, tm.exchange, e.name
-- FROM ticker_mappings tm
-- JOIN entities e ON tm.entity_id = e.id
-- WHERE e.name = 'Apple';

-- Get market data with 7-day price change:
-- SELECT ticker, date, close_price,
--        close_price - LAG(close_price, 7) OVER (PARTITION BY ticker ORDER BY date) AS price_change_7d
-- FROM market_data
-- WHERE ticker = 'AAPL'
-- ORDER BY date DESC
-- LIMIT 10;

-- Get articles with high urgency from AI analysis:
-- SELECT id, title, ai_analysis->>'urgency_level' AS urgency
-- FROM articles
-- WHERE ai_analysis->>'urgency_level' IN ('high', 'critical')
-- ORDER BY published_at DESC
-- LIMIT 20;

-- ============================================================================
-- Migration Complete
-- ============================================================================

COMMIT;

-- Verification queries (run after migration)
-- \dt ticker_mappings
-- \dt market_data
-- \d articles
