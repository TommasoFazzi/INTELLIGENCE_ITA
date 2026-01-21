-- ============================================================================
-- MIGRATION 010: Financial Intelligence v2 - Quantitative Validation
-- ============================================================================
-- Description: Adds intelligence scoring columns to trade_signals and creates
--              sector_pe_medians cache table for valuation comparisons.
-- Author: Intelligence ITA Team
-- Date: 2025-01-18
-- Dependencies: 005_add_trade_signals.sql, 009_add_openbb_schema.sql
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. Extend trade_signals with Financial Intelligence columns
-- ============================================================================

-- Intelligence score (0-100): Combines LLM confidence with market validation
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS intelligence_score INTEGER
    CHECK (intelligence_score >= 0 AND intelligence_score <= 100);

-- SMA200 deviation percentage: ((Price - SMA200) / SMA200) * 100
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS sma_200_deviation NUMERIC(5,2);

-- PE relative valuation: Ticker PE / Sector PE median
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS pe_rel_valuation NUMERIC(5,2);

-- Valuation rating based on PE and technical analysis
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS valuation_rating VARCHAR(20)
    CHECK (valuation_rating IN ('UNDERVALUED', 'FAIR', 'OVERVALUED', 'BUBBLE', 'LOSS_MAKING', 'UNKNOWN'));

-- Data quality flag for metrics availability
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS data_quality VARCHAR(20) DEFAULT 'FULL'
    CHECK (data_quality IN ('FULL', 'PARTIAL', 'INSUFFICIENT'));

-- Macro topics associated with the signal
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS macro_topics TEXT[];

-- ============================================================================
-- 2. Sector PE Medians Cache Table
-- ============================================================================
-- Caches calculated sector PE medians to avoid repeated calculations.
-- Expires after 7 days and is recalculated from company_fundamentals.

CREATE TABLE IF NOT EXISTS sector_pe_medians (
    sector VARCHAR(100) PRIMARY KEY,
    pe_median NUMERIC(10,2) NOT NULL,
    sample_size INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cache_expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days')
);

COMMENT ON TABLE sector_pe_medians IS 'Cache for sector PE medians used in valuation scoring';
COMMENT ON COLUMN sector_pe_medians.pe_median IS 'Median PE ratio for the sector';
COMMENT ON COLUMN sector_pe_medians.sample_size IS 'Number of companies used to calculate median';
COMMENT ON COLUMN sector_pe_medians.cache_expires_at IS 'When this cache entry should be recalculated';

-- ============================================================================
-- 3. Indexes for Financial Intelligence queries
-- ============================================================================

-- Index for sorting/filtering by intelligence score
CREATE INDEX IF NOT EXISTS idx_trade_signals_intelligence
    ON trade_signals(intelligence_score DESC)
    WHERE intelligence_score IS NOT NULL;

-- Index for valuation-based queries
CREATE INDEX IF NOT EXISTS idx_trade_signals_valuation
    ON trade_signals(valuation_rating)
    WHERE valuation_rating IS NOT NULL;

-- Composite index for conviction board: high intelligence + recent
CREATE INDEX IF NOT EXISTS idx_trade_signals_conviction
    ON trade_signals(intelligence_score DESC, created_at DESC)
    WHERE intelligence_score IS NOT NULL;

-- Index for sector PE cache expiry checks
CREATE INDEX IF NOT EXISTS idx_sector_pe_expires
    ON sector_pe_medians(cache_expires_at);

-- ============================================================================
-- 4. Comments on new columns
-- ============================================================================

COMMENT ON COLUMN trade_signals.intelligence_score IS 'Combined score (0-100) from LLM confidence + market validation';
COMMENT ON COLUMN trade_signals.sma_200_deviation IS 'Price deviation from 200-day SMA as percentage';
COMMENT ON COLUMN trade_signals.pe_rel_valuation IS 'PE ratio relative to sector median (1.0 = same as sector)';
COMMENT ON COLUMN trade_signals.valuation_rating IS 'Valuation category: UNDERVALUED, FAIR, OVERVALUED, BUBBLE, LOSS_MAKING, UNKNOWN';
COMMENT ON COLUMN trade_signals.data_quality IS 'Quality of market data used: FULL, PARTIAL, INSUFFICIENT';
COMMENT ON COLUMN trade_signals.macro_topics IS 'Array of macro themes associated with signal (e.g., AI_BOOM, RATES)';

-- ============================================================================
-- 5. Helper Function: Get high-conviction signals
-- ============================================================================

CREATE OR REPLACE FUNCTION get_high_conviction_signals(
    p_min_score INTEGER DEFAULT 70,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE(
    id INTEGER,
    ticker VARCHAR,
    signal TEXT,
    timeframe TEXT,
    intelligence_score INTEGER,
    valuation_rating VARCHAR,
    sma_200_deviation NUMERIC,
    confidence DECIMAL,
    rationale TEXT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ts.id,
        ts.ticker,
        ts.signal,
        ts.timeframe,
        ts.intelligence_score,
        ts.valuation_rating,
        ts.sma_200_deviation,
        ts.confidence,
        ts.rationale,
        ts.created_at
    FROM trade_signals ts
    WHERE ts.intelligence_score >= p_min_score
      AND ts.created_at > NOW() - (p_days || ' days')::INTERVAL
    ORDER BY ts.intelligence_score DESC, ts.created_at DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_high_conviction_signals IS 'Returns signals with intelligence_score >= threshold from last N days';

-- ============================================================================
-- 6. Helper Function: Refresh sector PE median
-- ============================================================================

CREATE OR REPLACE FUNCTION refresh_sector_pe_median(p_sector VARCHAR)
RETURNS NUMERIC AS $$
DECLARE
    v_median NUMERIC;
    v_count INTEGER;
BEGIN
    -- Calculate median PE from company_fundamentals
    SELECT
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pe_ratio),
        COUNT(*)
    INTO v_median, v_count
    FROM company_fundamentals
    WHERE sector = p_sector
      AND pe_ratio > 0
      AND pe_ratio < 100
      AND last_updated > NOW() - INTERVAL '30 days';

    -- Update or insert cache
    IF v_count >= 5 THEN
        INSERT INTO sector_pe_medians (sector, pe_median, sample_size, cache_expires_at)
        VALUES (p_sector, v_median, v_count, NOW() + INTERVAL '7 days')
        ON CONFLICT (sector) DO UPDATE SET
            pe_median = EXCLUDED.pe_median,
            sample_size = EXCLUDED.sample_size,
            last_updated = NOW(),
            cache_expires_at = NOW() + INTERVAL '7 days';
        RETURN v_median;
    ELSE
        RETURN NULL;
    END IF;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_sector_pe_median IS 'Recalculates and caches sector PE median from company_fundamentals';

-- ============================================================================
-- Migration Complete
-- ============================================================================

COMMIT;

-- Verification queries (run after migration)
-- \d trade_signals
-- \d sector_pe_medians
-- \df get_high_conviction_signals
-- \df refresh_sector_pe_median
