-- =============================================================================
-- Migration 035 — Macro Intelligence Layer
-- Phase 1 of the Strategic Intelligence Layer upgrade.
--
-- Creates two tables:
--   1. macro_indicator_metadata  — data quality tracking per indicator
--   2. macro_regime_history      — 60-day rolling macro regime history (Phase 4)
--
-- Apply: psql $DATABASE_URL -f migrations/035_macro_intelligence_layer.sql
-- =============================================================================

-- =============================================================================
-- 1. macro_indicator_metadata
--    One row per indicator key. Tracks real data date, staleness, fetch status.
--    Fixes the NICKEL/monthly mislabeling bug: data_date (not fetch date)
--    is now the source of truth.
-- =============================================================================

CREATE TABLE IF NOT EXISTS macro_indicator_metadata (
    key                 VARCHAR(60) PRIMARY KEY,
    expected_frequency  VARCHAR(20) NOT NULL DEFAULT 'daily',
    expected_gap_days   INTEGER     NOT NULL DEFAULT 1,

    -- Real date of the last data point (not the fetch date)
    last_updated        DATE,
    last_source         VARCHAR(40),
        -- 'fred' | 'yfinance' | 'openbb' | 'crypto' | 'fallback'

    -- Staleness
    staleness_days      INTEGER,
    is_stale            BOOLEAN DEFAULT FALSE,

    -- Fetch tracking
    last_fetch_date     DATE,
    fetch_attempted     BOOLEAN DEFAULT FALSE,
    fetch_succeeded     BOOLEAN DEFAULT FALSE,

    -- Reliability classification
    reliability         VARCHAR(20) DEFAULT 'normal',
        -- 'normal' | 'restricted' | 'legacy' | 'context_only'
    reliability_note    TEXT,

    -- Additional metadata
    release_pattern     VARCHAR(50),
    notes               TEXT,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mim_stale
    ON macro_indicator_metadata (is_stale, expected_frequency);

CREATE INDEX IF NOT EXISTS idx_mim_reliability
    ON macro_indicator_metadata (reliability);

-- Additional indexes for Phase 2 freshness queries
CREATE INDEX IF NOT EXISTS idx_macro_metadata_stale
    ON macro_indicator_metadata (is_stale)
    WHERE is_stale = TRUE;

CREATE INDEX IF NOT EXISTS idx_macro_metadata_reliability
    ON macro_indicator_metadata (reliability);


-- =============================================================================
-- 2. macro_regime_history
--    One row per analysis day. Populated by MacroRegimePersistence (Phase 4).
--    Indexed for Oracle 2.0 queries (streak days, SC signals, etc.)
-- =============================================================================

CREATE TABLE IF NOT EXISTS macro_regime_history (
    id                      SERIAL PRIMARY KEY,
    date                    DATE NOT NULL UNIQUE,

    -- Indexed projections (avoid JSON parsing in hot queries)
    risk_regime             VARCHAR(40) NOT NULL,
        -- values: risk_on | risk_off | transition | stagflation |
        --         monetary_tightening | recession_signal | neutral
    regime_confidence       NUMERIC(4,3),           -- 0.000 – 1.000

    -- Active convergences (array for ANY/overlap queries)
    active_convergence_ids  TEXT[],
        -- e.g. ARRAY['risk_off_systemic', 'carry_trade_unwind_jpy']

    -- Active supply chain signals (array of sector keys)
    active_sc_sectors       TEXT[],
        -- e.g. ARRAY['semiconductors', 'energy', 'automotive_ev']

    -- Brief macro narrative (useful for Oracle without RAG)
    macro_narrative         TEXT,

    -- Full Layer 2 output JSON (source of truth)
    analysis_json           JSONB NOT NULL,

    -- Data quality snapshot at analysis time
    data_quality_snapshot   JSONB,

    -- Metadata
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    data_freshness_gap_days SMALLINT DEFAULT 0
        -- 0 = same-day data, 1 = weekend (Friday data), etc.
);

-- Indexes for frequent queries
CREATE INDEX IF NOT EXISTS idx_mrh_date
    ON macro_regime_history (date DESC);

CREATE INDEX IF NOT EXISTS idx_mrh_regime
    ON macro_regime_history (risk_regime, date DESC);

CREATE INDEX IF NOT EXISTS idx_mrh_convergences
    ON macro_regime_history USING GIN (active_convergence_ids);

CREATE INDEX IF NOT EXISTS idx_mrh_sc_sectors
    ON macro_regime_history USING GIN (active_sc_sectors);

-- JSONB index for occasional field queries
CREATE INDEX IF NOT EXISTS idx_mrh_json
    ON macro_regime_history USING GIN (analysis_json jsonb_path_ops);
