-- ============================================================================
-- Migration 009: OpenBB Financial Intelligence Schema
--
-- Adds tables for macro indicators, shipping data, and company fundamentals
-- to support OpenBB v4+ integration replacing yfinance.
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. Macro Indicators (Daily global context)
-- ============================================================================
CREATE TABLE IF NOT EXISTS macro_indicators (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    indicator_key VARCHAR(50) NOT NULL,
    value DECIMAL(15, 4) NOT NULL,
    unit VARCHAR(20),
    category VARCHAR(50) NOT NULL,
    source VARCHAR(50) DEFAULT 'openbb',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(date, indicator_key)
);

CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_indicators(date DESC);
CREATE INDEX IF NOT EXISTS idx_macro_category ON macro_indicators(category, date DESC);
CREATE INDEX IF NOT EXISTS idx_macro_key ON macro_indicators(indicator_key, date DESC);

COMMENT ON TABLE macro_indicators IS 'Global macro indicators from OpenBB (FRED, Yahoo, etc.)';

-- ============================================================================
-- 2. Shipping Indicators (Supply chain stress)
-- ============================================================================
CREATE TABLE IF NOT EXISTS shipping_indicators (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    indicator_key VARCHAR(50) NOT NULL,
    value DECIMAL(15, 4) NOT NULL,
    unit VARCHAR(20),
    route VARCHAR(100),
    source VARCHAR(50) DEFAULT 'openbb',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- RIMOSSO UNIQUE constraint inline che causava errore
);

-- CORREZIONE: Indice univoco creato separatamente per supportare COALESCE
CREATE UNIQUE INDEX IF NOT EXISTS idx_shipping_unique 
ON shipping_indicators (date, indicator_key, COALESCE(route, ''));

CREATE INDEX IF NOT EXISTS idx_shipping_date ON shipping_indicators(date DESC);
CREATE INDEX IF NOT EXISTS idx_shipping_key ON shipping_indicators(indicator_key, date DESC);

COMMENT ON TABLE shipping_indicators IS 'Shipping rates and Baltic Dry Index from OpenBB';

-- ============================================================================
-- 3. Company Fundamentals (Cached ticker metadata)
-- ============================================================================
CREATE TABLE IF NOT EXISTS company_fundamentals (
    ticker VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(200),
    sector VARCHAR(100),
    industry VARCHAR(100),

    -- Valuation
    market_cap BIGINT,
    pe_ratio DECIMAL(10, 2),
    pb_ratio DECIMAL(10, 2),
    ps_ratio DECIMAL(10, 2),
    ev_ebitda DECIMAL(10, 2),

    -- Financial Health
    debt_to_equity DECIMAL(10, 2),
    current_ratio DECIMAL(10, 2),
    quick_ratio DECIMAL(10, 2),

    -- Profitability
    profit_margin DECIMAL(10, 4),
    operating_margin DECIMAL(10, 4),
    roe DECIMAL(10, 4),
    roa DECIMAL(10, 4),

    -- Growth
    revenue_growth DECIMAL(10, 4),
    earnings_growth DECIMAL(10, 4),

    -- Dividends
    dividend_yield DECIMAL(5, 4),
    payout_ratio DECIMAL(10, 4),

    -- Metadata
    data_source VARCHAR(50) DEFAULT 'openbb',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cache_expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_sector ON company_fundamentals(sector);
CREATE INDEX IF NOT EXISTS idx_fundamentals_industry ON company_fundamentals(industry);
CREATE INDEX IF NOT EXISTS idx_fundamentals_updated ON company_fundamentals(last_updated);
CREATE INDEX IF NOT EXISTS idx_fundamentals_cache_expires ON company_fundamentals(cache_expires_at);

COMMENT ON TABLE company_fundamentals IS 'Cached fundamental metrics from OpenBB equity module';

-- ============================================================================
-- 4. Trigger for auto-updating timestamps
-- ============================================================================
CREATE OR REPLACE FUNCTION update_macro_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_macro_indicators_updated_at ON macro_indicators;
CREATE TRIGGER update_macro_indicators_updated_at
    BEFORE UPDATE ON macro_indicators
    FOR EACH ROW
    EXECUTE FUNCTION update_macro_updated_at();

-- ============================================================================
-- 5. View for latest macro snapshot
-- ============================================================================
CREATE OR REPLACE VIEW v_latest_macro AS
SELECT DISTINCT ON (indicator_key)
    indicator_key,
    value,
    unit,
    category,
    date,
    updated_at
FROM macro_indicators
ORDER BY indicator_key, date DESC;

COMMENT ON VIEW v_latest_macro IS 'Latest value for each macro indicator';

-- ============================================================================
-- 6. Function to get macro dashboard
-- ============================================================================
CREATE OR REPLACE FUNCTION get_macro_dashboard(target_date DATE DEFAULT CURRENT_DATE)
RETURNS TABLE (
    indicator_key VARCHAR(50),
    value DECIMAL(15, 4),
    unit VARCHAR(20),
    category VARCHAR(50),
    prev_value DECIMAL(15, 4),
    change_pct DECIMAL(10, 4)
) AS $$
BEGIN
    RETURN QUERY
    WITH current_values AS (
        SELECT
            m.indicator_key,
            m.value,
            m.unit,
            m.category
        FROM macro_indicators m
        WHERE m.date = target_date
    ),
    previous_values AS (
        SELECT DISTINCT ON (m.indicator_key)
            m.indicator_key,
            m.value as prev_value
        FROM macro_indicators m
        WHERE m.date < target_date
        ORDER BY m.indicator_key, m.date DESC
    )
    SELECT
        c.indicator_key,
        c.value,
        c.unit,
        c.category,
        p.prev_value,
        CASE
            WHEN p.prev_value IS NOT NULL AND p.prev_value != 0
            THEN ((c.value - p.prev_value) / p.prev_value) * 100
            ELSE NULL
        END as change_pct
    FROM current_values c
    LEFT JOIN previous_values p ON c.indicator_key = p.indicator_key
    ORDER BY c.category, c.indicator_key;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_macro_dashboard IS 'Returns macro indicators with day-over-day change for LLM context';

COMMIT;