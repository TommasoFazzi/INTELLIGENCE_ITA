-- Migration 033: Trade Flow Indicators
-- Stores bilateral/unilateral trade metrics (export/import volumes, trade balance)
-- Populated by future trade data loaders (UNCTAD, WTO)

CREATE TABLE IF NOT EXISTS trade_flow_indicators (
    id              SERIAL PRIMARY KEY,
    reporter_iso3   CHAR(3) NOT NULL,
    partner_iso3    CHAR(3),                 -- NULL for total/world
    indicator_code  TEXT NOT NULL,            -- EXPORT_VALUE, IMPORT_VALUE, TRADE_BALANCE
    year            INTEGER NOT NULL,
    value           NUMERIC(20,2),
    unit            TEXT DEFAULT 'USD',
    commodity_code  TEXT,                     -- HS2 or SITC classification (optional)
    source          TEXT,
    last_updated    TIMESTAMP DEFAULT NOW(),
    UNIQUE (reporter_iso3, partner_iso3, indicator_code, year, commodity_code)
);

CREATE INDEX IF NOT EXISTS idx_trade_reporter ON trade_flow_indicators(reporter_iso3);
CREATE INDEX IF NOT EXISTS idx_trade_partner ON trade_flow_indicators(partner_iso3);
CREATE INDEX IF NOT EXISTS idx_trade_year ON trade_flow_indicators(year);
