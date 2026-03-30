-- Migration 032: IMF WEO Macro Forecasts
-- Stores forward-looking macro projections from IMF World Economic Outlook
-- Populated by: scripts/load_imf_weo.py

CREATE TABLE IF NOT EXISTS macro_forecasts (
    id              SERIAL PRIMARY KEY,
    iso3            CHAR(3) NOT NULL,
    indicator_code  TEXT NOT NULL,            -- IMF WEO code (NGDP_RPCH, PCPIPCH, etc.)
    indicator_name  TEXT,
    year            INTEGER NOT NULL,
    value           NUMERIC(15,4),
    unit            TEXT,
    source          TEXT DEFAULT 'IMF_WEO',
    vintage         TEXT,                     -- e.g. 'April2024', 'October2024'
    last_updated    TIMESTAMP DEFAULT NOW(),
    UNIQUE (iso3, indicator_code, year, vintage)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_iso3 ON macro_forecasts(iso3);
CREATE INDEX IF NOT EXISTS idx_forecasts_indicator ON macro_forecasts(indicator_code);
CREATE INDEX IF NOT EXISTS idx_forecasts_year ON macro_forecasts(year);
