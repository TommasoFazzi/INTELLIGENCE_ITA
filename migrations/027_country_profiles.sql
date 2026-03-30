-- Migration 027: Country Profiles reference table
-- Pure structured data — no PostGIS dependency
-- Populated by scripts/load_world_bank.py (World Bank API v2)

CREATE TABLE IF NOT EXISTS country_profiles (
    iso3                CHAR(3) PRIMARY KEY,
    iso2                CHAR(2),
    name                TEXT NOT NULL,
    capital             TEXT,
    region              TEXT,
    income_group        TEXT,
    population          BIGINT,
    gdp_usd             NUMERIC(20,2),
    gdp_per_capita      NUMERIC(12,2),
    gdp_growth          NUMERIC(6,3),
    inflation           NUMERIC(6,3),
    unemployment        NUMERIC(6,3),
    debt_to_gdp         NUMERIC(6,3),
    current_account_pct NUMERIC(6,3),
    internet_access_pct NUMERIC(6,3),
    governance_score    NUMERIC(5,3),
    data_year           INTEGER,
    last_updated        TIMESTAMP DEFAULT NOW()
);

-- Index for name lookup
CREATE INDEX IF NOT EXISTS idx_country_profiles_name ON country_profiles(name);
CREATE INDEX IF NOT EXISTS idx_country_profiles_region ON country_profiles(region);
