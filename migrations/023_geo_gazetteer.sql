-- Migration 023: GeoNames Gazetteer table
-- Purpose: Local geographic reference DB for entity disambiguation + geocoding
-- Source: https://download.geonames.org/export/dump/
-- Populated by: scripts/load_geonames.py (one-time, ~2-3M rows, ~500MB)
-- Used by: scripts/geocode_geonames.py (replaces Photon-first approach)

CREATE TABLE IF NOT EXISTS geo_gazetteer (
    geoname_id    INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    ascii_name    TEXT,
    alternate_names TEXT[] DEFAULT '{}',   -- aggregated from alternateNames.txt
    latitude      DECIMAL(10, 8) NOT NULL,
    longitude     DECIMAL(11, 8) NOT NULL,
    feature_class CHAR(1),                 -- A=admin, P=city, H=water, L=area, S=spot
    feature_code  VARCHAR(10),             -- PCLI, PPLA, PPL, SEA, ADM1, etc.
    country_code  CHAR(2),                 -- ISO 3166-1 alpha-2
    population    BIGINT DEFAULT 0,
    timezone      TEXT
);

-- Canonical name search (ascii, case-insensitive)
CREATE INDEX IF NOT EXISTS idx_gazetteer_ascii
    ON geo_gazetteer (lower(ascii_name));

-- Full-text search on original name
CREATE INDEX IF NOT EXISTS idx_gazetteer_name_fts
    ON geo_gazetteer USING GIN (to_tsvector('simple', name));

-- Alternate name array search (GIN for ANY/overlap queries)
CREATE INDEX IF NOT EXISTS idx_gazetteer_altnames
    ON geo_gazetteer USING GIN (alternate_names);

-- Country + feature type filtering (used in Gemini-guided lookup)
CREATE INDEX IF NOT EXISTS idx_gazetteer_country
    ON geo_gazetteer (country_code);

CREATE INDEX IF NOT EXISTS idx_gazetteer_feature
    ON geo_gazetteer (feature_class, feature_code);

-- Population descending (used to pick most relevant match among duplicates)
CREATE INDEX IF NOT EXISTS idx_gazetteer_pop
    ON geo_gazetteer (population DESC NULLS LAST);

-- Composite: country + ascii + feature (most common query pattern after Gemini disambiguates)
CREATE INDEX IF NOT EXISTS idx_gazetteer_country_ascii_feature
    ON geo_gazetteer (country_code, lower(ascii_name), feature_class);

COMMENT ON TABLE geo_gazetteer IS
    'GeoNames geographic gazetteer: ~2-3M rows filtered to feature classes A/P/H/L. '
    'Alternate names aggregated as TEXT[] for multilingual lookup. '
    'Primary source for geocode_geonames.py entity resolution pipeline.';
