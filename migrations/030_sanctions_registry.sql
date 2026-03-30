-- Migration 030: Sanctions Registry (OpenSanctions)
-- Pure JSONB data — no PostGIS dependency
-- Populated by: scripts/load_opensanctions.py (NDJSON FtM entities)

CREATE TABLE IF NOT EXISTS sanctions_registry (
    id              TEXT PRIMARY KEY,
    caption         TEXT,
    schema_type     TEXT,                    -- FtM: Person, Company, Organization, LegalEntity
    aliases         TEXT[],
    countries       CHAR(2)[],               -- ISO2 codes
    datasets        TEXT[],                  -- e.g. eu_fsf, us_ofac_sdn, un_sc_sanctions
    properties      JSONB,                   -- Full FtM properties blob
    first_seen      DATE,
    last_seen       DATE,
    last_updated    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sanctions_caption ON sanctions_registry(caption);
CREATE INDEX IF NOT EXISTS idx_sanctions_countries ON sanctions_registry USING GIN(countries);
CREATE INDEX IF NOT EXISTS idx_sanctions_datasets ON sanctions_registry USING GIN(datasets);
CREATE INDEX IF NOT EXISTS idx_sanctions_schema ON sanctions_registry(schema_type);
