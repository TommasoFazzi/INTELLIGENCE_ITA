-- Migration 028: Country Boundaries (PostGIS MultiPolygon geometry)
-- Requires: PostGIS extension (migration 026)
-- Populated by: scripts/load_natural_earth.sh (Natural Earth 50m via ogr2ogr)

CREATE TABLE IF NOT EXISTS country_boundaries (
    iso3        CHAR(3) PRIMARY KEY,
    iso2        CHAR(2),
    name        TEXT NOT NULL,
    geom        GEOMETRY(MultiPolygon, 4326),
    area_km2    NUMERIC(12,2),
    pop_est     BIGINT,
    continent   TEXT,
    subregion   TEXT,
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_boundaries_geom ON country_boundaries USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_boundaries_continent ON country_boundaries(continent);
