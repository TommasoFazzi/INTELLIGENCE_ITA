-- Migration 031: Strategic Infrastructure (PostGIS Points + Lines)
-- Requires: PostGIS extension (migration 026)
-- Populated by: scripts/load_telegeography.py + atlas_pipeline.py

CREATE TABLE IF NOT EXISTS strategic_infrastructure (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    infra_type      TEXT NOT NULL CHECK (infra_type IN (
        'SUBMARINE_CABLE', 'CABLE_LANDING_POINT', 'PORT', 'AIRPORT',
        'PIPELINE', 'REFINERY', 'LNG_TERMINAL', 'MINE',
        'DATA_CENTER', 'POWER_PLANT', 'MILITARY_BASE'
    )),
    geom            GEOMETRY(Point, 4326),
    route_geom      GEOMETRY(LineString, 4326),
    country_code    CHAR(2),
    operator        TEXT,
    capacity_metric TEXT,
    strategic_tier  SMALLINT CHECK (strategic_tier IN (1, 2, 3)),
    status          TEXT DEFAULT 'operational',
    properties      JSONB,
    source_dataset  TEXT,
    last_verified   DATE
);

CREATE INDEX IF NOT EXISTS idx_infra_geom ON strategic_infrastructure USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_infra_route ON strategic_infrastructure USING GIST(route_geom);
CREATE INDEX IF NOT EXISTS idx_infra_type ON strategic_infrastructure(infra_type);
CREATE INDEX IF NOT EXISTS idx_infra_tier ON strategic_infrastructure(strategic_tier);
CREATE INDEX IF NOT EXISTS idx_infra_country ON strategic_infrastructure(country_code);
