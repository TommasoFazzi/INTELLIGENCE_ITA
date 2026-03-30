-- Migration 029: Conflict Events (UCDP GED)
-- Requires: PostGIS extension (migration 026)
-- Populated by: scripts/load_ucdp.py (UCDP API v24.1)

CREATE TABLE IF NOT EXISTS conflict_events (
    id              SERIAL PRIMARY KEY,
    event_date      DATE NOT NULL,
    event_type      TEXT NOT NULL,           -- UCDP: type_of_violence (1=state, 2=non-state, 3=one-sided)
    country         TEXT,                    -- UCDP: country
    location        TEXT,                    -- UCDP: where_description
    geom            GEOMETRY(Point, 4326),   -- UCDP: latitude + longitude
    actor1          TEXT,                    -- UCDP: side_a
    actor2          TEXT,                    -- UCDP: side_b
    fatalities      INTEGER,                -- UCDP: best estimate
    fatalities_low  INTEGER,                -- UCDP: low estimate
    fatalities_high INTEGER,                -- UCDP: high estimate
    notes           TEXT,                    -- UCDP: source_article + source_office
    source          TEXT DEFAULT 'UCDP_GED',
    data_source_id  TEXT UNIQUE             -- UCDP: id field for dedup
);

CREATE INDEX IF NOT EXISTS idx_conflict_events_geom ON conflict_events USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_conflict_events_date ON conflict_events(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_conflict_events_type ON conflict_events(event_type);
CREATE INDEX IF NOT EXISTS idx_conflict_events_country ON conflict_events(country);
