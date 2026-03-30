-- Migration 026: Enable PostGIS extension
-- Prerequisite for all spatial/geo tables (country_boundaries, conflict_events, strategic_infrastructure)
-- Idempotent: IF NOT EXISTS protects against re-runs
-- Requires: postgresql-17-postgis-3 installed in container (deploy/postgres/Dockerfile)

CREATE EXTENSION IF NOT EXISTS postgis;

-- Verify installation
DO $$
BEGIN
    RAISE NOTICE 'PostGIS version: %', PostGIS_Version();
END $$;
