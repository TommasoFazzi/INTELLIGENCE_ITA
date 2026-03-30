-- Migration 026 Rollback: Remove PostGIS extension
-- WARNING: CASCADE will drop ALL PostGIS-dependent objects:
--   - geometry columns, spatial indexes, spatial_ref_sys table
--   - country_boundaries, conflict_events, strategic_infrastructure tables
-- Only use if no PostGIS tables have been created yet

DROP EXTENSION IF EXISTS postgis CASCADE;
