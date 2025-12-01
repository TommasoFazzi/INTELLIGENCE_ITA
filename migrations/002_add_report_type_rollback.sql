-- Rollback Migration: Remove report_type from reports table
-- Date: 2025-11-30
-- Description: Rollback for 002_add_report_type.sql

-- Drop index
DROP INDEX IF EXISTS idx_reports_type;

-- Remove column
ALTER TABLE reports DROP COLUMN IF EXISTS report_type;
