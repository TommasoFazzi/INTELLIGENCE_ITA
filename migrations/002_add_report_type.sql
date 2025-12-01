-- Migration: Add report_type to reports table
-- Date: 2025-11-30
-- Description: Add report_type column to distinguish between daily and weekly reports

-- Add report_type column with default 'daily'
ALTER TABLE reports 
ADD COLUMN IF NOT EXISTS report_type TEXT DEFAULT 'daily';

-- Add index for faster weekly report queries
CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);

-- Backfill existing records (set all existing reports to 'daily')
UPDATE reports SET report_type = 'daily' WHERE report_type IS NULL;

-- Add comment to column
COMMENT ON COLUMN reports.report_type IS 'Type of report: daily or weekly (for meta-analysis)';
