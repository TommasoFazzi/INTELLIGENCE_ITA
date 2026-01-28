-- ============================================================================
-- MIGRATION 011: Audit Trail for Data Transparency
-- ============================================================================
-- Description: Adds audit trail columns to trade_signals for data source
--              transparency in the Intelligence Scores UI.
-- Author: Intelligence ITA Team
-- Date: 2025-01-27
-- Dependencies: 010_financial_intel_v2.sql
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. Add audit trail columns to trade_signals
-- ============================================================================

-- Price data source (yfinance, openbb, cache, unavailable)
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS price_source VARCHAR(20);

-- SMA200 calculation method (calculated_200d, proxy_mean, unavailable)
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS sma_source VARCHAR(20);

-- P/E ratio source (openbb, yfinance, benchmark_etf, unavailable)
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS pe_source VARCHAR(20);

-- Sector PE source (database, calculated, benchmark_etf, unavailable)
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS sector_pe_source VARCHAR(20);

-- Timestamp when data was fetched from external APIs
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMP WITH TIME ZONE;

-- Number of trading days available for technical analysis
ALTER TABLE trade_signals
ADD COLUMN IF NOT EXISTS days_of_history INTEGER;

-- ============================================================================
-- 2. Comments for documentation
-- ============================================================================

COMMENT ON COLUMN trade_signals.price_source IS 'Data source for price: yfinance, openbb, cache, unavailable';
COMMENT ON COLUMN trade_signals.sma_source IS 'SMA200 calculation method: calculated_200d (200+ days), proxy_mean (<200 days), unavailable';
COMMENT ON COLUMN trade_signals.pe_source IS 'P/E ratio source: openbb, yfinance, benchmark_etf, unavailable';
COMMENT ON COLUMN trade_signals.sector_pe_source IS 'Sector PE source: database (cache), calculated, benchmark_etf, unavailable';
COMMENT ON COLUMN trade_signals.fetched_at IS 'Timestamp when market data was fetched from external APIs';
COMMENT ON COLUMN trade_signals.days_of_history IS 'Number of trading days available for technical indicators calculation';

-- ============================================================================
-- Migration Complete
-- ============================================================================

COMMIT;

-- Verification queries (run after migration)
-- \d trade_signals
-- SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'trade_signals' AND column_name IN ('price_source', 'sma_source', 'pe_source', 'sector_pe_source', 'fetched_at', 'days_of_history');
