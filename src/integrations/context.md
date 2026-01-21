# Integrations Context

## Purpose
External service wrappers for market data and financial APIs. Provides unified interfaces for Yahoo Finance (price data) and OpenBB (fundamentals, macro indicators) with caching and database persistence.

## Architecture Role
Data acquisition layer for financial intelligence. Used by `src/finance/` for trade signal scoring and by `src/llm/report_generator.py` for macro context injection. All data persisted to PostgreSQL via `src/storage/`.

## Key Files

- `market_data.py` - Yahoo Finance integration
  - `MarketDataService` class - Price and technical data
  - `fetch_ticker_data(ticker, period)` - OHLCV with derived metrics
    - 7-day volatility (std dev of daily returns)
    - Relative volume (volume / 30-day average)
  - `fetch_batch(tickers)` - Batch fetching with rate limiting
  - `fetch_with_sma200(ticker)` - Price with 200-day SMA for technical analysis
  - 1-hour in-memory cache to avoid rate limits
  - Database persistence to `market_data` table
  - Uses yfinance 0.2.66+ with curl_cffi for anti-bot protection

- `openbb_service.py` - OpenBB v4+ integration
  - `OpenBBMarketService` class - Macro and fundamentals
  - **Macro Indicators** (stored daily):
    - US 10Y Treasury Yield (FRED: DGS10)
    - VIX Volatility Index (Yahoo: ^VIX)
    - Brent Crude Oil (Yahoo: BZ=F)
    - EUR/USD Exchange Rate (Yahoo: EURUSD=X)
    - Baltic Dry Index when available
  - `ensure_daily_macro_data()` - Fetch and persist macro indicators
  - `get_macro_context_text(date)` - Formatted text for LLM prompt injection
  - **Company Fundamentals** (7-day cache):
    - P/E ratio, forward P/E
    - Debt/Equity ratio
    - Sector classification
    - Profit margins
  - `fetch_fundamentals(ticker)` - Cached fundamental data
  - API key configuration via environment variables:
    - `FRED_API_KEY` - Federal Reserve Economic Data
    - `FMP_API_KEY` - Financial Modeling Prep (optional)
    - `INTRINIO_API_KEY` - Intrinio (optional)

## Dependencies

- **Internal**: `src/storage/database`, `src/utils/logger`
- **External**:
  - `yfinance` (0.2.66+) - Yahoo Finance with curl_cffi
  - `openbb` (v4+) - OpenBB unified API
  - `pandas` - Data manipulation
  - `python-dotenv` - Environment configuration

## Data Flow

- **Input**:
  - Ticker symbols from trade signals
  - API requests with caching

- **Output**:
  - `market_data` table - OHLCV time series
  - `macro_indicators` table - Daily macro snapshots
  - `ticker_fundamentals` table - Cached fundamentals
  - In-memory cache for rate limit protection
