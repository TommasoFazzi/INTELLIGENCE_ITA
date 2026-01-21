# Finance Context

## Purpose
Financial intelligence scoring system that evaluates trade signals extracted by the LLM. Combines market metrics (technical and fundamental) with LLM confidence to produce actionable intelligence scores. Part of Sprint 3 enhancements.

## Architecture Role
Validation and enrichment layer for trade signals. Receives raw signals from `src/llm/report_generator.py`, fetches market data via `src/integrations/`, calculates penalties/bonuses, and returns enriched signals with intelligence scores (0-100).

## Key Files

- `scoring.py` - Score calculation functions
  - `calculate_technical_penalty(metrics)` - SMA200 deviation penalty
    - Minor (<15%): 0 points
    - Moderate (15-30%): Linear penalty
    - Extreme (>30%): Non-linear penalty (z-proxy formula)
    - Max penalty: 40 points
  - `calculate_fundamental_score(metrics)` - P/E valuation adjustment
    - Loss-making (PE < 0): -15 points
    - Bubble (PE > 2x sector): -20 points
    - Overvalued (PE > 1.5x sector): -10 points
    - Undervalued (PE < 0.8x sector): +10 points
  - `calculate_intelligence_score(llm_confidence, metrics)` - Final score
    - Formula: `base = llm_confidence * 100 - technical_penalty + fundamental_score`
    - Capped 0-100

- `validator.py` - Valuation engine for metrics aggregation
  - `ValuationEngine` class - Builds `TickerMetrics` from multiple sources
  - Data sources: MarketDataService (yfinance), OpenBBMarketService (fundamentals)
  - Graceful degradation: Returns partial metrics if some data unavailable
  - Region-aware: Uses appropriate benchmarks for non-US stocks
  - Sector PE median caching with LRU

- `types.py` - Data type definitions
  - `TickerMetrics` dataclass - All metrics for a ticker
    - `price`, `sma_200`, `sma_200_deviation_pct`
    - `pe_ratio`, `sector`, `sector_pe_median`, `pe_rel_valuation`
    - `is_loss_making`, `data_quality` flag

- `constants.py` - Threshold configurations
  - `THRESHOLDS` dict - Scoring thresholds (SMA deviation, PE ratios, penalties)
  - `SECTOR_BENCHMARK_MAP` - Sector to benchmark ETF mappings
  - `get_region(ticker)` - Determine region from ticker suffix
  - `get_sector_benchmark(sector)` - Get benchmark for sector

## Dependencies

- **Internal**: `src/storage/database`, `src/integrations/market_data`, `src/integrations/openbb_service`, `src/utils/logger`
- **External**: None (uses integrations for data)

## Data Flow

- **Input**:
  - Trade signals from LLM with ticker and confidence
  - Market data from Yahoo Finance (price, SMA200)
  - Fundamentals from OpenBB (P/E, sector)

- **Output**:
  - Enriched signals with:
    - `intelligence_score` (0-100)
    - `technical_penalty`, `fundamental_score`
    - `data_quality` flag
    - Scoring rationale text
