"""
Financial Intelligence Module v2

Provides quantitative validation for LLM-generated trade signals:
- TickerMetrics: Data structure for market metrics
- ValuationEngine: Fetches and aggregates market data
- Scoring functions: Calculate intelligence_score with penalties/bonuses
"""

from .types import TickerMetrics
from .constants import (
    TICKER_REGION_MAP,
    SECTOR_BENCHMARK_MAP,
    THRESHOLDS,
    get_region,
)
from .scoring import (
    calculate_technical_penalty,
    calculate_fundamental_score,
    calculate_intelligence_score,
    enrich_signal_with_intelligence,
)
from .validator import ValuationEngine

__all__ = [
    # Types
    "TickerMetrics",
    # Constants
    "TICKER_REGION_MAP",
    "SECTOR_BENCHMARK_MAP",
    "THRESHOLDS",
    "get_region",
    # Scoring
    "calculate_technical_penalty",
    "calculate_fundamental_score",
    "calculate_intelligence_score",
    "enrich_signal_with_intelligence",
    # Validator
    "ValuationEngine",
]
