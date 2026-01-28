"""
Financial data types for Intelligence v2.

Defines TickerMetrics dataclass for aggregating market data
with proper handling of missing/partial data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal


@dataclass
class TickerMetrics:
    """
    Comprehensive metrics for a single ticker.

    Used by ValuationEngine to calculate intelligence_score.
    Handles edge cases like missing data, non-US tickers, and loss-making companies.

    Attributes:
        ticker: Stock symbol (e.g., 'LMT', 'RHM.DE')
        price: Current price
        currency: Currency code (USD, EUR, etc.)
        region: Market region (US, EU, ASIA, OTHER)
        sma_200: 200-day simple moving average
        sma_200_deviation_pct: ((Price - SMA) / SMA) * 100
        rsi_14: 14-day RSI (optional, for future use)
        pe_ratio: Price-to-earnings ratio (negative if loss-making)
        pe_sector_median: Sector median PE for comparison
        pe_rel_valuation: TickerPE / SectorPE (>1 = expensive, <1 = cheap)
        data_quality: Indicates completeness of data
        days_of_history: Number of trading days available
    """
    ticker: str
    price: float
    currency: str = "USD"
    region: Literal["US", "EU", "ASIA", "OTHER"] = "US"

    # Technical indicators
    sma_200: Optional[float] = None
    sma_200_deviation_pct: Optional[float] = None
    rsi_14: Optional[float] = None

    # Fundamental metrics
    pe_ratio: Optional[float] = None
    pe_sector_median: Optional[float] = None
    pe_rel_valuation: Optional[float] = None

    # Metadata
    data_quality: Literal["FULL", "PARTIAL", "INSUFFICIENT"] = "FULL"
    days_of_history: int = 0

    # Audit trail - traccia provenienza dati
    price_source: Literal["yfinance", "openbb", "cache", "unavailable"] = "yfinance"
    sma_source: Literal["calculated_200d", "proxy_mean", "unavailable"] = "calculated_200d"
    pe_source: Literal["openbb", "yfinance", "benchmark_etf", "unavailable"] = "openbb"
    sector_pe_source: Literal["database", "calculated", "benchmark_etf", "unavailable"] = "database"
    fetched_at: Optional[datetime] = None

    @property
    def is_loss_making(self) -> bool:
        """Company has negative P/E (negative earnings)."""
        return self.pe_ratio is not None and self.pe_ratio < 0

    @property
    def is_bubble_territory(self) -> bool:
        """
        Price >50% above SMA200 - extreme deviation.

        This triggers a hard cap on intelligence_score.
        """
        if self.sma_200_deviation_pct is None:
            return False
        return self.sma_200_deviation_pct > 50.0

    @property
    def has_sufficient_data(self) -> bool:
        """At least price and SMA200 available for technical scoring."""
        return self.price > 0 and self.sma_200 is not None

    @property
    def has_valuation_data(self) -> bool:
        """PE and sector PE available for fundamental scoring."""
        return (
            self.pe_ratio is not None
            and self.pe_sector_median is not None
            and self.pe_sector_median > 0
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "price": self.price,
            "currency": self.currency,
            "region": self.region,
            "sma_200": self.sma_200,
            "sma_200_deviation_pct": self.sma_200_deviation_pct,
            "rsi_14": self.rsi_14,
            "pe_ratio": self.pe_ratio,
            "pe_sector_median": self.pe_sector_median,
            "pe_rel_valuation": self.pe_rel_valuation,
            "data_quality": self.data_quality,
            "days_of_history": self.days_of_history,
            "is_loss_making": self.is_loss_making,
            "is_bubble_territory": self.is_bubble_territory,
            # Audit trail
            "price_source": self.price_source,
            "sma_source": self.sma_source,
            "pe_source": self.pe_source,
            "sector_pe_source": self.sector_pe_source,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
        }
