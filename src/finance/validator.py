"""
Valuation Engine for Financial Intelligence v2.

Aggregates data from multiple sources to build TickerMetrics,
with graceful degradation for missing data.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache

from ..storage.database import DatabaseManager
from ..utils.logger import get_logger
from .types import TickerMetrics
from .constants import get_region, get_sector_benchmark, SECTOR_BENCHMARK_MAP

logger = get_logger(__name__)


class ValuationEngine:
    """
    Builds TickerMetrics by aggregating data from multiple sources.

    Data Sources:
    1. MarketDataService (yfinance) - Price, SMA200
    2. OpenBBMarketService - Fundamentals (P/E, sector)
    3. Database cache - sector_pe_medians

    Features:
    - Graceful degradation: Returns partial metrics if some data unavailable
    - Caching: In-memory LRU cache for sector PE medians
    - Region-aware: Uses appropriate benchmarks for non-US stocks
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        market_service: Optional[Any] = None,
        openbb_service: Optional[Any] = None
    ):
        """
        Initialize ValuationEngine.

        Args:
            db: DatabaseManager instance (creates new if None)
            market_service: MarketDataService instance (lazy-loaded if None)
            openbb_service: OpenBBMarketService instance (lazy-loaded if None)
        """
        self.db = db
        self._market_service = market_service
        self._openbb_service = openbb_service

        # In-memory cache for sector PE medians
        self._sector_pe_cache: Dict[str, tuple] = {}  # sector -> (pe, expiry)

        logger.debug("ValuationEngine initialized")

    @property
    def market(self):
        """Lazy-load MarketDataService."""
        if self._market_service is None:
            from ..integrations.market_data import MarketDataService
            if self.db is None:
                self.db = DatabaseManager()
            self._market_service = MarketDataService(self.db)
        return self._market_service

    @property
    def openbb(self):
        """Lazy-load OpenBBMarketService."""
        if self._openbb_service is None:
            from ..integrations.openbb_service import OpenBBMarketService
            if self.db is None:
                self.db = DatabaseManager()
            self._openbb_service = OpenBBMarketService(self.db)
        return self._openbb_service

    def build_ticker_metrics(self, ticker: str) -> TickerMetrics:
        """
        Fetch all metrics for a ticker with graceful degradation.

        Order of operations:
        1. Fetch price + SMA200 from yfinance
        2. Fetch fundamentals from OpenBB (cached 7 days)
        3. Calculate sector PE median
        4. Build TickerMetrics with data_quality flag

        Args:
            ticker: Stock symbol (e.g., 'LMT', 'RHM.DE')

        Returns:
            TickerMetrics with available data (data_quality indicates completeness)
        """
        region = get_region(ticker)

        # 1. Fetch price data with SMA200
        price_data = self._fetch_price_with_sma200(ticker)

        if not price_data or price_data.get("price", 0) <= 0:
            logger.warning(f"No market data available for {ticker}")
            return TickerMetrics(
                ticker=ticker,
                price=0.0,
                region=region,
                data_quality="INSUFFICIENT"
            )

        # 2. Fetch fundamentals
        pe_ratio = None
        sector = None
        try:
            fundamentals = self.openbb.fetch_fundamentals(ticker)
            if fundamentals:
                pe_ratio = fundamentals.get("pe_ratio")
                if pe_ratio is not None:
                    pe_ratio = float(pe_ratio)
                sector = fundamentals.get("sector")
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")

        # 3. Get sector PE median
        pe_sector_median = None
        pe_rel_valuation = None
        if sector and pe_ratio and pe_ratio > 0:
            pe_sector_median = self._get_sector_pe_cached(sector, region)
            if pe_sector_median and pe_sector_median > 0:
                pe_rel_valuation = pe_ratio / pe_sector_median

        # 4. Determine data quality
        data_quality = self._assess_data_quality(
            has_price=price_data.get("price", 0) > 0,
            has_sma=price_data.get("sma_200") is not None,
            has_pe=pe_ratio is not None,
            has_sector_pe=pe_sector_median is not None,
            days_available=price_data.get("days_of_history", 0)
        )

        # Build metrics
        return TickerMetrics(
            ticker=ticker,
            price=float(price_data.get("price", 0)),
            currency="USD" if region in ("US", "OTHER") else ("EUR" if region == "EU" else "LOCAL"),
            region=region,
            sma_200=price_data.get("sma_200"),
            sma_200_deviation_pct=price_data.get("sma_200_deviation_pct"),
            pe_ratio=pe_ratio,
            pe_sector_median=pe_sector_median,
            pe_rel_valuation=pe_rel_valuation,
            data_quality=data_quality,
            days_of_history=price_data.get("days_of_history", 0)
        )

    def _fetch_price_with_sma200(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch price data with SMA200 calculation.

        Uses yfinance to get 1-year history for SMA200 calculation.
        Falls back to shorter periods if full history unavailable.

        Args:
            ticker: Stock symbol

        Returns:
            Dict with price, sma_200, sma_200_deviation_pct, days_of_history
        """
        try:
            # Check if market service has SMA200 method
            if hasattr(self.market, "fetch_ticker_with_sma200"):
                return self.market.fetch_ticker_with_sma200(ticker)

            # Fallback: use yfinance directly
            import yfinance as yf
            import pandas as pd

            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty:
                logger.warning(f"No historical data for {ticker}")
                return None

            current_price = float(hist["Close"].iloc[-1])
            days_available = len(hist)

            # Calculate SMA200
            sma_200 = None
            sma_200_deviation_pct = None

            if days_available >= 200:
                sma_200 = float(hist["Close"].rolling(window=200).mean().iloc[-1])
                sma_200_deviation_pct = ((current_price - sma_200) / sma_200) * 100
            elif days_available >= 50:
                # Use available data as approximation
                sma_200 = float(hist["Close"].mean())
                sma_200_deviation_pct = ((current_price - sma_200) / sma_200) * 100
                logger.debug(f"{ticker}: Only {days_available} days, using mean as SMA proxy")

            return {
                "ticker": ticker,
                "price": current_price,
                "sma_200": sma_200,
                "sma_200_deviation_pct": sma_200_deviation_pct,
                "days_of_history": days_available,
            }

        except Exception as e:
            logger.error(f"Failed to fetch price with SMA200 for {ticker}: {e}")
            return None

    def _get_sector_pe_cached(
        self,
        sector: str,
        region: str,
        cache_ttl_hours: int = 1
    ) -> Optional[float]:
        """
        Get sector PE with in-memory caching.

        Checks:
        1. In-memory cache (1 hour)
        2. Database cache (7 days)
        3. Calculates from company_fundamentals
        4. Falls back to sector benchmark ETF

        Args:
            sector: GICS sector name
            region: Market region
            cache_ttl_hours: In-memory cache duration

        Returns:
            Sector median PE or None
        """
        cache_key = f"{region}:{sector}"
        now = datetime.now()

        # Check in-memory cache
        if cache_key in self._sector_pe_cache:
            pe, expiry = self._sector_pe_cache[cache_key]
            if now < expiry:
                return pe

        # Check database cache
        pe_from_db = self._get_sector_pe_from_db(sector)
        if pe_from_db is not None:
            self._sector_pe_cache[cache_key] = (
                pe_from_db,
                now + timedelta(hours=cache_ttl_hours)
            )
            return pe_from_db

        # Calculate from company_fundamentals
        pe_calculated = self._calculate_sector_pe_median(sector)
        if pe_calculated is not None:
            # Save to database cache
            self._save_sector_pe_to_db(sector, pe_calculated)
            self._sector_pe_cache[cache_key] = (
                pe_calculated,
                now + timedelta(hours=cache_ttl_hours)
            )
            return pe_calculated

        # Fallback: get benchmark ETF PE
        benchmark = get_sector_benchmark(sector, region)
        if benchmark:
            try:
                fundamentals = self.openbb.fetch_fundamentals(benchmark)
                if fundamentals and fundamentals.get("pe_ratio"):
                    pe = float(fundamentals["pe_ratio"])
                    self._sector_pe_cache[cache_key] = (
                        pe,
                        now + timedelta(hours=cache_ttl_hours)
                    )
                    return pe
            except Exception as e:
                logger.warning(f"Failed to fetch benchmark PE for {benchmark}: {e}")

        return None

    def _get_sector_pe_from_db(self, sector: str) -> Optional[float]:
        """Check sector_pe_medians table for cached value."""
        if self.db is None:
            return None

        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT pe_median
                        FROM sector_pe_medians
                        WHERE sector = %s
                          AND cache_expires_at > NOW()
                    """, (sector,))
                    row = cur.fetchone()
                    return float(row[0]) if row else None
        except Exception as e:
            logger.debug(f"Failed to get sector PE from DB: {e}")
            return None

    def _calculate_sector_pe_median(self, sector: str) -> Optional[float]:
        """Calculate median PE from company_fundamentals table."""
        if self.db is None:
            return None

        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT pe_ratio
                        FROM company_fundamentals
                        WHERE sector = %s
                          AND pe_ratio > 0
                          AND pe_ratio < 100
                          AND last_updated > NOW() - INTERVAL '30 days'
                    """, (sector,))
                    pe_values = [row[0] for row in cur.fetchall()]

                    if len(pe_values) >= 5:
                        import statistics
                        return float(statistics.median(pe_values))
                    return None

        except Exception as e:
            logger.debug(f"Failed to calculate sector PE: {e}")
            return None

    def _save_sector_pe_to_db(
        self,
        sector: str,
        pe_median: float,
        sample_size: int = 0
    ):
        """Save sector PE to database cache."""
        if self.db is None:
            return

        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sector_pe_medians (sector, pe_median, sample_size, cache_expires_at)
                        VALUES (%s, %s, %s, NOW() + INTERVAL '7 days')
                        ON CONFLICT (sector) DO UPDATE SET
                            pe_median = EXCLUDED.pe_median,
                            sample_size = EXCLUDED.sample_size,
                            last_updated = NOW(),
                            cache_expires_at = NOW() + INTERVAL '7 days'
                    """, (sector, pe_median, sample_size))
                    conn.commit()
        except Exception as e:
            logger.debug(f"Failed to save sector PE to DB: {e}")

    def _assess_data_quality(
        self,
        has_price: bool,
        has_sma: bool,
        has_pe: bool,
        has_sector_pe: bool,
        days_available: int
    ) -> str:
        """
        Determine data quality level.

        - FULL: All data available (price, SMA200, PE, sector PE)
        - PARTIAL: Price and at least one other metric
        - INSUFFICIENT: Only price or nothing

        Args:
            has_price: Price data available
            has_sma: SMA200 available
            has_pe: PE ratio available
            has_sector_pe: Sector PE available
            days_available: Trading days of history

        Returns:
            Data quality level
        """
        if has_price and has_sma and has_pe and has_sector_pe:
            return "FULL"
        elif has_price and (has_sma or has_pe):
            return "PARTIAL"
        else:
            return "INSUFFICIENT"
