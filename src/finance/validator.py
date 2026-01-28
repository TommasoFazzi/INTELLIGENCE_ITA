"""
Valuation Engine for Financial Intelligence v2.

Aggregates data from multiple sources to build TickerMetrics,
with graceful degradation for missing data.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache
import time

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
        1. Fetch price + SMA200 from yfinance (with retry)
        2. Fetch fundamentals from OpenBB (cached 7 days)
        3. Calculate sector PE median (with IQR outlier removal)
        4. Build TickerMetrics with data_quality flag and audit trail

        Args:
            ticker: Stock symbol (e.g., 'LMT', 'RHM.DE')

        Returns:
            TickerMetrics with available data (data_quality indicates completeness)
        """
        region = get_region(ticker)
        fetched_at = datetime.now()

        # Audit trail defaults
        price_source = "unavailable"
        sma_source = "unavailable"
        pe_source = "unavailable"
        sector_pe_source = "unavailable"

        # 1. Fetch price data with SMA200 (with retry)
        price_data = self._fetch_price_with_sma200(ticker)

        if not price_data or price_data.get("price", 0) <= 0:
            logger.warning(f"No market data available for {ticker}")
            return TickerMetrics(
                ticker=ticker,
                price=0.0,
                region=region,
                data_quality="INSUFFICIENT",
                price_source=price_source,
                sma_source=sma_source,
                pe_source=pe_source,
                sector_pe_source=sector_pe_source,
                fetched_at=fetched_at
            )

        price_source = "yfinance"
        sma_source = price_data.get("sma_source", "unavailable")

        # 2. Fetch fundamentals
        pe_ratio = None
        sector = None
        try:
            fundamentals = self.openbb.fetch_fundamentals(ticker)
            if fundamentals:
                pe_ratio = fundamentals.get("pe_ratio")
                if pe_ratio is not None:
                    pe_ratio = float(pe_ratio)
                    pe_source = "openbb"
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
                # Determine sector PE source (cache vs calculated vs benchmark)
                sector_pe_source = self._get_sector_pe_source(sector)

        # 4. Determine data quality
        data_quality = self._assess_data_quality(
            has_price=price_data.get("price", 0) > 0,
            has_sma=price_data.get("sma_200") is not None,
            has_pe=pe_ratio is not None,
            has_sector_pe=pe_sector_median is not None,
            days_available=price_data.get("days_of_history", 0)
        )

        # Build metrics with audit trail
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
            days_of_history=price_data.get("days_of_history", 0),
            # Audit trail
            price_source=price_source,
            sma_source=sma_source,
            pe_source=pe_source,
            sector_pe_source=sector_pe_source,
            fetched_at=fetched_at
        )

    def _fetch_price_with_sma200(
        self,
        ticker: str,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch price data with SMA200 calculation.

        Uses yfinance to get 1-year history for SMA200 calculation.
        Falls back to shorter periods if full history unavailable.
        Implements exponential backoff retry (1s, 2s, 4s) on failures.

        Args:
            ticker: Stock symbol
            max_retries: Number of retry attempts (default: 3)

        Returns:
            Dict with price, sma_200, sma_200_deviation_pct, days_of_history, sma_source
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                # Check if market service has SMA200 method
                if hasattr(self.market, "fetch_ticker_with_sma200"):
                    result = self.market.fetch_ticker_with_sma200(ticker)
                    if result:
                        return result

                # Fallback: use yfinance directly
                import yfinance as yf

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
                sma_source = "unavailable"

                if days_available >= 200:
                    sma_200 = float(hist["Close"].rolling(window=200).mean().iloc[-1])
                    sma_200_deviation_pct = ((current_price - sma_200) / sma_200) * 100
                    sma_source = "calculated_200d"
                elif days_available >= 50:
                    # Use available data as approximation
                    sma_200 = float(hist["Close"].mean())
                    sma_200_deviation_pct = ((current_price - sma_200) / sma_200) * 100
                    sma_source = "proxy_mean"
                    logger.debug(f"{ticker}: Only {days_available} days, using mean as SMA proxy")

                return {
                    "ticker": ticker,
                    "price": current_price,
                    "sma_200": sma_200,
                    "sma_200_deviation_pct": sma_200_deviation_pct,
                    "days_of_history": days_available,
                    "sma_source": sma_source,
                }

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {ticker} "
                        f"after {delay}s: {e}"
                    )
                    time.sleep(delay)

        logger.error(f"All {max_retries} retries failed for {ticker}: {last_error}")
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
        """
        Calculate median PE from company_fundamentals table with IQR outlier removal.

        Uses Interquartile Range (IQR) method to filter outliers instead of
        hardcoded PE < 100 threshold. This adapts dynamically to sector-specific
        PE distributions.

        Args:
            sector: GICS sector name

        Returns:
            Median PE after outlier removal, or None if insufficient data
        """
        if self.db is None:
            return None

        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Fetch all positive PE ratios (no hardcoded upper limit)
                    cur.execute("""
                        SELECT pe_ratio
                        FROM company_fundamentals
                        WHERE sector = %s
                          AND pe_ratio > 0
                          AND last_updated > NOW() - INTERVAL '30 days'
                    """, (sector,))
                    pe_values = [float(row[0]) for row in cur.fetchall()]

                    if len(pe_values) < 5:
                        return None

                    # IQR outlier removal
                    import numpy as np
                    q1 = float(np.percentile(pe_values, 25))
                    q3 = float(np.percentile(pe_values, 75))
                    iqr = q3 - q1

                    # Standard IQR bounds (1.5 * IQR)
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    # Filter outliers
                    filtered = [pe for pe in pe_values if lower_bound <= pe <= upper_bound]

                    if len(filtered) < 3:
                        logger.debug(
                            f"Sector {sector}: Too few values after IQR filter "
                            f"({len(filtered)} from {len(pe_values)})"
                        )
                        return None

                    import statistics
                    median_pe = float(statistics.median(filtered))
                    logger.debug(
                        f"Sector {sector}: PE median {median_pe:.2f} "
                        f"(IQR filtered {len(pe_values)} → {len(filtered)})"
                    )
                    return median_pe

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

    def _get_sector_pe_source(self, sector: str) -> str:
        """
        Determine the source of sector PE data for audit trail.

        Returns:
            'database' if from cache, 'calculated' if freshly computed,
            'benchmark_etf' if from ETF fallback, 'unavailable' if not found
        """
        cache_key = f"{sector}"

        # Check if from in-memory cache
        if cache_key in self._sector_pe_cache:
            return "database"  # In-memory cache is populated from DB or calculation

        # Check if in database
        if self.db:
            try:
                with self.db.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT 1 FROM sector_pe_medians
                            WHERE sector = %s AND cache_expires_at > NOW()
                        """, (sector,))
                        if cur.fetchone():
                            return "database"
            except Exception:
                pass

        return "calculated"

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

        - FULL: All data available (price, SMA200 with 200+ days, PE, sector PE)
        - PARTIAL: Price and at least one metric, OR SMA based on proxy (<200 days)
        - INSUFFICIENT: Only price, or history < 100 days

        Args:
            has_price: Price data available
            has_sma: SMA200 available
            has_pe: PE ratio available
            has_sector_pe: Sector PE available
            days_available: Trading days of history

        Returns:
            Data quality level
        """
        # Storia troppo breve per SMA affidabile
        if days_available < 100:
            return "INSUFFICIENT"

        # SMA basata su proxy mean (100-199 giorni) → downgrade da FULL a PARTIAL
        if days_available < 200:
            if has_price and has_sma and has_pe and has_sector_pe:
                return "PARTIAL"  # Ha tutti i dati ma SMA è un proxy
            elif has_price and (has_sma or has_pe):
                return "PARTIAL"
            else:
                return "INSUFFICIENT"

        # Storia completa (≥200 giorni)
        if has_price and has_sma and has_pe and has_sector_pe:
            return "FULL"
        elif has_price and (has_sma or has_pe):
            return "PARTIAL"
        else:
            return "INSUFFICIENT"
