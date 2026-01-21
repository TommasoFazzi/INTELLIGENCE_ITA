#!/usr/bin/env python3
"""
Yahoo Finance Integration for Market Data (Sprint 3)

Features:
- Fetch OHLCV data for tickers mentioned in trade signals
- Calculate 7-day volatility (std dev of daily returns)
- Calculate relative volume (vs 30-day average)
- Database persistence with conflict handling
- 1-hour caching to avoid rate limits
- Batch processing with rate limiting
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

import yfinance as yf
import pandas as pd

from ..storage.database import DatabaseManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataService:
    """
    Service for fetching and storing market data from Yahoo Finance.

    Includes database integration, caching, and derived metrics calculation.
    """

    def __init__(self, db: Optional[DatabaseManager] = None, cache_ttl_hours: int = 1):
        """
        Initialize market data service.

        Args:
            db: DatabaseManager instance (creates new if None)
            cache_ttl_hours: Cache TTL in hours (default: 1 hour)
        """
        self.db = db or DatabaseManager()
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Note: yfinance 0.2.66+ uses curl_cffi internally for anti-bot protection
        # No need to create custom session - yfinance handles it automatically

        logger.info(f"MarketDataService initialized (cache TTL: {cache_ttl_hours}h, yfinance 0.2.66+ with curl_cffi)")

    def fetch_ticker_data(
        self,
        ticker: str,
        period: str = "1mo",
        use_cache: bool = True,
        save_to_db: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch market data for a single ticker with caching and database persistence.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSM')
            period: Data period (default: '1mo' for volatility calculation)
            use_cache: Use cached data if available (default: True)
            save_to_db: Save to database after fetching (default: True)

        Returns:
            Dictionary with:
            - ticker: str
            - date: datetime (most recent trading day)
            - open_price, high_price, low_price, close_price: Decimal
            - volume: int
            - volatility_7d: Decimal (7-day rolling std dev of returns)
            - relative_volume: Decimal (volume / 30-day avg volume)
            - source: str ('yahoo_finance')
            - fetched_at: datetime

            Returns None if ticker not found or error occurs.
        """
        # Check cache
        if use_cache and ticker in self._cache:
            cached_data = self._cache[ticker]
            cache_age = datetime.now() - cached_data['fetched_at']
            if cache_age < self.cache_ttl:
                logger.debug(f"Cache HIT for {ticker} (age: {cache_age.seconds}s)")
                return cached_data['data']
            else:
                logger.debug(f"Cache EXPIRED for {ticker} (age: {cache_age.seconds}s)")

        # Fetch from Yahoo Finance
        try:
            logger.info(f"Fetching {ticker} from Yahoo Finance (period: {period})...")
            stock = yf.Ticker(ticker)  # yfinance 0.2.66+ handles session internally with curl_cffi
            hist = stock.history(period=period)

            if hist.empty:
                logger.warning(f"No data found for ticker: {ticker}")
                return None

            # Extract latest data
            latest_date = hist.index[-1]
            latest_row = hist.iloc[-1]

            # Calculate 7-day volatility (std dev of daily returns)
            volatility_7d = self._calculate_volatility(hist, window=7)

            # Calculate relative volume (current volume / 30-day avg)
            relative_volume = self._calculate_relative_volume(hist, window=30)

            # Prepare data dictionary
            data = {
                'ticker': ticker,
                'date': latest_date.to_pydatetime().date(),
                'open_price': Decimal(str(round(latest_row['Open'], 4))),
                'high_price': Decimal(str(round(latest_row['High'], 4))),
                'low_price': Decimal(str(round(latest_row['Low'], 4))),
                'close_price': Decimal(str(round(latest_row['Close'], 4))),
                'volume': int(latest_row['Volume']),
                'volatility_7d': volatility_7d,
                'relative_volume': relative_volume,
                'source': 'yahoo_finance',
                'fetched_at': datetime.now()
            }

            logger.info(f"✓ Fetched {ticker}: ${data['close_price']} "
                       f"(vol: {data['volatility_7d']:.2%}, rel_vol: {data['relative_volume']:.2f}x)")

            # Cache the result
            self._cache[ticker] = {
                'data': data,
                'fetched_at': datetime.now()
            }

            # Save to database
            if save_to_db:
                self._save_to_database(data)

            return data

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return None

    def _calculate_volatility(self, hist: pd.DataFrame, window: int = 7) -> Decimal:
        """
        Calculate rolling volatility (std dev of daily returns).

        Args:
            hist: Historical price dataframe
            window: Rolling window in days (default: 7)

        Returns:
            Decimal volatility (e.g., 0.0234 = 2.34% daily volatility)
        """
        if len(hist) < 2:
            return Decimal('0.0000')

        # Calculate daily returns
        returns = hist['Close'].pct_change().dropna()

        if len(returns) < window:
            # Not enough data for rolling window, use all available
            volatility = returns.std()
        else:
            # Use last N days for rolling calculation
            volatility = returns.tail(window).std()

        return Decimal(str(round(volatility, 4))) if not pd.isna(volatility) else Decimal('0.0000')

    def _calculate_relative_volume(self, hist: pd.DataFrame, window: int = 30) -> Decimal:
        """
        Calculate relative volume (current volume / N-day average).

        Args:
            hist: Historical price dataframe
            window: Average window in days (default: 30)

        Returns:
            Decimal relative volume (e.g., 1.5 = 50% above average)
        """
        if len(hist) < 2:
            return Decimal('1.00')

        latest_volume = hist['Volume'].iloc[-1]

        if len(hist) < window:
            # Not enough data for full window, use all available
            avg_volume = hist['Volume'].mean()
        else:
            # Use last N days (excluding today)
            avg_volume = hist['Volume'].iloc[:-1].tail(window).mean()

        if avg_volume == 0 or pd.isna(avg_volume):
            return Decimal('1.00')

        relative_vol = latest_volume / avg_volume
        return Decimal(str(round(relative_vol, 2)))

    def _save_to_database(self, data: Dict[str, Any]) -> bool:
        """
        Save market data to database with ON CONFLICT DO UPDATE.

        Args:
            data: Market data dictionary

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Insert or update market data
                    cursor.execute("""
                        INSERT INTO market_data (
                            ticker, date,
                            open_price, high_price, low_price, close_price, volume,
                            volatility_7d, relative_volume,
                            created_at, updated_at
                        ) VALUES (
                            %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s,
                            NOW(), NOW()
                        )
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            volume = EXCLUDED.volume,
                            volatility_7d = EXCLUDED.volatility_7d,
                            relative_volume = EXCLUDED.relative_volume,
                            updated_at = NOW()
                    """, (
                        data['ticker'], data['date'],
                        data['open_price'], data['high_price'], data['low_price'],
                        data['close_price'], data['volume'],
                        data['volatility_7d'], data['relative_volume']
                    ))

                    conn.commit()

            logger.debug(f"Saved {data['ticker']} to database")
            return True

        except Exception as e:
            logger.error(f"Failed to save {data['ticker']} to database: {e}")
            return False

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        delay_seconds: float = 0.5,
        use_cache: bool = True,
        save_to_db: bool = True
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetch market data for multiple tickers with rate limiting.

        Args:
            tickers: List of ticker symbols
            delay_seconds: Delay between API calls (default: 0.5s)
            use_cache: Use cached data if available
            save_to_db: Save to database after fetching

        Returns:
            Dictionary mapping ticker -> data (or None if failed)
        """
        results = {}

        logger.info(f"Fetching {len(tickers)} tickers with {delay_seconds}s delay...")

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] Processing {ticker}...")

            data = self.fetch_ticker_data(
                ticker=ticker,
                use_cache=use_cache,
                save_to_db=save_to_db
            )

            results[ticker] = data

            # Rate limiting (skip delay for last ticker)
            if i < len(tickers):
                time.sleep(delay_seconds)

        # Summary
        success_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"Fetch complete: {success_count}/{len(tickers)} successful")

        return results

    def get_latest_market_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get latest market data from database (no API call).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with market data or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT
                            ticker, date,
                            open_price, high_price, low_price, close_price, volume,
                            volatility_7d, relative_volume,
                            created_at, updated_at
                        FROM market_data
                        WHERE ticker = %s
                        ORDER BY date DESC
                        LIMIT 1
                    """, (ticker,))

                    row = cursor.fetchone()

            if not row:
                return None

            return {
                'ticker': row[0],
                'date': row[1],
                'open_price': row[2],
                'high_price': row[3],
                'low_price': row[4],
                'close_price': row[5],
                'volume': row[6],
                'volatility_7d': row[7],
                'relative_volume': row[8],
                'created_at': row[9],
                'updated_at': row[10]
            }

        except Exception as e:
            logger.error(f"Failed to get latest market data for {ticker}: {e}")
            return None

    def fetch_ticker_with_sma200(
        self,
        ticker: str,
        period: str = "1y",
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch OHLCV with SMA200 calculation for Financial Intelligence v2.

        Args:
            ticker: Stock symbol (e.g., 'LMT', 'RHM.DE')
            period: Data period (default '1y' for 200+ trading days)
            use_cache: Use cached data if available

        Returns:
            Dict with:
            - ticker, price, sma_200, sma_200_deviation_pct
            - days_of_history (for data quality assessment)
            - close_price, fetched_at, source
        """
        # Check cache
        cache_key = f"{ticker}_sma200"
        if use_cache and cache_key in self._cache:
            cached_data = self._cache[cache_key]
            cache_age = datetime.now() - cached_data['fetched_at']
            if cache_age < self.cache_ttl:
                logger.debug(f"SMA200 cache HIT for {ticker}")
                return cached_data['data']

        try:
            logger.info(f"Fetching {ticker} with SMA200 (period: {period})...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                logger.warning(f"No historical data for {ticker}")
                return None

            current_price = float(hist['Close'].iloc[-1])
            days_available = len(hist)

            # Calculate SMA200
            sma_200 = None
            sma_200_deviation_pct = None

            if days_available >= 200:
                sma_200 = float(hist['Close'].rolling(window=200).mean().iloc[-1])
                sma_200_deviation_pct = ((current_price - sma_200) / sma_200) * 100
                logger.debug(f"{ticker}: SMA200={sma_200:.2f}, deviation={sma_200_deviation_pct:.1f}%")
            elif days_available >= 50:
                # Partial data: use available mean as approximation
                sma_200 = float(hist['Close'].mean())
                sma_200_deviation_pct = ((current_price - sma_200) / sma_200) * 100
                logger.info(f"{ticker}: Only {days_available} days, using mean as SMA proxy")
            else:
                logger.warning(f"{ticker}: Insufficient history ({days_available} days)")

            data = {
                'ticker': ticker,
                'price': current_price,
                'close_price': Decimal(str(round(current_price, 4))),
                'sma_200': sma_200,
                'sma_200_deviation_pct': sma_200_deviation_pct,
                'days_of_history': days_available,
                'source': 'yahoo_finance',
                'fetched_at': datetime.now()
            }

            # Cache the result
            self._cache[cache_key] = {
                'data': data,
                'fetched_at': datetime.now()
            }

            return data

        except Exception as e:
            logger.error(f"Failed to fetch {ticker} with SMA200: {e}")
            return None

    def get_sector_pe_median(
        self,
        sector: str,
        region: str = "US",
        cache_ttl_days: int = 7
    ) -> Optional[float]:
        """
        Get median P/E for a sector with database caching.

        Calculation order:
        1. Check sector_pe_medians table cache
        2. Calculate from company_fundamentals if expired/missing
        3. Fallback to None if insufficient data

        Args:
            sector: GICS sector name (e.g., 'Technology', 'Industrials')
            region: Market region ('US', 'EU', 'ASIA', 'OTHER')
            cache_ttl_days: Cache duration in days

        Returns:
            Median P/E ratio or None if unavailable
        """
        import statistics

        # Check database cache first
        cached_pe = self._get_cached_sector_pe(sector)
        if cached_pe is not None:
            return cached_pe

        # Calculate from company_fundamentals
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

                    pe_values = [float(row[0]) for row in cur.fetchall()]

                    if len(pe_values) >= 5:
                        median_pe = statistics.median(pe_values)
                        self._save_sector_pe_cache(sector, median_pe, len(pe_values))
                        logger.info(f"Calculated sector PE for {sector}: {median_pe:.1f} (n={len(pe_values)})")
                        return median_pe
                    else:
                        logger.debug(f"Insufficient data for {sector} PE (n={len(pe_values)})")
                        return None

        except Exception as e:
            logger.error(f"Failed to calculate sector PE for {sector}: {e}")
            return None

    def _get_cached_sector_pe(self, sector: str) -> Optional[float]:
        """Check sector_pe_medians table for cached value."""
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
                    if row:
                        logger.debug(f"Sector PE cache HIT: {sector} = {row[0]}")
                        return float(row[0])
                    return None
        except Exception as e:
            logger.debug(f"Failed to get cached sector PE: {e}")
            return None

    def _save_sector_pe_cache(self, sector: str, pe_median: float, sample_size: int):
        """Save sector PE to database cache."""
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
                    logger.debug(f"Cached sector PE for {sector}: {pe_median}")
        except Exception as e:
            logger.error(f"Failed to cache sector PE: {e}")

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.info("Market data cache cleared")


# Standalone test
if __name__ == "__main__":
    # Test single ticker fetch
    service = MarketDataService()

    test_ticker = "AAPL"
    logger.info(f"Testing market data fetch for {test_ticker}...")

    data = service.fetch_ticker_data(test_ticker)

    if data:
        print("\n" + "=" * 80)
        print(f"MARKET DATA: {data['ticker']}")
        print("=" * 80)
        print(f"Date: {data['date']}")
        print(f"Close: ${data['close_price']}")
        print(f"Volume: {data['volume']:,}")
        print(f"Volatility (7d): {float(data['volatility_7d']):.2%}")
        print(f"Relative Volume: {float(data['relative_volume']):.2f}x")
        print("=" * 80)
    else:
        print(f"Failed to fetch data for {test_ticker}")
