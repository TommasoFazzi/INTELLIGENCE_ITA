#!/usr/bin/env python3
"""
Backfill Market Data from Sprint 2.2 Trade Signals

Extracts all tickers mentioned in articles.ai_analysis (Sprint 2.2 results)
and fetches their market data from Yahoo Finance.

Usage:
    python scripts/backfill_market_data.py                    # Dry-run mode (no DB writes)
    python scripts/backfill_market_data.py --execute           # Execute with DB writes
    python scripts/backfill_market_data.py --tickers LMT RTX   # Specific tickers only
"""

import sys
import argparse
from pathlib import Path
from typing import List, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from src.integrations.market_data import MarketDataService
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_tickers_from_articles(db: DatabaseManager) -> Set[str]:
    """
    Extract all unique tickers from articles.ai_analysis column.

    Returns:
        Set of ticker symbols (e.g., {'LMT', 'RTX', 'TSM'})
    """
    logger.info("Extracting tickers from articles.ai_analysis...")

    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # Query to extract tickers from JSONB array
                cursor.execute("""
                    SELECT DISTINCT
                        jsonb_array_elements(ai_analysis->'related_tickers')->>'ticker' AS ticker
                    FROM articles
                    WHERE ai_analysis IS NOT NULL
                      AND ai_analysis->'related_tickers' IS NOT NULL
                      AND jsonb_array_length(ai_analysis->'related_tickers') > 0
                    ORDER BY ticker
                """)

                rows = cursor.fetchall()

        tickers = {row[0] for row in rows if row[0]}

        logger.info(f"✓ Extracted {len(tickers)} unique tickers from database")
        return tickers

    except Exception as e:
        logger.error(f"Failed to extract tickers: {e}")
        return set()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Backfill market data from Sprint 2.2 trade signals"
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute backfill (default: dry-run mode)'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to fetch (default: extract from articles)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching (force fresh API calls)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MARKET DATA BACKFILL - Sprint 3")
    logger.info("=" * 80)

    # Initialize services
    db = DatabaseManager()
    service = MarketDataService(db=db)

    # Get tickers
    if args.tickers:
        tickers = set(args.tickers)
        logger.info(f"Using {len(tickers)} tickers from command line: {', '.join(sorted(tickers))}")
    else:
        tickers = extract_tickers_from_articles(db)
        if not tickers:
            logger.error("No tickers found in database. Run Sprint 2.2 analysis first:")
            logger.error("  python scripts/generate_report.py --analyze-articles")
            return 1

    logger.info(f"\nTickers to process: {', '.join(sorted(tickers))}")

    # Dry-run warning
    if not args.execute:
        logger.warning("\n⚠️  DRY-RUN MODE - No data will be saved to database")
        logger.warning("Use --execute flag to save data\n")

    # Fetch market data
    logger.info(f"\n[STEP 1] Fetching market data from Yahoo Finance...")
    logger.info(f"Rate limit: {args.delay}s delay between requests")

    results = service.fetch_multiple_tickers(
        tickers=list(tickers),
        delay_seconds=args.delay,
        use_cache=not args.no_cache,
        save_to_db=args.execute  # Only save if --execute flag is set
    )

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 80)

    success_tickers = [t for t, data in results.items() if data is not None]
    failed_tickers = [t for t, data in results.items() if data is None]

    success_rate = (len(success_tickers) / len(tickers) * 100) if tickers else 0

    logger.info(f"Total tickers: {len(tickers)}")
    logger.info(f"Successful: {len(success_tickers)} ({success_rate:.1f}%)")
    logger.info(f"Failed: {len(failed_tickers)}")

    if success_tickers:
        logger.info(f"\n✅ Successfully fetched: {', '.join(sorted(success_tickers))}")

    if failed_tickers:
        logger.warning(f"\n❌ Failed to fetch: {', '.join(sorted(failed_tickers))}")
        logger.warning("Common reasons:")
        logger.warning("  - Ticker delisted or suspended")
        logger.warning("  - Non-US ticker format (e.g., 'BA.L' for LSE)")
        logger.warning("  - API rate limit exceeded (try increasing --delay)")

    # Show sample data
    if success_tickers and results[success_tickers[0]]:
        sample_ticker = success_tickers[0]
        sample_data = results[sample_ticker]

        logger.info(f"\nSample Data ({sample_ticker}):")
        logger.info(f"  Date: {sample_data['date']}")
        logger.info(f"  Close: ${sample_data['close_price']}")
        logger.info(f"  Volume: {sample_data['volume']:,}")
        logger.info(f"  Volatility (7d): {float(sample_data['volatility_7d']):.2%}")
        logger.info(f"  Relative Volume: {float(sample_data['relative_volume']):.2f}x")

    # Database status
    if args.execute:
        logger.info(f"\n✅ Data saved to database (market_data table)")
        logger.info(f"You can now query market data for trade signal validation")
    else:
        logger.info(f"\n⚠️  DRY-RUN: No data was saved to database")
        logger.info(f"Re-run with --execute to save data")

    logger.info("=" * 80)

    # Next steps
    if args.execute and success_tickers:
        logger.info("\nNext Steps:")
        logger.info("1. Verify data in database:")
        logger.info("   SELECT ticker, date, close_price, volatility_7d FROM market_data ORDER BY date DESC LIMIT 10;")
        logger.info("2. Test market data queries:")
        logger.info("   python -c \"from src.integrations.market_data import MarketDataService; s=MarketDataService(); print(s.get_latest_market_data('LMT'))\"")
        logger.info("3. Schedule daily updates:")
        logger.info("   python scripts/backfill_market_data.py --execute --tickers LMT RTX TSM (add to cron)")

    return 0 if success_rate >= 50 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
