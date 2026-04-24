#!/usr/bin/env python3
"""
One-shot backfill: replace URA history with SRUUF in macro_indicators.

Steps:
  1. Delete all existing URANIUM rows (URA prices, invalid deltas)
  2. Download SRUUF daily closes from yfinance (last 90 days → keeps 63 trading days)
  3. Insert each row with previous_value = prior trading day close
  4. Flag today's report as draft so it doesn't enter the knowledge base

Run from repo root on the production server:
    python scripts/backfill_sruuf.py

The script is idempotent: re-running it will replace all URANIUM rows cleanly.
"""

import sys
from pathlib import Path
from datetime import date, timedelta

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import psycopg2
import psycopg2.extras
import os

DB_URL = os.environ["DATABASE_URL"]

INDICATOR_KEY = "URANIUM"
SYMBOL = "SRUUF"
UNIT = "USD"
CATEGORY = "COMMODITIES"
LOOKBACK_DAYS = 90  # yfinance window; keeps ~63 trading days


def main() -> int:
    print(f"Connecting to database...")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # ── 1. Delete all existing URANIUM rows ──────────────────────────
            cur.execute("SELECT COUNT(*) FROM macro_indicators WHERE indicator_key = %s",
                        (INDICATOR_KEY,))
            existing = cur.fetchone()[0]
            print(f"Deleting {existing} existing URANIUM rows (URA history)...")
            cur.execute("DELETE FROM macro_indicators WHERE indicator_key = %s",
                        (INDICATOR_KEY,))

            # ── 2. Download SRUUF history ─────────────────────────────────────
            print(f"Downloading {SYMBOL} history ({LOOKBACK_DAYS}d window)...")
            end_date = date.today()
            start_date = end_date - timedelta(days=LOOKBACK_DAYS)

            ticker = yf.Ticker(SYMBOL)
            hist = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())

            if hist.empty:
                print(f"ERROR: yfinance returned no data for {SYMBOL}")
                conn.rollback()
                return 1

            # Build sorted list of (date, close_price)
            rows = sorted(
                [(d.date(), float(row["Close"])) for d, row in hist.iterrows()],
                key=lambda x: x[0],
            )
            print(f"Got {len(rows)} trading days: {rows[0][0]} → {rows[-1][0]}")

            # ── 3. Insert with correct previous_value ─────────────────────────
            inserted = 0
            for i, (trading_date, close) in enumerate(rows):
                prev_value = rows[i - 1][1] if i > 0 else None

                cur.execute("""
                    INSERT INTO macro_indicators
                        (date, indicator_key, value, unit, category, previous_value)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, indicator_key)
                    DO UPDATE SET
                        value = EXCLUDED.value,
                        previous_value = EXCLUDED.previous_value,
                        updated_at = NOW()
                """, (trading_date, INDICATOR_KEY, close, UNIT, CATEGORY, prev_value))
                inserted += 1

            print(f"Inserted {inserted} SRUUF rows into macro_indicators.")

            # ── 4. Flag today's report as draft ───────────────────────────────
            today = date.today()
            cur.execute("""
                UPDATE reports
                SET status = 'draft',
                    content_embedding = NULL
                WHERE report_date = %s
                  AND report_type = 'daily'
            """, (today,))
            flagged = cur.rowcount
            if flagged:
                print(f"Flagged {flagged} report(s) for {today} as draft "
                      f"(will not enter knowledge base).")
            else:
                print(f"No report found for {today} to flag.")

            conn.commit()
            print("Done. Re-run the pipeline to generate a clean report.")
            return 0

    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
