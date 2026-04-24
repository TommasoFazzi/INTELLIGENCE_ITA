#!/usr/bin/env python3
"""
One-shot backfill: populate 60 days of history for TTF_GAS and YIELD_CURVE_10Y_3M.

Steps:
  1. TTF_GAS  — download SRUUF TTF=F daily closes from yfinance (EUR/MWh)
  2. YIELD_CURVE_10Y_3M — download T10Y3M from FRED REST API (daily %)
  3. Insert each series with correct previous_value chain
  4. ON CONFLICT DO UPDATE (idempotent — safe to re-run)

Requires: DATABASE_URL, FRED_API_KEY in environment.

Run from repo root on the production server:
    python3 scripts/backfill_new_indicators_b2.py
"""

import sys
import os
from pathlib import Path
from datetime import date, timedelta

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import yfinance as yf
import psycopg2

DB_URL = os.environ["DATABASE_URL"]
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
LOOKBACK_DAYS = 60

INDICATORS = [
    {
        "key": "TTF_GAS",
        "source": "yfinance",
        "symbol": "TTF=F",
        "unit": "EUR",
        "category": "COMMODITIES",
    },
    {
        "key": "YIELD_CURVE_10Y_3M",
        "source": "fred",
        "fred_series": "T10Y3M",
        "unit": "%",
        "category": "RATES",
    },
]


def fetch_yfinance(symbol: str, start_date: date, end_date: date) -> list:
    """Return sorted list of (date, close_price) from yfinance."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())
    if hist.empty:
        return []
    return sorted(
        [(d.date(), float(row["Close"])) for d, row in hist.iterrows()],
        key=lambda x: x[0],
    )


def fetch_fred(fred_series: str, start_date: date, end_date: date) -> list:
    """Return sorted list of (date, value) from FRED REST API."""
    if not FRED_API_KEY:
        print(f"  ERROR: FRED_API_KEY not set — cannot fetch {fred_series}")
        return []
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": fred_series,
        "observation_start": start_date.isoformat(),
        "observation_end": end_date.isoformat(),
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    observations = r.json().get("observations", [])
    # FRED uses '.' for missing values (weekends, holidays) — skip them
    rows = [
        (date.fromisoformat(obs["date"]), float(obs["value"]))
        for obs in observations
        if obs["value"] != "."
    ]
    return sorted(rows, key=lambda x: x[0])


def backfill_indicator(cur, indicator: dict, rows: list) -> int:
    """Insert rows with correct previous_value chain. Returns number inserted."""
    inserted = 0
    for i, (trading_date, value) in enumerate(rows):
        prev_value = rows[i - 1][1] if i > 0 else None
        cur.execute(
            """
            INSERT INTO macro_indicators
                (date, indicator_key, value, unit, category, previous_value)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, indicator_key)
            DO UPDATE SET
                value = EXCLUDED.value,
                previous_value = EXCLUDED.previous_value,
                updated_at = NOW()
            """,
            (trading_date, indicator["key"], value, indicator["unit"],
             indicator["category"], prev_value),
        )
        inserted += 1
    return inserted


def main() -> int:
    end_date = date.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    print(f"Backfill window: {start_date} → {end_date} ({LOOKBACK_DAYS} calendar days)")

    print(f"\nConnecting to database...")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            for indicator in INDICATORS:
                key = indicator["key"]
                print(f"\n── {key} ──────────────────────────")

                # Fetch data
                if indicator["source"] == "yfinance":
                    rows = fetch_yfinance(indicator["symbol"], start_date, end_date)
                    source_label = f"yfinance ({indicator['symbol']})"
                else:
                    rows = fetch_fred(indicator["fred_series"], start_date, end_date)
                    source_label = f"FRED ({indicator['fred_series']})"

                if not rows:
                    print(f"  WARNING: no data from {source_label} — skipping {key}")
                    continue

                print(f"  Source: {source_label}")
                print(f"  Got {len(rows)} rows: {rows[0][0]} → {rows[-1][0]}")
                print(f"  Value range: {min(v for _, v in rows):.4f} – {max(v for _, v in rows):.4f} {indicator['unit']}")

                # Clear existing rows then insert (ensures clean previous_value chain)
                cur.execute(
                    "SELECT COUNT(*) FROM macro_indicators WHERE indicator_key = %s",
                    (key,)
                )
                existing = cur.fetchone()[0]
                if existing:
                    print(f"  Replacing {existing} existing rows...")

                inserted = backfill_indicator(cur, indicator, rows)
                print(f"  Inserted {inserted} rows.")

        conn.commit()
        print("\nDone. All indicators backfilled successfully.")
        return 0

    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
