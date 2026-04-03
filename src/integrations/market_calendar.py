"""
Market calendar utilities for holiday-aware fetch scheduling.

Weekend skip (weekday >= 5) is handled upstream by fetch scripts.
This module adds NYSE holiday detection on weekdays, enabling:
  - Logging when fetching on a US market holiday
  - Backfill logic that skips weekdays where NYSE was closed

Usage:
    from src.integrations.market_calendar import fetch_mode, is_nyse_open, last_nyse_trading_day
"""
from datetime import date, timedelta
from typing import Optional

import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar('NYSE')


def is_nyse_open(target_date: date) -> bool:
    """Return True if NYSE is open on target_date (False for weekends AND US holidays)."""
    schedule = _NYSE.schedule(
        start_date=target_date.strftime('%Y-%m-%d'),
        end_date=target_date.strftime('%Y-%m-%d'),
    )
    return not schedule.empty


def last_nyse_trading_day(before: date) -> Optional[date]:
    """
    Return the most recent NYSE trading day strictly before `before`.

    Used as a reference date for equity/commodity indicators when fetching
    on a US market holiday (weekday, NYSE closed).

    Returns None only in pathological cases (e.g. no trading days in last 10 days).
    """
    start = before - timedelta(days=10)
    schedule = _NYSE.schedule(
        start_date=start.strftime('%Y-%m-%d'),
        end_date=(before - timedelta(days=1)).strftime('%Y-%m-%d'),
    )
    if schedule.empty:
        return None
    return schedule.index[-1].date()


def fetch_mode(target_date: date) -> str:
    """
    Classify the fetch mode for a given date:

      'skip'    — Saturday or Sunday (markets globally closed, no fetch)
      'normal'  — weekday, NYSE open
      'holiday' — weekday, NYSE closed (US market holiday)

    Note: yfinance already returns the last available close on holidays via
    ticker.history(period='5d'), so equity/commodity data is still collected
    with acceptable 1-2 day staleness. FRED and FX are unaffected.
    """
    if target_date.weekday() >= 5:
        return 'skip'
    if is_nyse_open(target_date):
        return 'normal'
    return 'holiday'
