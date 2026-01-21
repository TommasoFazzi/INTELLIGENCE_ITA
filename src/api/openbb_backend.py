"""
OpenBB Workspace Backend for Intelligence ITA

Exposes Intelligence ITA data as widgets for OpenBB Workspace Desktop.
Run with: uvicorn src.api.openbb_backend:app --reload --port 7779
"""

import sys
import json
from pathlib import Path
from datetime import date
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src.storage.database import DatabaseManager
from src.integrations.openbb_service import OpenBBMarketService
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligence ITA - OpenBB Backend",
    description="Custom backend for OpenBB Workspace integration",
    version="1.0.0"
)

# Configure CORS for OpenBB Workspace
origins = [
    "https://pro.openbb.co",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db = DatabaseManager()


# ===================================================================
# Widget Configuration Endpoint
# ===================================================================

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "name": "Intelligence ITA - OpenBB Backend",
        "version": "1.0.0",
        "widgets_config": "/widgets.json"
    }


@app.get("/widgets.json")
def get_widgets():
    """Return widget configuration for OpenBB Workspace"""
    widgets_path = Path(__file__).parent / "widgets.json"
    try:
        return JSONResponse(content=json.load(widgets_path.open()))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="widgets.json not found")


# ===================================================================
# Markdown Widgets
# ===================================================================

@app.get("/get_latest_report", response_class=PlainTextResponse)
def get_latest_report():
    """
    Return the latest intelligence report as markdown.
    Prefers final_content (human-reviewed) over draft_content.
    """
    try:
        # Get the most recent report
        reports = db.get_all_reports(limit=1)
        if not reports:
            return "# No Reports Available\n\nNo intelligence reports have been generated yet."

        # Get full report with content
        report = db.get_report(reports[0]['id'])
        if not report:
            return "# Report Not Found\n\nCould not retrieve report content."

        # Prefer reviewed content over draft
        content = report.get('final_content') or report.get('draft_content')
        if not content:
            return "# Empty Report\n\nReport exists but has no content."

        # Add metadata header
        report_date = report.get('report_date', 'Unknown')
        status = report.get('status', 'draft')
        status_emoji = {
            'draft': 'Draft',
            'reviewed': 'Reviewed',
            'approved': 'Approved'
        }.get(status, status)

        header = f"> **Report Date**: {report_date} | **Status**: {status_emoji}\n\n---\n\n"

        return header + content

    except Exception as e:
        logger.error(f"Error fetching latest report: {e}")
        return f"# Error\n\nCould not fetch report: {str(e)}"


@app.get("/get_macro_summary", response_class=PlainTextResponse)
def get_macro_summary():
    """
    Return macro economic indicators as formatted markdown.
    Uses OpenBBMarketService.get_macro_context_text().
    """
    try:
        service = OpenBBMarketService()
        macro_text = service.get_macro_context_text(date.today())

        if not macro_text or macro_text.strip() == "":
            return "# Macro Context\n\n*No macro data available. Run `fetch_daily_market_data.py` to populate.*"

        return f"# Macro Context\n\n{macro_text}"

    except Exception as e:
        logger.error(f"Error fetching macro summary: {e}")
        return f"# Macro Context\n\n*Error fetching data: {str(e)}*"


# ===================================================================
# Table Widget
# ===================================================================

@app.get("/get_conviction_board")
def get_conviction_board():
    """
    Return active trade signals as a table.
    Returns JSON array of signal objects for OpenBB table widget.
    """
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        ticker,
                        signal,
                        timeframe,
                        rationale,
                        confidence,
                        alignment_score,
                        signal_source,
                        category,
                        created_at
                    FROM trade_signals
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT 20
                """)

                signals = []
                for row in cur.fetchall():
                    # Signal emoji
                    signal_type = row[1].upper() if row[1] else 'NEUTRAL'
                    signal_emoji = {
                        'BULLISH': 'LONG',
                        'BEARISH': 'SHORT',
                        'NEUTRAL': 'HOLD',
                        'WATCHLIST': 'WATCH'
                    }.get(signal_type, signal_type)

                    signals.append({
                        'Ticker': row[0],
                        'Signal': signal_emoji,
                        'Timeframe': row[2] or '-',
                        'Rationale': (row[3][:80] + '...') if row[3] and len(row[3]) > 80 else (row[3] or '-'),
                        'Confidence': f"{row[4]:.0%}" if row[4] else '-',
                        'Alignment': f"{row[5]:.0%}" if row[5] else '-',
                        'Source': row[6] or '-',
                        'Category': row[7] or '-'
                    })

                if not signals:
                    # Return placeholder if no signals
                    return [{"Ticker": "-", "Signal": "No active signals", "Timeframe": "-",
                             "Rationale": "Generate a report with --macro-first flag",
                             "Confidence": "-", "Alignment": "-", "Source": "-", "Category": "-"}]

                return signals

    except Exception as e:
        logger.error(f"Error fetching conviction board: {e}")
        return [{"Ticker": "ERROR", "Signal": str(e), "Timeframe": "-",
                 "Rationale": "-", "Confidence": "-", "Alignment": "-", "Source": "-", "Category": "-"}]


# ===================================================================
# Metric Widgets
# ===================================================================

@app.get("/get_metric_articles_24h")
def get_metric_articles_24h():
    """Return count of articles processed in the last 24 hours."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM articles
                    WHERE published_date > NOW() - INTERVAL '24 hours'
                """)
                count = cur.fetchone()[0]

                # Also get 7-day average for comparison
                cur.execute("""
                    SELECT COUNT(*) / 7.0
                    FROM articles
                    WHERE published_date > NOW() - INTERVAL '7 days'
                """)
                avg = cur.fetchone()[0] or 0

                # Calculate delta
                delta = None
                if avg > 0:
                    delta_pct = ((count - avg) / avg) * 100
                    delta = f"{delta_pct:+.0f}%"

                return {
                    "value": count,
                    "label": "Articles (24h)",
                    "delta": delta
                }

    except Exception as e:
        logger.error(f"Error fetching articles metric: {e}")
        return {"value": 0, "label": "Articles (24h)", "delta": None}


@app.get("/get_metric_active_signals")
def get_metric_active_signals():
    """Return count of active trade signals (last 7 days)."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM trade_signals
                    WHERE created_at > NOW() - INTERVAL '7 days'
                """)
                count = cur.fetchone()[0]

                # Count by signal type for context
                cur.execute("""
                    SELECT signal, COUNT(*)
                    FROM trade_signals
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    GROUP BY signal
                """)
                breakdown = dict(cur.fetchall())
                bullish = breakdown.get('BULLISH', 0)
                bearish = breakdown.get('BEARISH', 0)

                delta = None
                if bullish > 0 or bearish > 0:
                    delta = f"{bullish} LONG / {bearish} SHORT"

                return {
                    "value": count,
                    "label": "Active Signals",
                    "delta": delta
                }

    except Exception as e:
        logger.error(f"Error fetching signals metric: {e}")
        return {"value": 0, "label": "Active Signals", "delta": None}


@app.get("/get_metric_sources_active")
def get_metric_sources_active():
    """Return count of unique sources active in the last 24 hours."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(DISTINCT source)
                    FROM articles
                    WHERE published_date > NOW() - INTERVAL '24 hours'
                """)
                count = cur.fetchone()[0]

                # Total configured sources for context
                cur.execute("""
                    SELECT COUNT(DISTINCT source)
                    FROM articles
                """)
                total = cur.fetchone()[0]

                delta = f"of {total} total" if total > 0 else None

                return {
                    "value": count,
                    "label": "Sources Active",
                    "delta": delta
                }

    except Exception as e:
        logger.error(f"Error fetching sources metric: {e}")
        return {"value": 0, "label": "Sources Active", "delta": None}


# ===================================================================
# Financial Intelligence v2 - Chart Overlay
# ===================================================================

@app.get("/api/v1/openbb/chart_overlay")
def get_chart_overlay(ticker: str, days: int = 365):
    """
    Generate OHLCV + signal annotations for chart overlay.

    Returns data suitable for OpenBB chart widget with signal markers.

    Args:
        ticker: Stock symbol (e.g., 'LMT')
        days: Number of days of history (default: 365)

    Returns:
        {
            "ticker": "LMT",
            "ohlcv": [...],  # Standard OHLCV array
            "sma_200": [...],  # SMA200 values
            "annotations": [
                {
                    "date": "2025-01-15",
                    "type": "signal",
                    "signal": "BULLISH",
                    "intelligence_score": 78,
                    "rationale": "..."
                }
            ]
        }
    """
    import yfinance as yf
    import pandas as pd

    try:
        # Fetch OHLCV from yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        # Build OHLCV array
        ohlcv = []
        for idx, row in hist.iterrows():
            ohlcv.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"])
            })

        # Calculate SMA200
        sma_200 = []
        if len(hist) >= 200:
            sma_series = hist['Close'].rolling(window=200).mean()
            for idx, val in sma_series.items():
                if not pd.isna(val):
                    sma_200.append({
                        "date": idx.strftime("%Y-%m-%d"),
                        "value": round(val, 2)
                    })

        # Fetch signal annotations from database
        annotations = []
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            ts.created_at::date,
                            ts.signal,
                            ts.intelligence_score,
                            ts.rationale,
                            ts.valuation_rating,
                            ts.sma_200_deviation,
                            ts.confidence
                        FROM trade_signals ts
                        JOIN reports r ON ts.report_id = r.id
                        WHERE ts.ticker = %s
                          AND ts.created_at > NOW() - INTERVAL '%s days'
                        ORDER BY ts.created_at DESC
                    """, (ticker.upper(), days))

                    for row in cur.fetchall():
                        annotations.append({
                            "date": row[0].strftime("%Y-%m-%d") if row[0] else None,
                            "type": "signal",
                            "signal": row[1],
                            "intelligence_score": row[2],
                            "rationale": row[3][:150] + "..." if row[3] and len(row[3]) > 150 else row[3],
                            "valuation_rating": row[4],
                            "sma_deviation": float(row[5]) if row[5] else None,
                            "confidence": float(row[6]) if row[6] else None
                        })
        except Exception as e:
            logger.warning(f"Failed to fetch annotations for {ticker}: {e}")

        return {
            "ticker": ticker.upper(),
            "data_points": len(ohlcv),
            "ohlcv": ohlcv,
            "sma_200": sma_200,
            "annotations": annotations,
            "annotation_count": len(annotations)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart overlay error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/openbb/high_conviction_signals")
def get_high_conviction_signals(min_score: int = 70, days: int = 7):
    """
    Get signals with intelligence_score above threshold.

    Args:
        min_score: Minimum intelligence score (default: 70)
        days: Lookback period in days (default: 7)

    Returns:
        List of high-conviction signals with full metadata.
    """
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        ts.ticker,
                        ts.signal,
                        ts.timeframe,
                        ts.intelligence_score,
                        ts.valuation_rating,
                        ts.sma_200_deviation,
                        ts.confidence,
                        ts.rationale,
                        ts.created_at::date,
                        ts.data_quality
                    FROM trade_signals ts
                    WHERE ts.intelligence_score >= %s
                      AND ts.created_at > NOW() - INTERVAL '%s days'
                    ORDER BY ts.intelligence_score DESC, ts.created_at DESC
                    LIMIT 50
                """, (min_score, days))

                signals = []
                for row in cur.fetchall():
                    signals.append({
                        "ticker": row[0],
                        "signal": row[1],
                        "timeframe": row[2],
                        "intelligence_score": row[3],
                        "valuation_rating": row[4],
                        "sma_deviation": float(row[5]) if row[5] else None,
                        "confidence": float(row[6]) if row[6] else None,
                        "rationale": row[7],
                        "date": row[8].strftime("%Y-%m-%d") if row[8] else None,
                        "data_quality": row[9]
                    })

        return {
            "min_score": min_score,
            "days": days,
            "count": len(signals),
            "signals": signals
        }

    except Exception as e:
        logger.error(f"High conviction signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# Health Check
# ===================================================================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Quick DB check
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  Intelligence ITA - OpenBB Workspace Backend")
    print("="*60)
    print("\n  Starting server on http://localhost:7779")
    print("  Widgets config: http://localhost:7779/widgets.json")
    print("\n  To connect in OpenBB Workspace:")
    print("  1. Right-click on dashboard → 'Add Widget'")
    print("  2. Enter: http://localhost:7779/widgets.json")
    print("  3. Drag 'Intelligence ITA' widgets to dashboard")
    print("\n" + "="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=7779, reload=True)
