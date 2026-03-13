"""
Ticker-based theme clustering service.

Shared logic for querying ticker-correlated storylines, used by both API endpoints and Oracle tools.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from ..storage.database import DatabaseManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Cache for ticker config (YAML loaded once)
_TICKERS_CACHE: Optional[Dict] = None
_CACHE_TIMESTAMP: Optional[float] = None
_CACHE_TTL_SECONDS = 3600  # 1 hour


def _get_project_root() -> Path:
    """Get the project root path (where config/ lives)."""
    return Path(__file__).parents[2]  # src/services/ticker_service.py -> project root


def load_tickers_config(config_path: str = "config/top_50_tickers.yaml") -> Dict:
    """
    Load ticker configuration from YAML file.

    Returns: Dict with structure:
    {
        "tickers": {
            "RTX": {"name": "Raytheon Technologies", "ticker": "RTX", "aliases": [...], "category": "defense"},
            ...
        },
        "categories": ["defense", "semiconductors", ...],
        "total": 29
    }

    Uses in-memory cache with 1-hour TTL to avoid repeated file reads.
    """
    global _TICKERS_CACHE, _CACHE_TIMESTAMP

    import time
    now = time.time()

    # Check cache
    if _TICKERS_CACHE is not None and _CACHE_TIMESTAMP is not None:
        if now - _CACHE_TIMESTAMP < _CACHE_TTL_SECONDS:
            return _TICKERS_CACHE

    # Load YAML
    config_file = _get_project_root() / config_path
    if not config_file.exists():
        logger.error(f"Ticker config not found: {config_file}")
        return {"tickers": {}, "categories": [], "total": 0}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        if not raw_config or 'tickers' not in raw_config:
            logger.warning(f"No 'tickers' section found in {config_path}")
            return {"tickers": {}, "categories": [], "total": 0}

        # Flatten structure: build dict keyed by uppercase ticker symbol
        tickers_dict = {}
        categories_set = set()

        for category, ticker_list in raw_config['tickers'].items():
            if not isinstance(ticker_list, list):
                continue

            categories_set.add(category)

            for entry in ticker_list:
                if 'ticker' not in entry:
                    continue

                ticker_key = entry['ticker'].upper()
                tickers_dict[ticker_key] = {
                    'ticker': entry['ticker'],
                    'name': entry.get('name', ''),
                    'exchange': entry.get('exchange', ''),
                    'aliases': entry.get('aliases', [entry['ticker']]),  # aliases or fallback to ticker
                    'category': category
                }

        result = {
            'tickers': tickers_dict,
            'categories': list(categories_set),
            'total': len(tickers_dict)
        }

        # Update cache
        _TICKERS_CACHE = result
        _CACHE_TIMESTAMP = now

        logger.info(f"Loaded {len(tickers_dict)} tickers from config (cache TTL: {_CACHE_TTL_SECONDS}s)")
        return result

    except Exception as e:
        logger.error(f"Failed to load ticker config: {e}", exc_info=True)
        return {"tickers": {}, "categories": [], "total": 0}


def get_themes_for_ticker(db: DatabaseManager, ticker: str, days: int = 30, top_n: int = 5) -> Dict:
    """
    Find storylines correlated to a specific ticker.

    Searches `key_entities` JSONB array in articles for ticker aliases, then aggregates
    associated storylines by article count and momentum score.

    Args:
        db: DatabaseManager instance
        ticker: Ticker symbol (case-insensitive, e.g. "RTX")
        days: Look back this many days from today
        top_n: Maximum number of storylines to return

    Returns:
        Dict with structure:
        {
            "ticker": "RTX",
            "name": "Raytheon Technologies",
            "themes": [
                {
                    "storyline_id": 123,
                    "title": "Arms Export Controls",
                    "momentum_score": 0.82,
                    "article_count": 12,
                    "community_id": 5
                },
                ...
            ],
            "days": 30,
            "total_themes": 3
        }

    Raises:
        ValueError: If ticker not found in configuration
    """
    # Load config
    config = load_tickers_config()
    ticker_upper = ticker.upper()

    if ticker_upper not in config['tickers']:
        raise ValueError(f"Ticker '{ticker}' not found in configuration")

    ticker_info = config['tickers'][ticker_upper]
    aliases = ticker_info['aliases']

    logger.info(f"Finding themes for {ticker_upper} with aliases: {aliases}")

    # Query: join v_active_storylines + article_storylines + articles
    # Filter by: published_date >= NOW() - INTERVAL 'N days'
    #           AND key_entities JSONB array contains any alias
    # Order by: article count desc, momentum desc
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT s.id, s.title, s.momentum_score, s.community_id,
                           COUNT(DISTINCT als.article_id) as article_count
                    FROM v_active_storylines s
                    JOIN article_storylines als ON als.storyline_id = s.id
                    JOIN articles a ON a.id = als.article_id
                    WHERE a.published_date >= NOW() - INTERVAL '%s days'
                      AND a.key_entities::jsonb ?| %s
                    GROUP BY s.id, s.title, s.momentum_score, s.community_id
                    ORDER BY COUNT(DISTINCT als.article_id) DESC, s.momentum_score DESC
                    LIMIT %s
                """, (days, aliases, top_n))

                rows = cur.fetchall()

                themes = []
                for row in rows:
                    themes.append({
                        'storyline_id': row[0],
                        'title': row[1],
                        'momentum_score': float(row[2]) if row[2] else 0.0,
                        'community_id': row[3],
                        'article_count': row[4]
                    })

                result = {
                    'ticker': ticker_upper,
                    'name': ticker_info['name'],
                    'themes': themes,
                    'days': days,
                    'total_themes': len(themes)
                }

                logger.info(f"Found {len(themes)} themes for {ticker_upper}")
                return result

    except Exception as e:
        logger.error(f"Failed to query themes for ticker {ticker}: {e}", exc_info=True)
        raise


def get_tickers_with_categories() -> Dict:
    """
    Return ticker list organized by category.

    Returns:
    {
        "defense": [
            {"ticker": "RTX", "name": "Raytheon Technologies", "exchange": "NYSE", "aliases": [...], "category": "defense"},
            ...
        ],
        "semiconductors": [...],
        ...
    }
    """
    config = load_tickers_config()

    result = {}
    for ticker_info in config['tickers'].values():
        category = ticker_info['category']
        if category not in result:
            result[category] = []
        result[category].append(ticker_info)

    return result
