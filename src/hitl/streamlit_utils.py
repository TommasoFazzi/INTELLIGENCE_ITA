"""
Streamlit Utilities for INTELLIGENCE_ITA Multi-Page App

Shared utilities including:
- Database singleton with caching
- Ticker detection from whitelist
- Common CSS styling
- Path setup helpers
"""

import sys
from pathlib import Path
from typing import List, Set

import streamlit as st
import yaml

# Setup project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATABASE SINGLETON (Critical for MPA)
# =============================================================================

@st.cache_resource
def get_db_manager() -> DatabaseManager:
    """
    Get singleton DatabaseManager instance.

    Uses @st.cache_resource to share connection across all pages and sessions.
    This prevents connection pool exhaustion.
    """
    logger.info("Initializing DatabaseManager singleton")
    return DatabaseManager()


@st.cache_resource
def get_embedding_model():
    """
    Get singleton SentenceTransformer instance for embedding generation.

    Uses @st.cache_resource to avoid reloading the model on each query.
    First call is slow (~3-5 sec), subsequent calls are instant.

    Returns:
        SentenceTransformer model instance (paraphrase-multilingual-MiniLM-L12-v2)
    """
    from sentence_transformers import SentenceTransformer
    logger.info("Loading SentenceTransformer model (paraphrase-multilingual-MiniLM-L12-v2)")
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# =============================================================================
# TICKER DETECTION (Whitelist-based)
# =============================================================================

@st.cache_data
def load_ticker_whitelist() -> Set[str]:
    """
    Load ticker whitelist from config/top_50_tickers.yaml.

    Extracts all tickers and their aliases for accurate detection
    without false positives.
    """
    config_path = PROJECT_ROOT / 'config' / 'top_50_tickers.yaml'

    if not config_path.exists():
        logger.warning(f"Ticker whitelist not found: {config_path}")
        return set()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        tickers = set()
        for category in data.values():
            if isinstance(category, list):
                for company in category:
                    if isinstance(company, dict):
                        ticker = company.get('ticker', '')
                        if ticker:
                            tickers.add(ticker.upper())
                        # Add aliases
                        aliases = company.get('aliases', [])
                        for alias in aliases:
                            if alias:
                                tickers.add(alias.upper())

        logger.info(f"Loaded {len(tickers)} tickers from whitelist")
        return tickers

    except Exception as e:
        logger.error(f"Failed to load ticker whitelist: {e}")
        return set()


def extract_tickers(text: str) -> List[str]:
    """
    Extract tickers mentioned in text using whitelist validation.

    Only returns tickers that exist in the whitelist to avoid
    false positives from common English words.

    Args:
        text: Text to search for tickers

    Returns:
        Sorted list of unique tickers found
    """
    if not text:
        return []

    whitelist = load_ticker_whitelist()
    if not whitelist:
        return []

    text_upper = text.upper()
    found = [t for t in whitelist if t in text_upper]
    return sorted(list(set(found)))


# =============================================================================
# CSS STYLING
# =============================================================================

CUSTOM_CSS = """
<style>
    /* Metric cards */
    .big-metric {
        font-size: 28px !important;
        font-weight: bold;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Status badges */
    .status-badge {
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
    }

    .status-draft {
        background-color: #ffeaa7;
        color: #2d3436;
    }

    .status-reviewed {
        background-color: #81ecec;
        color: #2d3436;
    }

    .status-approved {
        background-color: #55a630;
        color: white;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* War Room specific */
    .war-room-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
"""


def inject_custom_css():
    """Inject custom CSS into the Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# STATUS BADGE HELPER
# =============================================================================

def get_status_badge(status: str) -> str:
    """
    Generate HTML badge for report status.

    Args:
        status: One of 'draft', 'reviewed', 'approved'

    Returns:
        HTML string for the badge
    """
    status_lower = status.lower() if status else 'draft'

    labels = {
        'draft': 'BOZZA',
        'reviewed': 'REVISIONATO',
        'approved': 'APPROVATO'
    }

    label = labels.get(status_lower, status_lower.upper())
    css_class = f"status-{status_lower}"

    return f'<span class="status-badge {css_class}">{label}</span>'


def get_status_emoji(status: str) -> str:
    """Get emoji for report status."""
    status_lower = status.lower() if status else 'draft'
    emojis = {
        'draft': '📝',
        'reviewed': '👀',
        'approved': '✅'
    }
    return emojis.get(status_lower, '📄')


# =============================================================================
# SESSION STATE HELPERS
# =============================================================================

def init_session_state():
    """Initialize common session state variables."""
    if 'current_report_id' not in st.session_state:
        st.session_state.current_report_id = None
    if 'show_success_message' not in st.session_state:
        st.session_state.show_success_message = False


# =============================================================================
# FRESHNESS INDICATORS (for The Oracle)
# =============================================================================

def get_freshness_badge(doc_date) -> str:
    """
    Return colored badge based on document age.

    Args:
        doc_date: Document date (datetime or date object)

    Returns:
        Emoji badge indicating freshness:
        - Green: < 7 days (Fresh Intelligence)
        - Yellow: < 30 days (Recent Intelligence)
        - Red: > 30 days (Old Intelligence)
        - White: Unknown date
    """
    from datetime import datetime, date

    if not doc_date:
        return "⚪"  # Unknown

    # Handle both datetime and date objects
    if hasattr(doc_date, 'date'):
        doc_date = doc_date.date()
    elif isinstance(doc_date, str):
        try:
            doc_date = datetime.fromisoformat(doc_date.replace('Z', '+00:00')).date()
        except:
            return "⚪"

    today = date.today()
    age = (today - doc_date).days

    if age <= 7:
        return "🟢"  # Fresh: < 7 days
    elif age <= 30:
        return "🟡"  # Recent: < 30 days
    else:
        return "🔴"  # Old: > 30 days


def get_freshness_label(doc_date) -> str:
    """
    Return human-readable freshness label.

    Args:
        doc_date: Document date

    Returns:
        Freshness description string
    """
    from datetime import datetime, date

    if not doc_date:
        return "Data sconosciuta"

    if hasattr(doc_date, 'date'):
        doc_date = doc_date.date()
    elif isinstance(doc_date, str):
        try:
            doc_date = datetime.fromisoformat(doc_date.replace('Z', '+00:00')).date()
        except:
            return "Data sconosciuta"

    today = date.today()
    age = (today - doc_date).days

    if age == 0:
        return "Oggi"
    elif age == 1:
        return "Ieri"
    elif age <= 7:
        return f"{age} giorni fa"
    elif age <= 30:
        return f"{age // 7} settimane fa"
    else:
        return f"{age // 30} mesi fa"
