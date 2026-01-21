"""
Constants for Financial Intelligence v2.

Defines region mappings, sector benchmarks, and scoring thresholds.
"""

from typing import Dict

# ============================================================================
# Ticker Region Detection
# ============================================================================
# Maps ticker suffix to market region for appropriate benchmark selection.
# Tickers without suffix are assumed to be US.

TICKER_REGION_MAP: Dict[str, str] = {
    # European exchanges
    ".L": "EU",       # London Stock Exchange
    ".DE": "EU",      # XETRA Germany
    ".PA": "EU",      # Euronext Paris
    ".AS": "EU",      # Euronext Amsterdam
    ".MI": "EU",      # Borsa Italiana
    ".SW": "EU",      # SIX Swiss Exchange
    ".BR": "EU",      # Euronext Brussels
    ".VI": "EU",      # Vienna Stock Exchange
    ".OL": "EU",      # Oslo Stock Exchange
    ".HE": "EU",      # Helsinki Stock Exchange
    ".CO": "EU",      # Copenhagen Stock Exchange
    ".ST": "EU",      # Stockholm Stock Exchange

    # Asian exchanges
    ".T": "ASIA",     # Tokyo Stock Exchange
    ".HK": "ASIA",    # Hong Kong Stock Exchange
    ".KS": "ASIA",    # Korea Stock Exchange
    ".TW": "ASIA",    # Taiwan Stock Exchange
    ".SI": "ASIA",    # Singapore Stock Exchange
    ".SS": "ASIA",    # Shanghai Stock Exchange
    ".SZ": "ASIA",    # Shenzhen Stock Exchange

    # Other markets
    ".SR": "OTHER",   # Saudi Tadawul
    ".AX": "OTHER",   # Australian Stock Exchange
    ".TO": "OTHER",   # Toronto Stock Exchange
    ".MX": "OTHER",   # Mexican Stock Exchange
    ".SA": "OTHER",   # Sao Paulo Stock Exchange
}


def get_region(ticker: str) -> str:
    """
    Determine region from ticker suffix.

    Examples:
        'LMT' -> 'US'
        'RHM.DE' -> 'EU'
        '005930.KS' -> 'ASIA'
        'ARAMCO.SR' -> 'OTHER'

    Args:
        ticker: Stock ticker symbol

    Returns:
        Region code: 'US', 'EU', 'ASIA', or 'OTHER'
    """
    for suffix, region in TICKER_REGION_MAP.items():
        if ticker.upper().endswith(suffix):
            return region
    return "US"  # Default to US for unsuffixed tickers


# ============================================================================
# Sector Benchmark ETFs
# ============================================================================
# Maps Region -> Sector -> Benchmark ETF for PE comparison.
# Falls back to regional index if sector not found.

SECTOR_BENCHMARK_MAP: Dict[str, Dict[str, str]] = {
    "US": {
        # SPDR Select Sector ETFs
        "Technology": "XLK",
        "Information Technology": "XLK",
        "Industrials": "XLI",
        "Energy": "XLE",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Health Care": "XLV",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication Services": "XLC",
        # Special sectors
        "Aerospace & Defense": "ITA",
        "Defense": "ITA",
        "Semiconductors": "SOXX",
        "Cybersecurity": "CIBR",
        # Fallback
        "DEFAULT": "SPY",
    },
    "EU": {
        # iShares STOXX Europe 600 Sector ETFs
        "Technology": "EXV1.DE",
        "Information Technology": "EXV1.DE",
        "Industrials": "EXH1.DE",
        "Energy": "EXH2.DE",
        "Financials": "EXV2.DE",
        "Healthcare": "EXV4.DE",
        "Defense": "EXV1.DE",  # No specific defense ETF, use industrials proxy
        "Aerospace & Defense": "EXV1.DE",
        # Fallback
        "DEFAULT": "EXS1.DE",  # STOXX 600
    },
    "ASIA": {
        # Limited sector ETFs for Asia, use regional index
        "Technology": "AAXJ",
        "Semiconductors": "SOXX",  # Use US as proxy
        # Fallback
        "DEFAULT": "AAXJ",  # All Asia ex-Japan
    },
    "OTHER": {
        # Emerging markets proxy
        "DEFAULT": "VWO",
    },
}


def get_sector_benchmark(sector: str, region: str) -> str:
    """
    Get benchmark ETF for a sector in a region.

    Args:
        sector: GICS sector name
        region: Market region ('US', 'EU', 'ASIA', 'OTHER')

    Returns:
        Benchmark ETF ticker
    """
    region_map = SECTOR_BENCHMARK_MAP.get(region, SECTOR_BENCHMARK_MAP["OTHER"])
    return region_map.get(sector, region_map.get("DEFAULT", "SPY"))


# ============================================================================
# Scoring Thresholds
# ============================================================================
# Constants for penalty/bonus calculations.

THRESHOLDS = {
    # SMA200 deviation thresholds (percentage points)
    "SMA_MINOR_DEVIATION": 15.0,      # No penalty below this
    "SMA_MODERATE_DEVIATION": 30.0,   # Non-linear penalty starts
    "SMA_EXTREME_DEVIATION": 50.0,    # Hard cap trigger
    "SMA_MAX_PENALTY": 40,            # Maximum technical penalty points

    # PE valuation thresholds (relative to sector)
    "PE_UNDERVALUED": 0.8,            # PE < 0.8x sector = undervalued
    "PE_OVERVALUED": 1.5,             # PE > 1.5x sector = overvalued
    "PE_BUBBLE": 2.0,                 # PE > 2x sector = bubble

    # Scoring adjustments
    "UNDERVALUED_BONUS": 10,          # Bonus for undervalued
    "OVERVALUED_PENALTY": -10,        # Penalty for overvalued
    "BUBBLE_PENALTY": -20,            # Penalty for bubble PE
    "LOSS_MAKING_PENALTY": -15,       # Penalty for negative PE

    # Hard caps
    "BUBBLE_SCORE_CAP": 50,           # Max score if >50% over SMA
}


# ============================================================================
# Macro Topics
# ============================================================================
# Standard macro themes for signal classification.

MACRO_TOPICS = [
    "AI_BOOM",
    "RATE_HIKE",
    "RATE_CUT",
    "INFLATION",
    "GEOPOLITICS",
    "WAR",
    "SUPPLY_CHAIN",
    "ENERGY_CRISIS",
    "CHINA_RISK",
    "TECH_REGULATION",
    "CYBER_THREAT",
    "DEFENSE_SPENDING",
    "GREEN_TRANSITION",
]
