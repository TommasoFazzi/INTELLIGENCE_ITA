#!/usr/bin/env python3
"""
Load country profiles from World Bank API v2.

Fetches key macroeconomic and governance indicators for ~217 countries
and upserts into the country_profiles table.

Uses tenacity exponential backoff for API resilience.

Usage:
    python scripts/load_world_bank.py
    python scripts/load_world_bank.py --dry-run
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# World Bank indicator codes → country_profiles columns
WB_INDICATORS = {
    'NY.GDP.MKTP.CD':     'gdp_usd',           # GDP (current US$)
    'NY.GDP.PCAP.CD':     'gdp_per_capita',     # GDP per capita (current US$)
    'NY.GDP.MKTP.KD.ZG':  'gdp_growth',         # GDP growth (annual %)
    'FP.CPI.TOTL.ZG':     'inflation',          # Inflation, consumer prices (annual %)
    'SL.UEM.TOTL.ZS':     'unemployment',        # Unemployment, total (% of labor force)
    'GC.DOD.TOTL.GD.ZS':  'debt_to_gdp',        # Central govt debt (% of GDP)
    'BN.CAB.XOKA.GD.ZS':  'current_account_pct', # Current account balance (% of GDP)
    'IT.NET.USER.ZS':     'internet_access_pct', # Individuals using Internet (%)
    'SP.POP.TOTL':        'population',          # Population, total
}

WB_API_BASE = "https://api.worldbank.org/v2"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )),
)
def fetch_wb_indicator(indicator_code: str, year: int = 2023) -> dict:
    """Fetch a single World Bank indicator for all countries.

    Returns dict mapping ISO3 → value.
    """
    url = f"{WB_API_BASE}/country/all/indicator/{indicator_code}"
    params = {
        "format": "json",
        "per_page": 300,
        "date": f"{year - 2}:{year}",  # Fetch 3-year window for data availability
        "source": 2,                    # World Development Indicators
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if len(data) < 2:
        return {}

    results = {}
    for entry in data[1]:
        iso3 = entry.get('countryiso3code')
        value = entry.get('value')
        if iso3 and value is not None and len(iso3) == 3:
            # Keep most recent year's data
            if iso3 not in results:
                results[iso3] = value
    return results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )),
)
def fetch_country_metadata() -> list:
    """Fetch country metadata (iso3, iso2, name, capital, region, income group)."""
    url = f"{WB_API_BASE}/country/all"
    params = {"format": "json", "per_page": 300}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if len(data) < 2:
        return []

    countries = []
    for entry in data[1]:
        # Filter out aggregates (regions, world, etc.)
        if entry.get('region', {}).get('value') == 'Aggregates':
            continue
        countries.append({
            'iso3': entry.get('id', ''),
            'iso2': entry.get('iso2Code', ''),
            'name': entry.get('name', ''),
            'capital': entry.get('capitalCity', ''),
            'region': entry.get('region', {}).get('value', ''),
            'income_group': entry.get('incomeLevel', {}).get('value', ''),
        })
    return countries


def main():
    parser = argparse.ArgumentParser(description="Load World Bank country profiles")
    parser.add_argument('--dry-run', action='store_true', help='Fetch but do not save')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("WORLD BANK COUNTRY PROFILES LOADER")
    logger.info("=" * 80)

    # Step 1: Fetch country metadata
    logger.info("\n[STEP 1] Fetching country metadata...")
    countries = fetch_country_metadata()
    logger.info(f"  ✓ {len(countries)} countries found")

    if not countries:
        logger.error("No countries returned from API")
        return 1

    # Build profiles dict keyed by iso3
    profiles = {c['iso3']: c for c in countries}

    # Step 2: Fetch each indicator
    logger.info(f"\n[STEP 2] Fetching {len(WB_INDICATORS)} indicators...")
    for indicator_code, column_name in WB_INDICATORS.items():
        logger.info(f"  → {indicator_code} → {column_name}")
        try:
            values = fetch_wb_indicator(indicator_code)
            found = 0
            for iso3, value in values.items():
                if iso3 in profiles:
                    profiles[iso3][column_name] = value
                    found += 1
            logger.info(f"    ✓ {found} countries with data")
        except Exception as e:
            logger.warning(f"    ✗ Failed: {e}")
        time.sleep(0.5)  # Base rate limit between indicators

    # Step 3: Upsert to database
    if args.dry_run:
        logger.info(f"\n[DRY RUN] Would upsert {len(profiles)} country profiles")
        for iso3, p in list(profiles.items())[:5]:
            logger.info(f"  {iso3}: {p.get('name', '?')} — GDP: ${p.get('gdp_usd', 0):,.0f}")
        return 0

    logger.info(f"\n[STEP 3] Upserting {len(profiles)} profiles to database...")
    db = DatabaseManager()

    upsert_sql = """
        INSERT INTO country_profiles (iso3, iso2, name, capital, region, income_group,
            population, gdp_usd, gdp_per_capita, gdp_growth, inflation, unemployment,
            debt_to_gdp, current_account_pct, internet_access_pct, data_year, last_updated)
        VALUES (%(iso3)s, %(iso2)s, %(name)s, %(capital)s, %(region)s, %(income_group)s,
            %(population)s, %(gdp_usd)s, %(gdp_per_capita)s, %(gdp_growth)s, %(inflation)s,
            %(unemployment)s, %(debt_to_gdp)s, %(current_account_pct)s, %(internet_access_pct)s,
            %(data_year)s, NOW())
        ON CONFLICT (iso3) DO UPDATE SET
            iso2 = EXCLUDED.iso2,
            name = EXCLUDED.name,
            capital = EXCLUDED.capital,
            region = EXCLUDED.region,
            income_group = EXCLUDED.income_group,
            population = EXCLUDED.population,
            gdp_usd = EXCLUDED.gdp_usd,
            gdp_per_capita = EXCLUDED.gdp_per_capita,
            gdp_growth = EXCLUDED.gdp_growth,
            inflation = EXCLUDED.inflation,
            unemployment = EXCLUDED.unemployment,
            debt_to_gdp = EXCLUDED.debt_to_gdp,
            current_account_pct = EXCLUDED.current_account_pct,
            internet_access_pct = EXCLUDED.internet_access_pct,
            data_year = EXCLUDED.data_year,
            last_updated = NOW()
    """

    saved = 0
    errors = 0
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            for iso3, profile in profiles.items():
                try:
                    params = {
                        'iso3': iso3,
                        'iso2': profile.get('iso2'),
                        'name': profile.get('name'),
                        'capital': profile.get('capital'),
                        'region': profile.get('region'),
                        'income_group': profile.get('income_group'),
                        'population': profile.get('population'),
                        'gdp_usd': profile.get('gdp_usd'),
                        'gdp_per_capita': profile.get('gdp_per_capita'),
                        'gdp_growth': profile.get('gdp_growth'),
                        'inflation': profile.get('inflation'),
                        'unemployment': profile.get('unemployment'),
                        'debt_to_gdp': profile.get('debt_to_gdp'),
                        'current_account_pct': profile.get('current_account_pct'),
                        'internet_access_pct': profile.get('internet_access_pct'),
                        'data_year': 2023,
                    }
                    cur.execute(upsert_sql, params)
                    saved += 1
                except Exception as e:
                    logger.warning(f"  ✗ {iso3}: {e}")
                    errors += 1
            conn.commit()

    db.close()

    logger.info(f"\n  ✓ Saved: {saved}, Errors: {errors}")
    logger.info("✓ World Bank data loaded successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
