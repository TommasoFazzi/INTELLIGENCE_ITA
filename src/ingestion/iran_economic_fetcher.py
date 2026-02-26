#!/usr/bin/env python3
"""
Iran Economic Data Fetcher

Scrapes live economic indicators for Iran from:
  - Bonbast.com: parallel exchange rates (USD, EUR, AED, USDT, CNY, RUB, TRY, Gold 18k, NIMA)
  - Trading Economics: macro indicators (inflation, oil production, GDP) via meta-description parsing
  - World Bank Open Data API: fallback for inflation/GDP (free, no auth required)
  - config/iran_static_data.json: static military OOB and geopolitical constants (annual)

Design principles:
  - ALL fetch methods are wrapped in independent try/except blocks.
    A failure in any one never blocks the others. Script never crashes due to this layer.
  - Math safety: _parse_price() always strips commas before float() — no ValueError on "714,000".
  - Regex safety: broad pattern r'([0-9]+\\.?[0-9]*)\\s*(?:percent|%)' captures any phrasing change.
  - Sanity bounds: reject values outside plausible ranges (no garbage injected into prompts).

Usage (standalone test):
    cd INTELLIGENCE_ITA/INTELLIGENCE_ITA
    venv/bin/python3 -c "
    from src.ingestion.iran_economic_fetcher import IranEconomicFetcher
    f = IranEconomicFetcher()
    data = f.fetch_all()
    result = f.format_for_prompt(data)
    print(result['summary_line'])
    print()
    print(result['detail_block'])
    "
"""

import re
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

_BONBAST_URL = "https://www.bonbast.com/"

_TE_URLS: Dict[str, str] = {
    "inflation":      "https://tradingeconomics.com/iran/inflation-cpi",
    "oil_production": "https://tradingeconomics.com/iran/crude-oil-production",
    "gdp_growth":     "https://tradingeconomics.com/iran/gdp-growth-annual",
}

# World Bank Open Data API — free, no key, JSON
_WB_BASE = "https://api.worldbank.org/v2/country/IRN/indicator/{indicator}?format=json&mrv=1"
_WB_INDICATORS: Dict[str, str] = {
    "inflation":  "FP.CPI.TOTL.ZG",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
}

# Currency codes to extract from Bonbast
_CURRENCY_CODES = ["USD", "EUR", "AED", "USDT", "CNY", "RUB", "TRY"]

# Plausible sanity bounds for IRR rates (reject garbage scrape values)
_RATE_BOUNDS: Dict[str, tuple] = {
    "usd":      (300_000,    2_000_000),
    "eur":      (300_000,    2_500_000),
    "aed":      (80_000,       600_000),
    "usdt":     (300_000,    2_000_000),
    "cny":      (40_000,       300_000),
    "rub":      (2_000,         30_000),
    "try":      (8_000,         80_000),
    "gold_18k": (10_000_000, 200_000_000),  # per gram in IRR
    "nima_usd": (100_000,      700_000),
}

# Full browser headers — required for Bonbast anti-bot bypass
_BROWSER_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.google.com/",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

_STATIC_DATA_PATH = Path(__file__).parent.parent.parent / "config" / "iran_static_data.json"


# ── Main Class ─────────────────────────────────────────────────────────────────

class IranEconomicFetcher:
    """
    Fetches and formats live + static economic indicators for the Iran Intelligence Brief.
    All fetch methods are fully independent — any one can fail without affecting others.
    """

    # ── Math Safety Utilities ──────────────────────────────────────────────

    @staticmethod
    def _parse_price(text: Any) -> Optional[float]:
        """
        Parse a price string to float — handles Iranian number formatting.

        MATH SAFETY: Always strips commas before float() to avoid ValueError on "714,000".
        Handles: "714,000" → 714000.0 | "34.7" → 34.7 | None → None | "N/A" → None

        Args:
            text: String, int, or float input.
        Returns:
            Parsed float, or None if unparseable.
        """
        if text is None:
            return None
        if isinstance(text, (int, float)):
            return float(text) if text > 0 else None

        cleaned = str(text).strip()
        cleaned = cleaned.replace(",", "")   # "714,000" → "714000"  [MATH SAFETY]
        cleaned = cleaned.replace(" ", "")
        cleaned = re.sub(r"[^\d.]", "", cleaned)  # strip any trailing % or other chars

        if not cleaned:
            return None
        try:
            value = float(cleaned)
            return value if value > 0 else None
        except ValueError:
            return None

    @staticmethod
    def _sanity_check(value: Optional[float], currency: str) -> Optional[float]:
        """
        Reject values outside plausible bounds to prevent garbage injection into prompts.
        Returns None if out of range, logs a warning.
        """
        if value is None:
            return None
        bounds = _RATE_BOUNDS.get(currency.lower())
        if bounds and not (bounds[0] <= value <= bounds[1]):
            logger.warning(
                f"  [EconData] {currency} = {value:,.0f} is outside "
                f"bounds [{bounds[0]:,} – {bounds[1]:,}] — discarding"
            )
            return None
        return value

    # ── Bonbast.com ────────────────────────────────────────────────────────

    def fetch_bonbast(self) -> Dict[str, Any]:
        """
        Fetch Iran parallel exchange rates from Bonbast.com.

        Uses three strategies in order:
          1. Element-level: look for elements with id=currency_code (e.g. id="usd")
          2. Table rows: scan <tr> rows for cells containing currency codes
          3. Text-based: get_text(separator='|') + regex (most permissive fallback)

        Also extracts:
          - Gold 18k IRR (cross-validator for USD rate reliability)
          - NIMA rate (government official rate — spread signals economic distress)

        Computes derived metrics:
          - USDT premium % = ((USDT - USD) / USD) * 100
          - NIMA spread %  = ((USD_free - NIMA) / NIMA) * 100
        """
        from bs4 import BeautifulSoup

        logger.info("  [Bonbast] Fetching parallel exchange rates...")
        session = requests.Session()
        session.headers.update(_BROWSER_HEADERS)

        resp = session.get(_BONBAST_URL, timeout=12)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        rates: Dict[str, Any] = {}

        def _try_parse_validated(val_str: str, key: str) -> Optional[float]:
            """Parse a string and apply sanity check."""
            return self._sanity_check(self._parse_price(val_str), key)

        # ── Strategy 1: look for elements with id matching currency code ───
        for code in _CURRENCY_CODES:
            el = soup.find(id=code.lower()) or soup.find(id=code.upper())
            if el:
                # Try to get a number from the element or its siblings
                siblings = list(el.find_next_siblings())
                for sib in siblings[:3]:
                    rate = _try_parse_validated(sib.get_text(strip=True), code.lower())
                    if rate:
                        rates[code.lower()] = rate
                        logger.info(f"  [Bonbast/id] {code}/IRR: {rate:,.0f}")
                        break

        # ── Strategy 2: table row scan for currencies not yet found ────────
        missing = [c for c in _CURRENCY_CODES if c.lower() not in rates]
        if missing:
            for row in soup.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                first_cell = cells[0].get_text(strip=True).upper()
                for code in list(missing):
                    if code in first_cell:
                        # Try cells 1, 2, 3 for the rate (buy/mid/sell)
                        for cell in cells[1:4]:
                            rate = _try_parse_validated(cell.get_text(strip=True), code.lower())
                            if rate:
                                rates[code.lower()] = rate
                                logger.info(f"  [Bonbast/tr] {code}/IRR: {rate:,.0f}")
                                missing.remove(code)
                                break

        # ── Strategy 3: text-based fallback for any still missing ──────────
        still_missing = [c for c in _CURRENCY_CODES if c.lower() not in rates]
        if still_missing:
            page_text = soup.get_text(separator="|", strip=True)
            for code in still_missing:
                # Look for: CODE | number | number (buy | sell layout)
                pattern = rf"\b{re.escape(code)}\b[^|]{{0,30}}\|([\d,. ]+)\|([\d,. ]+)"
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    # Try sell rate (group 2) first, then buy rate (group 1)
                    rate = (
                        _try_parse_validated(match.group(2).strip(), code.lower())
                        or _try_parse_validated(match.group(1).strip(), code.lower())
                    )
                    if rate:
                        rates[code.lower()] = rate
                        logger.info(f"  [Bonbast/text] {code}/IRR: {rate:,.0f}")

        # ── Gold 18k ───────────────────────────────────────────────────────
        page_text = page_text if "page_text" in dir() else soup.get_text(separator="|", strip=True)
        gold_patterns = [
            r"18[Kk][^|]{0,30}\|([\d,. ]+)\|([\d,. ]+)",
            r"[Gg]old[^|]{0,30}\|([\d,. ]+)\|([\d,. ]+)",
            r"طلا[^|]{0,30}\|([\d,. ]+)\|([\d,. ]+)",
        ]
        for gp in gold_patterns:
            m = re.search(gp, page_text)
            if m:
                rate = (
                    _try_parse_validated(m.group(2).strip(), "gold_18k")
                    or _try_parse_validated(m.group(1).strip(), "gold_18k")
                )
                if rate:
                    rates["gold_18k"] = rate
                    logger.info(f"  [Bonbast] Gold 18k/IRR: {rate:,.0f}")
                break

        # ── NIMA rate ──────────────────────────────────────────────────────
        nima_match = re.search(
            r"\bNIMA\b[^|]{0,30}\|([\d,. ]+)\|([\d,. ]+)", page_text, re.IGNORECASE
        )
        if nima_match:
            rate = (
                _try_parse_validated(nima_match.group(1).strip(), "nima_usd")
            )
            if rate:
                rates["nima_usd"] = rate
                logger.info(f"  [Bonbast] NIMA USD/IRR: {rate:,.0f}")
        else:
            logger.debug("  [Bonbast] NIMA rate not found on page (may not be listed)")

        # ── Derived metrics ────────────────────────────────────────────────
        usd  = rates.get("usd")
        usdt = rates.get("usdt")
        nima = rates.get("nima_usd")

        if usd and usdt:
            # USDT premium: positive → capital leaving Iran; negative → unusual/rate suppression
            rates["usdt_premium_pct"] = round(((usdt - usd) / usd) * 100, 2)
            logger.info(f"  [Bonbast] USDT premium: {rates['usdt_premium_pct']:+.2f}%")

        if usd and nima:
            # NIMA spread: how much cheaper is the government rate vs. market
            # >30% = HIGH distress; >15% = moderate; <15% = low
            rates["nima_spread_pct"] = round(((usd - nima) / nima) * 100, 1)
            logger.info(f"  [Bonbast] NIMA spread: {rates['nima_spread_pct']:.1f}%")

        fetched_count = sum(1 for k, v in rates.items()
                            if isinstance(v, (int, float))
                            and k not in ("usdt_premium_pct", "nima_spread_pct"))
        rates["as_of"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        rates["source"] = "Bonbast.com"
        logger.info(f"  [Bonbast] ✓ {fetched_count} rates fetched")
        return rates

    # ── Trading Economics + World Bank fallback ────────────────────────────

    def _extract_te_value(self, meta_text: str, indicator: str) -> Optional[float]:
        """
        Extract a numeric value from a Trading Economics meta description string.

        Different indicators use different units in the TE meta tag:
          - inflation, gdp_growth: reported as "34.70 percent" or "2.8%"
          - oil_production:        reported as "3100 BBL/D/1K" (thousands of barrels/day)
                                   -> must divide by 1000 to convert to mbpd

        Returns None if no match found.
        """
        if indicator == "oil_production":
            # TE format: "decreased to 3100 BBL/D/1K in December from 3120..."
            m = re.search(r"(?:to|at|of)\s+([\d,]+)\s*BBL/D/1K", meta_text, re.IGNORECASE)
            if m:
                val = self._parse_price(m.group(1))
                if val:
                    return round(val / 1000.0, 3)  # convert thousands bbl/day → mbpd
            # Fallback: try "million barrels per day" phrasing
            m = re.search(r"([\d.]+)\s*(?:million barrels|mbpd|mbd)", meta_text, re.IGNORECASE)
            if m:
                return self._parse_price(m.group(1))
            # Last resort: any "to/at N" where N > 100 → assume BBL/D/1K
            m = re.search(r"(?:to|at)\s+([\d,]+)", meta_text, re.IGNORECASE)
            if m:
                val = self._parse_price(m.group(1))
                if val and val > 100:
                    return round(val / 1000.0, 3)
        else:
            # inflation, gdp_growth — look for number near "percent" or "%"
            m = re.search(r"([\d]+\.?\d*)\s*(?:percent|%)", meta_text, re.IGNORECASE)
            if m:
                return self._parse_price(m.group(1))
        return None

    def fetch_trading_economics(self) -> Dict[str, Any]:
        """
        Fetch macro indicators via Trading Economics meta-description parsing.

        KEY INSIGHT: TE embeds the current value in <meta name="description"> for SEO.
        This tag is present in raw HTML — no JavaScript rendering required.
        Example: "Inflation Rate in Iran decreased to 34.70 percent in December..."

        REGEX SAFETY: Uses broad pattern r'([0-9]+\\.?[0-9]*)\\s*(?:percent|%)'
        instead of grammar-dependent patterns. Captures any phrasing variation.

        Fallback: World Bank Open Data API (free, no auth, JSON) for inflation/GDP
        if meta tag is absent or regex finds no match.
        Oil production has no WB fallback (WB lacks monthly mbpd data).
        """
        import cloudscraper
        from bs4 import BeautifulSoup

        scraper = cloudscraper.create_scraper()
        macro: Dict[str, Any] = {}

        for indicator, url in _TE_URLS.items():
            try:
                logger.info(f"  [TE] Fetching {indicator}...")
                resp = scraper.get(url, timeout=15)
                resp.raise_for_status()

                soup = BeautifulSoup(resp.text, "lxml")
                meta = soup.find("meta", attrs={"name": "description"})
                meta_text = meta.get("content", "") if meta else ""

                if meta_text:
                    logger.debug(f"  [TE] meta description: {meta_text[:120]}...")

                value = self._extract_te_value(meta_text, indicator)
                if value is not None:
                    macro[indicator] = value
                    logger.info(f"  [TE] {indicator}: {value} (meta-desc)")
                else:
                    logger.warning(
                        f"  [TE] No numeric value found in meta for {indicator} "
                        f"— falling back to World Bank"
                    )
                    raise ValueError("Meta regex found no match")

            except Exception as e:
                logger.warning(f"  [TE→WB] {indicator} TE failed ({type(e).__name__}: {e})")
                wb_value = self._fetch_world_bank(indicator)
                if wb_value is not None:
                    macro[indicator] = wb_value
                    macro[f"{indicator}_source"] = "WorldBank"
                    logger.info(f"  [TE→WB] {indicator}: {wb_value} (World Bank fallback)")
                else:
                    macro[indicator] = None
                    logger.warning(f"  [TE→WB] {indicator}: unavailable from both sources")

        macro["as_of"] = datetime.now().strftime("%Y-%m-%d")
        macro["source"] = "TradingEconomics/WorldBank"
        return macro

    def _fetch_world_bank(self, indicator_name: str) -> Optional[float]:
        """
        Fetch a single indicator from World Bank Open Data API.
        Free, no authentication, returns most recent annual value.
        Lag: ~1 year (latest available), suitable for baseline context.
        """
        wb_code = _WB_INDICATORS.get(indicator_name)
        if not wb_code:
            return None
        try:
            url = _WB_BASE.format(indicator=wb_code)
            resp = requests.get(
                url, timeout=10,
                headers={"User-Agent": "IranBriefBot/1.0 (intelligence-research)"}
            )
            resp.raise_for_status()
            data = resp.json()
            # WB response format: [metadata_dict, [{"value": float, "date": "2024"}, ...]]
            if isinstance(data, list) and len(data) > 1 and data[1]:
                entry = data[1][0]
                value = entry.get("value")
                if value is not None:
                    logger.debug(f"  [WB] {indicator_name}: {value} ({entry.get('date', '?')})")
                    return float(value)
        except Exception as e:
            logger.debug(f"  [WB] {indicator_name} failed: {e}")
        return None

    # ── Static Data ────────────────────────────────────────────────────────

    def load_static_data(self) -> Dict:
        """
        Load military OOB and geopolitical constants from config/iran_static_data.json.
        Returns empty dict if file is missing (non-fatal).
        """
        if not _STATIC_DATA_PATH.exists():
            logger.warning(f"  [Static] {_STATIC_DATA_PATH} not found — skipping")
            return {}
        with open(_STATIC_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        keys = [k for k in data.keys() if not k.startswith("_")]
        logger.info(f"  [Static] Loaded {_STATIC_DATA_PATH.name}: {keys}")
        return data

    # ── Orchestration ──────────────────────────────────────────────────────

    def fetch_all(self) -> Dict[str, Any]:
        """
        Run all three data sources independently.
        Each can fail without affecting the others.

        Returns:
            Dict with keys 'bonbast', 'macro', 'static' — each is a dict or None on failure.
        """
        result: Dict[str, Any] = {}
        sources = [
            ("bonbast", self.fetch_bonbast),
            ("macro",   self.fetch_trading_economics),
            ("static",  self.load_static_data),
        ]
        for name, fetcher in sources:
            try:
                result[name] = fetcher()
                logger.info(f"  [EconData] ✓ {name} complete")
            except Exception as e:
                logger.warning(f"  [EconData] {name} failed (non-fatal, will use N/A): {e}")
                result[name] = None
        return result

    # ── Prompt Formatting ──────────────────────────────────────────────────

    def format_for_prompt(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Format fetched data into two text blocks:

        'summary_line': 1-2 lines → injected into shared_context (all 4 pillars see this)
        'detail_block': full structured table → injected only into Section 2 (Geoeconomics)

        None values are shown as "N/A" with an instruction to extract from article corpus.
        """
        bonbast = data.get("bonbast") or {}
        macro   = data.get("macro")   or {}
        static  = data.get("static")  or {}
        today   = datetime.now().strftime("%Y-%m-%d")

        def _rate(key: str, label: str) -> str:
            v = bonbast.get(key)
            return f"{label} {v:,.0f}" if v else f"{label} N/A"

        def _pct(key: str, label: str) -> str:
            v = bonbast.get(key)
            return f"{label} {v:+.2f}%" if v is not None else f"{label} N/A"

        def _macro(key: str, label: str, unit: str = "") -> str:
            v = macro.get(key)
            src = macro.get(f"{key}_source", "TE")
            return f"{label} {v}{unit} [{src}]" if v is not None else f"{label} N/A"

        # ── summary_line (shared_context — all pillars) ────────────────────
        usd_str    = _rate("usd", "USD/IRR:")
        usdt_prem  = bonbast.get("usdt_premium_pct")
        nima_sprd  = bonbast.get("nima_spread_pct")
        inf_str    = _macro("inflation", "Inflation:", "%")
        oil_str    = _macro("oil_production", "Oil:", " mbpd")

        prem_str = ""
        if usdt_prem is not None:
            prem_label = "HIGH capital flight" if usdt_prem > 2 else "low pressure"
            prem_str = f"USDT premium {usdt_prem:+.2f}% ({prem_label})"

        nima_str = ""
        if nima_sprd is not None:
            nima_label = "HIGH distress" if nima_sprd > 30 else ("MODERATE" if nima_sprd > 15 else "LOW")
            nima_str = f"NIMA spread {nima_sprd:.1f}% ({nima_label})"

        summary_parts = [usd_str, prem_str, inf_str, oil_str, nima_str]
        summary_line = (
            f"**Economic snapshot** [{today}]: "
            + " | ".join(p for p in summary_parts if p and "N/A" not in p)
        )
        if "N/A" not in summary_line and len(summary_line) < 50:
            summary_line = f"**Economic snapshot** [{today}]: Live data unavailable — see article corpus."

        # ── detail_block (Section 2 only — full structured context) ───────
        lines = []

        # Free market rates
        lines.append(f"[FREE MARKET RATES — Bonbast.com, {today}]")
        lines.append(
            f"{_rate('usd', 'USD/IRR:')} | {_rate('eur', 'EUR/IRR:')} | "
            f"{_rate('aed', 'AED/IRR:')} (Dubai hub — proxy for import cost pressure)"
        )
        usdt_line = _rate("usdt", "USDT/IRR:")
        if prem_str:
            usdt_line += f" ({prem_str})"
        lines.append(
            f"{usdt_line} | {_rate('cny', 'CNY/IRR:')} | "
            f"{_rate('rub', 'RUB/IRR:')} (Russia-Iran trade proxy) | {_rate('try', 'TRY/IRR:')}"
        )

        # Gold cross-validation
        gold = bonbast.get("gold_18k")
        if gold:
            lines.append("")
            lines.append("[INFLATION PROXY & RATE VALIDATION]")
            lines.append(
                f"Gold 18k/IRR: {gold:,.0f} — Cross-validate against USD rate: "
                "if gold rises while USD/IRR stays flat, treat USD rate as lagged or suppressed."
            )

        # NIMA rate
        if nima_sprd is not None:
            nima_val = bonbast.get("nima_usd")
            lines.append("")
            lines.append("[GOVERNMENT NIMA RATE]")
            nima_val_str = f"{nima_val:,.0f}" if isinstance(nima_val, float) else "N/A"
            severity = "HIGH" if nima_sprd > 30 else ("MODERATE" if nima_sprd > 15 else "LOW")
            lines.append(
                f"NIMA USD/IRR: {nima_val_str} | Free market spread: {nima_sprd:.1f}% → {severity}"
            )
            lines.append(
                "  (Spread >30% historically correlates with black market expansion and protest risk)"
            )

        # Macro indicators
        lines.append("")
        lines.append("[MACRO INDICATORS — TradingEconomics / World Bank fallback]")
        lines.append(
            f"Inflation CPI: {macro.get('inflation', 'N/A')}% | "
            f"Oil production: {macro.get('oil_production', 'N/A')} mbpd | "
            f"GDP growth: {macro.get('gdp_growth', 'N/A')}%"
        )

        # Military baseline (static)
        mil = static.get("military_baseline") or {} if static else {}
        geo = static.get("geopolitical_constants") or {} if static else {}

        if mil or geo:
            lines.append("")
            lines.append("[MILITARY BASELINE — IISS 2025, static annual data]")
        if mil:
            active = mil.get("active_personnel")
            active_str = f"{active:,}" if isinstance(active, int) else str(active or "N/A")
            irgc = mil.get("irgc_personnel")
            irgc_str = f"{irgc:,}" if isinstance(irgc, int) else str(irgc or "N/A")
            lines.append(
                f"Active personnel: {active_str} — {mil.get('qualitative_note', '')}"
            )
            lines.append(
                f"IRGC: {irgc_str} — {mil.get('irgc_qualitative_note', '')}"
            )
            af = mil.get("artesh_air_force") or {}
            if af:
                lines.append(
                    f"Air force: {af.get('fighters', 'N/A')} fighters — {af.get('qualitative_note', '')}"
                )
            missiles = mil.get("irgc_aerospace_force") or {}
            if missiles:
                lines.append(
                    f"Ballistic missiles: ~{missiles.get('ballistic_missiles_estimated', 'N/A')} — "
                    f"{missiles.get('qualitative_note', '')}"
                )
        if geo:
            hormuz = geo.get("hormuz_strait_transit_daily_mbpd")
            if hormuz:
                lines.append(
                    f"Hormuz transit: {hormuz} mbpd daily — {geo.get('hormuz_note', '')}"
                )

        # Citation and missing-data instructions
        lines.append("")
        lines.append(
            f"CITATION: Cite scraped live numbers as [EconData: Bonbast/TE, {today}]. "
            "Static military data as [EconData: IISS 2025]."
        )
        lines.append(
            "MISSING DATA: If any value shows N/A, search the article corpus for quantitative "
            "mentions: 'exports increased', 'Rial hit record', 'insurance premiums', "
            "'war risk surcharge', 'oil revenue'. Extract and cite as [Unverified/Reported]."
        )

        detail_block = "\n".join(lines)

        # Total failure fallback
        if not bonbast and not macro:
            summary_line = (
                f"**Economic snapshot** [{today}]: Live data unavailable this cycle. "
                "Extract economic figures from article corpus."
            )
            detail_block = (
                f"[ECONOMIC DATA — Scraping unavailable {today}]\n"
                "All live sources failed. Extract all economic figures from the article corpus.\n"
                "Mark extracted estimates as [Unverified/Reported].\n"
                f"\n{detail_block}"  # still include static military data if available
            )

        return {"summary_line": summary_line, "detail_block": detail_block}
