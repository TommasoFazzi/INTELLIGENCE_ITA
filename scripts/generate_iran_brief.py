#!/usr/bin/env python3
"""
Iran Weekly Intelligence Brief Generator

Generates a structured 4-pillar intelligence brief on Iran using:
- Filtered articles from DB (keyword + entity matching, last 7 days)
- RAG semantic search with Iran-specific queries (last 90 days)
- Active storylines filtered for Iran entities (v_active_storylines)
- Gemini: 4 section calls + 1 synthesis call + 1 scenarios call
- WeasyPrint for PDF export (optional — falls back gracefully)

Usage:
    python scripts/generate_iran_brief.py                        # Full run
    python scripts/generate_iran_brief.py --days 7               # Explicit window
    python scripts/generate_iran_brief.py --dry-run              # Collect data, skip LLM
    python scripts/generate_iran_brief.py --check-coverage       # Count Iran articles and exit
    python scripts/generate_iran_brief.py --no-db-save           # Skip DB storage
    python scripts/generate_iran_brief.py --output-dir reports/iran
    python scripts/generate_iran_brief.py --model gemini-3-flash-preview
"""

import sys
import os
import json
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2.extras
import google.generativeai as genai

from src.llm.report_generator import ReportGenerator
from src.ingestion.iran_economic_fetcher import IranEconomicFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── IRAN ENTITY TAXONOMY ─────────────────────────────────────────────────────

IRAN_TITLE_KEYWORDS = [
    # Geographic / General
    'iran', 'iranian', 'tehran', 'persian gulf', 'hormuz',
    # Nuclear sites and tech
    'natanz', 'fordow', 'bushehr', 'arak', 'parchin', 'karaj',
    'uranium', 'enrichment', 'centrifuge', 'iaea', 'jcpoa', 'nuclear deal', 'atomic energy',
    'ir-6', 'ir-8', 'breakout', 'safeguards', 'npt',
    # Military / IRGC
    'irgc', 'quds force', 'sepah', 'basij', 'artesh', 'revolutionary guard',
    # Weapons systems
    'shahed', 'mohajer', 'fattah', 'kheibar', 'emad', 'ballistic missile',
    'cruise missile', 'hypersonic',
    # Cyber / APT groups
    'apt33', 'apt34', 'apt35', 'muddywater', 'charming kitten', 'oilrig', 'phosphorus',
    # Political figures
    'khamenei', 'pezeshkian', 'raisi', 'qaani', 'hossein salami', 'mojtaba',
    # Proxies / Axis of Resistance
    'hezbollah', 'houthi', 'ansar allah', "kata'ib hezbollah", 'hashd al-shaabi',
    'fatemiyoun', 'pmf', 'popular mobilization',
    # Sanctions / Economy
    'dark fleet', 'shadow fleet', 'iran sanctions', 'iran oil',
]

IRAN_GPE_ENTITIES = [
    'Iran', 'Tehran', 'Isfahan', 'Qom', 'Bushehr', 'Natanz', 'Arak',
    'Fordow', 'Karaj', 'Parchin', 'Strait of Hormuz', 'Persian Gulf',
]

IRAN_ORG_ENTITIES = [
    'IRGC', 'Islamic Revolutionary Guard Corps', 'Quds Force', 'Basij', 'AEOI',
    'IAEA', 'Hezbollah', 'Ansar Allah', 'Houthis', 'PMF', "Kata'ib Hezbollah",
    'APT33', 'APT34', 'MuddyWater',
]

IRAN_PERSON_ENTITIES = [
    'Ali Khamenei', 'Ebrahim Raisi', 'Masoud Pezeshkian', 'Esmail Qaani',
    'Hossein Salami', 'Rafael Grossi', 'Mojtaba Khamenei',
]

IRAN_RAG_QUERIES = [
    "Iran nuclear enrichment IAEA inspections violations centrifuge Natanz Fordow uranium stockpile",
    "IRGC Quds Force regional operations proxy forces Hezbollah Houthi PMF Iraq Syria Yemen",
    "Iran oil exports China sanctions evasion Dark Fleet tanker SWIFT Rial inflation",
    "APT33 APT34 MuddyWater Charming Kitten Iran cyber operations critical infrastructure espionage",
    "Iran diplomatic negotiations JCPOA nuclear deal US backchannel Oman snapback E3",
]

IRAN_FOCUS_AREAS = [
    "Iran nuclear program IAEA inspections uranium enrichment JCPOA centrifuge Natanz Fordow breakout",
    "IRGC Revolutionary Guard Quds Force proxy Hezbollah Houthi PMF regional military security",
    "Iran oil exports sanctions evasion Dark Fleet China Russia economy Rial inflation",
    "Iran cyber operations APT33 APT34 MuddyWater critical infrastructure espionage",
    "Iran diplomacy Trump nuclear deal negotiations Khamenei Pezeshkian Middle East",
]


# ─── SECTION DEFINITIONS ──────────────────────────────────────────────────────

_SECTION_DEFS: Dict[int, Dict[str, str]] = {
    1: {
        "header": "## 1. ☢️ Nuclear & Regional Security",
        "word_target": "800-1000 words",
        "focus": (
            "- IAEA inspections status, enrichment levels (%), centrifuge installations, "
            "uranium stockpile quantities (kg)\n"
            "- IRGC military posture: missile tests, naval operations (Hormuz Strait), "
            "IRGC Aerospace/Navy exercises\n"
            "- Regional proxy activity: Hezbollah (Lebanon/Syria), Houthi (Yemen/Red Sea), PMF (Iraq)\n"
            "- Air defense deployments, ballistic/cruise missile program developments"
        ),
        "requirements": (
            "- Cite specific enrichment %, uranium stockpile kg, centrifuge IR-type numbers\n"
            "- Identify military units by name (IRGC Navy/Aerospace Force/Quds Force brigades)\n"
            "- Map each proxy operation to Iranian strategic objectives "
            "(deterrence, power projection, economic coercion)\n"
            "- Use confidence levels: [High confidence] / [Medium confidence] / [Unverified/Reported]\n"
            "- **This section must be the most detailed** (target: 40-50% of total brief)\n"
            "- SOURCE FIDELITY: use the source's exact verb — boarding ≠ seizure, talks ≠ agreement\n"
            "- TEMPORAL: specify concrete date ranges from event dates, not vague terms like 'imminent'"
        ),
    },
    2: {
        "header": "## 2. 🛢️ Geoeconomics & Sanctions",
        "word_target": "500-700 words",
        "focus": (
            "- Oil export volumes, \"Dark Fleet\" tanker routes, Malaysia/UAE/China re-flagging schemes\n"
            "- Rial exchange rate, inflation indicators, subsidy cuts, economic stress signals\n"
            "- BRICS/SCO integration, trade deals circumventing SWIFT\n"
            "- Internal economic stress: strikes (bazaar, oil workers, teachers), protests"
        ),
        "requirements": (
            "- Link oil revenue directly to IRGC/proxy funding capability (causal chain)\n"
            "- Name specific vessels or companies in sanctions evasion if publicly confirmed\n"
            "- Connect economic pressure trends to regime stability indicators\n"
            "- Economic figures: always note the reference period (e.g., 'inflation at X% as of [month]')\n"
            "- SOURCE FIDELITY: use the source's exact verb — inspection ≠ seizure\n"
            "- TEMPORAL: specify concrete date ranges from event dates, not vague terms"
        ),
    },
    3: {
        "header": "## 3. 💻 Cyberwarfare & Asymmetric Threats",
        "word_target": "500-700 words",
        "focus": (
            "- APT activity: APT33/Elfin (Oil & Gas), APT34/OilRig (Telecom/Gov), "
            "APT35/Charming Kitten (Espionage), MuddyWater (Gov/Military)\n"
            "- Target sectors: critical infrastructure, ransomware, disinformation, influence operations\n"
            "- IRGC-linked Telegram channels, psychological warfare, social engineering campaigns\n"
            "- Iranian defensive cyber posture against Western/Israeli operations"
        ),
        "requirements": (
            "- Specific malware families or CVEs if confirmed; attribution confidence level\n"
            "- Target organizations (name publicly attributed ones)\n"
            "- Operational intent: espionage vs. pre-positioning for sabotage vs. revenue\n"
            "- SOURCE FIDELITY: attribution is never certain — use 'attributed to', 'linked to', not 'by'\n"
            "- TEMPORAL: specify concrete date ranges from event dates, not vague terms"
        ),
    },
    4: {
        "header": "## 4. 🤝 Diplomatic Monitor",
        "word_target": "400-600 words",
        "focus": (
            "- JCPOA revival/nuclear talks status, US-Iran backchannel (Oman/Qatar mediation)\n"
            "- Iran-Russia: Shahed drone cooperation, military tech transfers, Syria coordination\n"
            "- Iran-China: 25-year pact implementation, energy/infrastructure deals\n"
            "- EU/E3 posture: diplomatic freeze, hostage negotiations, snapback mechanism status"
        ),
        "requirements": (
            "- Track negotiation rounds, name mediators and officials quoted\n"
            "- Assess leverage shifts: what changed diplomatically this week?\n"
            "- Identify pressure points (prisoner swaps, humanitarian access, consular)\n"
            "- SOURCE FIDELITY: 'talks scheduled' ≠ 'agreement reached'; 'signaled' ≠ 'committed'\n"
            "- TEMPORAL: specify concrete date ranges from event dates, not vague terms"
        ),
    },
}

_GUARDRAIL = (
    "\n**GUARDRAIL**: If there are no relevant developments in the last 7 days for this specific "
    "section, write:\n"
    "> \"No significant developments in this reporting period for [domain].\"\n"
    "Then briefly note the most recent historical context from RAG sources only. "
    "Do NOT invent content, do NOT pad with unrelated or stale material. "
    "Stop writing when the section's content is complete — do not fill space.\n"
)

_CITATION_RULES = (
    "**CITATION DISCIPLINE**: Every factual claim → [Article N]. "
    "Historical RAG context → [RAG: Title, Date]. Analytical inference without direct source → [Assessment].\n"
    "**CONFIDENCE**: Tag key claims: [High confidence] = confirmed by ≥2 independent sources; "
    "[Medium confidence] = single credible source, plausible; [Unverified/Reported] = single source, unconfirmed.\n"
    "**EPISTEMIC DISCIPLINE** — MANDATORY:\n"
    "  • VERIFIED FACT (named source, directly stated): cite plainly → [Article N]\n"
    "  • REPORTED/UNVERIFIED (single source, not independently confirmed): "
    "→ 'reportedly', 'according to [Source]', 'is alleged to'\n"
    "  • ASSESSMENT (analytical inference): → 'appears to', 'suggests', 'likely', 'may indicate' → [Assessment]\n"
    "  • FORECAST: ALWAYS conditional → 'could', 'may', 'if X then Y'. "
    "NEVER state future events as certain.\n"
    "**SOURCE FIDELITY** — FEW-SHOT:\n"
    "  ✗ WRONG: 'Danish authorities seized the Nora'\n"
    "  ✓ RIGHT: 'Danish authorities boarded the Nora for inspection' [use the source's exact verb]\n"
    "  ✗ WRONG: 'Iran and the US reached a deal'\n"
    "  ✓ RIGHT: 'Iran and the US are reportedly working toward a framework'\n"
    "  ✗ WRONG: 'The IRGC launched a strike'\n"
    "  ✓ RIGHT: 'The IRGC claimed responsibility for a strike, according to state media [Article N]'\n"
    "  RULE: Use the source's verb. Boarding ≠ seizure. Scheduled ≠ agreed. Claimed ≠ confirmed.\n"
    "**TEMPORAL PRECISION**: Calculate concrete deadlines from event dates, NOT from the report date. "
    "Example: 'ultimatum issued Feb 20 with 10–15 day window → expires approximately March 2–7'. "
    "Do NOT write 'in 10 days' for an event dated during the analysis period.\n"
    "**HISTORICAL DATA**: Economic/statistical figures are time-lagged — always note the reference period. "
    "Write 'Inflation at 48.6% (as of [month/period from source])' — not as a real-time reading.\n"
    "**STYLE**: Analytical, neutral, professional. ZERO moral/narrative language: "
    "'brutal', 'reckless', 'alarming', 'desperate', 'provocative'. "
    "Report events; do not dramatize. Forecast ≠ certainty.\n"
    "**TERMINOLOGY**: IAEA safeguards, Additional Protocol, JCPOA snapback, "
    "SIGINT, HUMINT, kinetic options, nuclear threshold state, breakout timeline, SWU."
)


# ─── PROMPT TEMPLATES ─────────────────────────────────────────────────────────

def build_section_prompt(
    section_num: int,
    shared_context: str,
    previous_sections: str,
    report_date: str,
    start_date: str,
    end_date: str,
    econ_detail: str = "",
) -> str:
    """
    Build a focused prompt for a single section of the Iran brief.

    Args:
        econ_detail: Live economic data block — injected only for section 2 (Geoeconomics).
                     Ignored for all other sections.
    """
    defn = _SECTION_DEFS[section_num]
    prev_block = ""
    if previous_sections.strip():
        prev_block = (
            "\n**SECTIONS ALREADY WRITTEN** (read for context and consistency — "
            "do not repeat their content):\n"
            f"{previous_sections.strip()}\n\n---\n"
        )

    # Economic data block — injected only for Section 2 (Geoeconomics)
    econ_block = ""
    if section_num == 2 and econ_detail.strip():
        econ_block = (
            "\n**LIVE ECONOMIC DATA** "
            "(use these numbers directly — cite as [EconData: Bonbast/TE, date]):\n"
            f"{econ_detail}\n\n"
        )

    return (
        "You are a senior intelligence analyst specializing in Iran and Middle East security "
        "(ISW/CTP/CSIS caliber).\n\n"
        f"**DATE**: {report_date}\n"
        f"**ANALYSIS PERIOD**: {start_date} to {end_date} (7 days)\n"
        f"{shared_context}"
        f"{prev_block}\n"
        f"{econ_block}"
        f"**YOUR TASK**: Write ONLY the following section ({defn['word_target']}):\n\n"
        f"{defn['header']}\n\n"
        f"**Focus Areas**:\n{defn['focus']}\n\n"
        f"**Analysis Requirements**:\n{defn['requirements']}\n\n"
        "End this section with a **\"Cross-Domain Implications\"** paragraph "
        "linking this domain to the other three pillars.\n\n"
        f"{_GUARDRAIL}\n"
        f"{_CITATION_RULES}\n\n"
        f"Output ONLY this section. Begin directly with \"{defn['header']}\"."
    )


def build_synthesis_prompt(
    s1: str,
    s2: str,
    s3: str,
    s4: str,
    report_date: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Build synthesis prompt that generates the report wrapper:
    title block + threat dashboard + executive summary + intelligence gaps.
    Runs AFTER all 4 sections are written.
    """
    def trunc(text: str, chars: int = 2500) -> str:
        return text[:chars] + "\n[...truncated...]" if len(text) > chars else text

    return (
        "You are a senior intelligence analyst. Four analytical sections of a Weekly Iran "
        "Intelligence Brief have been produced by specialized sub-analysts. "
        "Your task: write the OPENING BLOCK and CLOSING section of this report.\n\n"
        "**SECTION SUMMARIES** (for reference — synthesize from these):\n\n"
        f"[SECTION 1 — Nuclear & Regional Security]:\n{trunc(s1)}\n\n"
        f"[SECTION 2 — Geoeconomics & Sanctions]:\n{trunc(s2)}\n\n"
        f"[SECTION 3 — Cyber & Asymmetric Threats]:\n{trunc(s3)}\n\n"
        f"[SECTION 4 — Diplomatic Monitor]:\n{trunc(s4)}\n\n"
        "---\n\n"
        "Generate the following elements in EXACTLY this order. "
        "Begin immediately with \"# Weekly Iran Intelligence Brief\":\n\n"
        "1. **Title block** (metadata header)\n"
        "2. **Threat Assessment Dashboard** (table)\n"
        "3. **Executive Summary** (200-250 words, BLUF)\n"
        "4. **Intelligence Gaps** (3-5 bullets)\n\n"
        "Use this exact format:\n\n"
        "# Weekly Iran Intelligence Brief\n"
        f"**Analysis Period**: {start_date} — {end_date}\n"
        "**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY\n"
        "**Produced by**: MacroIntel Automated Intelligence Platform\n\n"
        "---\n\n"
        "## Threat Assessment Dashboard\n"
        "| Domain | Risk Level | Trend | Key Driver |\n"
        "|--------|-----------|-------|------------|\n"
        "| Nuclear | [Critical/High/Medium/Low] | [↑/→/↓] | [1 sentence max] |\n"
        "| Regional Security | [Critical/High/Medium/Low] | [↑/→/↓] | [1 sentence max] |\n"
        "| Cyber | [Critical/High/Medium/Low] | [↑/→/↓] | [1 sentence max] |\n"
        "| Geoeconomic | [Critical/High/Medium/Low] | [↑/→/↓] | [1 sentence max] |\n"
        "| Diplomatic | [Critical/High/Medium/Low] | [↑/→/↓] | [1 sentence max] |\n\n"
        "---\n\n"
        "## Executive Summary (200-250 words)\n"
        "[BLUF — synthesize the single highest-priority development from each section. "
        "Bold key actors. Use [Article N] style references. Active voice; hedge single-source "
        "or unverified claims with 'reportedly', 'allegedly'. Do not dramatize.]\n\n"
        "---\n\n"
        "## 5. 🔍 Intelligence Gaps\n"
        "[List 3-5 critical intelligence gaps — what CANNOT be assessed from open sources "
        "this week. Examples: IAEA access to undeclared sites, IRGC command changes, "
        "actual proxy operational orders, Khamenei health status. "
        "This section distinguishes professional analysis from news aggregation.]\n\n"
        f"**Today**: {report_date}\n"
        "Output ONLY the 4 elements above. Do not add preamble or commentary."
    )


def build_scenarios_prompt(brief_content: str, report_date: str) -> str:
    return f"""You are a strategic forecasting analyst specializing in Iran at a senior think-tank level (ISW/RAND/IISS caliber).

**TODAY**: {report_date}

**BASIS**: The following intelligence brief was produced this week. Your scenarios MUST be grounded in it — cite specific developments. Do not invent facts not present in the brief.

---
{brief_content[:6000]}
---

**TASK**: Generate 5-7 actionable scenarios across three time horizons. These are NOT speculation — they are plausible pathways based on current trajectory and verifiable historical precedent.

# Scenarios & Hot Spots

---

## Short-Term Scenarios (1-3 Months)

For EACH scenario, use this exact format:

**Scenario**: [Descriptive title]
**Probability**: [Low ~25% / Medium ~50% / High ~70%]
**Trigger**: [Specific catalyst, with date/event if applicable]
**Indicators to Watch**:
- [Concrete, monitorable metric or observable event]
- [Concrete, monitorable metric or observable event]
**Regional Implications**: [Impact on neighboring states, Gulf actors, Israel, Iraq]
**Global Implications**: [Impact on oil markets, US posture, great-power dynamics]

Required scenarios to assess (add others if the brief warrants):
- IAEA Board of Governors action on Iran non-compliance (March 2026 meeting)
- US-Iran military escalation (direct strike or proxy escalation)
- Houthi/Hezbollah operational intensification (Red Sea, Lebanon border)
- Nuclear talks breakthrough or collapse

---

## Medium-Term Scenarios (3-12 Months)

Focus on structural trends. Use same format. Required:
- Iran approaches nuclear threshold (30-day breakout capability)
- Economic collapse threshold (Rial/inflation tipping point, regime stability impact)
- IRGC proxy network consolidation or degradation after regional dynamics
- US maximum pressure policy impact on regime behavior and internal politics

---

## Long-Term Scenarios (1-3 Years)

Focus on strategic breakpoints. Use same format. Required:
- Nuclear weapons declaration + NPT exit
- Regime instability (succession crisis, mass uprising, military intervention)
- Regional war (Iran-Israel direct conflict, Hormuz closure, US military intervention)
- JCPOA 2.0 comprehensive deal and normalization pathway

---

**METHODOLOGY**:
- Ground each scenario in specific intelligence from the brief above
- Use historical precedent: 2019 Aramco attacks, 2020 Soleimani strike, 2015 JCPOA deal, 2022-2024 Iran-Israel shadow war, 2023 Gaza war spillover
- Identify the tipping point: one specific event that makes each scenario inevitable vs. improbable
- Leading indicators must be concrete and observable (IAEA reports, exchange rates, UN votes, satellite imagery, diplomatic statements)

**STYLE**: Analytical, confidence-calibrated probabilities. No fiction, no vague "could happen". Active voice.

Begin directly with "# Scenarios & Hot Spots" — no preamble."""


# ─── MAIN CLASS ───────────────────────────────────────────────────────────────

class IranBriefGenerator:
    """
    Orchestrates the Iran Weekly Intelligence Brief pipeline.
    Wraps ReportGenerator and reuses its DB/NLP/LLM infrastructure.
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        logger.info("Initializing IranBriefGenerator...")
        self.model_name = model_name
        self.rg = ReportGenerator(model_name=model_name)
        logger.info("✓ IranBriefGenerator ready")

    # ── Step 1: Article Collection ─────────────────────────────────────────

    def collect_iran_articles(self, days: int = 7) -> List[Dict]:
        """Fetch recent articles and filter for Iran relevance via keyword + JSONB entity matching."""
        logger.info(f"[STEP 1] Collecting articles from last {days} days...")
        all_articles = self.rg.db.get_recent_articles(days=days)
        logger.info(f"  Total articles in window: {len(all_articles)}")
        iran_articles = self._filter_iran(all_articles)
        logger.info(f"  Iran-relevant after filter: {len(iran_articles)}")
        return iran_articles

    def _filter_iran(self, articles: List[Dict]) -> List[Dict]:
        """
        Post-fetch Iran filter:
        1. Keyword match in title (fast path)
        2. Entity JSONB match (GPE / ORG / PERSON)
        """
        result = []
        kw_lower = [k.lower() for k in IRAN_TITLE_KEYWORDS]
        gpe_lower = {e.lower() for e in IRAN_GPE_ENTITIES}
        org_lower = {e.lower() for e in IRAN_ORG_ENTITIES}
        person_lower = {e.lower() for e in IRAN_PERSON_ENTITIES}

        for a in articles:
            title = (a.get('title') or '').lower()

            # Fast path: keyword in title
            if any(kw in title for kw in kw_lower):
                result.append(a)
                continue

            # Entity JSONB check
            by_type = (a.get('entities') or {}).get('by_type', {})
            gpe = {e.lower() for e in by_type.get('GPE', [])}
            org = {e.lower() for e in by_type.get('ORG', [])}
            person = {e.lower() for e in by_type.get('PERSON', [])}

            if (gpe & gpe_lower) or (org & org_lower) or (person & person_lower):
                result.append(a)

        return result

    # ── Step 2a: RAG Context ───────────────────────────────────────────────

    def get_iran_rag_context(self, top_k_per_query: int = 5) -> List[Dict]:
        """Run 5 Iran-specific RAG queries and return deduplicated chunks."""
        logger.info("[STEP 2a] Fetching RAG context (Iran-specific queries)...")
        all_chunks = []
        for query in IRAN_RAG_QUERIES:
            chunks = self.rg.get_rag_context(query=query, top_k=top_k_per_query)
            all_chunks.extend(chunks)
            logger.info(f"  '{query[:55]}...' → {len(chunks)} chunks")

        deduped = self.rg.deduplicate_chunks_advanced(all_chunks)
        logger.info(f"  RAG total after dedup: {len(deduped)} chunks")
        return deduped

    # ── Step 2b: Iran Storylines ───────────────────────────────────────────

    def get_iran_storylines(self, top_n: int = 10) -> List[Dict]:
        """Query v_active_storylines filtered for Iran-related key_entities."""
        logger.info("[STEP 2b] Fetching Iran-relevant active storylines...")
        storylines = []
        try:
            with self.rg.db.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("""
                    SELECT id, title, summary, narrative_status, momentum_score,
                           article_count, key_entities, start_date, last_update
                    FROM v_active_storylines
                    WHERE EXISTS (
                        SELECT 1 FROM jsonb_array_elements_text(key_entities) e
                        WHERE e = ANY(ARRAY[
                            'Iran','Tehran','IRGC','Quds Force','Hezbollah',
                            'Houthi','IAEA','Natanz','Fordow','Pezeshkian',
                            'Khamenei','Basij','AEOI','PMF','APT33','APT34'
                        ])
                    )
                    ORDER BY momentum_score DESC
                    LIMIT %s
                """, (top_n,))
                rows = cursor.fetchall()
                storylines = [dict(r) for r in rows]
                logger.info(f"  Found {len(storylines)} Iran storylines")
        except Exception as e:
            logger.warning(f"  Could not fetch storylines (non-fatal): {e}")
        return storylines

    def _format_storylines_xml(self, storylines: List[Dict]) -> str:
        """Format Iran storylines as XML block for LLM prompt injection."""
        if not storylines:
            return "<storylines>No active Iran storylines currently tracked.</storylines>"

        parts = ["<storylines>"]
        for s in storylines:
            entities = s.get('key_entities', [])
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                except Exception:
                    entities = []
            summary = (s.get('summary') or 'No summary available')[:500]
            parts.append(
                f'  <storyline id="{s["id"]}" status="{s.get("narrative_status", "active")}" '
                f'momentum="{float(s.get("momentum_score", 0)):.2f}" '
                f'articles="{s.get("article_count", 0)}">\n'
                f'    <title>{s["title"]}</title>\n'
                f'    <summary>{summary}</summary>\n'
                f'    <key_entities>{", ".join(str(e) for e in entities[:10])}</key_entities>\n'
                f'    <since>{s.get("start_date", "")}</since>\n'
                f'    <last_update>{s.get("last_update", "")}</last_update>\n'
                f'  </storyline>'
            )
        parts.append("</storylines>")
        return "\n".join(parts)

    # ── LLM Helper ────────────────────────────────────────────────────────

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 3072,
        timeout: int = 120,
        label: str = "LLM",
    ) -> str:
        """Shared LLM call with standardized logging. Always includes timeout."""
        logger.info(
            f"  [{label}] prompt={len(prompt):,} chars | "
            f"T={temperature} | max_tokens={max_tokens} | timeout={timeout}s"
        )
        response = self.rg.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            request_options={"timeout": timeout},
        )
        result = response.text
        logger.info(f"  [{label}] ✓ {len(result.split()):,} words")
        return result

    # ── Source Index + Citation Linking ───────────────────────────────────

    @staticmethod
    def _build_source_index(
        url_map: Dict[int, Dict],
        rag_chunks: List[Dict],
    ) -> str:
        """
        Build a markdown source appendix for the brief.

        Articles (numbered 1..N matching [Article N] citations) and RAG historical
        context chunks are listed in separate tables. URLs become clickable markdown links;
        sources without a URL are listed as plain text so the reader knows they exist.
        """
        lines = ["", "---", "", "## 📎 Fonti — News (Periodo di Analisi)",
                 "| # | Titolo | Fonte | Data |",
                 "|---|--------|-------|------|"]

        for n, entry in sorted(url_map.items()):
            title  = (entry.get("title") or f"Article {n}")[:90]
            source = entry.get("source", "")
            date   = entry.get("date", "")
            url    = entry.get("url", "")
            title_cell = f"[{title}]({url})" if url else title
            lines.append(f"| {n} | {title_cell} | {source} | {date} |")

        # RAG historical sources
        rag_rows = []
        for chunk in rag_chunks:
            title  = (chunk.get("title") or "Unknown")[:90]
            source = chunk.get("source", "")
            date   = str(chunk.get("published_date", ""))[:10]
            url    = chunk.get("link", "")
            title_cell = f"[{title}]({url})" if url else title
            rag_rows.append(f"| {title_cell} | {source} | {date} |")

        if rag_rows:
            lines += ["", "## 📎 Fonti — Contesto Storico (RAG)",
                      "| Titolo | Fonte | Data |",
                      "|--------|-------|------|"]
            lines.extend(rag_rows)

        return "\n".join(lines)

    @staticmethod
    def _linkify_citations(html_body: str, url_map: Dict[int, Dict]) -> str:
        """
        Replace [Article N] and [Article N, M, ...] in HTML with clickable links.

        Commas separate distinct articles (not a range): [Article 6, 25] → two links.
        Articles without a URL in url_map stay as plain text.
        Each link carries a tooltip (title attribute) for quick preview on hover.
        """
        def replace_match(m: re.Match) -> str:
            nums = re.findall(r'\d+', m.group(0))
            parts = []
            for n_str in nums:
                idx   = int(n_str)
                entry = url_map.get(idx, {})
                url   = entry.get("url", "")
                tip   = (entry.get("title") or "")[:80].replace('"', "'")
                if url:
                    parts.append(
                        f'<a href="{url}" target="_blank" rel="noopener" '
                        f'class="src-link" title="{tip}">{n_str}</a>'
                    )
                else:
                    parts.append(n_str)
            return '[Article ' + ', '.join(parts) + ']'

        return re.sub(r'\[Article [\d,\s]+\]', replace_match, html_body)

    # ── Step 3a: Generate Main Brief ──────────────────────────────────────

    def generate_main_brief(
        self,
        iran_articles: List[Dict],
        rag_context: List[Dict],
        storylines: List[Dict],
        url_map: Optional[Dict] = None,
        econ_data: Optional[Dict] = None,
        days: int = 7,
    ) -> str:
        """
        Generate the 4-pillar Iran brief via 5 sequential LLM calls:
          Call 1 — Nuclear & Regional Security     (full context)
          Call 2 — Geoeconomics & Sanctions        (full context + S1, avoids repetition)
          Call 3 — Cyber & Asymmetric Threats      (full context + S1+S2)
          Call 4 — Diplomatic Monitor              (full context + S1+S2+S3)
          Call 5 — Synthesis                       (header + dashboard + exec summary + intel gaps)

        Final assembly order: synthesis_header | S1 | S2 | S3 | S4 | intel_gaps
        """
        logger.info("[STEP 3a] Generating main brief (5 sequential LLM calls)...")

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)
        report_date = end_dt.strftime('%B %d, %Y')
        start_date = start_dt.strftime('%B %d, %Y')
        end_date = end_dt.strftime('%B %d, %Y')

        # Shared context block — formatted once, reused across all 4 section calls
        recent_formatted = self.rg.format_recent_articles(iran_articles)
        rag_formatted = self.rg.format_rag_context(rag_context)
        narrative_xml = self._format_storylines_xml(storylines)

        shared_context = (
            f"\n**INTELLIGENCE SOURCES** (Last 7 Days):\n{recent_formatted}\n\n"
            f"**HISTORICAL CONTEXT** (RAG — Prior 90 Days):\n{rag_formatted}\n\n"
            f"**ACTIVE STORYLINES** (Narrative Tracking):\n{narrative_xml}\n"
        )

        # ── Economic data enrichment (Stage 2c) ───────────────────────────
        econ_detail = ""
        if econ_data:
            fetcher = IranEconomicFetcher()
            econ_formatted = fetcher.format_for_prompt(econ_data)
            # summary_line → all pillars see a compact economic snapshot
            shared_context += f"\n{econ_formatted['summary_line']}\n"
            # detail_block → passed only to Section 2 (Geoeconomics) call below
            econ_detail = econ_formatted["detail_block"]
            logger.info(f"  Economic context: {len(econ_detail):,} chars injected into S2")

        logger.info(
            f"  Shared context: {len(shared_context):,} chars | "
            f"Articles: {len(iran_articles)} | RAG: {len(rag_context)} chunks | "
            f"Storylines: {len(storylines)}"
        )

        # ── Call 1: Nuclear & Regional Security ───────────────────────────
        logger.info("  [3a-1/5] Section 1: Nuclear & Regional Security")
        s1 = self._call_llm(
            build_section_prompt(1, shared_context, "", report_date, start_date, end_date),
            temperature=0.2, max_tokens=8192, timeout=240, label="S1-Nuclear",
        )

        # ── Call 2: Geoeconomics (+ S1 for consistency + live economic data) ─
        logger.info("  [3a-2/5] Section 2: Geoeconomics & Sanctions")
        s2 = self._call_llm(
            build_section_prompt(
                2, shared_context, s1, report_date, start_date, end_date,
                econ_detail=econ_detail,
            ),
            temperature=0.2, max_tokens=8192, timeout=240, label="S2-Geo",
        )

        # ── Call 3: Cyber (+ S1+S2) ───────────────────────────────────────
        logger.info("  [3a-3/5] Section 3: Cyber & Asymmetric Threats")
        s3 = self._call_llm(
            build_section_prompt(
                3, shared_context, s1 + "\n\n" + s2, report_date, start_date, end_date
            ),
            temperature=0.2, max_tokens=8192, timeout=240, label="S3-Cyber",
        )

        # ── Call 4: Diplomacy (+ S1+S2+S3) ───────────────────────────────
        logger.info("  [3a-4/5] Section 4: Diplomatic Monitor")
        s4 = self._call_llm(
            build_section_prompt(
                4, shared_context,
                s1 + "\n\n" + s2 + "\n\n" + s3,
                report_date, start_date, end_date,
            ),
            temperature=0.2, max_tokens=8192, timeout=240, label="S4-Diplomacy",
        )

        # ── Call 5: Synthesis (header + dashboard + exec summary + gaps) ──
        logger.info("  [3a-5/5] Synthesis: header + threat dashboard + exec summary + intel gaps")
        synthesis = self._call_llm(
            build_synthesis_prompt(s1, s2, s3, s4, report_date, start_date, end_date),
            temperature=0.15, max_tokens=4096, timeout=120, label="S5-Synthesis",
        )

        # ── Assemble: synthesis_header | S1 | S2 | S3 | S4 | intel_gaps ──
        intel_gaps_marker = "## 5. 🔍 Intelligence Gaps"
        if intel_gaps_marker in synthesis:
            synth_parts = synthesis.split(intel_gaps_marker, 1)
            synth_header = synth_parts[0].rstrip()
            intel_gaps_block = intel_gaps_marker + synth_parts[1]
        else:
            synth_header = synthesis.strip()
            intel_gaps_block = ""

        sep = "\n\n---\n\n"
        brief = (
            synth_header
            + sep + s1.strip()
            + sep + s2.strip()
            + sep + s3.strip()
            + sep + s4.strip()
        )
        if intel_gaps_block:
            brief += sep + intel_gaps_block.strip()

        # ── Append source index (links to original URLs) ───────────────────
        if url_map:
            brief += self._build_source_index(url_map, rag_context)

        logger.info(f"  ✓ Main brief assembled: {len(brief.split()):,} words")
        return brief

    # ── Step 3b: Generate Scenarios ───────────────────────────────────────

    def generate_scenarios(self, main_brief: str) -> str:
        """Generate scenario analysis as a separate LLM call (T=0.35 for forecasting)."""
        logger.info("[STEP 3b] Generating scenarios (T=0.35)...")
        return self._call_llm(
            build_scenarios_prompt(main_brief, datetime.now().strftime('%B %d, %Y')),
            temperature=0.35, max_tokens=8192, timeout=240, label="Scenarios",
        )

    # ── Step 4: Validation ────────────────────────────────────────────────

    def validate_brief(self, full_content: str, url_map: Optional[Dict] = None) -> Dict:
        """Citation audit + word count check per section."""
        logger.info("[STEP 4] Validating brief quality...")
        report: Dict[str, Any] = {}

        # Citation count
        citations = re.findall(r'\[Article \d+\]|\[Assessment\]|\[RAG:', full_content)
        report['citation_count'] = len(citations)

        # Word count per section
        sections = re.split(r'^## ', full_content, flags=re.MULTILINE)
        section_wc = {}
        for section in sections:
            if not section.strip():
                continue
            first_line = section.split('\n')[0].strip()[:40]
            wc = len(section.split())
            section_wc[first_line] = wc

        report['section_word_counts'] = section_wc
        report['total_word_count'] = len(full_content.split())

        # Log summary
        logger.info(f"  Total words: {report['total_word_count']}")
        logger.info(f"  Citations:   {report['citation_count']}")
        for s, wc in section_wc.items():
            logger.info(f"  [{s}]: {wc} words")

        # Warnings — word count evaluated on prose only (exclude source index appendix)
        _FONTI_MARKER = "## 📎 Fonti"
        prose_content = full_content.split(_FONTI_MARKER)[0] if _FONTI_MARKER in full_content else full_content
        prose_word_count = len(prose_content.split())
        if report['citation_count'] < 10:
            logger.warning("  ⚠️  Low citation count (<10) — brief may lack source rigor")
        if prose_word_count < 2500:
            logger.warning(f"  ⚠️  Brief short ({prose_word_count} words prose) — target 3000-4200")
        if prose_word_count > 5500:
            logger.warning(f"  ⚠️  Brief long ({prose_word_count} words prose) — consider trimming")

        # ── Language quality checks (prose only — exclude source index) ────
        # 1. Dangling citations: [Article N] with N not in url_map
        if url_map:
            cited_nums = set(
                int(n) for n in re.findall(r'\[Article (\d+)', prose_content)
            )
            missing = cited_nums - set(url_map.keys())
            if missing:
                logger.warning(
                    f"  ⚠️  Dangling citations: [Article {sorted(missing)}] "
                    "— numbers exceed article count, may be hallucinated"
                )
            report['dangling_citations'] = sorted(missing)

        # 2. Banned narrative/moral terms — scanned on prose only to avoid
        #    false positives from article titles/URLs in source index.
        _BANNED_TERMS = [
            "seized", "brutal", "alarming", "desperate",
            "reckless", "provocative", "terrifying",
        ]
        flagged = {t: prose_content.lower().count(t) for t in _BANNED_TERMS if t in prose_content.lower()}
        if flagged:
            for term, count in flagged.items():
                logger.warning(
                    f"  ⚠️  Narrative term '{term}' found {count}× in prose — verify context"
                )
        report['flagged_terms'] = flagged

        return report

    # ── Step 5: Export ────────────────────────────────────────────────────

    def export_html(self, content: str, output_path: str, url_map: Optional[Dict] = None) -> str:
        """Convert markdown to styled HTML report with optional clickable citations."""
        # ── Pre-process: risk level coloring in dashboard table ────────────
        # Done on the markdown string (stable) before HTML conversion.
        for label, css_class in [
            ('| Critical |', '| <span class="risk-critical">Critical</span> |'),
            ('| High |',     '| <span class="risk-high">High</span> |'),
            ('| Medium |',   '| <span class="risk-medium">Medium</span> |'),
            ('| Low |',      '| <span class="risk-low">Low</span> |'),
        ]:
            content = content.replace(label, css_class)

        try:
            import markdown as md_lib
            html_body = md_lib.markdown(
                content,
                extensions=['tables', 'fenced_code', 'nl2br'],
            )
        except ImportError:
            logger.warning("  markdown library not installed — using plain text fallback")
            html_body = f"<pre>{content}</pre>"

        # ── Post-process: domain header color classes ──────────────────────
        # Sections have the format: <h2>N. ☢️ Nuclear &amp; Regional Security</h2>
        # so we match h2 tags that CONTAIN the emoji (not necessarily first char).
        domain_map = {
            '☢️': 'domain-nuclear',
            '🛢️': 'domain-geo',
            '💻': 'domain-cyber',
            '🤝': 'domain-diplo',
            '🔍': 'domain-gaps',
        }
        for emoji, cls in domain_map.items():
            html_body = re.sub(
                r'<h2>([^<]*' + re.escape(emoji) + r')',
                f'<h2 class="{cls}">\\1',
                html_body,
            )

        # ── Post-process: clickable [Article N] citations ──────────────────
        if url_map:
            html_body = self._linkify_citations(html_body, url_map)

        date_str = datetime.now().strftime('%d %B %Y')
        ts_str   = datetime.now().strftime('%Y-%m-%d %H:%M')

        css = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            max-width: 960px; margin: 0 auto; padding: 32px 48px 48px;
            color: #111827; line-height: 1.8; font-size: 14px;
            background: #ffffff;
        }

        /* ── Classification banner ── */
        .classif-banner {
            background: #dc2626; color: #ffffff;
            text-align: center; font-size: 10px; font-weight: 700;
            letter-spacing: 2px; text-transform: uppercase;
            padding: 5px 0; margin: -32px -48px 0;
        }
        .header-bar {
            background: #111827; color: #f9fafb; padding: 12px 24px;
            margin: 0 -48px 32px; font-size: 11px;
            letter-spacing: 1px; text-transform: uppercase; font-weight: 500;
            display: flex; justify-content: space-between; align-items: center;
        }

        /* ── Typography ── */
        h1 {
            font-size: 24px; font-weight: 700; color: #111827;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 12px; margin: 24px 0 6px;
        }
        h2 {
            font-size: 15px; font-weight: 600; color: #111827;
            border-left: 4px solid #6b7280;
            padding-left: 14px; margin: 36px 0 14px;
        }
        /* domain-specific h2 colors */
        h2.domain-nuclear { border-color: #dc2626; }
        h2.domain-geo     { border-color: #0369a1; }
        h2.domain-cyber   { border-color: #7c3aed; }
        h2.domain-diplo   { border-color: #059669; }
        h2.domain-gaps    { border-color: #d97706; }

        h3 { font-size: 13px; font-weight: 600; color: #374151; margin: 20px 0 8px; }
        p  { margin: 10px 0; }
        strong { font-weight: 600; }
        em     { font-style: italic; color: #374151; }

        /* ── Tables ── */
        table {
            border-collapse: collapse; width: 100%; margin: 16px 0;
            font-size: 12.5px;
        }
        th {
            background: #111827; color: #f9fafb; padding: 9px 14px;
            text-align: left; font-weight: 600; font-size: 11px;
            letter-spacing: 0.5px; text-transform: uppercase;
        }
        td { padding: 8px 14px; border: 1px solid #e5e7eb; vertical-align: top; }
        tr:nth-child(even) td { background: #f9fafb; }
        tr:hover td { background: #f0f9ff; }

        /* ── Risk level colors ── */
        .risk-critical { color: #dc2626; font-weight: 700; }
        .risk-high     { color: #ea580c; font-weight: 600; }
        .risk-medium   { color: #ca8a04; font-weight: 500; }
        .risk-low      { color: #16a34a; }

        /* ── Citation badges ── */
        .src-link {
            display: inline-block;
            background: #eff6ff; color: #1d4ed8;
            border: 1px solid #bfdbfe; border-radius: 3px;
            padding: 0 4px; font-size: 11px; font-weight: 500;
            text-decoration: none; line-height: 1.6;
        }
        .src-link:hover { background: #dbeafe; border-color: #93c5fd; }

        /* ── Confidence tags ── */
        blockquote {
            border-left: 3px solid #e5e7eb; margin: 12px 0;
            padding: 8px 16px; background: #f9fafb; color: #374151;
        }
        code {
            background: #f3f4f6; padding: 1px 5px; border-radius: 3px;
            font-family: 'Menlo', 'Consolas', monospace; font-size: 12px;
        }

        ul, ol { margin: 8px 0; padding-left: 24px; }
        li { margin: 5px 0; }
        hr { border: none; border-top: 1px solid #e5e7eb; margin: 28px 0; }

        /* ── Source index table ── */
        .src-index th  { background: #374151; }
        .src-index td  { font-size: 12px; }
        .src-index a   { color: #1d4ed8; text-decoration: underline; }
        .src-index a:hover { color: #1e40af; }

        /* ── Footer ── */
        .footer {
            border-top: 1px solid #e5e7eb; margin-top: 48px; padding-top: 14px;
            font-size: 11px; color: #9ca3af; display: flex;
            justify-content: space-between;
        }

        /* ── Print / PDF ── */
        @media print {
            .classif-banner, .header-bar { display: none; }
            body { padding: 0; font-size: 11px; max-width: 100%; }
            h2 { page-break-before: always; border-left-width: 3px; }
            h2:first-of-type, .domain-gaps { page-break-before: avoid; }
            table { font-size: 11px; }
            .src-link { border: none; padding: 0; color: #1d4ed8; }
            .footer { display: block; }
        }
        """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Iran Intelligence Brief — {date_str}</title>
  <style>{css}</style>
</head>
<body>
  <div class="classif-banner">UNCLASSIFIED // FOR OFFICIAL USE ONLY</div>
  <div class="header-bar">
    <span>MacroIntel &nbsp;·&nbsp; Weekly Iran Intelligence Brief</span>
    <span>{date_str}</span>
  </div>
  {html_body}
  <div class="footer">
    <span>UNCLASSIFIED // FOR OFFICIAL USE ONLY &nbsp;·&nbsp; MacroIntel v2.0</span>
    <span>Generated: {ts_str} UTC &nbsp;·&nbsp; Distribution: Pietro / RID Internal</span>
  </div>
</body>
</html>"""

        Path(output_path).write_text(html, encoding='utf-8')
        logger.info(f"  ✓ HTML: {output_path}")
        return html

    def export_pdf(self, html_content: str, output_path: str) -> bool:
        """
        Convert HTML to PDF via WeasyPrint.
        Falls back gracefully if WeasyPrint is not installed or fails
        (e.g., missing system graphics libs on Linux/CI servers).
        Markdown file is always saved first and is the primary output.
        """
        try:
            from weasyprint import HTML as WP_HTML
            WP_HTML(string=html_content).write_pdf(output_path)
            size_kb = Path(output_path).stat().st_size // 1024
            logger.info(f"  ✓ PDF: {output_path} ({size_kb} KB)")
            return True
        except ImportError:
            logger.warning("  PDF skipped — weasyprint not installed (pip install weasyprint)")
            return False
        except Exception as e:
            logger.warning(f"  PDF skipped — WeasyPrint error (non-fatal): {e}")
            return False

    def save_to_db(self, content: str, metadata: Dict) -> Optional[int]:
        """Store the brief in the reports table as report_type='iran_weekly'."""
        logger.info("[STEP 5c] Saving to database (report_type=iran_weekly)...")
        try:
            report_id = self.rg.db.save_report({
                'report_date': date.today(),
                'model_used': self.model_name,
                'draft_content': content,
                'final_content': None,
                'status': 'draft',
                'report_type': 'iran_weekly',
                'metadata': metadata,
            })
            logger.info(f"  ✓ Saved with ID: {report_id}")
            return report_id
        except Exception as e:
            logger.error(f"  ✗ DB save failed (non-fatal): {e}")
            return None

    # ── Coverage Check ─────────────────────────────────────────────────────

    def check_coverage(self, days: int = 7) -> Dict:
        """Count Iran-relevant articles in DB for given window."""
        articles = self.collect_iran_articles(days=days)
        sources: Dict[str, int] = {}
        for a in articles:
            src = a.get('source', 'Unknown')
            sources[src] = sources.get(src, 0) + 1
        return {
            'days': days,
            'total_iran_articles': len(articles),
            'sources': dict(sorted(sources.items(), key=lambda x: -x[1])),
        }

    # ── Main Run ──────────────────────────────────────────────────────────

    def run(
        self,
        days: int = 7,
        output_dir: str = 'reports',
        dry_run: bool = False,
        save_db: bool = True,
    ) -> Dict:
        """Full 5-stage pipeline: collect → enrich → generate → validate → export."""
        logger.info("=" * 70)
        logger.info("IRAN WEEKLY INTELLIGENCE BRIEF — PIPELINE START")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Window: {days} days")
        logger.info("=" * 70)

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d')

        # ── STAGE 1: Data Collection ───────────────────────────────────────
        iran_articles = self.collect_iran_articles(days=days)
        if len(iran_articles) < 5:
            logger.warning(f"⚠️  Only {len(iran_articles)} Iran articles found.")
            logger.warning("    Consider running ingestion pipeline or adding Iran-specific feeds.")

        # Build URL map: {article_number: {title, url, source, date}}
        # article N in the prompt ↔ iran_articles[N-1] (format_recent_articles numbers from 1)
        url_map: Dict[int, Dict] = {
            i + 1: {
                "title":  (a.get("title") or f"Article {i + 1}"),
                "url":    a.get("link", ""),
                "source": a.get("source", ""),
                "date":   str(a.get("published_date", ""))[:10],
            }
            for i, a in enumerate(iran_articles)
        }

        # ── STAGE 2: Semantic Enrichment ──────────────────────────────────
        rag_context = self.get_iran_rag_context()
        storylines = self.get_iran_storylines()

        # ── STAGE 2c: Live Economic Data ───────────────────────────────────
        logger.info("[STEP 2c] Fetching live economic data (Bonbast + TE + static)...")
        econ_fetcher = IranEconomicFetcher()
        econ_data = econ_fetcher.fetch_all()

        if dry_run:
            logger.info("[DRY RUN] Data collected. Skipping LLM calls.")
            logger.info(f"  Articles: {len(iran_articles)} | RAG: {len(rag_context)} | Storylines: {len(storylines)}")
            econ_preview = econ_fetcher.format_for_prompt(econ_data)
            logger.info(f"  Econ: {econ_preview['summary_line']}")
            return {
                'dry_run': True,
                'articles': len(iran_articles),
                'rag_chunks': len(rag_context),
                'storylines': len(storylines),
                'econ_summary': econ_preview['summary_line'],
            }

        # ── STAGE 3: LLM Synthesis ────────────────────────────────────────
        main_brief = self.generate_main_brief(
            iran_articles, rag_context, storylines,
            url_map=url_map, econ_data=econ_data, days=days,
        )
        scenarios = self.generate_scenarios(main_brief)
        full_content = main_brief + "\n\n---\n\n" + scenarios

        # ── STAGE 4: Validation ───────────────────────────────────────────
        validation = self.validate_brief(full_content, url_map=url_map)

        # ── STAGE 5: Output Generation ────────────────────────────────────
        # Markdown is saved FIRST — it is the primary output and the lifeline
        md_path = str(output_dir_path / f"iran_brief_{date_str}.md")
        html_path = str(output_dir_path / f"iran_brief_{date_str}.html")
        pdf_path = str(output_dir_path / f"iran_brief_{date_str}.pdf")

        Path(md_path).write_text(full_content, encoding='utf-8')
        logger.info(f"  ✓ Markdown (primary): {md_path}")

        html_content = self.export_html(full_content, html_path, url_map=url_map)
        self.export_pdf(html_content, pdf_path)

        metadata = {
            'iran_articles_count': len(iran_articles),
            'rag_chunks_count': len(rag_context),
            'storylines_count': len(storylines),
            'validation': validation,
            'output_files': {'html': html_path, 'pdf': pdf_path, 'md': md_path},
        }

        if save_db:
            self.save_to_db(full_content, metadata)

        logger.info("=" * 70)
        logger.info("✓ IRAN BRIEF COMPLETE")
        logger.info(f"  Markdown : {md_path}")
        logger.info(f"  HTML     : {html_path}")
        logger.info(f"  PDF      : {pdf_path}")
        logger.info(f"  Words    : {validation.get('total_word_count', 'N/A')}")
        logger.info(f"  Citations: {validation.get('citation_count', 'N/A')}")
        logger.info("=" * 70)

        return {
            'success': True,
            'files': {'html': html_path, 'pdf': pdf_path, 'md': md_path},
            'validation': validation,
        }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Weekly Iran Intelligence Brief",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--days', type=int, default=7, help='Article window in days (default: 7)')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.getenv('REPORT_OUTPUT_DIR', 'reports'),
        help='Output directory (default: $REPORT_OUTPUT_DIR or reports)',
    )
    parser.add_argument('--dry-run', action='store_true', help='Run pipeline without LLM calls')
    parser.add_argument('--check-coverage', action='store_true', help='Count Iran articles in DB and exit')
    parser.add_argument('--no-db-save', action='store_true', help='Skip saving to database')
    parser.add_argument(
        '--model', type=str, default='gemini-3-flash-preview',
        help='Gemini model (default: gemini-3-flash-preview)',
    )
    args = parser.parse_args()

    if not os.getenv('GEMINI_API_KEY'):
        logger.error("GEMINI_API_KEY not set. Add to .env or: export GEMINI_API_KEY='...'")
        return 1

    generator = IranBriefGenerator(model_name=args.model)

    if args.check_coverage:
        logger.info("Coverage check mode")
        for window in [7, 30, 90]:
            result = generator.check_coverage(days=window)
            print(f"\nLast {window} days: {result['total_iran_articles']} Iran articles")
            for src, n in result['sources'].items():
                print(f"  {n:3d}  {src}")
        return 0

    result = generator.run(
        days=args.days,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        save_db=not args.no_db_save,
    )

    return 0 if result.get('success') or result.get('dry_run') else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
