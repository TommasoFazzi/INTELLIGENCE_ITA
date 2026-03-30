"""
ReferenceTool — structured reference data lookup for Oracle 2.0.

Unlike SQLTool (LLM-generated queries) or SpatialTool (composable PostGIS templates),
this tool uses hardcoded parameterized queries with type-safe inputs.

No LLM-generated SQL, no EXPLAIN cost check needed.

Supported lookups:
    - country_profile: by ISO3 code
    - country_by_name: by country name (ILIKE match)
    - country_by_region: by World Bank region
    - sanctions_search: by name/alias
    - sanctions_by_country: by country ISO2 code
"""

import json
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ReferenceTool(BaseTool):
    """
    Lookup structured reference data (country profiles, sanctions).

    SECURITY: All queries use psycopg2 parameterized queries (%(key)s).
    No user input is interpolated into SQL strings.
    """

    name = "reference_lookup"
    description = "Look up structured reference data: country profiles, sanctions registries"
    parameters = {
        "lookup_type": "string (country_profile|country_by_name|country_by_region|sanctions_search|sanctions_by_country)",
        "query": "string (ISO3 code, country name, or search term)",
    }

    # Pre-approved parameterized queries — no LLM SQL generation
    SAFE_QUERIES = {
        "country_profile": {
            "sql": """
                SELECT iso3, iso2, name, capital, region, income_group,
                       population, gdp_usd, gdp_per_capita, gdp_growth,
                       inflation, unemployment, debt_to_gdp,
                       current_account_pct, governance_score, data_year
                FROM country_profiles WHERE iso3 = %(query)s
            """,
            "description": "Country profile by ISO3 code",
        },
        "country_by_name": {
            "sql": """
                SELECT iso3, iso2, name, capital, region, income_group,
                       population, gdp_usd, gdp_per_capita, gdp_growth,
                       inflation, unemployment, debt_to_gdp,
                       current_account_pct, governance_score, data_year
                FROM country_profiles
                WHERE name ILIKE %(query)s OR name ILIKE %(query_fuzzy)s
                ORDER BY name LIMIT 10
            """,
            "description": "Country profile by name search",
        },
        "country_by_region": {
            "sql": """
                SELECT iso3, name, population, gdp_usd, gdp_per_capita,
                       gdp_growth, inflation, income_group
                FROM country_profiles
                WHERE region ILIKE %(query)s
                ORDER BY gdp_usd DESC NULLS LAST LIMIT 30
            """,
            "description": "Countries by region",
        },
        "sanctions_search": {
            "sql": """
                SELECT id, caption, schema_type, aliases, datasets, first_seen, last_seen
                FROM sanctions_registry
                WHERE caption ILIKE %(query_fuzzy)s OR %(query)s = ANY(aliases)
                ORDER BY last_seen DESC NULLS LAST LIMIT 20
            """,
            "description": "Sanctions entity search",
        },
        "sanctions_by_country": {
            "sql": """
                SELECT id, caption, schema_type, datasets, first_seen, last_seen
                FROM sanctions_registry
                WHERE %(query)s = ANY(countries)
                ORDER BY last_seen DESC NULLS LAST LIMIT 50
            """,
            "description": "Sanctions by country ISO2",
        },
    }

    def _execute(self, lookup_type: str = "country_profile", query: str = "", **kwargs) -> ToolResult:
        """Execute a reference lookup."""
        if lookup_type not in self.SAFE_QUERIES:
            return ToolResult(
                success=False, data=None,
                error=f"Unknown lookup_type: {lookup_type}. Valid: {list(self.SAFE_QUERIES.keys())}"
            )

        if not query or not query.strip():
            return ToolResult(success=False, data=None, error="Empty query")

        query_clean = query.strip().upper() if lookup_type in ("country_profile", "sanctions_by_country") else query.strip()
        query_config = self.SAFE_QUERIES[lookup_type]

        params = {
            "query": query_clean,
            "query_fuzzy": f"%{query_clean}%",
        }

        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '5s'")
                    cur.execute(query_config["sql"], params)
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    conn.rollback()  # Read-only — release locks

            results = [dict(zip(columns, row)) for row in rows]

            return ToolResult(
                success=True,
                data={"results": results, "count": len(results)},
                metadata={
                    "lookup_type": lookup_type,
                    "query": query_clean,
                    "description": query_config["description"],
                },
            )
        except Exception as e:
            logger.error(f"ReferenceTool error: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    def _format_success(self, data: Any, metadata: Dict) -> str:
        """Format reference lookup results for LLM injection."""
        if not data or not data.get("results"):
            return f"[REFERENCE: No results for '{metadata.get('query', '')}']"

        results = data["results"]
        lookup_type = metadata.get("lookup_type", "")

        if lookup_type in ("country_profile", "country_by_name", "country_by_region"):
            return self._format_country_profiles(results)
        elif lookup_type in ("sanctions_search", "sanctions_by_country"):
            return self._format_sanctions(results)
        else:
            return json.dumps(results, indent=2, default=str)

    def _format_country_profiles(self, results: List[Dict]) -> str:
        """Format country profiles as readable text."""
        lines = []
        for r in results:
            parts = [f"## {r.get('name', '?')} ({r.get('iso3', '?')})"]
            if r.get('capital'):
                parts.append(f"Capitale: {r['capital']}")
            if r.get('region'):
                parts.append(f"Regione: {r['region']}")
            if r.get('income_group'):
                parts.append(f"Income: {r['income_group']}")
            if r.get('population'):
                parts.append(f"Popolazione: {r['population']:,}")
            if r.get('gdp_usd'):
                parts.append(f"GDP: ${r['gdp_usd']:,.0f}")
            if r.get('gdp_per_capita'):
                parts.append(f"GDP/capita: ${r['gdp_per_capita']:,.0f}")
            if r.get('gdp_growth') is not None:
                parts.append(f"Crescita GDP: {r['gdp_growth']:.1f}%")
            if r.get('inflation') is not None:
                parts.append(f"Inflazione: {r['inflation']:.1f}%")
            if r.get('unemployment') is not None:
                parts.append(f"Disoccupazione: {r['unemployment']:.1f}%")
            if r.get('debt_to_gdp') is not None:
                parts.append(f"Debito/GDP: {r['debt_to_gdp']:.1f}%")
            if r.get('governance_score') is not None:
                parts.append(f"Governance: {r['governance_score']:.2f}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _format_sanctions(self, results: List[Dict]) -> str:
        """Format sanctions results."""
        lines = [f"Trovate {len(results)} entità sanzionate:"]
        for r in results:
            datasets = ", ".join(r.get('datasets', [])[:3]) if r.get('datasets') else "N/A"
            line = f"- **{r.get('caption', '?')}** ({r.get('schema_type', '?')}) — Datasets: {datasets}"
            if r.get('first_seen'):
                line += f" — Dal {r['first_seen']}"
            lines.append(line)
        return "\n".join(lines)
