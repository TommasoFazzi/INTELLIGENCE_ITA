"""SQLTool — executes LLM-generated SQL with 5-layer safety validation."""

import re
from typing import Any, Dict, List, Optional, Set

import sqlparse
import sqlparse.tokens as T

from .base import BaseTool, ToolResult
from ...utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_TABLES: Set[str] = {
    "articles",
    "chunks",
    "reports",
    "storylines",
    "entities",
    "entity_mentions",
    "trade_signals",
    "macro_indicators",
    "market_data",
    "article_storylines",
    "storyline_edges",
    "v_active_storylines",
    "v_storyline_graph",
    # Knowledge Base expansion (Sprint 3-10)
    "country_profiles",
    "v_sanctions_public",  # PII-sanitized view (migration 034) — use instead of sanctions_registry
    "conflict_events",
    "country_boundaries",
    "strategic_infrastructure",
    "macro_forecasts",
    "trade_flow_indicators",
}

FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "EXEC",
    "COPY", "VACUUM", "ANALYZE", "CLUSTER", "REINDEX",
}

MAX_JOINS = 3
MAX_COST = 10000.0
STATEMENT_TIMEOUT_MS = 5000


class SQLTool(BaseTool):
    name = "sql_query"
    description = (
        "Execute a read-only SQL SELECT query against the intelligence PostgreSQL database. "
        "Use for PATH ANALYTICAL: counting articles, trends, distributions, statistics. "
        "NEVER use rag_search to count things — use sql_query with GROUP BY. "
        "5-layer safety validation: parse → forbidden keywords → max 3 JOINs → LIMIT enforcement → EXPLAIN cost check. "
        "\n\nALLOWED TABLES: articles, chunks, reports, storylines, entities, entity_mentions, "
        "trade_signals, macro_indicators, market_data, article_storylines, storyline_edges, "
        "v_active_storylines, v_storyline_graph, country_profiles, v_sanctions_public, "
        "conflict_events, country_boundaries, strategic_infrastructure, macro_forecasts, trade_flow_indicators. "
        "\n\nKEY SCHEMA:\n"
        "- articles: id, title, source, category, published_date DATE, url, content\n"
        "- storylines: id, title, summary, momentum_score, narrative_status, community_id\n"
        "- trade_signals: id, ticker, signal (BULLISH/BEARISH/NEUTRAL/WATCHLIST), timeframe, rationale, confidence, signal_date\n"
        "- entities: id, name, entity_type, intelligence_score\n"
        "- v_active_storylines: id, title, momentum_score, narrative_status (view)\n"
        "- reports: id, report_date, status, report_type, title\n"
        "- conflict_events: event_date DATE, event_type ('1'=state,'2'=non-state,'3'=one-sided), "
        "country TEXT, location TEXT, actor1, actor2, fatalities INT, geom GEOMETRY(Point,4326)\n"
        "- macro_forecasts: iso3 CHAR(3), indicator_code (NGDP_RPCH=GDP growth, PCPIPCH=inflation, "
        "LUR=unemployment, GGXWDG_NGDP=debt/GDP), year INT, value NUMERIC, vintage TEXT\n"
        "- v_sanctions_public: id TEXT, caption TEXT, schema_type TEXT, aliases TEXT[], "
        "countries CHAR(2)[], datasets TEXT[], first_seen DATE, last_seen DATE\n"
        "- country_profiles: iso3 CHAR(3), iso2 CHAR(2), name TEXT, region TEXT, "
        "population BIGINT, gdp_usd NUMERIC, gdp_growth NUMERIC, inflation NUMERIC, debt_to_gdp NUMERIC\n"
        "- trade_flow_indicators: reporter_iso3, partner_iso3, indicator_code "
        "(EXPORT_VALUE/IMPORT_VALUE/TRADE_BALANCE), year INT, value NUMERIC\n"
        "- country_boundaries: iso3, name, geom GEOMETRY(MultiPolygon,4326), continent\n"
        "\n\nSQL EXAMPLES (follow these patterns):\n"
        "-- Count articles by category:\n"
        "SELECT category, COUNT(*) AS cnt FROM articles "
        "WHERE published_date >= CURRENT_DATE - INTERVAL '30 days' GROUP BY category ORDER BY cnt DESC LIMIT 20;\n\n"
        "-- Conflicts with fatalities (region, recent):\n"
        "SELECT event_date, country, location, actor1, actor2, fatalities, event_type "
        "FROM conflict_events WHERE country ILIKE '%Sudan%' "
        "AND event_date >= CURRENT_DATE - INTERVAL '365 days' ORDER BY event_date DESC LIMIT 50;\n\n"
        "-- IMF GDP forecasts — ALWAYS filter latest vintage with subquery:\n"
        "SELECT mf.year, mf.value, mf.unit FROM macro_forecasts mf "
        "WHERE mf.iso3 = 'DEU' AND mf.indicator_code = 'NGDP_RPCH' "
        "AND mf.vintage = (SELECT MAX(vintage) FROM macro_forecasts WHERE iso3 = 'DEU') "
        "ORDER BY mf.year LIMIT 10;\n\n"
        "-- Sanctions by country (ALWAYS use v_sanctions_public, NEVER sanctions_registry):\n"
        "SELECT caption, schema_type, datasets, first_seen FROM v_sanctions_public "
        "WHERE 'RU' = ANY(countries) ORDER BY last_seen DESC NULLS LAST LIMIT 30;\n\n"
        "-- Trade flows for a country:\n"
        "SELECT tf.year, tf.partner_iso3, cp.name AS partner, tf.value, tf.unit "
        "FROM trade_flow_indicators tf LEFT JOIN country_profiles cp ON tf.partner_iso3 = cp.iso3 "
        "WHERE tf.reporter_iso3 = 'CHN' AND tf.indicator_code = 'EXPORT_VALUE' "
        "ORDER BY tf.year DESC, tf.value DESC NULLS LAST LIMIT 30;\n\n"
        "-- Countries near a point (PostGIS):\n"
        "SELECT cb.iso3, cb.name FROM country_boundaries cb "
        "WHERE ST_DWithin(cb.geom::geography, ST_Point(37.9, 21.5)::geography, 500000) LIMIT 20;\n\n"
        "RULES: PostgreSQL syntax only (NOW() - INTERVAL '7 days'). Max 3 JOINs. "
        "Add LIMIT 50 for non-aggregate queries. Default to last 365 days for conflict_events."
    )
    parameters = {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": (
                    "Think step-by-step: why is sql_query needed here (PATH ANALYTICAL)? "
                    "Which table(s) will you query? What GROUP BY / aggregation will you write? "
                    "What WHERE conditions will capture the user's intent?"
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "PostgreSQL SELECT query. Follow the schema and examples above. "
                    "Always use CURRENT_DATE for date comparisons. "
                    "For sanctions data use v_sanctions_public (NOT sanctions_registry). "
                    "For macro_forecasts always include vintage subquery."
                ),
            },
        },
        "required": ["rationale", "query"],
    }

    def _execute(self, **kwargs) -> ToolResult:
        raw_query: Optional[str] = kwargs.get("query")
        if not raw_query:
            return ToolResult(success=False, data=None, error="No SQL query provided")

        # ── Layer 1: sqlparse parsing ────────────────────────────────────────
        try:
            statements = sqlparse.parse(raw_query.strip())
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"SQL parse error: {e}")

        if len(statements) != 1:
            return ToolResult(
                success=False, data=None,
                error="Only a single SQL statement is allowed"
            )
        parsed = statements[0]

        # ── Layer 2: Forbidden keyword check (token-level, not regex) ────────
        forbidden_found = self._check_forbidden_keywords(parsed)
        if forbidden_found:
            return ToolResult(
                success=False, data=None,
                error=f"Forbidden SQL keyword(s): {', '.join(forbidden_found)}"
            )

        # Ensure it's a SELECT statement
        stmt_type = parsed.get_type()
        if stmt_type != "SELECT":
            return ToolResult(
                success=False, data=None,
                error=f"Only SELECT statements are allowed (got: {stmt_type})"
            )

        # ── Layer 3: Max JOIN complexity ──────────────────────────────────────
        join_count = self._count_joins(parsed)
        if join_count > MAX_JOINS:
            return ToolResult(
                success=False, data=None,
                error=f"Too many JOINs: {join_count} (max {MAX_JOINS})"
            )

        # ── Layer 4: LIMIT enforcement ────────────────────────────────────────
        safe_query = self._enforce_limit(raw_query.strip(), parsed)

        # ── Layer 5: EXPLAIN cost pre-check + execution ───────────────────────
        return self._run_with_safety(safe_query)

    # ── Helper methods ────────────────────────────────────────────────────────

    def _check_forbidden_keywords(self, parsed) -> List[str]:
        found = []
        flat = list(parsed.flatten())
        for token in flat:
            if token.ttype in (T.Keyword, T.Keyword.DML, T.Keyword.DDL):
                upper = token.normalized.upper()
                if upper in FORBIDDEN_KEYWORDS:
                    found.append(upper)
        return found

    def _count_joins(self, parsed) -> int:
        count = 0
        for token in parsed.flatten():
            if token.ttype is T.Keyword and token.normalized.upper() == "JOIN":
                count += 1
        return count

    def _enforce_limit(self, query: str, parsed) -> str:
        has_limit = any(
            t.ttype is T.Keyword and t.normalized.upper() == "LIMIT"
            for t in parsed.flatten()
        )
        if not has_limit:
            # Strip trailing semicolon before appending LIMIT
            q = query.rstrip().rstrip(";")
            return f"{q} LIMIT 1000"
        return query

    def _run_with_safety(self, query: str) -> ToolResult:
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Set statement timeout (read-only enforcement)
                    cur.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT_MS}'")
                    conn.commit()

                    # EXPLAIN cost check
                    cur.execute(f"EXPLAIN {query}")
                    explain_rows = cur.fetchall()
                    explain_output = "\n".join(str(r[0]) for r in explain_rows)
                    cost_match = re.search(r"cost=\d+\.\d+\.\.(\d+\.\d+)", explain_output)
                    if cost_match and float(cost_match.group(1)) > MAX_COST:
                        conn.rollback()
                        return ToolResult(
                            success=False, data=None,
                            error=f"Query too complex (estimated cost {float(cost_match.group(1)):.0f} > {MAX_COST:.0f})"
                        )

                    # Execute
                    cur.execute(query)
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                    conn.rollback()  # ensure read-only — no side-effects

            data = {"results": [dict(zip(columns, row)) for row in rows], "columns": columns}
            return ToolResult(
                success=True, data=data,
                metadata={"rows_returned": len(rows), "columns": columns}
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def _format_success(self, data: Any, metadata: Dict) -> str:
        rows = data.get("results", [])
        columns = data.get("columns", [])
        if not rows:
            return "[SQL: query returned 0 rows]"

        header = " | ".join(columns)
        sep = "-" * len(header)
        lines = [header, sep]
        for row in rows[:20]:
            lines.append(" | ".join(str(row.get(c, "")) for c in columns))

        suffix = f"\n[...{len(rows) - 20} more rows]" if len(rows) > 20 else ""
        return "\n".join(lines) + suffix
