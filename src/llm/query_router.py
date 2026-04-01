"""
query_router — legacy config constants (Oracle 2.0 agentic migration).

The QueryRouter class and all LLM-based routing logic have been removed.
Routing is now handled by the Gemini Function Calling agentic loop in
oracle_orchestrator.py using SOPs encoded in the system prompt.

Constants below are kept for reference and backward-compat with eval scripts.
"""

# Few-shot examples per intent category (documentation / eval reference)
INTENT_EXAMPLES = {
    "factual": [
        "Cosa è successo a Taiwan negli ultimi 7 giorni?",
        "Ultime notizie sulla Cina e Taiwan",
        "Cosa ha dichiarato Biden sull'Ucraina?",
    ],
    "analytical": [
        "Quanti articoli sulla Cina negli ultimi 30 giorni?",
        "Qual è il trend delle notizie sull'energia nel 2024?",
        "Distribuzione degli articoli per categoria nell'ultimo mese",
    ],
    "narrative": [
        "Come si è evoluta la narrativa sulla guerra in Ucraina?",
        "Quali storyline sono collegate alla crisi energetica europea?",
        "Mostrami la rete di connessioni tra le storie sul Medio Oriente",
    ],
    "market": [
        "Quali sono i segnali di trading più forti ora?",
        "Mostrami le opportunità BUY con alto intelligence score",
        "Come si correlano gli indicatori macro con gli eventi in Russia?",
    ],
    "comparative": [
        "Confronta la copertura su Cina vs USA negli ultimi 3 mesi",
        "Come è cambiata la situazione in Iran rispetto a 6 mesi fa?",
        "Differenze tra la narrativa europea e americana sulla NATO",
    ],
    "ticker": [
        "Quali sono i temi principali per RTX?",
        "Mostrami le storyline correlate a NVDA",
        "Temi associati al ticker tecnologico",
    ],
    "overview": [
        "Give me a geopolitical overview of Myanmar",
        "Panorama geopolitico dell'Iran",
        "Quadro generale della situazione in Ucraina",
        "Excursus sulla crisi energetica europea",
        "Situazione geopolitica del Medio Oriente",
    ],
}

# Keywords used for keyword-based complexity/intent heuristics (kept for reference)
ANALYTICAL_KEYWORDS = {
    "quanti", "quante", "conteggio", "trend", "distribuzione",
    "statistiche", "analisi", "report", "percentuale", "media",
    "totale", "aggregazione",
}
COMPLEX_KEYWORDS = {
    "confronta", "vs", "versus", "come si è evoluto", "come è cambiato",
    "rispetto a", "paragona", "differenze", "simile a",
}
TICKER_KEYWORDS = {
    "ticker", "azione", "stock", "rtx", "nvda", "msft", "tsm", "temi",
    "storyline", "associato", "correlato", "correlate", "correlati",
}
OVERVIEW_KEYWORDS = {
    "panorama", "panoramica", "overview", "landscape", "excursus",
    "quadro generale", "situazione generale", "quadro geopolitico",
    "geopolitical overview", "geopolitical landscape", "country profile",
    "scenario complessivo", "storia di", "contesto storico",
    "analisi paese", "country analysis", "comprehensive analysis",
}

# Few-Shot SQL examples per table (migrated into SQLTool.description for agentic loop)
# Kept here for eval script backward-compat and documentation.
SQL_EXAMPLES = {
    "conflict_events": (
        "-- Conflicts in a region with fatalities, ordered most recent first:\n"
        "SELECT event_date, country, location, actor1, actor2, fatalities, event_type\n"
        "FROM conflict_events\n"
        "WHERE country ILIKE '%Sudan%'\n"
        "  AND event_date >= CURRENT_DATE - INTERVAL '365 days'\n"
        "ORDER BY event_date DESC LIMIT 50;"
    ),
    "macro_forecasts": (
        "-- Latest IMF GDP growth forecasts for a country (auto-select latest vintage):\n"
        "SELECT mf.year, mf.value, mf.unit, mf.vintage\n"
        "FROM macro_forecasts mf\n"
        "WHERE mf.iso3 = 'DEU' AND mf.indicator_code = 'NGDP_RPCH'\n"
        "  AND mf.vintage = (SELECT MAX(vintage) FROM macro_forecasts WHERE iso3 = 'DEU')\n"
        "ORDER BY mf.year LIMIT 10;"
    ),
    "v_sanctions_public": (
        "-- Sanctioned entities in a country (ISO2), most recent first:\n"
        "SELECT caption, schema_type, datasets, first_seen\n"
        "FROM v_sanctions_public\n"
        "WHERE 'RU' = ANY(countries)\n"
        "ORDER BY last_seen DESC NULLS LAST LIMIT 30;"
    ),
    "country_profiles": (
        "-- Compare GDP and debt for countries in a region:\n"
        "SELECT name, iso3, gdp_usd, gdp_growth, debt_to_gdp, inflation\n"
        "FROM country_profiles\n"
        "WHERE region = 'Middle East & North Africa'\n"
        "ORDER BY gdp_usd DESC NULLS LAST LIMIT 20;"
    ),
    "trade_flow_indicators": (
        "-- Export flows for a country, most recent year first:\n"
        "SELECT tf.year, tf.partner_iso3, cp.name AS partner, tf.value, tf.unit\n"
        "FROM trade_flow_indicators tf\n"
        "LEFT JOIN country_profiles cp ON tf.partner_iso3 = cp.iso3\n"
        "WHERE tf.reporter_iso3 = 'CHN' AND tf.indicator_code = 'EXPORT_VALUE'\n"
        "ORDER BY tf.year DESC, tf.value DESC NULLS LAST LIMIT 30;"
    ),
    "country_boundaries": (
        "-- Countries whose territory is within 500km of a point (PostGIS):\n"
        "SELECT cb.iso3, cb.name, cb.continent\n"
        "FROM country_boundaries cb\n"
        "WHERE ST_DWithin(cb.geom::geography, ST_Point(37.9, 21.5)::geography, 500000)\n"
        "LIMIT 20;"
    ),
}

# Backward-compat alias used in some eval scripts
_SQL_EXAMPLES = SQL_EXAMPLES
