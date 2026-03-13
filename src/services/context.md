# Services — Context

## Purpose

This module provides reusable business logic services used by both the REST API (`src/api/`) and Oracle tools (`src/llm/tools/`). Services encapsulate domain-specific operations and keep concerns separate from LLM integration, tool implementations, and API routing.

## Architecture Role

Services sit between the API/tools layer and the storage/database layer:

```
API routers / Oracle tools → Services (business logic) → Database / External APIs
```

Each service provides a clean, testable interface for domain operations without tight coupling to HTTP or LLM specifics.

## Key Files

| File | Description |
|------|-------------|
| `report_compare_service.py` | **Report comparison via LLM delta analysis.** Compares two intelligence reports chronologically and generates structured delta identifying new developments, resolved topics, trend shifts, and persistent themes using Gemini 2.5-Flash. Handles LLM-specific concerns: report ordering by date, content truncation (35k chars per report), XML-safe extraction via regex before ElementTree parsing. |
| `ticker_service.py` | **Ticker theme clustering.** Finds storylines correlated to a market ticker symbol. Queries the database for articles mentioning the ticker within a date range, groups by storyline, scores by momentum, and returns top-N results with metadata. Used by `TickerThemesTool` and dashboard ticker analysis. |

## Key Functions

### `report_compare_service.py`

**`compare_reports(report_a: dict, report_b: dict) -> dict`**
- **Input**: Two report dicts with keys: `id`, `report_date`, `report_type`, `status`, `draft_content`, `final_content`
- **Output**: Dict with keys: `new_developments`, `resolved_topics`, `trend_shifts`, `persistent_themes` (all lists of strings)
- **Logic**:
  1. Orders reports chronologically (older = "Precedente", newer = "Recente")
  2. Selects content: `final_content or draft_content`
  3. Truncates to 35000 chars per report (token budget safety)
  4. Calls Gemini 2.5-Flash with structured XML prompt
  5. Extracts `<analysis>` block with regex (handles LLM "chatter")
  6. Parses XML safely, returns 4-section delta dict
- **Timeout**: 90 seconds (two 35k-char reports)
- **Error handling**: Logs and raises `ValueError` on XML parse failures; LLM errors propagate to caller

**`_extract_delta_from_xml(xml_str: str) -> dict`** (private)
- Extracts delta from Gemini XML response
- Regex pattern: `<analysis>.*?</analysis>` with `re.DOTALL` (handles multiline)
- Falls back gracefully if LLM adds text before/after XML block
- Returns dict: `{new_developments: [...], resolved_topics: [...], ...}`

### `ticker_service.py`

**`get_themes_for_ticker(db: DatabaseManager, ticker: str, days: int = 30, top_n: int = 5) -> dict`**
- **Input**: DatabaseManager instance, ticker symbol (case-insensitive), lookback days, limit (1–20)
- **Output**: Dict with keys: `ticker`, `name`, `days`, `total_themes`, `themes` (list of storyline dicts)
- **Logic**:
  1. Validates ticker against `config/top_50_tickers.yaml`
  2. Queries articles mentioning ticker in past N days
  3. Groups by storyline, aggregates momentum and article count
  4. Sorts by momentum DESC, returns top-N
- **Error handling**: Raises `ValueError` if ticker not found in config

## Dependencies

**Internal:**
- `src/storage/database` — `DatabaseManager` for DB access
- `src/utils/logger` — Logging utility
- `config/top_50_tickers.yaml` — Ticker whitelist and metadata

**External:**
- `google-generativeai` — Gemini API (gemini-2.5-flash for report_compare_service)
- `xml.etree.ElementTree` — XML parsing
- `re` — Regex for safe XML extraction
- `datetime` — Date/time operations

## Data Flow

### Report Comparison Flow

```
API endpoint /api/v1/reports/compare
    ↓
routers/reports.py → compare_two_reports()
    ↓
[Fetch two reports from DB]
    ↓
report_compare_service.compare_reports()
    │
    ├─ Order reports by date
    ├─ Truncate content to 35k chars
    ├─ Call Gemini 2.5-Flash with structured prompt
    ├─ Extract <analysis> block with regex
    ├─ Parse XML → delta dict
    │
    ↓
{new_developments, resolved_topics, trend_shifts, persistent_themes}
    ↓
Return as APIResponse to frontend
```

### Ticker Theme Clustering Flow

```
API endpoint /api/v1/stories/{id}/themes (future)
    ↓
Oracle tool: TickerThemesTool._execute()
    ↓
ticker_service.get_themes_for_ticker()
    │
    ├─ Validate ticker in config
    ├─ Query articles mentioning ticker (recent N days)
    ├─ Group by storyline
    ├─ Sort by momentum
    │
    ↓
{ticker, name, days, themes: [{title, momentum_score, article_count}, ...]}
    ↓
Return as tool result
```

## Known Gotchas / Important Notes

**XML Safety in report_compare_service**: The Gemini model sometimes adds explanatory text before/after the `<analysis>` block. The regex extraction `re.search(r'<analysis>.*?</analysis>', response_text, re.DOTALL)` safely handles this without requiring the entire response to be valid XML.

**Content truncation is intentional**: Both reports are truncated to 35000 chars to stay within the Gemini 2.5-Flash token budget (150k input tokens for ~2 reports worth of context). Daily reports average 23k chars, weekly ~13k; the 35k limit covers ~99% of reports.

**Timezone-naive dates**: Reports' `report_date` field is treated as a date without timezone info. Date comparison for "older vs. newer" uses simple string comparison when both are ISO 8601 dates, which is safe for chronological ordering.

**Ticker whitelist enforcement**: `get_themes_for_ticker()` validates against `config/top_50_tickers.yaml` before querying, preventing arbitrary ticker searches that could be slow or expose undefined behavior.

**No caching in services**: These services do no caching. Caching happens at the API layer (SWR on frontend, slowapi on backend). If the same comparison is requested twice, both LLM calls will execute. This ensures freshness but is expensive for report comparisons.
