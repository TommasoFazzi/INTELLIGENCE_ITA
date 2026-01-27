# API Context

## Purpose
FastAPI REST backend providing HTTP endpoints for the Intelligence Map visualization and related services. Serves GeoJSON entities to the Next.js frontend.

## Architecture Role
HTTP interface layer between the database and frontend applications. Exposes entity data with coordinates for map visualization. Designed to run alongside the Streamlit HITL dashboard.

## Key Files

- `main.py` - FastAPI application entry point
  - `app` - FastAPI instance with CORS middleware
  - CORS configured for localhost:3000-3002 (Next.js dev servers)

  **Endpoints:**
  - `GET /` - API root with endpoint listing
  - `GET /api/v1/map/entities` - GeoJSON FeatureCollection of geocoded entities
    - Query params: `limit` (default: 1000)
    - Returns entities with lat/lng coordinates for Mapbox
  - `GET /api/v1/map/entities/{id}` - Single entity detail

  **Pydantic Models:**
  - `EntityProperties` - Entity metadata (id, name, type, mention_count)
  - `EntityFeature` - GeoJSON Feature wrapper
  - `EntityCollection` - GeoJSON FeatureCollection response

- `openbb_backend.py` - OpenBB Workspace backend (port 7779)
  - CORS configured for `pro.openbb.co`, `localhost:1420`
  - `widgets.json` - Widget configuration for OpenBB Workspace

  **Endpoints:**
  - `GET /widgets.json` - Widget configuration for OpenBB Workspace
  - `GET /get_latest_report` - Markdown: latest intelligence report
  - `GET /get_macro_summary` - Markdown: macro economic indicators
  - `GET /get_conviction_board` - Table: active trade signals
  - `GET /get_metric_articles_24h` - Metric: articles processed
  - `GET /get_metric_active_signals` - Metric: active signals count
  - `GET /get_metric_sources_active` - Metric: RSS sources active
  - `GET /api/v1/openbb/chart_overlay` - Chart OHLCV + signal annotations
  - `GET /api/v1/openbb/high_conviction_signals` - High-conviction signals (score >= 70)
  - `GET /api/v1/openbb/intelligence_scores` - **Full scoring breakdown**
    - Query params: `days` (default: 7), `min_score` (default: 0)
    - Returns: ticker, company_name, sector, signal, intelligence_score,
      llm_confidence, price, sma_200, sma_200_deviation_pct, pe_ratio,
      pe_rel_valuation, valuation_rating, data_quality, rationale

## Dependencies

- **Internal**: `src/storage/database`, `src/utils/logger`
- **External**:
  - `fastapi` - Web framework
  - `uvicorn` - ASGI server (run with `uvicorn src.api.main:app`)
  - `pydantic` - Request/response validation

## Data Flow

- **Input**:
  - HTTP requests from Next.js frontend (`intelligence-map/`)
  - Query parameters for filtering

- **Output**:
  - GeoJSON responses for map rendering
  - Entity metadata for dossier display

## Running

```bash
# Development
uvicorn src.api.main:app --reload --port 8000

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
