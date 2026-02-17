# API Context

## Purpose
FastAPI REST backend providing HTTP endpoints for the Intelligence ITA platform: dashboard stats, report management, entity visualization, and **narrative storyline graph**. Serves data to the Next.js frontend.

## Architecture Role
HTTP interface layer between the database and frontend applications. Modular router architecture with Pydantic schema validation and API key authentication.

## Key Files

### Core
- `main.py` - FastAPI application entry point
  - `app` - FastAPI instance with CORS middleware (configurable via `ALLOWED_ORIGINS` env var)
  - Rate limiting via `slowapi` (30 req/min on map entities, configurable per endpoint)
  - Registers 3 routers: dashboard, reports, stories
  - Direct endpoints for map entities (`/api/v1/map/entities`)

- `auth.py` - Shared authentication module
  - `verify_api_key()` — dependency for all protected endpoints
  - Timing-safe comparison via `secrets.compare_digest()`
  - `INTELLIGENCE_API_KEY` env var required for production (warns in dev mode)
  - All router endpoints use `Depends(verify_api_key)`

### Routers (`routers/`)
- `dashboard.py` - Dashboard statistics
  - `GET /api/v1/dashboard/stats` - KPIs: total articles, entities, reports, coverage

- `reports.py` - Intelligence reports
  - `GET /api/v1/reports` - Paginated report list with filters (status, type, date range)
  - `GET /api/v1/reports/{id}` - Full report detail with sources and feedback

- `stories.py` - **Storyline graph & narrative data**
  - `GET /api/v1/stories/graph` - Full graph network: nodes (active storylines) + links (edges). Queries `v_active_storylines` and `v_storyline_graph` views. Response structured for react-force-graph.
  - `GET /api/v1/stories` - Paginated storyline list, ordered by momentum_score DESC. Filters: `status` (emerging/active/stabilized/archived)
  - `GET /api/v1/stories/{id}` - Storyline detail with related storylines (via edges) and recent articles (via article_storylines join)

### Schemas (`schemas/`)
- `common.py` - `APIResponse[T]` generic wrapper, `PaginationMeta`
- `dashboard.py` - `DashboardStats`, `OverviewStats`, `ArticleStats`, `EntityStats`, `QualityStats`
- `reports.py` - `ReportListItem`, `ReportDetail`, `ReportFilters`, `ReportContent`, `ReportSource`
- `stories.py` - `StorylineNode`, `StorylineEdge`, `GraphStats`, `GraphNetwork`, `StorylineDetail`, `RelatedStoryline`, `LinkedArticle`

## Security
- All endpoints require `X-API-Key` header (via `auth.py:verify_api_key`)
- Error responses use generic `"Internal server error"` (no `str(e)` leaking)
- Rate limiting via slowapi on expensive endpoints
- CORS origins configurable via `ALLOWED_ORIGINS` env var

## Dependencies

- **Internal**: `src/storage/database`, `src/utils/logger`, `src/api/auth`
- **External**: `fastapi`, `uvicorn`, `pydantic`, `slowapi`

## Data Flow

- **Input**: HTTP requests from Next.js frontend (via server-side proxy at `/api/proxy/`)
- **Output**: JSON responses wrapped in `APIResponse[T]`, GeoJSON for map, Graph network for storyline visualization

## Running

```bash
# Development
uvicorn src.api.main:app --reload --port 8000

# Production (Docker)
docker compose up backend
```
