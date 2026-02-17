# API Context

## Purpose
FastAPI REST backend providing HTTP endpoints for the Intelligence ITA platform: dashboard stats, report management, entity visualization, and **narrative storyline graph**. Serves data to the Next.js frontend.

## Architecture Role
HTTP interface layer between the database and frontend applications. Modular router architecture with Pydantic schema validation and API key authentication. Runs alongside the Streamlit HITL dashboard.

## Key Files

### Core
- `main.py` - FastAPI application entry point
  - `app` - FastAPI instance with CORS middleware (localhost:3000-3002)
  - API key authentication via `X-API-Key` header
  - Registers 3 routers: dashboard, reports, stories
  - Direct endpoints for map entities (`/api/v1/map/entities`)

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

### OpenBB Integration
- `openbb_backend.py` - OpenBB Workspace backend (port 7779)
  - Widget endpoints for OpenBB Pro dashboard
  - Intelligence scores, trade signals, macro summary

## Dependencies

- **Internal**: `src/storage/database`, `src/utils/logger`
- **External**:
  - `fastapi` - Web framework
  - `uvicorn` - ASGI server
  - `pydantic` - Request/response validation

## Data Flow

- **Input**:
  - HTTP requests from Next.js frontend
  - Database queries via `DatabaseManager`

- **Output**:
  - JSON responses wrapped in `APIResponse[T]`
  - GeoJSON for map rendering
  - Graph network (nodes + links) for storyline visualization
  - Paginated lists with `PaginationMeta`

## Running

```bash
# Development
uvicorn src.api.main:app --reload --port 8000

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
