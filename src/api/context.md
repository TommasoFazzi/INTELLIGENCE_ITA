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

- `openbb_backend.py` - OpenBB widget backend (optional)
  - Integration point for OpenBB widgets
  - `widgets.json` - Widget configuration

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
