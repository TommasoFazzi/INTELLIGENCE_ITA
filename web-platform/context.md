# Intelligence Map Context

## Purpose
Modern Next.js/React frontend for interactive intelligence visualization on a tactical map. Displays geocoded entities from the database on a Mapbox-powered map with military-style HUD overlays.

## Architecture Role
Advanced visualization layer consuming data from `src/api/` REST endpoints. Provides interactive exploration of geopolitical entities with their locations, mention counts, and relationships. Separate from the Streamlit dashboard for specialized map visualization.

## Key Files

### App Structure
- `app/layout.tsx` - Root Next.js layout
- `app/globals.css` - Global styles with animations
- `app/page.tsx` - Landing page
- `app/map/page.tsx` - Tactical map route (SSR metadata + dynamic import)
- `app/dashboard/page.tsx` - Dashboard route (SWR data fetching)

### Components

#### Map Components (`components/IntelligenceMap/`)
- `TacticalMap.tsx` - Main Mapbox GL component with clustering
- `MapLoader.tsx` - Client wrapper for dynamic import (ssr: false)
- `MapSkeleton.tsx` - Loading skeleton for map
- `GridOverlay.tsx` - Tactical grid visualization
- `HUDOverlay.tsx` - HUD elements (ZULU clock, coordinates)
- `EntityDossier.tsx` - Entity detail panel

#### Dashboard Components (`components/dashboard/`)
- `StatsCard.tsx` - Individual KPI card
- `StatsGrid.tsx` - Grid of stats cards
- `ReportsTable.tsx` - Paginated reports table
- `DashboardSkeleton.tsx` - Loading skeletons
- `ErrorState.tsx` - Error handling states

#### UI Components (`components/ui/`)
- Shadcn components: Button, Card, Skeleton, Table, Badge

### Configuration
- `.env.local` - Environment variables:
  - `NEXT_PUBLIC_MAPBOX_TOKEN` - Mapbox API token
  - `NEXT_PUBLIC_API_URL` - Backend API URL
  - `NEXT_PUBLIC_API_KEY` - API authentication key
- `next.config.ts` - Next.js configuration
- `package.json` - Dependencies
- `tsconfig.json` - TypeScript config

### Types & Utils
- `types/entities.ts` - Entity TypeScript interfaces
- `types/dashboard.ts` - Dashboard TypeScript interfaces
- `utils/api.ts` - API client for backend communication
- `hooks/useDashboard.ts` - SWR hooks for dashboard data

## Dependencies

- **Internal**: Consumes `src/api/` endpoints
- **External**:
  - `next` (16.x) - React framework with App Router
  - `react` (19.x) - UI library
  - `mapbox-gl` (3.x) - Map rendering
  - `swr` - Data fetching with polling
  - `framer-motion` - Animations
  - `tailwindcss` (4.x) - Styling
  - `lucide-react` - Icons

## Data Flow

- **Input**:
  - GeoJSON from `GET /api/v1/map/entities`
  - Entity details from `GET /api/v1/map/entities/{id}`
  - Dashboard stats from `GET /api/v1/dashboard/stats`
  - Reports from `GET /api/v1/reports`
  - Mapbox token from environment

- **Output**:
  - Interactive map with entity clustering
  - Entity dossier panels on click
  - Real-time ZULU time display
  - Dashboard with live KPIs and reports table

## Running

```bash
cd web-platform
npm install
npm run dev
# Routes:
#   http://localhost:3000/          - Landing page
#   http://localhost:3000/map       - Tactical map
#   http://localhost:3000/dashboard - Dashboard
```

## Map Controls

- **Drag**: Pan map
- **Scroll**: Zoom
- **Ctrl + Drag**: Rotate bearing
- **Shift + Drag**: Pitch (3D tilt)
- **Click marker**: Open entity dossier
- **Click cluster**: Zoom to expand
