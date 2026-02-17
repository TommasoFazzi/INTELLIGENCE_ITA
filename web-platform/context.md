# Web Platform Context

## Purpose
Modern Next.js/React frontend for interactive intelligence visualization. Provides a tactical intelligence map (Mapbox), a **narrative storyline graph** (force-directed), a dashboard with reports, and a landing page. Consumes data from the FastAPI backend.

## Architecture Role
Advanced visualization layer consuming data from `src/api/` REST endpoints. Provides interactive exploration of geopolitical entities (map), **narrative storyline network** (graph), and intelligence reports (dashboard). Separate from the Streamlit HITL dashboard.

## Key Files

### App Structure
- `app/layout.tsx` - Root Next.js layout
- `app/globals.css` - Global styles with animations
- `app/page.tsx` - Landing page
- `app/map/page.tsx` - Tactical map route (SSR metadata + dynamic import)
- `app/dashboard/page.tsx` - Dashboard route (SWR data fetching)
- `app/dashboard/report/[id]/page.tsx` - Report detail route
- **`app/stories/page.tsx`** - Storyline graph route (SSR metadata + dynamic import)

### Components

#### Map Components (`components/IntelligenceMap/`)
- `TacticalMap.tsx` - Main Mapbox GL component with clustering
- `MapLoader.tsx` - Client wrapper for dynamic import (ssr: false)
- `MapSkeleton.tsx` - Loading skeleton for map
- `GridOverlay.tsx` - Tactical grid visualization
- `HUDOverlay.tsx` - HUD elements (ZULU clock, coordinates)
- `EntityDossier.tsx` - Entity detail panel

#### **Storyline Graph Components (`components/StorylineGraph/`)**
- `StorylineGraph.tsx` - Main force-directed graph (react-force-graph-2d)
  - Custom `paintNode`: radius by momentum (4-16px), color by status (emerging=#FF6B35, active=#00A8E8, stabilized=#666)
  - Custom `paintLink`: thickness by weight (0.5-3px), gray with opacity
  - HUD overlay with stats (total nodes, edges, avg momentum)
  - Click node → opens StorylineDossier
- `GraphLoader.tsx` - Client wrapper for dynamic import (ssr: false, same pattern as MapLoader)
- `GraphSkeleton.tsx` - Loading skeleton with orange accent theme
- `StorylineDossier.tsx` - Storyline detail side panel (follows EntityDossier pattern)
  - Momentum analysis (score + bar), summary, key entities (badges)
  - Connected storylines (clickable, navigates graph)
  - Recent articles list
  - `onNavigate(id)` callback for graph navigation

#### Dashboard Components (`components/dashboard/`)
- `StatsCard.tsx` - Individual KPI card
- `StatsGrid.tsx` - Grid of stats cards
- `ReportsTable.tsx` - Paginated reports table
- `DashboardSkeleton.tsx` - Loading skeletons
- `ErrorState.tsx` - Error handling states

#### Landing Components (`components/landing/`)
- `Navbar.tsx` - Navigation with links to Dashboard, **Storylines**, Intelligence Map
- `Hero.tsx`, `Features.tsx`, `Footer.tsx` - Landing page sections

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

### Types & Hooks
- `types/entities.ts` - Entity TypeScript interfaces
- `types/dashboard.ts` - Dashboard TypeScript interfaces
- **`types/stories.ts`** - Storyline graph TypeScript interfaces (StorylineNode, StorylineEdge, GraphNetwork, StorylineDetail)
- `utils/api.ts` - API client for backend communication
- `hooks/useDashboard.ts` - SWR hooks for dashboard data
- **`hooks/useStories.ts`** - SWR hooks for storyline graph data
  - `useGraphNetwork()` → `GET /api/v1/stories/graph` (60s polling)
  - `useStorylineDetail(id)` → `GET /api/v1/stories/{id}` (on-demand)

## Dependencies

- **Internal**: Consumes `src/api/` endpoints
- **External**:
  - `next` (16.x) - React framework with App Router
  - `react` (19.x) - UI library
  - `mapbox-gl` (3.x) - Map rendering
  - **`react-force-graph-2d`** - Force-directed graph visualization (d3-force based)
  - `swr` - Data fetching with polling
  - `framer-motion` - Animations
  - `tailwindcss` (4.x) - Styling
  - `lucide-react` - Icons (GitBranch for Storylines nav)

## Data Flow

- **Input**:
  - GeoJSON from `GET /api/v1/map/entities`
  - Dashboard stats from `GET /api/v1/dashboard/stats`
  - Reports from `GET /api/v1/reports`
  - **Graph network from `GET /api/v1/stories/graph`** (nodes + links)
  - **Storyline detail from `GET /api/v1/stories/{id}`**
  - Mapbox token from environment

- **Output**:
  - Interactive map with entity clustering
  - **Force-directed narrative graph with momentum-scaled nodes**
  - **Storyline dossier panels on node click**
  - Dashboard with live KPIs and reports table

## Running

```bash
cd web-platform
npm install
npm run dev
# Routes:
#   http://localhost:3000/          - Landing page
#   http://localhost:3000/stories   - Storyline graph
#   http://localhost:3000/map       - Tactical intelligence map
#   http://localhost:3000/dashboard - Dashboard with reports
```
