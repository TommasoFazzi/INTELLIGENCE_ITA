# Intelligence Map Context

## Purpose
Modern Next.js/React frontend for interactive intelligence visualization on a tactical map. Displays geocoded entities from the database on a Mapbox-powered 3D map with military-style HUD overlays.

## Architecture Role
Advanced visualization layer consuming data from `src/api/` REST endpoints. Provides interactive exploration of geopolitical entities with their locations, mention counts, and relationships. Separate from the Streamlit dashboard for specialized map visualization.

## Key Files

### App Structure
- `app/layout.tsx` - Root Next.js layout
- `app/globals.css` - Global styles with animations
- `app/intelligence-map/page.tsx` - Main map page route

### Components (see `components/IntelligenceMap/context.md`)
- `TacticalMap.tsx` - Main Mapbox GL component
- `GridOverlay.tsx` - Tactical grid visualization
- `HUDOverlay.tsx` - HUD elements (ZULU clock, coordinates)
- `EntityDossier.tsx` - Entity detail panel

### Configuration
- `.env.local` - Mapbox token (`NEXT_PUBLIC_MAPBOX_TOKEN`)
- `next.config.ts` - Next.js configuration
- `package.json` - Dependencies
- `tsconfig.json` - TypeScript config

### Types & Utils
- `types/entities.ts` - Entity TypeScript interfaces
- `utils/api.ts` - API client for backend communication

## Dependencies

- **Internal**: Consumes `src/api/` endpoints
- **External**:
  - `next` (16.x) - React framework
  - `react` (19.x) - UI library
  - `mapbox-gl` (3.x) - Map rendering
  - `axios` - HTTP client
  - `swr` - Data fetching
  - `framer-motion` - Animations
  - `tailwindcss` - Styling

## Data Flow

- **Input**:
  - GeoJSON from `GET /api/v1/map/entities`
  - Entity details from `GET /api/v1/map/entities/{id}`
  - Mapbox token from environment

- **Output**:
  - Interactive 3D map with entity markers
  - Entity dossier panels on click
  - Real-time ZULU time display
  - Mouse coordinate tracking

## Running

```bash
cd intelligence-map
npm install
npm run dev
# Open http://localhost:3000/intelligence-map
```

## Map Controls

- **Drag**: Pan map
- **Scroll**: Zoom
- **Ctrl + Drag**: Rotate bearing
- **Shift + Drag**: Pitch (3D tilt)
