# IntelligenceMap Components Context

## Purpose
React/TypeScript components for the tactical intelligence map visualization. Provides the core UI elements for displaying geocoded entities on a Mapbox GL map with military-style HUD overlays.

## Architecture Role
Presentation components consumed by the `app/intelligence-map/page.tsx` route. Each component handles a specific visualization concern following React composition patterns.

## Key Files

- `TacticalMap.tsx` - Main map component
  - Initializes Mapbox GL with dark military style
  - Fetches entities from API (`/api/v1/map/entities`)
  - Implements clustering for large entity counts (5000+ limit)
  - Handles entity selection on click
  - Manages map state (latitude, longitude, zoom)
  - Composes GridOverlay, HUDOverlay, and EntityDossier
  - Default center: Rome (41.9028, 12.4964)

- `GridOverlay.tsx` - Tactical grid visualization
  - SVG-based grid overlay on top of map
  - Military/tactical aesthetic grid lines
  - Responsive to map dimensions

- `HUDOverlay.tsx` - Head-Up Display elements
  - Real-time ZULU (UTC) clock display
  - Mouse coordinate tracking (lat/lng)
  - Corner bracket framing (tactical style)
  - Scanline visual effect

- `EntityDossier.tsx` - Entity detail panel
  - Displays selected entity information
  - Shows: name, type, mention count, first/last seen dates
  - Lists related articles
  - Slide-in panel animation

## Dependencies

- **Internal**: `@/utils/api` for entity fetching, `@/types/entities` for interfaces
- **External**:
  - `mapbox-gl` - Map rendering engine
  - `react` - Component framework
  - `framer-motion` - Animations (for dossier panel)

## Data Flow

- **Input**:
  - GeoJSON features from API
  - Mapbox access token from environment
  - User interactions (clicks, pan, zoom)

- **Output**:
  - Rendered map with entity markers
  - Entity clusters at low zoom
  - Individual markers at high zoom
  - Selected entity dossier panel

## Entity Marker Types

| Type | Color | Description |
|------|-------|-------------|
| GPE | Cyan | Geopolitical entities (countries, cities) |
| ORG | Orange | Organizations |
| PERSON | Purple | People |
| Default | Gray | Other entity types |
