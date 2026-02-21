# StorylineGraph Components Context

## Purpose
React/TypeScript components for the force-directed narrative storyline graph visualization. Renders the network of active intelligence storylines and their inter-connections (Jaccard-weighted edges from the Narrative Engine) as an interactive Canvas-based force graph.

## Architecture Role
Presentation components consumed by `app/stories/page.tsx`. Uses the same dynamic import / SSR-disabled pattern as `IntelligenceMap/` because `react-force-graph-2d` requires the Canvas API and `requestAnimationFrame`, which are unavailable in the Node.js SSR environment. Each component has a single clear responsibility following React composition.

## Key Files

### `GraphLoader.tsx` — Client-side dynamic loader
- Marked `'use client'`
- Uses `next/dynamic` with `{ ssr: false, loading: () => <GraphSkeleton /> }`
- Enables code-splitting for the heavy `react-force-graph-2d` + d3-force bundle
- Shows `GraphSkeleton` while the bundle downloads; transparent to the parent `StoriesPage`

### `GraphSkeleton.tsx` — Loading skeleton
- Marked `'use client'`; full-screen dark background (`#0A1628`)
- Animated orange (#FF6B35) spinning ring + "INITIALIZING STORYLINE GRAPH" label
- HUD-corner skeleton bars (top-left, top-right) matching the real HUD layout
- Subtle orange grid overlay (60 px × 60 px, `opacity-10`)
- Corner bracket decorations matching `StorylineGraph`'s brackets (visual continuity)

### `StorylineGraph.tsx` — Main force-directed graph
Marked `'use client'`. Orchestrates `ForceGraph2D`, HUD overlays, tooltip, and `StorylineDossier`.

**Data flow:**
1. Calls `useGraphNetwork()` (SWR, 60 s polling) → `graph: GraphNetwork | undefined`
2. `useMemo` transforms `graph.nodes` → `GraphNode[]` and `graph.links` → `GraphLink[]` (filters out links whose source/target is not in the node set)
3. Passes `graphData` to `<ForceGraph2D>`

**Node rendering — `paintNode` (Canvas 2D):**
- Radius = `4 + momentum_score * 12` → range **4–16 px**
- Color by `narrative_status`: `emerging=#FF6B35`, `active=#00A8E8`, `stabilized=#666666`
- Selected node: filled white, color border (2 px); glow ring at `radius + 4` px with 20% alpha
- Hovered node: same glow ring
- Label: drawn only when `globalScale > 1.5` OR `momentum_score > 0.7` OR node is selected/hovered; truncated at 30 chars; monospace font at `max(10/globalScale, 3)` px; dark background pill (`rgba(10,22,40,0.85)`)

**Hit area — `nodePointerAreaPaint`:**
- Extends the clickable area by +4 px beyond the visual radius so small low-momentum nodes remain easy to click

**Link rendering — `paintLink` (Canvas 2D):**
- Color: `rgba(100, 100, 100, 0.2 + weight * 0.6)` — more opaque for stronger connections
- Width: `0.5 + weight * 2.5` px — thicker for stronger connections

**d3-force simulation parameters:**
- `cooldownTicks=100` — simulation stabilizes after 100 ticks
- `d3AlphaDecay=0.02` — slow cooling for a more settled layout
- `d3VelocityDecay=0.3` — moderate friction
- `linkDirectionalParticles=0` — no animated particles (performance)
- Drag, zoom, and pan all enabled

**Interaction:**
- `onNodeClick`: toggles `selectedId` (clicking the same node again deselects it → `setSelectedId(prev => prev === node.id ? null : node.id)`)
- `onNodeHover`: sets `hoveredNode` state
- `handleNavigate(id)`: called from `StorylineDossier`'s "Connected Storylines" buttons; sets `selectedId` and animates camera via `graphRef.current.centerAt(x, y, 500)` + `.zoom(3, 500)`

**Overlays (all `pointer-events-none` except Dossier and Retry button):**
- Top-left HUD: "NARRATIVE GRAPH" label + `graph.stats` (total_nodes, total_edges, avg_momentum)
- Top-right Legend: status color key; hidden when a node is selected (Dossier takes that space)
- Bottom-left Tooltip: shown when `hoveredNode` is set and no node is selected; displays title, momentum (2 dp), article count, category
- Loading overlay: spinner + "Loading graph data…" while `isLoading && !graph`
- Error overlay: "Connection Error" message + RETRY button (calls `refresh()`) while `error && !graph`
- Empty state: "No Active Storylines" message when graph loaded but `graph.nodes.length === 0`
- Corner brackets: decorative CSS borders matching `GraphSkeleton`

### `StorylineDossier.tsx` — Detail side panel
Marked `'use client'`. Follows the same design language as `EntityDossier.tsx` in `IntelligenceMap/`.

**Props:**
```ts
{ storylineId: number | null; onClose: () => void; onNavigate: (id: number) => void }
```
Returns `null` immediately when `storylineId` is null (no render cost).

**Data:**  Calls `useStorylineDetail(storylineId)` internally (SWR, on-demand, no polling).

**Layout:** Fixed panel `right-4 top-4 bottom-4 w-[450px]`, `z-50`, dark `gray-900/95` background with orange border and backdrop blur.

**Sections (scrollable content area):**

| Section | Content |
|---------|---------|
| Header | Animated pulse dot, "Storyline Dossier" label, title, status badge (color-coded), ID |
| Momentum Analysis | Score (2 dp), HIGH/MEDIUM/LOW/MINIMAL label, article count, days active, animated progress bar |
| Summary | LLM-generated summary text |
| Key Entities | Orange badge list (`key_entities[]`) |
| Connected Storylines | Clickable list; each triggers `onNavigate(id)` → graph camera centers on that node |
| Recent Articles | Scrollable (max 300 px), Italian date format (`it-IT`), source label |

**Momentum thresholds:**
- ≥ 0.8 → HIGH (red-400)
- ≥ 0.5 → MEDIUM (yellow-400)
- ≥ 0.3 → LOW (gray-400)
- < 0.3 → MINIMAL (gray-600)

**Footer:** "CLASSIFIED | NARRATIVE ENGINE | \<today's ISO date\>" monospace label.

## Loading Architecture

```
app/stories/page.tsx  (Server Component)
    ├── Metadata/SEO (rendered server-side)
    └── <GraphLoader>  ('use client')
            └── dynamic(() => import('./StorylineGraph'), { ssr: false })
                    ├── <GraphSkeleton>  (bundle download)
                    └── <StorylineGraph>  (Canvas, browser-only)
                            └── <StorylineDossier>  (on node click)
```

## Dependencies

- **Internal**: `@/hooks/useStories` (`useGraphNetwork`, `useStorylineDetail`), `@/types/stories` (`NarrativeStatus`, `GraphNetwork`, `StorylineDetailData`)
- **External**:
  - `react-force-graph-2d` — Canvas-based force-directed graph (d3-force under the hood)
  - `next/dynamic` — Dynamic imports with SSR control
  - `lucide-react` — Icons (X, TrendingUp, FileText, GitBranch, Calendar, Tag)
  - `react` — `useCallback`, `useRef`, `useState`, `useMemo`

## Data Flow

```
useGraphNetwork()
    └── SWR GET /api/proxy/stories/graph (60 s poll)
            └── ForceGraph2D (graphData: { nodes, links })
                    └── node click → setSelectedId(id)
                            └── useStorylineDetail(id)
                                    └── SWR GET /api/proxy/stories/<id>
                                            └── StorylineDossier (detail panel)
                                                    └── onNavigate(id) → handleNavigate → graphRef.centerAt + zoom
```

## Status Color Reference

| Status | Canvas fill | Tailwind text | Tailwind badge bg |
|--------|-------------|---------------|-------------------|
| emerging | `#FF6B35` | `text-[#FF6B35]` | `bg-[#FF6B35]/20 border-[#FF6B35]/40` |
| active | `#00A8E8` | `text-[#00A8E8]` | `bg-[#00A8E8]/20 border-[#00A8E8]/40` |
| stabilized | `#666666` | `text-gray-400` | `bg-gray-500/20 border-gray-500/40` |
