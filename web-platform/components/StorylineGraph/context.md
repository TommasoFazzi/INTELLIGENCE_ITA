# StorylineGraph Components Context

## Purpose
React/TypeScript components for the force-directed narrative storyline graph visualization. Renders the network of active intelligence storylines and their inter-connections (TF-IDF weighted Jaccard edges from the Narrative Engine) as an interactive Canvas-based force graph. Supports **community coloring** (Louvain clusters), **ego network** exploration, and **momentum-based filtering**.

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

### `StorylineGraph.tsx` (~479 lines) — Main force-directed graph
Marked `'use client'`. Orchestrates `ForceGraph2D`, HUD overlays, tooltip, ego network highlighting, **Top-N community coloring** with momentum brightness, legend with "Others" aggregation, momentum slider, and `StorylineDossier`.

**Data flow:**
1. Calls `useGraphNetwork()` (SWR, 60 s polling) → `graph: GraphNetwork | undefined`
2. On node click, calls `useEgoNetwork(selectedId, 0.05)` → `egoData` (direct neighbors + connecting edges)
3. `useMemo` transforms `graph.nodes` → `GraphNode[]` and `graph.links` → `GraphLink[]` (filters out links whose source/target is not in the node set)
4. Momentum slider state (`momentumFilter`, default 0) filters nodes below threshold
5. Passes `graphData` to `<ForceGraph2D>`

**Node rendering — `paintNode` (Canvas 2D):**
- Radius = `4 + momentum_score * 12` → range **4–16 px**
- **Color by `community_id`** (primary): **Top-N strategy** — 15-color `COMMUNITY_PALETTE` assigned by community size rank. Top 15 communities (by node count) get unique perceptually-distinct colors; all others render in `OTHER_COLOR = '#2A3A4A'` (neutral dark gray). Color assignment computed in `useMemo` via `communityColorMap` (Map<community_id, hex>). Nodes without `community_id` fall back to **status color**: `emerging=#FF6B35`, `active=#00A8E8`, `stabilized=#666666`
- **Momentum-as-brightness**: Node opacity = `Math.max(0.5, Math.min(1.0, 0.5 + momentum_score * 0.5))` — range [0.5, 1.0]. High-momentum storylines appear brighter; low-momentum ones are dimmer but always visible (minimum 50% opacity).
- **Ego network highlighting**: When ego mode is active (a node is selected and ego data is loaded), non-neighbor nodes are drawn with `globalAlpha=0.08` (heavily dimmed). Neighbor nodes and the selected node remain at full opacity. **Ghost highlight**: neighbor nodes that are normally gray (`OTHER_COLOR`) highlight to `EGO_HIGHLIGHT = '#FFFFFF'` (white) during ego drill-down.
- Selected node: filled white, color border (2 px); glow ring at `radius + 4` px with 20% alpha
- Hovered node: same glow ring
- Label: drawn only when `globalScale > 1.5` OR `momentum_score > 0.7` OR node is selected/hovered; truncated at 30 chars; monospace font at `max(10/globalScale, 3)` px; dark background pill (`rgba(10,22,40,0.85)`)

**Hit area — `nodePointerAreaPaint`:**
- Extends the clickable area by +4 px beyond the visual radius so small low-momentum nodes remain easy to click

**Link rendering — `paintLink` (Canvas 2D):**
- Default: `rgba(100, 100, 100, 0.2 + weight * 0.6)` — more opaque for stronger connections
- **Ego network edges**: highlighted in orange with `alpha=0.9`; non-ego edges dimmed further when ego mode is active
- Width: `0.5 + weight * 2.5` px — thicker for stronger connections

**Community labels — `paintFramePost` callback:**
- After each frame, computes the centroid (average x,y) of all nodes in each community
- Draws community labels at centroids: font `18px`, opacity `22%` (subtle background labels)
- Labels use community number or LLM-generated community label if available

**d3-force simulation parameters:**
- `warmupTicks=300` — let simulation settle before rendering
- `cooldownTicks=0` — never stops (continuous physics)
- `d3AlphaDecay=0.05` — moderate cooling
- `d3VelocityDecay=0.4` — moderate friction
- `linkDirectionalParticles=0` — no animated particles (performance)
- Drag, zoom, and pan all enabled

**Interaction:**
- `onNodeClick`: toggles `selectedId` (clicking the same node again deselects it → `setSelectedId(prev => prev === node.id ? null : node.id)`); triggers ego network fetch
- `onNodeHover`: sets `hoveredNode` state
- `handleNavigate(id)`: called from `StorylineDossier`'s "Connected Storylines" buttons; sets `selectedId` and animates camera via `graphRef.current.centerAt(x, y, 500)` + `.zoom(3, 500)`

**Overlays (all `pointer-events-none` except Dossier, Retry button, and momentum slider):**
- **Top-left HUD**: "NARRATIVE GRAPH" label + stats: NODES, EDGES, **COMMUNITIES**, AVG MOMENTUM, **EDGES/NODE**
- **Top-right Legend (when no node selected)**: **Community legend** — dynamic list of top 15 communities (by size) with colored dots and entity-based labels (most frequent entity among top-5 `key_entities` per community). **"Others (N)" row** at the bottom with a separator line, aggregating all minor communities and their total node count, displayed in `OTHER_COLOR`.
- **Top-right Momentum slider**: Interactive range slider (0–1, step 0.1) for filtering nodes by minimum momentum score
- Bottom-left Tooltip: shown when `hoveredNode` is set and no node is selected; displays title, momentum (2 dp), article count, category
- Loading overlay: spinner + "Loading graph data…" while `isLoading && !graph`
- Error overlay: "Connection Error" message + RETRY button (calls `refresh()`) while `error && !graph`
- Empty state: "No Active Storylines" message when graph loaded but `graph.nodes.length === 0`
- Corner brackets: decorative CSS borders matching `GraphSkeleton`

### `StorylineDossier.tsx` (~291 lines) — Detail side panel
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
                            ├── useGraphNetwork()  (graph data, 60s poll)
                            ├── useEgoNetwork(selectedId, 0.05)  (ego subgraph)
                            └── <StorylineDossier>  (on node click)
```

## Dependencies

- **Internal**: `@/hooks/useStories` (`useGraphNetwork`, `useEgoNetwork`, `useStorylineDetail`), `@/types/stories` (`NarrativeStatus`, `GraphNetwork`, `StorylineDetailData`, `EgoNetworkData`)
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
                    ├── community coloring (node.community_id → color palette)
                    ├── community labels (paintFramePost → centroid labels)
                    ├── momentum slider → filter nodes below threshold
                    └── node click → setSelectedId(id)
                            ├── useEgoNetwork(id, 0.05)
                            │       └── SWR GET /api/proxy/stories/<id>/network
                            │               └── highlight neighbors, dim non-neighbors (alpha=0.08)
                            │                   ego edges drawn in orange (alpha=0.9)
                            └── useStorylineDetail(id)
                                    └── SWR GET /api/proxy/stories/<id>
                                            └── StorylineDossier (detail panel)
                                                    └── onNavigate(id) → handleNavigate → graphRef.centerAt + zoom
```

## Color Reference

### Community Colors (15-color `COMMUNITY_PALETTE`, assigned by size rank)

The top 15 communities by node count each receive a unique color from the palette. All remaining communities are rendered in `OTHER_COLOR`.

| Rank | Color | Hex |
|------|-------|-----|
| 0 | Orange | `#FF6B35` |
| 1 | Blue | `#00A8E8` |
| 2 | Purple | `#7B68EE` |
| 3 | Teal | `#00CED1` |
| 4 | Gold | `#FFD700` |
| 5 | Pink | `#FF69B4` |
| 6 | Green | `#32CD32` |
| 7 | Red-Orange | `#FF4500` |
| 8 | Coral | `#FF7F7F` |
| 9 | Lime | `#ADFF2F` |
| 10 | Sky Blue | `#87CEEB` |
| 11 | Orchid | `#DA70D6` |
| 12 | Spring Green | `#00FA9A` |
| 13 | Salmon | `#FA8072` |
| 14 | Steel Blue | `#4682B4` |

### Special Colors

| Purpose | Hex | Usage |
|---------|-----|-------|
| Other (minor communities) | `#2A3A4A` | Neutral dark gray for communities outside top 15 |
| Ego Highlight | `#FFFFFF` | White — ghost nodes (normally `OTHER_COLOR`) highlight to this during ego drill-down |

### Status Colors (fallback when `community_id` is null)

| Status | Canvas fill | Tailwind text | Tailwind badge bg |
|--------|-------------|---------------|-------------------|
| emerging | `#FF6B35` | `text-[#FF6B35]` | `bg-[#FF6B35]/20 border-[#FF6B35]/40` |
| active | `#00A8E8` | `text-[#00A8E8]` | `bg-[#00A8E8]/20 border-[#00A8E8]/40` |
| stabilized | `#666666` | `text-gray-400` | `bg-gray-500/20 border-gray-500/40` |
