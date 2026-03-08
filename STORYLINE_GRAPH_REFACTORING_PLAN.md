# Storyline Graph Refactoring Plan

> **Data creazione:** 7 marzo 2026  
> **Stato:** Da implementare  
> **Obiettivo:** Trasformare il grafo narrativo da un hairball illeggibile (1262 nodi, 75k archi, 406 micro-communities) in uno strumento di analisi con 200-400 nodi, 2-5k archi e 15-30 macro-communities significative.

---

## Architettura Attuale (State of Art)

### Data Flow Completo

```
RSS (33 feeds)
  → NLP (spaCy + sentence-transformers embeddings)
  → PostgreSQL/pgvector (articles + embeddings 384-dim)
  → NarrativeProcessor.process_daily_batch() — 6 stages:
      Stage 1: Micro-clustering (cosine ≥ 0.90 → eventi unici)
      Stage 2: Adaptive matching (hybrid score ≥ 0.75 → match a storyline)
      Stage 3: HDBSCAN discovery (orphaned events → nuove storylines)
      Stage 4: LLM evolution (Gemini 2.0 Flash → titolo + summary italiano)
      Stage 4b: Relevance validation (regex scope/off-topic → archive)
      Stage 5: Graph builder (TF-IDF weighted Jaccard → storyline_edges)
      Stage 6: Decay (momentum × 0.7 weekly, lifecycle transitions)
  → compute_communities.py (Louvain community detection, post-pipeline)
  → API FastAPI (/stories/graph, /{id}/network, /{id}, list)
  → Frontend Next.js (react-force-graph-2d + ego-network + dossier panel)
```

### File Coinvolti

| File | Ruolo | Linee |
|------|-------|-------|
| `src/nlp/narrative_processor.py` | Engine core: 6 stage pipeline | ~1163 |
| `src/storage/database.py` | DB operations + `refresh_entity_idf()` | ~1996 |
| `scripts/process_narratives.py` | CLI wrapper per NarrativeProcessor | ~170 |
| `scripts/compute_communities.py` | Louvain community detection | ~170 |
| `scripts/daily_pipeline.py` | Pipeline orchestrator (7 steps) | ~806 |
| `src/api/routers/stories.py` | API endpoints graph/ego/list/detail | ~443 |
| `src/api/schemas/stories.py` | Pydantic models API responses | ~65 |
| `web-platform/components/StorylineGraph/StorylineGraph.tsx` | Force graph canvas rendering | ~339 |
| `web-platform/components/StorylineGraph/StorylineDossier.tsx` | Side panel dettaglio storyline | ~268 |
| `web-platform/components/StorylineGraph/GraphLoader.tsx` | Dynamic import wrapper (SSR off) | ~27 |
| `web-platform/hooks/useStories.ts` | SWR hooks per API calls | ~117 |
| `web-platform/types/stories.ts` | TypeScript types | ~87 |
| `migrations/008_add_storylines.sql` | Schema base storylines + article_storylines | — |
| `migrations/012_narrative_graph.sql` | storyline_edges + narrative_status + views | — |
| `migrations/015_tfidf_graph_community.sql` | entity_idf + community_id + views update | — |

### Schema DB Rilevante

```sql
-- storylines (colonne chiave per il refactoring)
storylines (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    original_embedding VECTOR(384),   -- immutabile
    current_embedding VECTOR(384),    -- drift: 0.85*old + 0.15*new
    key_entities JSONB DEFAULT '[]',  -- max 20 entities
    narrative_status VARCHAR(20),     -- emerging → active → stabilized → archived
    article_count INTEGER DEFAULT 0,
    momentum_score FLOAT DEFAULT 1.0, -- 1.0=hot, <0.3=dormant
    category VARCHAR(50),
    community_id INTEGER DEFAULT NULL,
    start_date DATE,
    last_update TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
)

-- storyline_edges
storyline_edges (
    source_story_id INTEGER REFERENCES storylines(id),
    target_story_id INTEGER REFERENCES storylines(id),
    weight FLOAT CHECK (weight >= 0.0 AND weight <= 1.0),
    relation_type VARCHAR(30) DEFAULT 'relates_to',
    UNIQUE (source_story_id, target_story_id),
    CHECK (source_story_id != target_story_id)
)

-- entity_idf (materialized view)
entity_idf AS
SELECT entity, COUNT(*)::integer AS doc_freq,
       LN(total_active_storylines / doc_freq) AS idf
FROM storylines, jsonb_array_elements_text(key_entities)
WHERE narrative_status IN ('emerging', 'active', 'stabilized')
GROUP BY entity;

-- v_active_storylines (view → usata da API /graph)
-- v_storyline_graph (view → edges con join su storylines attive)
```

### Metriche Attuali (7 marzo 2026)

| Metrica | Valore | Problema |
|---------|--------|----------|
| Nodi (emerging+active) | ~1262 | Troppi per visualizzazione utile |
| Archi totali DB | 75,631 | 60 archi/nodo media → hairball |
| Communities Louvain | 406 | ~3 nodi/community → frammentato |
| Community detection | TIMEOUT 120s | DB write lento (UPDATE singoli) |
| ENTITY_JACCARD_THRESHOLD | 0.10 | Troppo bassa per TF-IDF weighted |
| Decay emerging senza match | 14 giorni | Troppo lento → zombie accumulati |
| API default min_edge_weight | 0.35 | OK ma DB ha 75k archi sotto |
| Pipeline narrative step | 133s | OK |
| Archi duplicati bidirezionali | Sì | A→B e B→A per stessa relazione |

---

## Piano di Esecuzione

---

## FASE 1: Pulizia Dati (Backend)

**Obiettivo:** Ridurre nodi ~1262 → 200-400, archi 75k → 2-5k.

---

### 1.1 — Decay aggressivo per storylines zombie

**Problema:** Storylines "emerging" con 1-2 articoli che non ricevono nuovi match restano attive per 14 giorni, gonfiando il grafo con nodi inutili.

**File:** `src/nlp/narrative_processor.py`  
**Metodo:** `_apply_decay()` (riga ~1105)  
**Modifica:** Cambiare `INTERVAL '14 days'` → `INTERVAL '5 days'`

**Codice attuale:**
```python
# 4. Emerging for 14 days without reaching 3 articles → archived
cur.execute("""
    UPDATE storylines
    SET narrative_status = 'archived'
    WHERE narrative_status = 'emerging'
    AND article_count < 3
    AND created_at < NOW() - INTERVAL '14 days'
    RETURNING id
""")
```

**Codice nuovo:**
```python
# 4. Emerging for 5 days without reaching 3 articles → archived
cur.execute("""
    UPDATE storylines
    SET narrative_status = 'archived'
    WHERE narrative_status = 'emerging'
    AND article_count < 3
    AND created_at < NOW() - INTERVAL '5 days'
    RETURNING id
""")
```

**Impatto atteso:** Riduzione di ~500-700 storylines zombie.

---

### 1.2 — Alzare threshold archi da 0.10 → 0.20

**Problema:** Con TF-IDF weighted Jaccard, threshold 0.10 genera archi tra storylines che condividono solo 1-2 entità comuni. Questi archi deboli creano il hairball.

**File:** `src/nlp/narrative_processor.py`  
**Posizione:** Class constant (riga ~93)  
**Modifica:** `ENTITY_JACCARD_THRESHOLD = 0.10` → `0.20`

**Codice attuale:**
```python
ENTITY_JACCARD_THRESHOLD = 0.10  # Min TF-IDF weighted Jaccard for graph edges
```

**Codice nuovo:**
```python
ENTITY_JACCARD_THRESHOLD = 0.20  # Min TF-IDF weighted Jaccard for graph edges
```

**Impatto atteso:** Riduzione archi da 75k → ~10-20k (gli archi con weight 0.10-0.20 sono la maggioranza).

---

### 1.3 — Deduplica archi bidirezionali

**Problema:** Quando storyline A viene processata, crea edge A→B. Quando B viene processata, crea B→A. Il grafo è semanticamente non-diretto (entity overlap è simmetrico), ma gli archi vengono duplicati.

**File:** `src/nlp/narrative_processor.py`  
**Metodo:** `_update_graph_connections()` (riga ~979)

**Logica attuale:**
```python
# Replace all outgoing edges in one DELETE, then bulk-insert valid ones
cur.execute("DELETE FROM storyline_edges WHERE source_story_id = %s", (storyline_id,))
if new_edges:
    execute_values(cur, """
        INSERT INTO storyline_edges (source_story_id, target_story_id, weight, relation_type)
        VALUES %s
    """, [(s, t, w, 'relates_to') for s, t, w in new_edges])
    edges_modified = len(new_edges)
```

**Logica nuova:**
```python
# Delete all outgoing edges for this storyline
cur.execute("DELETE FROM storyline_edges WHERE source_story_id = %s", (storyline_id,))

if new_edges:
    # Check which reverse edges already exist (B→A where we want A→B)
    target_ids = [t for _, t, _ in new_edges]
    cur.execute("""
        SELECT source_story_id, weight
        FROM storyline_edges
        WHERE source_story_id = ANY(%s) AND target_story_id = %s
    """, (target_ids, storyline_id))
    reverse_edges = {row[0]: row[1] for row in cur.fetchall()}

    filtered_edges = []
    for s, t, w in new_edges:
        reverse_weight = reverse_edges.get(t, None)
        if reverse_weight is not None:
            # Reverse edge exists (B→A). Keep whichever has higher weight.
            if w > reverse_weight:
                # Our edge is stronger: delete reverse, insert ours
                cur.execute(
                    "DELETE FROM storyline_edges WHERE source_story_id = %s AND target_story_id = %s",
                    (t, storyline_id)
                )
                filtered_edges.append((s, t, w))
            # else: reverse is stronger or equal, skip this edge
        else:
            # No reverse edge exists: insert normally
            filtered_edges.append((s, t, w))

    if filtered_edges:
        execute_values(cur, """
            INSERT INTO storyline_edges (source_story_id, target_story_id, weight, relation_type)
            VALUES %s
        """, [(s, t, w, 'relates_to') for s, t, w in filtered_edges])
    edges_modified = len(filtered_edges)
```

**Impatto atteso:** Riduzione archi di ~40-50% (eliminazione duplicati).

---

### 1.4 — Migration SQL di cleanup immediato

**File da creare:** `migrations/016_graph_cleanup.sql`

Questa migration applica il cleanup una tantum al DB esistente. Le modifiche al codice (1.1-1.3) prevengono il re-accumulo futuro.

```sql
-- Migration 016: Graph cleanup — reduce zombie storylines and weak/duplicate edges
-- Run: psql $DATABASE_URL -f migrations/016_graph_cleanup.sql

-- 1. Archive zombie storylines (emerging, <3 articles, >5 days old)
UPDATE storylines
SET narrative_status = 'archived'
WHERE narrative_status = 'emerging'
  AND article_count < 3
  AND created_at < NOW() - INTERVAL '5 days';

-- 2. Delete weak edges (below new threshold 0.20)
DELETE FROM storyline_edges WHERE weight < 0.20;

-- 3. Delete duplicate bidirectional edges (keep the one with higher weight)
--    If A→B and B→A both exist, delete the one with lower weight.
--    If equal weight, delete the one with higher id.
DELETE FROM storyline_edges e1
USING storyline_edges e2
WHERE e1.source_story_id = e2.target_story_id
  AND e1.target_story_id = e2.source_story_id
  AND (e1.weight < e2.weight OR (e1.weight = e2.weight AND e1.id > e2.id));

-- 4. Refresh entity_idf to reflect new active storylines set
REFRESH MATERIALIZED VIEW entity_idf;

-- 5. Null out community_id (will be recomputed by compute_communities.py)
UPDATE storylines SET community_id = NULL;

-- Verify results:
-- SELECT COUNT(*) FROM storylines WHERE narrative_status IN ('emerging', 'active');
-- SELECT COUNT(*) FROM storyline_edges;
-- SELECT AVG(weight), MIN(weight), MAX(weight) FROM storyline_edges;
```

**Deploy:** Applicare via `psql $DATABASE_URL -f migrations/016_graph_cleanup.sql` sul server di produzione prima del prossimo run della pipeline.

---

## FASE 2: Community Detection Robusto (Backend)

**Obiettivo:** Da 406 micro-communities → 15-30 macro-communities. Fix timeout.

---

### 2.1 — Fix timeout: batch UPDATE in compute_communities.py

**Problema:** 1262 `UPDATE` singoli dentro una transazione. Lento + timeout 120s.

**File:** `scripts/compute_communities.py`  
**Metodo:** `compute_and_save_communities()` (riga ~111)

**Codice attuale:**
```python
with db.get_connection() as conn:
    with conn.cursor() as cur:
        for storyline_id, cid in partition.items():
            cur.execute(
                "UPDATE storylines SET community_id = %s WHERE id = %s",
                (cid, storyline_id)
            )
        # Null out any storyline not in partition
        if partition:
            cur.execute(
                "UPDATE storylines SET community_id = NULL "
                "WHERE id != ALL(%s) AND community_id IS NOT NULL",
                (list(partition.keys()),)
            )
    conn.commit()
```

**Codice nuovo:**
```python
from psycopg2.extras import execute_values

with db.get_connection() as conn:
    with conn.cursor() as cur:
        # Batch update all community assignments in one query
        execute_values(cur, """
            UPDATE storylines AS s
            SET community_id = v.cid
            FROM (VALUES %s) AS v(sid, cid)
            WHERE s.id = v.sid
        """, [(sid, cid) for sid, cid in partition.items()])

        # Null out any storyline not in partition
        if partition:
            cur.execute(
                "UPDATE storylines SET community_id = NULL "
                "WHERE id != ALL(%s) AND community_id IS NOT NULL",
                (list(partition.keys()),)
            )
    conn.commit()
```

**Aggiungere import** in cima al file:
```python
from psycopg2.extras import execute_values
```

---

### 2.2 — Aumentare timeout a 300s nella pipeline

**File:** `scripts/daily_pipeline.py`  
**Posizione:** Step `community_detection` (riga ~123)

**Codice attuale:**
```python
PipelineStep(
    name="community_detection",
    command="python scripts/compute_communities.py",
    description="Community detection (Louvain) sul grafo narrativo",
    timeout_seconds=120,  # 2 min
    continue_on_failure=True
),
```

**Codice nuovo:**
```python
PipelineStep(
    name="community_detection",
    command="python scripts/compute_communities.py --min-weight 0.25",
    description="Community detection (Louvain) sul grafo narrativo",
    timeout_seconds=300,  # 5 min safety net
    continue_on_failure=True
),
```

---

### 2.3 — Alzare min_weight default per Louvain e aggiungere resolution

**File:** `scripts/compute_communities.py`  
**Posizione:** argparse section (riga ~134) e Louvain call (riga ~91)

**Modifica 1 — Default min_weight:**
```python
# PRIMA
parser.add_argument("--min-weight", type=float, default=0.15)

# DOPO
parser.add_argument("--min-weight", type=float, default=0.25)
```

**Modifica 2 — Aggiungere resolution parameter:**
```python
# Aggiungere argomento
parser.add_argument(
    "--resolution", type=float, default=0.8,
    help="Louvain resolution: lower = larger communities (default: 0.8)"
)
```

**Modifica 3 — Passare resolution a Louvain:**
```python
# PRIMA
partition = community_louvain.best_partition(G, random_state=42, weight='weight')

# DOPO
partition = community_louvain.best_partition(
    G, random_state=42, weight='weight', resolution=args.resolution
)
```

**Nota:** L'argomento `resolution` deve essere propagato dalla funzione `main()` alla funzione `compute_and_save_communities()`. Aggiungere `resolution` come parametro della funzione.

---

### 2.4 — Logging modularity score

**File:** `scripts/compute_communities.py`  
**Posizione:** Dopo la chiamata Louvain (riga ~95)

**Aggiungere dopo `partition = community_louvain.best_partition(...)`:**
```python
modularity = community_louvain.modularity(partition, G, weight='weight')
logger.info(
    "Louvain found %d communities from %d nodes (%d edges, min_weight=%.2f, resolution=%.2f) — modularity=%.3f",
    stats["communities"], stats["nodes"], stats["edges_loaded"],
    min_weight, resolution, modularity
)
stats["modularity"] = modularity
```

**E nel print finale di `main()`:**
```python
print(f"Modularity:          {stats.get('modularity', 'N/A')}")
```

**Target modularity:** > 0.4 indica community structure significativa.

---

## FASE 3: API Intelligence Layer (Backend)

**Obiettivo:** API restituisce dati actionable, non dump grezzo.

---

### 3.1 — Alzare min_edge_weight default API

**File:** `src/api/routers/stories.py`  
**Posizione:** endpoint `get_graph_network()` (riga ~53)

**Codice attuale:**
```python
min_edge_weight: float = Query(0.35, description="Min TF-IDF weighted Jaccard for global view (default: 0.35)")
```

**Codice nuovo:**
```python
min_edge_weight: float = Query(0.40, description="Min TF-IDF weighted Jaccard for global view (default: 0.40)")
```

---

### 3.2 — Arricchire GraphStats con metriche utili

**File:** `src/api/schemas/stories.py`

**Codice attuale:**
```python
class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    avg_momentum: float
```

**Codice nuovo:**
```python
class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    avg_momentum: float
    communities_count: int = 0
    avg_edges_per_node: float = 0.0
```

**File:** `src/api/routers/stories.py` — calcolo stats (riga ~128)

**Codice attuale:**
```python
graph = GraphNetwork(
    nodes=nodes,
    links=links,
    stats=GraphStats(
        total_nodes=len(nodes),
        total_edges=len(links),
        avg_momentum=avg_momentum,
    ),
)
```

**Codice nuovo:**
```python
# Compute community count from node data
community_ids = set(n.community_id for n in nodes if n.community_id is not None)
avg_epn = round(len(links) / len(nodes), 1) if nodes else 0.0

graph = GraphNetwork(
    nodes=nodes,
    links=links,
    stats=GraphStats(
        total_nodes=len(nodes),
        total_edges=len(links),
        avg_momentum=avg_momentum,
        communities_count=len(community_ids),
        avg_edges_per_node=avg_epn,
    ),
)
```

---

### 3.3 — Nuovo endpoint `/stories/communities`

**File:** `src/api/routers/stories.py` — aggiungere nuovo endpoint

**Nuovo schema in `src/api/schemas/stories.py`:**
```python
class CommunityInfo(BaseModel):
    """Summary of a graph community."""
    community_id: int
    size: int
    label: str  # Most frequent entity across storylines in this community
    top_storylines: list[StorylineNode] = []
    key_entities: list[str] = []
    avg_momentum: float = 0.0
```

**Nuovo endpoint:**
```python
@router.get("/communities")
async def list_communities(
    api_key: str = Depends(verify_api_key),
):
    """
    List all detected communities with their top storylines and key entities.
    Communities are sorted by size (largest first).
    """
    db = get_db()
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT community_id,
                           COUNT(*) AS size,
                           AVG(momentum_score) AS avg_momentum,
                           ARRAY_AGG(id ORDER BY momentum_score DESC) AS storyline_ids,
                           ARRAY_AGG(title ORDER BY momentum_score DESC) AS titles,
                           ARRAY_AGG(key_entities ORDER BY momentum_score DESC) AS all_entities
                    FROM storylines
                    WHERE narrative_status IN ('emerging', 'active')
                      AND community_id IS NOT NULL
                    GROUP BY community_id
                    ORDER BY COUNT(*) DESC
                """)
                rows = cur.fetchall()

        communities = []
        for r in rows:
            cid, size, avg_mom, sids, titles, all_ents = r

            # Aggregate entities across all storylines in community → pick top 10
            entity_counter = Counter()
            for ent_list in all_ents:
                if isinstance(ent_list, list):
                    entity_counter.update(e.lower() for e in ent_list)
                elif isinstance(ent_list, str):
                    try:
                        parsed = json.loads(ent_list)
                        if isinstance(parsed, list):
                            entity_counter.update(e.lower() for e in parsed)
                    except (json.JSONDecodeError, TypeError):
                        pass
            top_entities = [e for e, _ in entity_counter.most_common(10)]
            label = top_entities[0].title() if top_entities else f"Community {cid}"

            # Top 5 storylines by momentum
            top_storylines = [
                StorylineNode(
                    id=sids[i], title=titles[i],
                    narrative_status="active", momentum_score=0,
                    article_count=0,
                )
                for i in range(min(5, len(sids)))
            ]

            communities.append({
                "community_id": cid,
                "size": size,
                "label": label,
                "top_storylines": [s.model_dump() for s in top_storylines],
                "key_entities": top_entities,
                "avg_momentum": round(avg_mom or 0, 3),
            })

        return {
            "success": True,
            "data": {"communities": communities, "total": len(communities)},
            "generated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Communities error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

**Nota:** Aggiungere `from collections import Counter` in cima al file se non presente.

---

### 3.4 — Aggiungere parametro `min_momentum` al graph endpoint

**File:** `src/api/routers/stories.py` — endpoint `get_graph_network()` (riga ~53)

**Aggiungere parametro:**
```python
@router.get("/graph")
async def get_graph_network(
    min_edge_weight: float = Query(0.40, description="Min TF-IDF weighted Jaccard"),
    min_momentum: float = Query(0.0, description="Exclude nodes below this momentum"),
    api_key: str = Depends(verify_api_key),
):
```

**Aggiungere filtro dopo la costruzione dei nodi:**
```python
# Filter by momentum threshold
if min_momentum > 0:
    nodes = [n for n in nodes if n.momentum_score >= min_momentum]
```

**Posizione:** dopo la riga che filtra nodi isolati (`nodes = [n for n in nodes if n.id in connected_ids]`).

---

## FASE 4: Frontend UX (Next.js)

**Obiettivo:** Grafo leggibile, navigabile, utile per analisi.

---

### 4.1 — Community legend dinamica

**File:** `web-platform/components/StorylineGraph/StorylineGraph.tsx`

**Rimuovere** la legenda statica Status Legend (righe ~250-265) e **sostituire** con una legenda dinamica basata sulle communities effettivamente presenti nei dati:

```tsx
// Compute community labels from graph data
const communityLabels = useMemo(() => {
  if (!graph) return [];
  const communityMap = new Map<number, { count: number; entities: Map<string, number> }>();

  for (const node of graph.nodes) {
    const cid = node.community_id;
    if (cid == null) continue;
    if (!communityMap.has(cid)) {
      communityMap.set(cid, { count: 0, entities: new Map() });
    }
    const entry = communityMap.get(cid)!;
    entry.count++;
    for (const e of node.key_entities.slice(0, 5)) {
      entry.entities.set(e, (entry.entities.get(e) || 0) + 1);
    }
  }

  return Array.from(communityMap.entries())
    .map(([cid, { count, entities }]) => {
      const topEntity = [...entities.entries()]
        .sort((a, b) => b[1] - a[1])[0]?.[0] || `Community ${cid}`;
      return { cid, label: topEntity, count };
    })
    .sort((a, b) => b.count - a.count)
    .slice(0, 10); // Show top 10 communities
}, [graph]);
```

**Legenda JSX (sostituire il blocco Status Legend):**
```tsx
<div className="absolute top-4 right-4 pointer-events-none">
  {!selectedId && (
    <div className="bg-[#0A1628]/80 backdrop-blur-sm border border-white/10 rounded px-4 py-3 max-h-[300px] overflow-y-auto">
      <div className="text-xs font-mono text-gray-500 mb-2 uppercase">Communities</div>
      <div className="space-y-1.5">
        {communityLabels.map(({ cid, label, count }) => (
          <div key={cid} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: COMMUNITY_COLORS[cid % COMMUNITY_COLORS.length] }}
            />
            <span className="text-xs font-mono text-gray-300 truncate max-w-[140px]">
              {label}
            </span>
            <span className="text-xs font-mono text-gray-600 ml-auto">{count}</span>
          </div>
        ))}
      </div>
    </div>
  )}
</div>
```

---

### 4.2 — HUD stats migliorati

**File:** `web-platform/components/StorylineGraph/StorylineGraph.tsx`  
**Posizione:** HUD Overlay block (righe ~234-248)

**Aggiungere dopo AVG MOMENTUM:**
```tsx
{graph?.stats && (
  <div className="space-y-1 text-xs font-mono text-gray-400">
    <div>NODES: <span className="text-white">{graph.stats.total_nodes}</span></div>
    <div>EDGES: <span className="text-white">{graph.stats.total_edges}</span></div>
    <div>COMMUNITIES: <span className="text-white">{graph.stats.communities_count || '—'}</span></div>
    <div>AVG MOMENTUM: <span className="text-white">{graph.stats.avg_momentum.toFixed(2)}</span></div>
    <div>EDGES/NODE: <span className="text-white">{graph.stats.avg_edges_per_node?.toFixed(1) || '—'}</span></div>
  </div>
)}
```

**Aggiornare TypeScript types:**

**File:** `web-platform/types/stories.ts`
```typescript
export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  avg_momentum: number;
  communities_count: number;       // NEW
  avg_edges_per_node: number;      // NEW
}
```

---

### 4.3 — Filtri interattivi (pannello collapsible)

**File:** Nuovo componente `web-platform/components/StorylineGraph/GraphFilters.tsx`

Pannello collapsible con:
- **Momentum slider**: filtra nodi con momentum sotto la soglia
- **Community toggle**: mostra/nascondi singole communities
- **Category dropdown**: filtra per categoria

```tsx
interface GraphFiltersProps {
  minMomentum: number;
  onMinMomentumChange: (value: number) => void;
  hiddenCommunities: Set<number>;
  onToggleCommunity: (cid: number) => void;
  communityLabels: Array<{ cid: number; label: string; count: number }>;
}
```

**Integrazione in StorylineGraph.tsx:**
- Aggiungere state: `const [minMomentum, setMinMomentum] = useState(0);`
- Aggiungere state: `const [hiddenCommunities, setHiddenCommunities] = useState<Set<number>>(new Set());`
- Filtrare `graphData` nel `useMemo` in base a questi filtri
- Renderizzare `<GraphFilters>` nel layout

---

### 4.4 — Community label nel canvas

**File:** `web-platform/components/StorylineGraph/StorylineGraph.tsx`

Aggiungere alla callback `paintNode` il rendering di label di community come testo grande semi-trasparente al centro del baricentro di ciascun cluster. Questo richiede:

1. Calcolare il baricentro (media x, y) dei nodi per ogni community
2. Renderizzare un `ctx.fillText()` con font size grande e alpha bassa

```tsx
// In un useMemo separato, calcola i baricentri:
const communityCentroids = useMemo(() => {
  const groups = new Map<number, { xs: number[]; ys: number[] }>();
  for (const node of graphData.nodes) {
    const cid = (node as GraphNode).community_id;
    if (cid == null || node.x == null || node.y == null) continue;
    if (!groups.has(cid)) groups.set(cid, { xs: [], ys: [] });
    groups.get(cid)!.xs.push(node.x);
    groups.get(cid)!.ys.push(node.y);
  }
  return new Map(
    [...groups.entries()]
      .filter(([, g]) => g.xs.length >= 3) // Only label communities with 3+ visible nodes
      .map(([cid, g]) => [cid, {
        x: g.xs.reduce((a, b) => a + b, 0) / g.xs.length,
        y: g.ys.reduce((a, b) => a + b, 0) / g.ys.length,
        label: communityLabels.find(c => c.cid === cid)?.label || '',
      }])
  );
}, [graphData.nodes, communityLabels]);
```

**Renderizzare via `onRenderFramePost` callback di ForceGraph2D** per disegnare dopo i nodi.

---

### 4.5 — Migliorare il dossier panel

**File:** `web-platform/components/StorylineGraph/StorylineDossier.tsx`

Aggiungere sezione "Community" dopo "Key Entities":

```tsx
{/* Community Info */}
{detail.storyline.community_id != null && (
  <div className="border border-purple-500/20 bg-gray-800/50 p-4 rounded">
    <h3 className="text-sm font-mono text-purple-400 uppercase mb-3 flex items-center gap-2">
      <GitBranch size={14} />
      Community
    </h3>
    <div className="flex items-center gap-2">
      <div
        className="w-4 h-4 rounded-full"
        style={{ backgroundColor: COMMUNITY_COLORS[detail.storyline.community_id % COMMUNITY_COLORS.length] }}
      />
      <span className="text-sm text-gray-300 font-mono">
        Community #{detail.storyline.community_id}
        {/* TODO: resolve community label from /communities endpoint */}
      </span>
    </div>
  </div>
)}
```

---

## Metriche di Successo

| Metrica | Prima | Target Post-Refactoring |
|---------|-------|-------------------------|
| Nodi attivi (emerging+active) | ~1262 | 200–400 |
| Archi totali DB | 75,631 | 2,000–5,000 |
| Archi visibili frontend (default) | ? | 200–600 |
| Communities Louvain | 406 | 15–30 |
| Modularity | sconosciuta | > 0.4 |
| Archi/nodo media | ~60 | 5–15 |
| Nodi/community media | ~3 | 8–25 |
| Community detection runtime | TIMEOUT | < 30s |
| Archi duplicati bidirezionali | Sì | No |

---

## Ordine di Deploy

```
1. Applicare migration 016 (cleanup DB) via psql
2. Deploy codice Fase 1 (narrative_processor.py modifiche)
3. Deploy codice Fase 2 (compute_communities.py + daily_pipeline.py)
4. Run manuale: python scripts/compute_communities.py --min-weight 0.25
5. Verificare metriche: nodi, archi, communities, modularity
6. Deploy codice Fase 3 (API changes)
7. Deploy codice Fase 4 (Frontend changes)
8. Verificare su web: grafo leggibile, communities colorate, filtri funzionanti
```

---

## Rischi e Mitigazioni

| Rischio | Mitigazione |
|---------|-------------|
| Threshold 0.20 troppo aggressiva → grafo troppo sparso | Testare con 0.15 prima, poi alzare. La migration 016 è reversibile (re-run pipeline per rigenerare archi). |
| Decay 5 giorni troppo aggressivo → perdita storylines valide | Le storylines con ≥3 articoli non sono toccate. Solo emerging con <3 articoli dopo 5 giorni senza match. |
| Louvain resolution 0.8 → communities troppo grandi | Parametro è tunable: provare 0.6, 0.8, 1.0 e confrontare modularity. |
| Archi bidirezionali → logica di deduplica complessa | Testare con dry-run prima. La deduplica SQL nella migration 016 è safe (tiene il peso maggiore). |
| Community labels imprecise | Il label è "entità più frequente" — funziona per communities tematiche ma potrebbe essere generico ("USA"). Valutare di escludere top-5 entità globali dal labeling. |
