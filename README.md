# INTELLIGENCE_ITA

End-to-end geopolitical intelligence platform: 33 RSS feeds → NLP → PostgreSQL/pgvector → Narrative Engine → RAG + LLM reports → FastAPI + Next.js frontend.

---

## Architecture

```
RSS Feeds (33 sources)
    │
    ▼
Ingestion ── async aiohttp (parallel) ── Trafilatura/Newspaper3k/Cloudscraper
    │        2-phase deduplication      Filtro 1: keyword blocklist
    │
    ▼
NLP Processing ── spaCy xx_ent_wiki_sm (NER) ── 384-dim embeddings
    │              Filtro 2: LLM relevance classification (Gemini 2.0 Flash)
    │
    ▼
PostgreSQL + pgvector (HNSW index)
    │
    ▼
Narrative Engine ─────────────────────────────────────────────────────────────┐
    │  Stage 1: Micro-clustering (cosine sim > 0.90)                          │
    │  Stage 2: Adaptive matching (hybrid score: cosine + entity boost - decay)│
    │  Stage 3: HDBSCAN discovery (orphan events → new storylines)            │
    │  Stage 4: LLM summary evolution (Gemini 2.5 Flash)                      │
    │  Stage 4b: Filtro 4 post-clustering validation (regex scope check)      │
    │  Stage 5: Jaccard entity-overlap graph edges                            │
    │  Stage 6: Momentum decay (weekly ×0.7)                                 │
    └──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
RAG + LLM Report Generation (Gemini 2.5 Flash)
    │  Stage 1: HNSW vector search → top-20 chunks
    │  Stage 2: Cross-encoder reranking (ms-marco-MiniLM) → top-10
    │  Narrative context: top-10 storylines injected as XML
    │  Report sections: Executive Summary, Key Developments, Trend Analysis,
    │                   Investment Implications, Strategic Storyline Tracker
    │
    ├──▶ Trade Signals (Macro-first pipeline)
    │        → Intelligence Scoring (0-100): LLM confidence - SMA200 penalty + PE score
    │
    └──▶ HITL Review (Streamlit dashboard)
             │
             ▼
         FastAPI backend (X-API-Key auth, slowapi rate limiting)
             │
             ▼
         Next.js 16 frontend
             ├── /dashboard  (reports list + detail)
             ├── /map        (Mapbox GL geospatial entity map)
             └── /stories    (react-force-graph-2d narrative network)
```

---

## Project Structure

```
INTELLIGENCE_ITA/                         ← inner working directory
├── config/
│   ├── feeds.yaml                        # 33 RSS feed definitions with categories
│   ├── top_50_tickers.yaml               # Geopolitical market movers whitelist
│   └── entity_blocklist.yaml             # Noise-filtering for extracted entities
├── src/
│   ├── ingestion/
│   │   ├── feed_parser.py                # Async RSS/Atom parser (aiohttp, parallel)
│   │   ├── content_extractor.py          # Full-text extraction (concurrent, semaphore)
│   │   └── pipeline.py                   # Orchestrated ingestion (Filtro 1 blocklist)
│   ├── nlp/
│   │   ├── processing.py                 # Text cleaning, NER, chunking, embeddings
│   │   ├── embeddings.py                 # Embedding generation (384-dim)
│   │   ├── narrative_processor.py        # Narrative Engine (~1072 lines)
│   │   ├── relevance_filter.py           # Filtro 2: LLM relevance classification
│   │   └── story_manager.py              # Legacy (superseded by narrative_processor)
│   ├── storage/
│   │   └── database.py                   # DatabaseManager (~1921 lines), pgvector ops
│   ├── llm/
│   │   ├── report_generator.py           # RAG pipeline + narrative context (~2700 lines)
│   │   ├── oracle_engine.py              # Interactive RAG Q&A chat engine
│   │   ├── query_analyzer.py             # Structured filter extraction from NL queries
│   │   └── schemas.py                    # Pydantic schemas for LLM structured output
│   ├── api/
│   │   ├── main.py                       # FastAPI app, CORS, rate limiter, map endpoints
│   │   ├── auth.py                       # X-API-Key auth (secrets.compare_digest)
│   │   ├── routers/
│   │   │   ├── dashboard.py              # Stats and KPIs
│   │   │   ├── reports.py                # Report list and detail
│   │   │   └── stories.py                # Storyline graph and detail
│   │   └── schemas/
│   │       ├── common.py                 # APIResponse[T], PaginationMeta
│   │       ├── dashboard.py              # DashboardStats Pydantic models
│   │       ├── reports.py                # Report Pydantic models
│   │       └── stories.py                # Storyline graph Pydantic models
│   ├── finance/
│   │   ├── scoring.py                    # Intelligence score calculation (0-100)
│   │   ├── validator.py                  # ValuationEngine: metrics aggregation
│   │   ├── types.py                      # TickerMetrics dataclass
│   │   └── constants.py                  # Score thresholds, sector benchmark map
│   ├── integrations/
│   │   ├── market_data.py                # Yahoo Finance (yfinance, OHLCV, SMA200)
│   │   └── openbb_service.py             # OpenBB v4: macro indicators + fundamentals
│   ├── hitl/
│   │   └── dashboard.py                  # Streamlit HITL review dashboard
│   └── utils/
│       └── logger.py                     # Centralized logging
├── scripts/
│   ├── daily_pipeline.py                 # Full automated pipeline orchestrator
│   ├── process_nlp.py                    # NLP processing step
│   ├── load_to_database.py               # DB load + schema init
│   ├── process_narratives.py             # Narrative Engine step
│   ├── generate_report.py                # LLM report generation
│   ├── pipeline_status_check.py          # Health check (launchd 9:00 AM)
│   ├── backfill_market_data.py           # Backfill Yahoo Finance history
│   └── check_setup.py                    # System prerequisites check
├── migrations/                           # 12+ incremental SQL files (applied manually)
├── web-platform/                         # Next.js 16 frontend
│   ├── app/
│   │   ├── dashboard/                    # Report list + detail pages
│   │   ├── map/                          # Geospatial entity map (Mapbox GL)
│   │   └── stories/                      # Narrative force-graph (react-force-graph-2d)
│   ├── components/
│   ├── Dockerfile                        # Standalone Next.js production image
│   └── package.json
├── deploy/
│   ├── nginx/conf.d/                     # nginx site configs (HTTP + HTTPS)
│   ├── systemd/                          # intelligence-pipeline.service/.timer, backup
│   ├── setup-firewall.sh                 # UFW firewall setup for Hetzner
│   └── backup-db.sh                      # PostgreSQL backup script
├── Home.py                               # Streamlit HITL entry point
├── Dockerfile                            # Python backend production image
├── docker-compose.yml                    # Four services: postgres, backend, frontend, nginx
├── requirements.txt
└── .env.example
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9+ (3.12 recommended) | Backend and pipeline |
| PostgreSQL | 14+ | With pgvector extension |
| Node.js | 16+ | Next.js frontend |
| Docker + Compose | any recent | Production deploy only |

**Required environment variables** (see `.env.example`):

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `INTELLIGENCE_API_KEY` | Yes (prod) | REST API shared secret |
| `FRED_API_KEY` | Optional | Federal Reserve Economic Data |
| `FMP_API_KEY` | Optional | Financial Modeling Prep |
| `ENVIRONMENT` | Optional | Set to `production` to enforce strict auth |
| `ALLOWED_ORIGINS` | Optional | CORS origins (default: localhost:3000) |

---

## Quick Start — Development

### Backend

```bash
# From INTELLIGENCE_ITA/ (inner directory)
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm

cp .env.example .env
# Edit .env: set DATABASE_URL and GEMINI_API_KEY

# Init database schema (first time only)
python scripts/load_to_database.py --init-only

# Verify setup
python scripts/check_setup.py

# Start FastAPI backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start HITL review dashboard
streamlit run Home.py
```

### Frontend

```bash
# From INTELLIGENCE_ITA/web-platform/
npm install
npm run dev    # http://localhost:3000
```

---

## Production Deploy — Docker

Four services: `postgres` (pgvector:pg17), `backend` (FastAPI), `frontend` (Next.js standalone), `nginx` (reverse proxy + TLS termination).

```bash
# From INTELLIGENCE_ITA/ (inner directory)
cp .env.example .env
# Edit .env: set all required vars including INTELLIGENCE_API_KEY

docker compose up -d

# Check health
docker compose ps
docker compose logs backend --tail 50
```

**Deploy utilities** (in `deploy/`):

```bash
# Hetzner firewall setup (run once on server)
bash deploy/setup-firewall.sh

# Manual database backup
bash deploy/backup-db.sh

# systemd timers (alternative to launchd on Linux)
# deploy/systemd/intelligence-pipeline.service + .timer
# deploy/systemd/intelligence-backup.service + .timer
```

---

## Pipeline

### Automated Daily Pipeline

```bash
python scripts/daily_pipeline.py
```

Six core steps, two conditional:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `ingestion.pipeline` | Async RSS ingestion + full-text extraction |
| 2 | `market_data` | Fetch macro indicators + ticker prices |
| 3 | `process_nlp.py` | NLP: NER, embeddings, relevance filter |
| 4 | `load_to_database.py` | Load enriched articles to PostgreSQL |
| 5 | `process_narratives.py` | Narrative Engine (continue on failure) |
| 6 | `generate_report.py` | RAG + LLM intelligence report |
| 7* | Weekly report | Sundays only |
| 8* | Monthly recap | After 4 weekly reports |

macOS scheduling via launchd: **8:00 AM** daily pipeline, **9:00 AM** status check (`pipeline_status_check.py`).

### Manual Step-by-Step

```bash
# 1. Ingest RSS feeds (last 24h)
python -m src.ingestion.pipeline

# 2. NLP processing
python scripts/process_nlp.py

# 3. Load to database
python scripts/load_to_database.py

# 4. Narrative Engine
python scripts/process_narratives.py

# 5. Generate report
python scripts/generate_report.py

# Options: --days 3, --macro-first, --skip-article-signals, --model gemini-2.5-flash
python scripts/generate_report.py --macro-first --days 3
```

---

## Key Features

### Narrative Engine

Tracks ongoing geopolitical storylines across articles using a 6-stage pipeline (`src/nlp/narrative_processor.py`):

1. **Micro-clustering** — groups near-duplicate articles (cosine sim > 0.90) into unique events
2. **Adaptive matching** — hybrid score (`cosine_sim - time_decay + entity_boost`) assigns events to existing storylines
3. **HDBSCAN discovery** — orphan events clustered into new storylines; noise points become individual threads
4. **LLM summary evolution** — Gemini 2.5 Flash generates/updates title + summary for each updated storyline
5. **Post-clustering validation (Filtro 4)** — archives storylines with no geopolitical scope keywords AND matching off-topic patterns
6. **Graph + decay** — Jaccard entity-overlap edges upserted in `storyline_edges`; momentum decays ×0.7 weekly

**Storyline lifecycle:** `emerging` (< 3 articles) → `active` → `stabilized` → `archived`

**Key tunable constants:**

| Constant | Default | Effect |
|----------|---------|--------|
| `MATCH_THRESHOLD` | 0.75 | Min hybrid score to match existing storyline |
| `MICRO_CLUSTER_THRESHOLD` | 0.90 | Cosine sim for near-duplicate grouping |
| `TIME_DECAY_FACTOR` | 0.05 | Score penalty per day of inactivity |
| `ENTITY_BOOST` | 0.10 | Bonus when entity Jaccard >= 0.30 |
| `MOMENTUM_DECAY_FACTOR` | 0.7 | Weekly decay multiplier |

### 3-Layer Content Filtering

Off-topic content rejected at three independent pipeline stages:

| Layer | Location | Method |
|-------|----------|--------|
| Filtro 1 | `src/ingestion/pipeline.py` | Keyword blocklist at ingestion (sports/entertainment/food) |
| Filtro 2 | `src/nlp/relevance_filter.py` | LLM classification: Gemini 2.0 Flash, conservative (borderline → RELEVANT) |
| Filtro 4 | `src/nlp/narrative_processor.py` | Post-clustering regex: must lack scope keywords AND match off-topic pattern to archive |

### RAG Pipeline

Two-stage retrieval for ~15–20% precision improvement over vector search alone:

- **Stage 1 (recall)** — HNSW approximate nearest neighbor on pgvector → top-20 chunks (~50 ms)
- **Stage 2 (precision)** — Cross-encoder reranking `cross-encoder/ms-marco-MiniLM-L-6-v2` → top-10 chunks (~3–4 s)

Narrative storyline context is injected as **XML** (top-10 active storylines by momentum) into the LLM prompt, enabling a dedicated **Strategic Storyline Tracker** report section.

### Trade Signals & Intelligence Scoring

**Macro-first pipeline** (`--macro-first` flag):

1. Generate macro report → condense context to ~500 tokens (vs. 5000+)
2. Extract report-level signals (high-conviction, multi-article synthesis)
3. Filter articles with whitelisted tickers
4. Extract article-level signals with macro alignment score
5. Persist to `trade_signals` table

**Signal schema:**

| Field | Type | Values |
|-------|------|--------|
| `ticker` | string | e.g. `LMT`, `TSM` |
| `signal` | enum | `BULLISH` / `BEARISH` / `NEUTRAL` / `WATCHLIST` |
| `timeframe` | enum | `SHORT_TERM` / `MEDIUM_TERM` / `LONG_TERM` |
| `confidence` | float | 0.0–1.0 |
| `alignment_score` | float | Alignment with macro narrative |

**Intelligence Score (0–100):**

```
base = llm_confidence × 100
     - SMA200 deviation penalty (0–40 pts, non-linear above 30%)
     + P/E valuation score (-20 to +10 pts)
```

Data sourced from Yahoo Finance (price, SMA200) and OpenBB v4 (P/E, sector, fundamentals).

### REST API

FastAPI backend at port 8000. All endpoints (except `/` and `/health`) require `X-API-Key` header authenticated with `secrets.compare_digest`. Rate limiting via slowapi.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API root |
| GET | `/health` | Health check |
| GET | `/api/v1/dashboard/stats` | Overview, articles, entities, quality KPIs |
| GET | `/api/v1/reports` | Paginated report list (filters: status, type, date range) |
| GET | `/api/v1/reports/{id}` | Full report: content, sources, feedback, metadata |
| GET | `/api/v1/stories/graph` | Full narrative graph (nodes + edges + stats) |
| GET | `/api/v1/stories` | Paginated storyline list |
| GET | `/api/v1/stories/{id}` | Storyline detail + related storylines + recent articles |
| GET | `/api/v1/map/entities` | GeoJSON FeatureCollection of geocoded entities |
| GET | `/api/v1/map/entities/{id}` | Single entity detail with related articles |

The frontend communicates through an internal proxy at `/api/proxy/[...path]` — no API key is exposed in the browser bundle.

### Web Frontend

Next.js 16 App Router, React 19, Tailwind CSS 4, Shadcn/ui (Radix), SWR, Framer Motion.

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/dashboard` | Intelligence reports list |
| `/dashboard/report/[id]` | Report detail view |
| `/map` | Geospatial entity map (Mapbox GL) |
| `/stories` | Narrative storyline force-graph (react-force-graph-2d) |

### HITL Review Dashboard

Streamlit multi-page dashboard (`streamlit run Home.py` → http://localhost:8501):

- **Daily Briefing**: view LLM draft, edit final version, star rating (1–5), save/approve workflow
- **Oracle RAG**: interactive Q&A over intelligence database (three search modes: factual, strategic, both)
- **Intelligence Scores**: scored articles and reports from the oracle engine

---

## Testing

```bash
# All tests
pytest tests/ -v

# By marker
pytest -m unit              # fast, no DB required
pytest -m integration       # requires live DB
pytest -m "not slow"

# Single file or function
pytest tests/test_llm/test_report_generator.py -v
pytest tests/test_llm/test_report_generator.py::test_fn -v

# Coverage
pytest tests/ --cov=src --cov-report=html

# Lint
black src/ scripts/
flake8 src/ scripts/ --max-line-length=120
ruff check src/
```

Markers defined in `pytest.ini`: `unit`, `integration`, `e2e`, `slow`. Mock HTTP with `responses`; mock datetime with `freezegun`; async tests with `pytest-asyncio`.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/feeds.yaml` | 33 RSS feed definitions with categories |
| `config/top_50_tickers.yaml` | Geopolitical market movers (aliases for NER matching) |
| `config/entity_blocklist.yaml` | Noisy entity suppression |
| `.env` | Runtime secrets and feature flags (see `.env.example`) |
| `migrations/` | 12+ incremental SQL files, applied via `psql` or `--init-only` |

---

## Feed Sources (33 active)

| Category | Count | Sources |
|----------|-------|---------|
| Breaking News | 1 | The Moscow Times |
| Intelligence & Geopolitics | 12 | ASEAN Beat, Asia Times, BleepingComputer, China Power, CyberScoop, Defense One, Diálogo Américas, ECFR, Foreign Affairs, POLITICO, Krebs on Security, The Diplomat, SpaceNews |
| Middle East & North Africa | 3 | Al Jazeera English, Middle East Eye, The Jerusalem Post |
| Defense & Military | 3 | Breaking Defense, War on the Rocks, Janes Defence Weekly |
| Think Tanks | 3 | CSIS, Council on Foreign Relations, Chatham House |
| Americas | 1 | Americas Quarterly |
| Africa | 2 | African Arguments, ISS Africa |
| Tech & Economy | 8 | Euronews Business, ECB Press Releases, ECB Monetary Policy, Il Sole 24 ORE, OilPrice, Ars Technica Policy, Supply Chain Dive, Semiconductor Engineering |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| NLP | spaCy `xx_ent_wiki_sm` |
| Clustering | scikit-learn HDBSCAN (≥ 1.3) |
| Vector DB | PostgreSQL 17 + pgvector (HNSW index) |
| Market Data | yfinance 0.2.66+ (curl_cffi), OpenBB v4, FRED |
| Backend | FastAPI + uvicorn + slowapi |
| Schema validation | Pydantic v2 |
| HITL dashboard | Streamlit |
| Frontend | Next.js 16, React 19, Tailwind CSS 4 |
| Frontend libs | react-force-graph-2d, Mapbox GL, Shadcn/ui, SWR, Framer Motion |
| Infrastructure | Docker, nginx, launchd (macOS), systemd (Linux) |

---

## Development Status

| Phase | Status | Key Deliverables |
|-------|--------|-----------------|
| 1 — Ingestion | Complete | 33 RSS feeds, async aiohttp, 2-phase dedup, keyword blocklist |
| 2 — NLP | Complete | spaCy NER, 384-dim embeddings, LLM relevance filter (Filtro 2) |
| 3 — Storage & RAG | Complete | pgvector HNSW, cross-encoder reranking (+15–20% precision) |
| 4 — LLM Reports | Complete | Gemini 2.5 Flash, macro-first pipeline, trade signals, Pydantic schemas |
| 5 — HITL | Complete | Streamlit dashboard, Oracle RAG, rating/feedback loop |
| 6 — Narrative Engine | Complete | HDBSCAN, LLM evolution, Jaccard graph, 3-layer filtering, momentum decay |
| 7 — API + Frontend | Complete | FastAPI (11 endpoints, auth, rate limiting) + Next.js (map, stories graph) |
| 8 — Automation + Deploy | Complete | Docker Compose, nginx, launchd (8AM), systemd timers, backup script |
| 9 — Financial Intelligence | Complete | Intelligence scoring (0–100), Yahoo Finance, OpenBB fundamentals |
| 10 — Advanced Analytics | Planned | Alert system, email automation, monitoring dashboard |

---

## Performance Notes

| Stage | Typical Duration |
|-------|-----------------|
| RSS ingestion (33 feeds, async) | 30–60 s |
| Full-text extraction (concurrent, semaphore 10) | 60–90 s |
| NLP processing per article batch | ~2–3 min total |
| RAG vector search (HNSW, top-20) | ~50 ms |
| Cross-encoder reranking (top-10) | ~3–4 s |
| Report generation (Gemini 2.5 Flash) | ~20–40 s |

---

**Status:** All phases 1–9 complete. Production-ready. Target: Hetzner CAX31 (8 GB ARM64).

**Last updated:** 2026-02-21
