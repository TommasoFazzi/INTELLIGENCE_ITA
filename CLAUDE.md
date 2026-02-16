# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

INTELLIGENCE_ITA is an end-to-end geopolitical intelligence news analysis platform. It ingests 33+ RSS feeds, processes articles with NLP (spaCy + sentence-transformers), stores them in PostgreSQL with pgvector for semantic search, generates intelligence reports via Google Gemini LLM with RAG, and provides a human-in-the-loop review dashboard. A Next.js web platform serves as the public-facing frontend.

## Common Commands

### Python Backend

```bash
# All commands run from the inner INTELLIGENCE_ITA/ directory (where src/, scripts/, requirements.txt live)

# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_ingestion/ -v
pytest tests/test_nlp/ -v
pytest tests/test_storage/ -v
pytest tests/test_llm/ -v

# Run a single test file
pytest tests/test_llm/test_report_generator.py -v

# Run a single test function
pytest tests/test_llm/test_report_generator.py::test_function_name -v

# Run by marker
pytest -m unit
pytest -m "not slow"

# Coverage
pytest tests/ --cov=src --cov-report=html

# Linting (tools are in requirements-dev.txt, some commented out)
black src/ scripts/
flake8 src/ scripts/ --max-line-length=120
ruff check src/

# Run pipeline steps individually
python -m src.ingestion.pipeline          # 1. Ingest RSS feeds
python scripts/process_nlp.py             # 2. NLP processing
python scripts/load_to_database.py        # 3. Load to database
python scripts/generate_report.py         # 4. Generate LLM report

# Full automated pipeline
python scripts/daily_pipeline.py

# Report generation with options
python scripts/generate_report.py --days 3 --macro-first --skip-article-signals

# HITL Dashboard (Streamlit)
streamlit run Home.py

# FastAPI backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# System health check
python scripts/check_setup.py
```

### Next.js Frontend (web-platform/)

```bash
cd web-platform
npm install
npm run dev       # Dev server at http://localhost:3000
npm run build     # Production build
npm run lint      # ESLint
```

## Architecture

### Data Pipeline Flow

```
RSS Feeds (33) → Ingestion → NLP Processing → PostgreSQL+pgvector → RAG+LLM → HITL Review
```

**Six phases:**
1. **Ingestion** (`src/ingestion/`): Async RSS parsing via aiohttp (parallel feed fetching + concurrent content extraction), full-text extraction (Trafilatura primary, Newspaper3k fallback, Cloudscraper for anti-bot sites), 2-phase deduplication (hash + content)
2. **NLP** (`src/nlp/`): spaCy multilingual NER (`xx_ent_wiki_sm`), semantic chunking (500-word sliding window), embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim)
3. **Storage** (`src/storage/database.py`): PostgreSQL + pgvector with HNSW indexing, connection pooling (psycopg2 SimpleConnectionPool)
4. **Report Generation** (`src/llm/`): Google Gemini 2.5 Flash, 2-stage RAG (vector search → cross-encoder reranking with ms-marco-MiniLM), trade signal extraction with ticker mapping, Pydantic schema validation
5. **HITL** (`src/hitl/`, `Home.py`): Streamlit dashboard for review, editing, rating, feedback loop
6. **Automation** (`scripts/daily_pipeline.py`): launchd (macOS) / systemd (Linux) scheduling

### Key Modules by Size/Complexity

- `src/llm/report_generator.py` (~2575 lines) — Core LLM integration, RAG pipeline, trade signals
- `src/storage/database.py` (~1921 lines) — All PostgreSQL/pgvector operations
- `src/nlp/story_manager.py` (~970 lines) — Cross-article narrative clustering
- `src/nlp/processing.py` (~610 lines) — NLP pipeline: cleaning, chunking, NER, embeddings
- `src/integrations/openbb_service.py` (~1026 lines) — OpenBB financial data integration
- `src/llm/oracle_engine.py` (~566 lines) — Intelligence scoring engine

### Web Platform (Next.js)

Located in `web-platform/`. Uses Next.js 16 App Router, React 19, Tailwind CSS 4, Shadcn/ui (Radix), Mapbox GL for intelligence map, SWR for data fetching, Framer Motion for animations.

**Routes:** `/` (landing), `/dashboard` (reports list), `/dashboard/report/[id]` (detail), `/map` (geospatial intelligence map)

**API communication:** Frontend → FastAPI backend (`src/api/main.py`) with `X-API-Key` header authentication.

## Configuration

- `config/feeds.yaml` — 33 RSS feed definitions with categories (breaking_news, intelligence, tech_economy, etc.)
- `config/top_50_tickers.yaml` — Geopolitical market movers with aliases for NER matching
- `config/entity_blocklist.yaml` — Noise filtering for extracted entities
- `.env` — Database URL, API keys (Gemini, FRED), app settings (see `.env.example`)
- `migrations/` — 11 incremental SQL migration files (001–011), applied manually via `psql` or through `load_to_database.py --init-only`

## Key Technical Patterns

- **RAG with reranking:** Vector search retrieves top-20, cross-encoder reranks to top-10 for ~15-20% precision improvement
- **Trade signal pipeline:** Macro-first approach — report → context condensation → structured signal extraction → Pydantic validation (BULLISH/BEARISH/NEUTRAL/WATCHLIST)
- **Async ingestion:** Single `asyncio.run()` in `pipeline.run()` orchestrates both feed parsing (aiohttp + `TCPConnector(limit=20, limit_per_host=3)`) and content extraction (`asyncio.Semaphore(10)` + `asyncio.to_thread()` for sync libraries). Sync wrappers (`parse_all_feeds`, `extract_batch`) kept for standalone use only.
- **Deduplication:** 2-phase — in-memory hash(link+title) then database content hash, reducing articles by 20-25%
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` for cross-language semantic similarity (Italian + English sources)
- **Schema validation:** Pydantic v2 models in `src/llm/schemas.py` for all LLM structured output

## Testing

Pytest markers defined in `pytest.ini`: `unit`, `integration`, `e2e`, `slow`. Tests mirror `src/` structure under `tests/`. Mock HTTP with `responses` library, mock datetime with `freezegun`. Async methods tested with `AsyncMock`; use `pytest-asyncio` for async test support.

## Environment Requirements

- Python 3.9+ (developed on 3.12.3)
- PostgreSQL 14+ with pgvector extension
- Node.js 16+ (for web-platform)
- spaCy model: `python -m spacy download xx_ent_wiki_sm`
- Required env vars: `DATABASE_URL`, `GEMINI_API_KEY`
