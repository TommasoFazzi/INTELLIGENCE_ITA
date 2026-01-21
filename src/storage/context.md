# Storage Context

## Purpose
PostgreSQL database layer with pgvector extension for the RAG (Retrieval-Augmented Generation) system. Handles connection pooling, schema initialization, article/chunk storage, vector similarity search, and HITL (Human-in-the-Loop) report management.

## Architecture Role
Central persistence layer between the processing pipeline and intelligence generation. All NLP-processed articles flow here for storage, and the LLM modules retrieve context via semantic search. Also stores generated reports and human feedback for the feedback loop.

## Key Files

- `database.py` - `DatabaseManager` class (1900+ lines)

  **Connection Management:**
  - `SimpleConnectionPool` (min=1, max=10 connections)
  - `get_connection()` context manager with auto-commit/rollback
  - `register_vector()` for pgvector type handling

  **Schema Initialization (`init_db()`):**
  - `articles` table - Full articles with NLP metadata, embeddings (384-dim)
  - `chunks` table - RAG chunks with embeddings for semantic search
  - `reports` table - LLM-generated intelligence reports
  - `report_feedback` table - Human corrections and ratings
  - `entities` table - Named entities with geocoding (for Intelligence Map)
  - HNSW indexes for fast approximate nearest neighbor search

  **Core Operations:**
  - `save_article()` / `batch_save()` - Store articles with content-hash deduplication
  - `semantic_search()` - Vector similarity search on chunks with filters
  - `full_text_search()` - PostgreSQL `ts_query` for keyword search
  - `hybrid_search()` - Combines vector + keyword with RRF fusion
  - `save_report()` / `update_report()` - Report lifecycle management
  - `save_feedback()` / `get_report_feedback()` - HITL feedback storage

  **Specialized Methods:**
  - `get_all_article_embeddings()` - For batch storyline clustering
  - `get_entities_with_coordinates()` - GeoJSON output for map
  - `semantic_search_reports()` - Search reports by embedding (for Oracle)
  - `get_reports_by_date_range()` - For weekly meta-analysis

## Dependencies

- **Internal**: `src/utils/logger`
- **External**:
  - `psycopg2` - PostgreSQL adapter
  - `psycopg2.pool.SimpleConnectionPool` - Connection pooling
  - `pgvector.psycopg2` - Vector type registration
  - `psycopg2.extras.Json` - JSONB handling

## Data Flow

- **Input**:
  - Processed articles with NLP data and embeddings from `src/nlp/`
  - Query embeddings for semantic search from `src/llm/`
  - Generated reports from `src/llm/report_generator.py`
  - Human feedback from `src/hitl/`

- **Output**:
  - Retrieved chunks for RAG context
  - Article metadata and statistics
  - Reports for review/editing
  - GeoJSON entities for Intelligence Map
  - Feedback data for prompt improvement

## Key Tables

| Table | Purpose |
|-------|---------|
| `articles` | Full articles, embeddings, NLP metadata, entities (JSONB) |
| `chunks` | 500-word chunks with 384-dim embeddings for RAG |
| `reports` | Generated intelligence reports (draft/final/status) |
| `report_feedback` | Human corrections, ratings, comments |
| `entities` | Named entities with coordinates for map |
| `storylines` | Narrative threads grouping related articles |
| `market_data` | OHLCV time series from Yahoo Finance |
| `ticker_mappings` | Entity → Stock ticker mappings |
