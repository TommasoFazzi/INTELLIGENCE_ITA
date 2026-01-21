# Migrations Context

## Purpose
SQL migration scripts that evolve the PostgreSQL database schema over time. Each migration is numbered sequentially and includes a rollback script for safe reversibility. Migrations add new capabilities without losing existing data.

## Architecture Role
Schema evolution layer that extends the core database as new features are added. Migrations must be run in order before using new functionality. The base schema is in `src/storage/database.py`, while migrations add incremental enhancements.

## Key Files

### Phase 2: Deduplication
- `001_add_content_hash.sql` - Adds `content_hash` (MD5) column to detect duplicate articles by content
- `001_add_content_hash_rollback.sql` - Removes the column

### Phase 3: Reporting
- `002_add_report_type.sql` - Adds `report_type` column (daily/weekly) to reports table
- `002_add_report_type_rollback.sql` - Removes the column

### Intelligence Map
- `003_add_entity_coordinates.sql` - Adds `latitude`, `longitude`, `geo_status` to entities table
- `003_add_entity_coordinates_rollback.sql` - Removes coordinate columns

### Market Intelligence (Sprint 3)
- `004_add_market_intelligence_schema.sql` - Creates `ticker_mappings` and `market_data` tables, adds `ai_analysis` JSONB to articles
- `005_add_trade_signals.sql` - Adds trade signal storage and scoring
- `010_financial_intel_v2.sql` - Enhanced financial intelligence tables

### RAG Enhancements
- `006_add_report_embeddings.sql` - Adds `content_embedding` vector column to reports for semantic search
- `007_add_full_text_search.sql` - Adds `content_tsv` tsvector column and GIN index for full-text search

### Storylines (Narrative Engine)
- `008_add_storylines.sql` - Creates `storylines` and `article_storylines` tables for narrative tracking
  - Dual embedding approach (original + current) to prevent semantic drift
  - `momentum_score` for story activity decay
  - Views: `v_active_storylines`, `v_articles_with_storylines`
- `008_add_storylines_rollback.sql` - Drops storyline tables and views

### OpenBB Integration
- `009_add_openbb_schema.sql` - Schema for OpenBB financial data API integration

## Dependencies

- **Internal**: `src/storage/database.py` (base schema)
- **External**: PostgreSQL 14+, pgvector extension

## Data Flow

- **Input**: SQL commands executed via `psql` CLI
- **Output**: Modified database schema

## Execution Order

```
001 → 002 → 003 → 004 → 005 → 006 → 007 → 008 → 009 → 010
```

Run with:
```bash
psql -d intelligence_ita -f migrations/XXX_migration_name.sql
```

Rollback with:
```bash
psql -d intelligence_ita -f migrations/XXX_migration_name_rollback.sql
```
