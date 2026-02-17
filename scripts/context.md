# Scripts Context

## Purpose
Automation and utility scripts for pipeline execution, data management, and maintenance tasks. Provides CLI tools for running the intelligence pipeline, backfilling data, cleaning entities, and generating reports.

## Architecture Role
Operational layer that orchestrates the core modules. Scripts tie together ingestion → NLP → database → LLM report generation.

**Primary orchestrator**: `daily_pipeline.py` executes 6 steps sequentially (+ conditional weekly/monthly) with logging, error handling, and configurable fail-fast behavior. Supports manual execution and automated scheduling via launchd (macOS, 8:00 AM daily).

## Key Files

### Setup & Verification
- `check_setup.py` - Verify system configuration (Python, env, DB, spaCy, models)

### Pipeline Execution
- `daily_pipeline.py` - **Orchestrator**: runs full pipeline in one command
  - Steps: 1.ingestion → 2.market_data → 3.nlp_processing → 4.load_to_database → **5.narrative_processing** → 6.generate_report → (7.weekly → 8.monthly)
  - `--dry-run` - Validate without executing
  - `--step N` - Run only step N (1-6)
  - `--from-step N` - Start from step N
  - `--verbose` - Enable DEBUG logging
  - `--skip-weekly` - Skip weekly/monthly reports
  - **Auto weekly**: Runs on Sundays
  - **Auto monthly**: Runs after 4 weekly reports
  - **narrative_processing** has `continue_on_failure=True` (report generated even if storylines fail)
- `process_nlp.py` - Run NLP processing on ingested articles (includes Filtro 2: LLM relevance)
- `process_narratives.py` - **Narrative Engine CLI**: runs storyline clustering, matching, LLM evolution, graph updates
  - `--days N` - Look back N days for unassigned articles
  - `--dry-run` - Validate without DB writes
  - `--verbose` - Enable DEBUG logging
- `load_to_database.py` - Load processed articles to PostgreSQL
- `generate_report.py` - Generate daily intelligence reports (now includes Storyline Tracker section)
  - `--macro-first` flag for serialized pipeline with trade signals
- `generate_weekly_report.py` - Generate weekly aggregated meta-analysis
- `generate_recap_report.py` - Generate recap reports for date ranges

### Market Data
- `backfill_market_data.py` - Backfill Yahoo Finance OHLCV data
- `fetch_daily_market_data.py` - Daily market data fetch

### Entity Management
- `extract_entities.py` - Run NER extraction on articles
- `backfill_entities.py` - Backfill entity data for older articles
- `clean_entities.py` - Clean garbage entities using blocklist
- `deep_clean_entities.py` - Deep deduplication of entities
- `add_sample_entities.py` - Load sample entities for testing

### Geocoding
- `geocode_entities.py` - Geocode entities with coordinates
- `geocode_batch.py` - Batch geocoding for efficiency
- `clean_geocoding.py` - Clean invalid geocoding data

### Embeddings & Search
- `backfill_report_embeddings.py` - Generate embeddings for existing reports

### Storylines / Narrative Engine
- `process_narratives.py` - **Primary**: Run NarrativeProcessor daily batch (HDBSCAN + LLM evolution + graph)
- `batch_storyline_clustering.py` - Legacy: Run DBSCAN clustering for storylines
- `test_storyline_clustering.py` - Legacy: Test storyline clustering

### Quality Auditing
- `audit_entity_quality.py` - Audit entity data quality

### Ticker Management
- `seed_tickers.py` - Seed ticker whitelist to database

### Dashboard & Scheduling
- `run_dashboard.sh` - Launch Streamlit dashboard
- `run_weekly_report.sh` - Cron script for weekly reports
- `com.intelligence-ita.daily-pipeline.plist` - launchd config for 8:00 AM scheduling (macOS)

### Migrations
- `run_migration_003.py` - Run specific migration

## Dependencies

- **Internal**: All `src/` modules
- **External**: CLI tools (argparse), scheduling (cron)

## Data Flow

- **Input**:
  - `data/` - Ingested article JSON files
  - `config/` - YAML configurations
  - Database tables

- **Output**:
  - `reports/` - Generated intelligence reports
  - Updated database tables
  - Log files in `logs/`

## Common Usage

```bash
# Full pipeline (one command - recommended)
python scripts/daily_pipeline.py

# Full pipeline (step by step)
python -m src.ingestion.pipeline
python scripts/fetch_daily_market_data.py
python scripts/process_nlp.py
python scripts/load_to_database.py
python scripts/process_narratives.py --days 1
python scripts/generate_report.py --macro-first

# Dry run (validate only)
python scripts/daily_pipeline.py --dry-run

# Resume from specific step
python scripts/daily_pipeline.py --from-step 3

# Enable automatic scheduling (8:00 AM daily)
launchctl load ~/Library/LaunchAgents/com.intelligence-ita.daily-pipeline.plist

# Weekly report
python scripts/generate_weekly_report.py

# Entity maintenance
python scripts/clean_entities.py
python scripts/geocode_entities.py

# Check system
python scripts/check_setup.py
```
