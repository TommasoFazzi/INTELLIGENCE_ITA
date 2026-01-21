# Scripts Context

## Purpose
Automation and utility scripts for pipeline execution, data management, and maintenance tasks. Provides CLI tools for running the intelligence pipeline, backfilling data, cleaning entities, and generating reports.

## Architecture Role
Operational layer that orchestrates the core modules. Scripts tie together ingestion → NLP → database → LLM report generation. Used for both manual execution and automated scheduling (cron).

## Key Files

### Setup & Verification
- `check_setup.py` - Verify system configuration (Python, env, DB, spaCy, models)

### Pipeline Execution
- `process_nlp.py` - Run NLP processing on ingested articles
- `load_to_database.py` - Load processed articles to PostgreSQL
- `generate_report.py` - Generate daily intelligence reports
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

### Storylines
- `batch_storyline_clustering.py` - Run DBSCAN clustering for storylines
- `test_storyline_clustering.py` - Test storyline clustering

### Quality Auditing
- `audit_entity_quality.py` - Audit entity data quality

### Ticker Management
- `seed_tickers.py` - Seed ticker whitelist to database

### Dashboard
- `run_dashboard.sh` - Launch Streamlit dashboard
- `run_weekly_report.sh` - Cron script for weekly reports

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
# Full pipeline
python -m src.ingestion.pipeline
python scripts/process_nlp.py
python scripts/load_to_database.py
python scripts/generate_report.py --macro-first

# Weekly report
python scripts/generate_weekly_report.py

# Entity maintenance
python scripts/clean_entities.py
python scripts/geocode_entities.py

# Check system
python scripts/check_setup.py
```
