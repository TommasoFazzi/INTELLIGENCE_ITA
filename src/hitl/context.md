# HITL Context

## Purpose
Human-in-the-Loop (HITL) interface for reviewing, correcting, and approving LLM-generated intelligence reports. Provides Streamlit utilities for the multi-page dashboard application.

## Architecture Role
Quality assurance layer between LLM report generation and final publication. Enables human analysts to review reports, provide corrections, and submit feedback that improves future LLM prompts. Phase 5 of the intelligence pipeline.

## Key Files

- `streamlit_utils.py` - Shared utilities for Streamlit MPA
  - `get_db_manager()` - Cached singleton DatabaseManager (prevents connection pool exhaustion)
  - `get_embedding_model()` - Cached SentenceTransformer for embedding queries
  - `load_ticker_whitelist()` - Load tickers from `config/top_50_tickers.yaml`
  - `detect_tickers_in_text(text)` - Find ticker mentions with word boundaries
  - Custom CSS styling functions
  - Path setup helpers for multi-page app

- `dashboard_legacy.py` - Original single-page dashboard (~26k lines)
  - Legacy implementation before multi-page architecture
  - Contains report review, feedback submission, statistics
  - Deprecated in favor of `pages/` structure

## Dependencies

- **Internal**: `src/storage/database`, `src/utils/logger`
- **External**:
  - `streamlit` - Web dashboard framework
  - `sentence-transformers` - Embedding model for queries
  - `pyyaml` - Config loading

## Data Flow

- **Input**:
  - Generated reports from `reports/` directory
  - User corrections and feedback
  - Search queries for RAG

- **Output**:
  - `report_feedback` table - Human corrections, ratings, comments
  - Updated report status (draft → reviewed → published)
  - Prompt improvement suggestions based on feedback patterns

## Related Files

- `Home.py` - Streamlit home page (project root)
- `pages/` - Multi-page dashboard structure (if exists)
- `scripts/run_dashboard.sh` - Dashboard launcher
