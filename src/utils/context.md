# Utils Context

## Purpose
Shared utility functions used across the entire INTELLIGENCE_ITA system. Provides centralized logging configuration and domain-specific query preprocessing for semantic search optimization.

## Architecture Role
Foundation layer that supports all other modules. The logger ensures consistent logging format across the pipeline, while the stopwords module improves RAG retrieval accuracy by filtering out intelligence-domain noise from user queries.

## Key Files

- `logger.py` - Centralized logging configuration
  - `setup_logger(name, log_level)` - Creates configured logger with console output
  - `get_logger(name)` - Retrieves or creates logger instance
  - Format: `YYYY-MM-DD HH:MM:SS - name - LEVEL - message`
  - Respects `LOG_LEVEL` environment variable (default: INFO)

- `stopwords.py` - Intelligence domain query cleaning
  - `INTELLIGENCE_STOPWORDS` - Set of domain terms to filter (e.g., "report", "briefing", "analysis")
  - `PRESERVE_TERMS` - Critical terms to keep (e.g., "nato", "cyber", "sanctions")
  - `QueryCleaner` class - Uses spaCy NER to preserve named entities while removing noise
  - `clean_query(query)` - Main function to clean user queries for semantic search
  - Example: `"latest intelligence report Taiwan"` → `"Taiwan"`

## Dependencies

- **Internal**: None (base layer)
- **External**:
  - `logging` (stdlib)
  - `spacy` - For NER-based entity preservation in query cleaning
  - `xx_ent_wiki_sm` - spaCy multilingual NER model

## Data Flow

- **Input**:
  - Logger name and optional log level
  - Raw user search queries
- **Output**:
  - Configured `logging.Logger` instances
  - Cleaned queries with stopwords removed but entities preserved
