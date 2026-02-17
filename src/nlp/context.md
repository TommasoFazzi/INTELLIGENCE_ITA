# NLP Context

## Purpose
Natural Language Processing module that transforms raw article text into structured, searchable data for RAG (Retrieval-Augmented Generation). Handles text cleaning, chunking, entity extraction, embedding generation, **LLM relevance classification**, and **narrative storyline processing**.

## Architecture Role
Processing layer between ingestion and storage. Takes JSON output from `src/ingestion/`, applies NLP pipeline, and produces enriched articles ready for vector database storage. The **Narrative Engine** (`narrative_processor.py`) clusters related articles into ongoing storylines, evolves summaries via LLM, maintains a graph of inter-storyline relationships, and enforces content relevance via post-clustering validation.

## Key Files

- `processing.py` - Core NLP pipeline (~600 lines)
  - `NLPProcessor` class - Hybrid NLP processor
  - Text Processing: `clean_text()`, `create_chunks()`, `preprocess_text()`
  - Entity Extraction: `extract_entities()` - spaCy NER (GPE, ORG, PERSON, LOC)
  - Embeddings: `generate_embedding()`, `generate_chunk_embeddings()` - 384-dim (`paraphrase-multilingual-MiniLM-L12-v2`)
  - Batch Processing: `process_article()`, `process_batch()`

- `narrative_processor.py` - **Narrative Engine** (~940 lines)
  - `NarrativeProcessor` class - Full storyline lifecycle
  - `process_daily_batch(days, dry_run)` - Main entry: micro-clustering → matching → HDBSCAN discovery → LLM evolution → **post-clustering validation** → graph → decay
  - `_micro_cluster_events()` - Groups near-duplicate articles into events
  - `_match_events_to_storylines()` - Embedding + entity similarity matching (threshold 0.60)
  - `_discover_new_storylines()` - HDBSCAN clustering of orphan events (min_cluster_size=2, min_samples=1)
  - `_evolve_storyline_summaries()` - LLM-driven summary evolution via Gemini
  - `_update_graph_connections()` - Jaccard entity-overlap edges (threshold 0.10)
  - `_apply_decay()` - Momentum decay for inactive storylines (half-life: 5 days)
  - `_validate_storyline_relevance()` - **Post-clustering filter** (Filtro 4): archives storylines with no scope keywords AND matching off-topic patterns
  - Module-level constants: `_SCOPE_KEYWORDS` (compiled regex), `_OFF_TOPIC_PATTERNS` (list of compiled regexes)

- `relevance_filter.py` - **LLM Relevance Classification** (Filtro 2)
  - `classify_article(title, source, snippet)` - Gemini-based RELEVANT/NOT_RELEVANT classification
  - `CLASSIFICATION_PROMPT` - System prompt with scope definition
  - `SCOPE_DESCRIPTION` / `OUT_OF_SCOPE` - Platform scope boundaries
  - Conservative: borderline → RELEVANT (prefer false positives)

- `story_manager.py` - Legacy narrative engine (~970 lines)
  - `StoryManager` class - Sequential storyline matching (superseded by `narrative_processor.py`)
  - `BatchClusterer` class - DBSCAN-based batch clustering

## 3-Layer Content Filtering

| Layer | File | Stage | Method |
|-------|------|-------|--------|
| Filtro 1 | `src/ingestion/pipeline.py` | Ingestion | Keyword blocklist (sports/entertainment/food) |
| Filtro 2 | `relevance_filter.py` | NLP processing | LLM classification (Gemini) |
| Filtro 4 | `narrative_processor.py` | Post-clustering | Regex scope keywords + off-topic patterns |

## Dependencies

- **Internal**: `src/storage/database`, `src/utils/logger`
- **External**:
  - `spacy` + `xx_ent_wiki_sm` - NER, tokenization
  - `sentence-transformers` - Embeddings (384-dim)
  - `hdbscan` - Density-based clustering
  - `numpy`, `scikit-learn` - Vector operations
  - `google-generativeai` - Gemini for LLM title generation, summary evolution, relevance classification

## Data Flow

- **Input**: JSON articles from `data/articles_{timestamp}.json`
- **Output**:
  - Enriched articles with `nlp_data` (chunks, entities, embeddings)
  - Storylines in `storylines` table with evolved summaries and momentum
  - Graph edges in `storyline_edges` table (Jaccard weights)
  - Article-storyline links in `article_storylines` junction table
