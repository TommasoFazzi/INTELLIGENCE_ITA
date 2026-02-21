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

- `narrative_processor.py` - **Narrative Engine** (~1072 lines)
  - `NarrativeProcessor` class - Full storyline lifecycle
  - **Key tunable constants:**
    - `MICRO_CLUSTER_THRESHOLD = 0.90` — cosine sim threshold for near-duplicate grouping
    - `MATCH_THRESHOLD = 0.75` — min hybrid score to match an event to an existing storyline
    - `TIME_DECAY_FACTOR = 0.05` — score penalty per day of storyline inactivity
    - `ENTITY_BOOST = 0.10` — bonus when entity Jaccard >= 0.3
    - `ENTITY_JACCARD_THRESHOLD = 0.30` — min Jaccard for entity boost and graph edges
    - `HDBSCAN_MIN_CLUSTER_SIZE = 2`, `HDBSCAN_MIN_SAMPLES = 2`
    - `DRIFT_WEIGHT_OLD = 0.85`, `DRIFT_WEIGHT_NEW = 0.15` — embedding drift weights
    - `MOMENTUM_DECAY_FACTOR = 0.7` — weekly decay multiplier for inactive storylines
    - `LLM_RATE_LIMIT_SECONDS = 0.5` — pause between Gemini calls
  - **Public interface:**
    - `process_daily_batch(days, dry_run)` — Main orchestrator: micro-clustering → matching → HDBSCAN discovery → LLM evolution → **post-clustering validation** → graph → decay
  - **Stage 1 — Micro-clustering:**
    - `_create_micro_clusters(articles)` — Groups near-duplicate articles (cosine sim > 0.90) into unique events using greedy clustering; returns list of event dicts with centroid embedding, merged entities, article_ids
    - `_article_to_event(article)` — Converts a single article to an event dict
    - `_articles_to_event(articles)` — Merges multiple articles into a single event (centroid embedding, union of entities, representative title from article with most entities)
  - **Stage 2 — Adaptive matching:**
    - `_load_active_storylines()` — Loads storylines with `narrative_status IN ('emerging', 'active')` ordered by momentum_score DESC
    - `_find_best_match(event, active_storylines)` — Hybrid score = `cosine_sim - time_decay_penalty + entity_boost`; returns best match above MATCH_THRESHOLD
    - `_assign_event_to_storyline(event, storyline_id)` — Links articles, applies embedding drift (85%/15%), merges entities (cap 20), bumps momentum, promotes emerging → active at article_count >= 3
  - **Stage 3 — HDBSCAN discovery:**
    - `_cluster_residuals(orphaned_events)` — Applies HDBSCAN (metric='euclidean' on unit vectors) to orphaned events; noise points become individual storylines; returns list of created storyline IDs
    - `_create_storyline_from_events(events)` — Creates a new storyline record from one or more events, initial `narrative_status='emerging'`, `momentum_score=0.5`
  - **Stage 4 — LLM summary evolution:**
    - `_evolve_narrative_summary(storyline_id)` — Calls Gemini 2.5 Flash; new storylines get title+summary from scratch, existing ones integrate new facts while preserving historical context; also encodes `summary_vector` via sentence-transformers
  - **Stage 4b — Post-clustering validation (Filtro 4):**
    - `_validate_storyline_relevance(storyline_ids)` — Archives storylines with no scope keywords AND matching off-topic patterns; runs after LLM evolution (so title+summary are available); sets `narrative_status='archived'`, `status='ARCHIVED'`
  - **Stage 5 — Graph builder:**
    - `_update_graph_connections(storyline_id)` — UPSERTs edges in `storyline_edges` where Jaccard >= 0.30; removes edges that drop below threshold; also updates `last_graph_update`
  - **Stage 6 — Decay:**
    - `_apply_decay()` — 4 rules applied each run:
      1. momentum *= 0.7 for emerging/active not updated in 7 days
      2. active + momentum < 0.3 → stabilized
      3. stabilized + no update for 30 days → archived
      4. emerging + article_count < 3 + older than 14 days → archived
  - **Helper:**
    - `_extract_entity_list(entities_json)` — Handles both new format (`clean.all`) and old format (`by_type.GPE/ORG/PERSON`)
  - **Module-level constants:** `_SCOPE_KEYWORDS` (compiled regex with geopolitical terms), `_OFF_TOPIC_PATTERNS` (list of compiled regexes for sports/entertainment/celebrity/food/tourism)

- `relevance_filter.py` - **LLM Relevance Classification** (Filtro 2)
  - `RelevanceFilter` class — initialized with `GEMINI_API_KEY`, uses `gemini-2.0-flash`
  - `classify_article(article)` — Returns `True` (relevant) or `False` (not relevant); on LLM error defaults to `True` (conservative)
  - `filter_batch(articles)` — Classifies a batch, returns `(relevant_articles, filtered_out_articles)` tuple; tags articles with `relevance_label` field; rate-limited at 0.15s between calls
  - `CLASSIFICATION_PROMPT` — Italian-language system prompt with scope definition
  - `SCOPE_DESCRIPTION` / `OUT_OF_SCOPE` — Platform scope boundaries
  - Conservative: borderline cases → RELEVANT (prefer false positives over missing intelligence)

- `story_manager.py` - Legacy narrative engine (~970 lines)
  - `StoryManager` class - Sequential storyline matching (superseded by `narrative_processor.py`)
  - `BatchClusterer` class - DBSCAN-based batch clustering

## 3-Layer Content Filtering

| Layer | File | Stage | Method |
|-------|------|-------|--------|
| Filtro 1 | `src/ingestion/pipeline.py` | Ingestion | Keyword blocklist (sports/entertainment/food) |
| Filtro 2 | `relevance_filter.py` | NLP processing | LLM classification (Gemini 2.0 Flash) |
| Filtro 4 | `narrative_processor.py` | Post-clustering | Regex scope keywords + off-topic patterns → archive |

**Filtro 4 logic (two conditions must both be true to archive):**
1. Title+summary contains NO match in `_SCOPE_KEYWORDS` (geopolitical terms, key countries, agencies, etc.)
2. Title+summary matches at least one `_OFF_TOPIC_PATTERNS` (sports leagues, entertainment awards, celebrity, food/travel)

If a storyline has no summary yet (LLM not yet run), it passes validation and is checked again on the next run.

## Dependencies

- **Internal**: `src/storage/database`, `src/utils/logger`
- **External**:
  - `spacy` + `xx_ent_wiki_sm` - NER, tokenization
  - `sentence-transformers` - Embeddings (384-dim) and `summary_vector` encoding
  - `sklearn.cluster.HDBSCAN` (scikit-learn >= 1.3) - Density-based clustering; gracefully degrades if unavailable
  - `numpy`, `scikit-learn` - Vector operations
  - `google-generativeai` - Gemini for LLM title generation, summary evolution (`gemini-2.5-flash`), relevance classification (`gemini-2.0-flash`)

## Data Flow

- **Input**: JSON articles from `data/articles_{timestamp}.json`
- **Output**:
  - Enriched articles with `nlp_data` (chunks, entities, embeddings)
  - Storylines in `storylines` table with evolved summaries and momentum
  - Graph edges in `storyline_edges` table (Jaccard weights)
  - Article-storyline links in `article_storylines` junction table

## Known Gotchas

- **HDBSCAN import**: Uses `sklearn.cluster.HDBSCAN` (added in scikit-learn 1.3), not the standalone `hdbscan` package. Falls back to individual storyline creation if unavailable.
- **Embedding model is lazy-loaded**: `NarrativeProcessor.embedding_model` property loads SentenceTransformer on first access. This avoids slow startup when LLM-only features are used.
- **`skip_llm` flag**: If `NarrativeProcessor(skip_llm=True)`, LLM summary evolution is disabled but all other stages run. Filtro 4 still runs (without summaries, new storylines with no scope keywords in the title but also no off-topic match will pass).
- **Momentum bump per article**: `min(1.0, 0.1 * len(event['article_ids']))` — capped at 1.0 total.
- **Entities capped at 20**: `_assign_event_to_storyline` merges entities with a hard cap at 20 items; `_create_storyline_from_events` uses top 15 by frequency.
- **`summary_vector` vs `current_embedding`**: `current_embedding` drifts with new article embeddings (semantic drift tracking); `summary_vector` is re-encoded from the LLM-generated summary text each evolution cycle.
