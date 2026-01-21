# NLP Context

## Purpose
Natural Language Processing module that transforms raw article text into structured, searchable data for RAG (Retrieval-Augmented Generation). Handles text cleaning, chunking, entity extraction, and embedding generation. Also manages storylines for narrative tracking.

## Architecture Role
Processing layer between ingestion and storage. Takes JSON output from `src/ingestion/`, applies NLP pipeline, and produces enriched articles ready for vector database storage. The storyline system enables "delta reporting" by grouping related articles into ongoing narratives.

## Key Files

- `processing.py` - Core NLP pipeline (~600 lines)
  - `NLPProcessor` class - Hybrid NLP processor

  **Text Processing:**
  - `clean_text(text)` - Remove HTML, URLs, scraping artifacts, normalize whitespace
  - `create_chunks(text)` - Sentence-aware chunking (500 words + 50 overlap)
  - `preprocess_text(text)` - Tokenization, lemmatization, POS tagging via spaCy

  **Entity Extraction:**
  - `extract_entities(text)` - spaCy NER (GPE, ORG, PERSON, LOC)
  - Returns `{entities: [...], by_type: {GPE: [...], ORG: [...]}, entity_count: N}`

  **Embeddings:**
  - `generate_embedding(text)` - Full-text embedding (384-dim)
  - `generate_chunk_embeddings(chunks)` - Batch chunk embeddings
  - Model: `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages)

  **Article Processing:**
  - `process_article(article)` - Complete pipeline for single article
  - `process_batch(articles)` - Batch processing with progress
  - `get_processing_stats(articles)` - Aggregate statistics

- `story_manager.py` - Narrative engine (~970 lines)
  - `StoryManager` class - Sequential storyline matching
    - `find_matching_storylines()` - Hybrid vector + entity matching
    - `create_storyline()` - New storyline from seed article
    - `add_article_to_storyline()` - Link article + vector drift
    - `assign_article()` - Main entrypoint: match or create
    - `apply_decay()` - Weekly momentum decay for inactive storylines
    - Thresholds: similarity=0.72, entity_boost=0.08, drift=0.9/0.1

  - `BatchClusterer` class - DBSCAN-based clustering
    - `cluster_articles(days, dry_run)` - Batch clustering on embeddings
    - `_generate_cluster_title()` - LLM title generation (Gemini)
    - `_generate_title_fallback()` - Entity-based heuristic fallback
    - `_create_storyline_from_cluster()` - Create storyline from cluster
    - `reset_storylines()` - Delete all for fresh clustering
    - Parameters: eps=0.28 (cosine distance), min_samples=2

## Dependencies

- **Internal**: `src/storage/database`, `src/utils/logger`
- **External**:
  - `spacy` - NER, tokenization, sentence segmentation
  - `xx_ent_wiki_sm` - Multilingual spaCy model (50+ languages)
  - `sentence-transformers` - Embedding generation
  - `numpy` - Vector operations
  - `scikit-learn` - DBSCAN clustering (optional, for BatchClusterer)
  - `google-generativeai` - Gemini for LLM title generation (optional)

## Data Flow

- **Input**:
  - JSON articles from `data/articles_{timestamp}.json`
  - `full_content.text` field from extraction

- **Output**:
  - Enriched articles with `nlp_data`:
    - `clean_text` - Cleaned article text
    - `chunks` - List of {text, embedding, word_count, sentence_count}
    - `entities` - {entities: [...], by_type: {...}}
    - `full_text_embedding` - 384-dim vector
    - `preprocessed` - tokens, lemmas, POS tags
  - Storylines in `storylines` table with embedding drift
  - Article-storyline links in `article_storylines` junction table
