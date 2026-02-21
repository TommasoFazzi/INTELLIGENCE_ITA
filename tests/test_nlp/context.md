# Test NLP — Context

## Purpose
Unit tests for the `src/nlp/processing.py` module (`NLPProcessor` class). Validates all stages of the NLP pipeline: text cleaning, semantic chunking, named entity recognition (NER), embedding generation, and the full article processing pipeline. All tests run in full isolation — spaCy and SentenceTransformer are mocked in every fixture so no model download is required.

## Test Coverage

### `test_text_cleaning.py`
Validates `NLPProcessor.clean_text()`.

Key test functions:
- `test_clean_text_removes_double_spaces` — collapses multiple consecutive spaces to one
- `test_clean_text_removes_tabs_and_newlines` — normalizes `\n` and `\t` to spaces
- `test_clean_text_removes_excessive_whitespace` — handles all mixed whitespace
- `test_clean_text_removes_markdown_links` — strips `[text]` bracket notation
- `test_clean_text_removes_brackets` — removes any `[...]` content
- `test_clean_text_removes_common_noise` (parametrized, 12 patterns) — removes boilerplate strings such as "Follow us on Twitter", "Click here to subscribe", "Advertisement", "Source:", "Photo:", "Subscribe to our newsletter", etc.
- `test_clean_text_case_insensitive_removal` — removal is case-insensitive
- `test_clean_text_handles_empty_string` / `test_clean_text_handles_none` / `test_clean_text_handles_whitespace_only` — edge cases return `""`
- `test_clean_text_preserves_punctuation` — commas, periods, `!`, `?` are retained
- `test_clean_text_preserves_unicode` — Italian characters (`è`, `Più`, `L'`) survive
- `test_clean_text_strips_leading_trailing_whitespace` — result is fully stripped
- `test_clean_text_realistic_article` — end-to-end realistic scraping artifact
- `test_clean_text_handles_html_like_content` — removes `[div]`/`[/div]` remnants

### `test_chunking.py`
Validates `NLPProcessor.create_chunks()`.

Key test functions:
- `test_create_chunks_handles_empty_text` / `test_create_chunks_handles_none` — return `[]`
- `test_create_chunks_single_sentence` — one sentence produces exactly one chunk
- `test_create_chunks_short_text` — text below `chunk_size` produces a single chunk
- `test_create_chunks_splits_long_text` — long text produces multiple chunks
- `test_chunk_has_required_metadata` — each chunk dict contains `text`, `word_count`, `sentence_count`
- `test_chunk_word_count_is_accurate` — `word_count` reflects actual word count
- `test_chunk_sentence_count_is_accurate` — `sentence_count` reflects sentence count
- `test_chunks_respect_size_limit` — no chunk significantly exceeds `chunk_size + 50` words
- `test_create_chunks_with_custom_size` — custom `chunk_size` parameter is respected
- `test_chunks_have_overlap` / `test_overlap_preserved_across_chunks` — overlap parameter is honoured
- `test_chunks_preserve_complete_sentences` — chunks never split a sentence mid-way
- `test_create_chunks_single_very_long_sentence` — single sentence longer than `chunk_size` emits one chunk (sentence boundary respected over size limit)
- `test_create_chunks_whitespace_only_sentences` — empty sentences are skipped
- `test_create_chunks_preserves_text_content` — total words across chunks ≥ original word count
- `test_create_chunks_returns_list_of_dicts` — return type is always `list[dict]`

### `test_entities.py`
Validates `NLPProcessor.extract_entities()`.

Key test functions:
- `test_extract_entities_returns_dict` — result has keys `entities`, `by_type`, `entity_count`
- `test_extract_entities_finds_organizations` — detects ORG entities (e.g. "Apple")
- `test_extract_entities_finds_locations` — detects LOC entities (e.g. "New York")
- `test_extract_entities_finds_persons` — detects PER entities (e.g. "John Smith")
- `test_extract_entities_multiple_types` — multiple entity types in one text
- `test_extract_entities_count_is_accurate` — `entity_count` matches found entities
- `test_entity_has_required_fields` — each entity dict has `text`, `label`, `start`, `end`
- `test_entity_positions_are_accurate` — `text[start:end]` matches `entity['text']`
- `test_entities_grouped_by_type` — `by_type` groups entities correctly per label
- `test_extract_entities_empty_text` / `test_extract_entities_none_input` / `test_extract_entities_non_string_input` — edge cases return empty result
- `test_extract_entities_italian_text` — multilingual NER on Italian text (e.g. "Roma" as LOC)
- `test_extract_entities_duplicate_entities` — handles repeated occurrences in text
- `test_extract_entities_handles_various_types` — supports DATE, MONEY, PERCENT entity labels

### `test_embeddings.py`
Validates `NLPProcessor.generate_embedding()` and `NLPProcessor.generate_chunk_embeddings()`.

Key test functions:
- `test_generate_embedding_returns_numpy_array` — output is `np.ndarray`
- `test_generate_embedding_correct_dimension` — length is 384 (paraphrase-multilingual-MiniLM-L12-v2)
- `test_generate_embedding_not_zero_vector` — valid text does not produce all-zeros
- `test_generate_embedding_handles_empty_text` / `test_generate_embedding_handles_none` / `test_generate_embedding_handles_non_string` — invalid input returns 384-dim zero vector
- `test_generate_chunk_embeddings_adds_embeddings` — each chunk dict gains an `embedding` key
- `test_generate_chunk_embeddings_correct_dimension` — `embedding` length = 384; `embedding_dim` field = 384
- `test_generate_chunk_embeddings_preserves_original_data` — original chunk fields are not mutated
- `test_generate_chunk_embeddings_batch_processing` — 10-chunk batch returns 10 enriched chunks
- `test_generate_chunk_embeddings_empty_list` — returns `[]`
- `test_chunk_embeddings_are_lists` — `embedding` is serialized as `list` (JSON-ready)
- `test_chunk_embeddings_contain_floats` — all embedding values are `float`
- `test_generate_embedding_consistent_for_same_text` — deterministic mock confirms identical output for same input
- `test_generate_chunk_embeddings_handles_malformed_chunks` — graceful handling or `KeyError` on missing `text` field

### `test_nlp_processor.py`
Validates the full `NLPProcessor` integration: `preprocess_text()`, `process_article()`, `process_batch()`, `get_processing_stats()`, and constructor initialization.

Key test functions:
- `test_preprocess_text_returns_dict` / `test_preprocess_text_has_required_fields` — output has `tokens`, `lemmas`, `pos_tags`, `sentences`, `num_tokens`, `num_sentences`
- `test_preprocess_text_tokenization` / `test_preprocess_text_lemmatization` / `test_preprocess_text_pos_tagging` — each linguistic step is exercised
- `test_preprocess_text_empty_input` / `test_preprocess_text_none_input` / `test_preprocess_text_non_string_input` — all return empty zeroed-out dict
- `test_process_article_enriches_with_nlp_data` — adds `nlp_data` and `nlp_processing` keys
- `test_process_article_nlp_data_structure` — `nlp_data` contains `clean_text`, `chunks`, `chunk_count`, `entities`, `preprocessed`, `full_text_embedding`, `embedding_dim`, `original_length`, `clean_length`, `processed_at`
- `test_process_article_marks_success` — `nlp_processing.success == True` and `timestamp` present on success
- `test_process_article_handles_no_content` — missing `full_content` sets `success == False` with `error`
- `test_process_article_extracts_from_full_content_dict` / `test_process_article_handles_full_content_string` — both dict and string forms of `full_content` are handled
- `test_process_article_falls_back_to_summary` — uses `summary` field when `full_content` is absent
- `test_process_article_creates_chunks` / `test_process_article_short_text_single_chunk` — chunking integrated into pipeline
- `test_process_article_chunks_have_embeddings` — every chunk has `embedding` and `embedding_dim`
- `test_process_article_generates_full_text_embedding` — 384-dim full-text embedding present
- `test_process_article_extracts_entities` — entity dict structure validated end-to-end
- `test_process_article_preprocesses_text` — `preprocessed` sub-dict has `tokens`, `lemmas`, `pos_tags`
- `test_process_article_handles_exception` — `full_content=None` sets `success == False`
- `test_process_batch_processes_all_articles` / `test_process_batch_returns_list` / `test_process_batch_empty_list` / `test_process_batch_tracks_success` — batch processing
- `test_get_processing_stats_*` — aggregated stats: `total_articles`, `successful_processing`, `success_rate`, `total_entities_extracted`, `entities_by_type`, `avg_tokens_per_article`, `total_chunks`, `avg_chunks_per_article`, `avg_chunk_size`, `embedding_dimension`
- `test_nlp_processor_initialization` — custom `chunk_size`, `chunk_overlap`, `batch_size` stored correctly
- `test_nlp_processor_adds_sentencizer` — `add_pipe("sentencizer")` called when not already in pipeline

## Key Test Patterns

- **Fixture-level mocking:** Every test file defines a local `nlp_processor` fixture that patches `spacy.load` and `src.nlp.processing.SentenceTransformer` inside a `with patch(...)` context manager. This prevents any real model loading, keeping tests fast and offline-safe.
- **Behaviour-driven spaCy mock:** The `mock_nlp_call` function in chunking, entity, and processor fixtures implements sentence splitting (on `.`) and entity detection (by substring match) to produce realistic but deterministic output without a real spaCy model.
- **SentenceTransformer mock:** `mock_encode` returns `np.random.rand(384)` for single strings and `np.random.rand(N, 384)` for lists, matching the real API signature.
- **Parametrize for noise patterns:** `test_clean_text_removes_common_noise` uses `@pytest.mark.parametrize` over 12 boilerplate strings.
- **Inline fixtures:** Some edge-case tests (e.g. `test_create_chunks_single_very_long_sentence`, `test_extract_entities_italian_text`) build their own processor inline rather than relying on the shared fixture, for full control over mock behaviour.
- **All tests marked `@pytest.mark.unit`** — all tests in this module are pure unit tests with no database or network I/O.

## Dependencies

- **Internal:** `src.nlp.processing.NLPProcessor`
- **External:**
  - `pytest` — test framework, `@pytest.mark.unit` marker
  - `unittest.mock` — `Mock`, `MagicMock`, `patch`
  - `numpy` — used in embedding tests (`np.ndarray`, `np.allclose`, `np.testing.assert_array_almost_equal`)
- **Fixtures from `conftest.py`:** `sample_article`, `sample_articles_batch`, `sample_embedding`, `sample_chunk` are available but most test files define their own local `nlp_processor` fixture and inline article dicts for isolation.
- **No database or network calls** — fully isolated unit tests.

## Known Gotchas

- **spaCy model not needed:** The `nlp_processor` fixture mocks `spacy.load` at the module level, so `xx_ent_wiki_sm` does not need to be installed to run these tests. Any test that creates an `NLPProcessor()` outside the fixture context (e.g. inline tests) must also patch `spacy.load` and `SentenceTransformer`, or the constructor will attempt to load the real model and fail.
- **Sentence splitting in chunking tests:** The mock uses Python `.split('.')` which is simpler than real spaCy segmentation. Tests that rely on exact sentence counts should account for this simplified behaviour.
- **Mocked `encode` randomness:** Most embedding tests use `np.random.rand` without a seed, so individual embedding values are non-deterministic between runs. Only `test_generate_embedding_consistent_for_same_text` uses a seeded mock for reproducibility.
- **Malformed chunk handling is implementation-dependent:** `test_generate_chunk_embeddings_handles_malformed_chunks` accepts either graceful handling or a `KeyError`; it does not enforce a specific contract.
- **`test_process_batch_tracks_success` uses a loose assertion** (`success_count >= 0`); the intent is just to confirm the field exists, not validate exact counts.
