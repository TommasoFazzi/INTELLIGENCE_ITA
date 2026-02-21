# Test E2E — Context

## Purpose
End-to-end integration tests for the complete INTELLIGENCE_ITA pipeline: RSS ingestion → NLP processing → PostgreSQL storage → Narrative Engine → LLM report generation. This directory is a placeholder for full-pipeline tests that exercise all components together, verifying that data flows correctly across module boundaries and that each pipeline step produces output consumable by the next.

## Test Coverage

This directory currently contains only `__init__.py` — no e2e test files have been written yet. The module is reserved for future full-pipeline tests. Based on the root `tests/context.md` and the overall pipeline architecture, the expected scope is:

- **Full pipeline smoke test:** Run `daily_pipeline.py` steps end-to-end against a test database and verify a report is produced.
- **Ingestion → DB round-trip:** Ingest mock RSS articles, run NLP, save to test DB, and query them back via `DatabaseManager`.
- **Narrative engine integration:** Verify that articles stored in the DB are correctly clustered into storylines by `NarrativeProcessor`, and that `v_active_storylines` and `v_storyline_graph` views return valid data.
- **Report generation integration:** Feed real articles from the test DB into `ReportGenerator` and verify structured output (trade signals, narrative section, Pydantic schema conformance).

## Key Test Patterns

When implemented, e2e tests in this directory are expected to follow these patterns:

- **`@pytest.mark.e2e`** marker to allow selective exclusion (`pytest -m "not e2e"`) during development cycles.
- **`@pytest.mark.slow`** marker since full-pipeline tests invoke LLMs, embedding models, and a live database.
- **Dedicated test database:** `TEST_DATABASE_URL` env var (configured in `conftest.py`) pointing to an isolated `intelligence_ita_test` DB to avoid polluting production data.
- **Transaction rollback or teardown fixtures** to reset DB state between test runs.
- **`AsyncMock` / `pytest-asyncio`** for the async ingestion pipeline (`asyncio.run()` entry point in `pipeline.run()`).
- **Environment guard:** Tests should be skipped automatically when `DATABASE_URL` or `GEMINI_API_KEY` are not set, to remain runnable in CI without secrets.

## Dependencies

- **Internal:** All `src/` pipeline modules (`src.ingestion.pipeline`, `src.nlp.processing`, `src.storage.database`, `src.nlp.narrative_processor`, `src.llm.report_generator`)
- **External:**
  - `pytest` — test framework
  - `pytest-asyncio` — async pipeline support
  - `pytest-mock` — mocking Gemini API calls
  - `responses` — mock HTTP for RSS feed fetching
  - PostgreSQL test instance with pgvector extension
- **Fixtures from `conftest.py`:** `test_db_url`, `sample_article`, `sample_articles_batch`

## Known Gotchas

- **No test files exist yet:** The directory contains only `__init__.py`. All items above describe intended future coverage.
- **Requires live database:** Unlike unit tests, e2e tests need a running PostgreSQL instance with the pgvector extension and all migrations applied (via `load_to_database.py --init-only` or `psql` against the `migrations/` SQL files). Tests will fail with a connection error if `TEST_DATABASE_URL` is not set.
- **Requires GEMINI_API_KEY:** The report generation step calls the Gemini API. Without a key, the LLM steps must be mocked or the test must be skipped.
- **spaCy model dependency:** `xx_ent_wiki_sm` must be installed (`python -m spacy download xx_ent_wiki_sm`) for NLP steps to run without mocking.
- **Pipeline step 5 (narrative_processing) has `continue_on_failure=True`** in `daily_pipeline.py`: e2e tests must explicitly assert narrative output is present rather than relying on the pipeline not raising an exception.
- **DB view dependency:** Tests that assert on storyline data must wait for `v_active_storylines` and `v_storyline_graph` to reflect inserted rows; these are non-materialized views so they update immediately, but the underlying `storylines` table must be populated first.
