# Test Ingestion Context

## Purpose
Unit tests for the `src/ingestion/` module. Validates RSS feed parsing, content extraction, deduplication logic, and async pipeline integration.

## Architecture Role
Test layer ensuring data collection reliability. Tests cover both happy paths and edge cases for feed parsing failures, extraction fallbacks, duplicate detection, and async method orchestration.

## Key Files

- `test_feed_parser.py` - RSS/Atom feed parsing tests
  - Valid feed parsing (sync `parse_feed`)
  - Malformed feed handling (bozo flag)
  - Fallback scraper activation
  - Article metadata extraction (title, link, published)
  - Category/subcategory assignment
  - `parse_all_feeds()` tests mock `_fetch_and_parse_feed` with `AsyncMock`

- `test_content_extractor.py` - Full-text extraction tests
  - Trafilatura extraction success
  - Newspaper3k fallback
  - Cloudscraper for protected sites
  - Empty content handling
  - Extraction method tracking
  - Batch extraction uses URL-dependent mock side_effects (not order-dependent) for concurrency safety

- `test_pipeline_dedup.py` - Deduplication and pipeline integration tests
  - Quick hash deduplication (MD5 of link + title)
  - Duplicate detection accuracy
  - Edge cases (empty titles, similar URLs)
  - Pipeline integration tests mock `_parse_all_feeds_async` and `_extract_batch_async` with `AsyncMock`
  - Test articles use `datetime.now()` to avoid age filtering

## Dependencies

- **Internal**: `src/ingestion/` modules
- **External**:
  - `pytest` - Test framework
  - `pytest-mock` - Mocking HTTP requests
  - `pytest-asyncio` - Async test support
  - `unittest.mock.AsyncMock` - Mocking async methods

## Test Data

- Mock RSS feed XML
- Sample article metadata
- Test URLs for extraction

## Running

```bash
pytest tests/test_ingestion/ -v
```
