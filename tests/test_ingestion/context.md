# Test Ingestion Context

## Purpose
Unit tests for the `src/ingestion/` module. Validates RSS feed parsing, content extraction, and deduplication logic.

## Architecture Role
Test layer ensuring data collection reliability. Tests cover both happy paths and edge cases for feed parsing failures, extraction fallbacks, and duplicate detection.

## Key Files

- `test_feed_parser.py` - RSS/Atom feed parsing tests
  - Valid feed parsing
  - Malformed feed handling (bozo flag)
  - Fallback scraper activation
  - Article metadata extraction (title, link, published)
  - Category/subcategory assignment

- `test_content_extractor.py` - Full-text extraction tests
  - Trafilatura extraction success
  - Newspaper3k fallback
  - Cloudscraper for protected sites
  - Empty content handling
  - Extraction method tracking

- `test_pipeline_dedup.py` - Deduplication tests
  - Quick hash deduplication (MD5 of link + title)
  - Duplicate detection accuracy
  - Edge cases (empty titles, similar URLs)

## Dependencies

- **Internal**: `src/ingestion/` modules
- **External**:
  - `pytest` - Test framework
  - `pytest-mock` - Mocking HTTP requests

## Test Data

- Mock RSS feed XML
- Sample article metadata
- Test URLs for extraction

## Running

```bash
pytest tests/test_ingestion/ -v
```
