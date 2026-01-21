# Ingestion Context

## Purpose
Data collection pipeline that fetches news articles from RSS/Atom feeds, extracts full-text content from URLs, and prepares data for NLP processing. This is Phase 1 of the intelligence pipeline.

## Architecture Role
Entry point for all external data. Reads feed configurations from `config/feeds.yaml`, parses RSS entries, extracts full article text using Trafilatura/Newspaper3k, and outputs JSON files to `data/` directory for downstream processing by `src/nlp/`.

## Key Files

- `feed_parser.py` - RSS/Atom feed parsing with fallback scraping
  - `FeedParser` class - Loads feeds from YAML config
  - `parse_feed(url, name)` - Parse single feed using `feedparser`
  - `parse_all_feeds(category)` - Parse all configured feeds
  - `scrape_fallback(feed_name)` - BeautifulSoup fallback for broken RSS
  - `FALLBACK_SCRAPERS` - Config for sites needing HTML scraping (Defense One, CFR, CSIS, ECFR, ISS Africa)
  - `cloudscraper` support for anti-bot protected sites (403 bypass)
  - User-Agent rotation to avoid blocks

- `content_extractor.py` - Full-text extraction from URLs
  - `ContentExtractor` class - Multi-method extraction
  - `extract_with_trafilatura(url)` - Primary method (fast, news-optimized)
  - `extract_with_newspaper(url)` - Newspaper3k fallback
  - `extract_with_cloudscraper(url)` - For anti-bot sites (e.g., politico.com)
  - `extract_batch(articles)` - Batch processing with progress logging
  - Extraction strategy: Trafilatura → Newspaper3k → Cloudscraper

- `pipeline.py` - Main orchestration
  - `IngestionPipeline` class - End-to-end workflow
  - `run(category, max_age_days)` - Main execution flow
  - `deduplicate_by_quick_hash()` - MD5 hash(link + title) deduplication (Phase 1)
  - `get_summary()` - Statistics by category/source
  - Auto-saves JSON to `data/articles_{timestamp}.json`

## Dependencies

- **Internal**: `src/utils/logger`
- **External**:
  - `feedparser` - RSS/Atom parsing
  - `trafilatura` - News article extraction (primary)
  - `newspaper3k` - Fallback extraction
  - `cloudscraper` - Anti-bot bypass (optional)
  - `beautifulsoup4` - HTML parsing for fallback scraping
  - `pyyaml` - Config loading
  - `requests` - HTTP client

## Data Flow

- **Input**:
  - `config/feeds.yaml` - RSS feed URLs and metadata (~33 feeds)
  - Live RSS/Atom feeds from web
  - Article URLs for full-text extraction

- **Output**:
  - `data/articles_{timestamp}.json` - Extracted articles with:
    - `title`, `link`, `published`, `source`, `category`, `subcategory`
    - `full_content.text` - Full article text
    - `extraction_success`, `extraction_method`
  - Statistics: total articles, by category, by source, extraction success rate
