# Ingestion Context

## Purpose
Data collection pipeline that fetches news articles from RSS/Atom feeds, extracts full-text content from URLs, and prepares data for NLP processing. This is Phase 1 of the intelligence pipeline.

## Architecture Role
Entry point for all external data. Reads feed configurations from `config/feeds.yaml`, fetches all RSS feeds in parallel via aiohttp, extracts full article text concurrently using Trafilatura/Newspaper3k/Cloudscraper, and outputs JSON files to `data/` directory for downstream processing by `src/nlp/`.

A single `asyncio.run()` in `pipeline.run()` orchestrates both feed parsing and content extraction. Sync libraries (feedparser, trafilatura, newspaper3k) are executed via `asyncio.to_thread()` to avoid blocking the event loop.

## Key Files

- `feed_parser.py` - RSS/Atom feed parsing with fallback scraping
  - `FeedParser` class - Loads feeds from YAML config
  - `parse_feed(url, name)` - Parse single feed using `feedparser` (sync)
  - `_fetch_and_parse_feed(session, url, name, category, subcategory)` - Async: fetch RSS via aiohttp, parse with `asyncio.to_thread(feedparser.parse, ...)`
  - `_scrape_fallback_async(feed_name, session)` - Async fallback scraper (cloudscraper via to_thread, aiohttp for simple gets)
  - `_parse_all_feeds_async(category)` - Async: `aiohttp.ClientSession` with `TCPConnector(limit=20, limit_per_host=3)`, launches all feeds via `asyncio.gather()`
  - `parse_all_feeds(category)` - Sync wrapper (`asyncio.run()`) for standalone use only
  - `scrape_fallback(feed_name)` - Sync BeautifulSoup fallback for broken RSS
  - `FALLBACK_SCRAPERS` - Config for sites needing HTML scraping (Defense One, CFR, CSIS, ECFR, ISS Africa)
  - `cloudscraper` support for anti-bot protected sites (403 bypass)
  - User-Agent rotation on every request to avoid blocks

- `content_extractor.py` - Full-text extraction from URLs
  - `ContentExtractor` class - Multi-method extraction with `max_concurrent=10`
  - `extract_with_trafilatura(url)` - Primary method (fast, news-optimized)
  - `extract_with_newspaper(url)` - Newspaper3k fallback
  - `extract_with_cloudscraper(url)` - For anti-bot sites (e.g., politico.com)
  - `_extract_content_async(semaphore, article, idx, total)` - Async: acquires semaphore, delegates to `asyncio.to_thread(self.extract_content, url)`
  - `_extract_batch_async(articles)` - Async: concurrent extraction via `asyncio.gather()` with `asyncio.Semaphore(max_concurrent)`
  - `extract_batch(articles)` - Sync wrapper (`asyncio.run()`) for standalone use only
  - Extraction strategy: Trafilatura → Newspaper3k → Cloudscraper

- `pipeline.py` - Main orchestration
  - `IngestionPipeline` class - End-to-end workflow
  - `_run_async(category, extract_content, max_age_days)` - Async core: calls `_parse_all_feeds_async()` and `_extract_batch_async()` directly (no sync wrappers)
  - `run(category, max_age_days)` - Single `asyncio.run(self._run_async(...))` entry point
  - `deduplicate_by_quick_hash()` - MD5 hash(link + title) deduplication (Phase 1)
  - `get_summary()` - Statistics by category/source
  - Auto-saves JSON to `data/articles_{timestamp}.json`

## Dependencies

- **Internal**: `src/utils/logger`
- **External**:
  - `aiohttp` - Async HTTP client for parallel feed fetching
  - `feedparser` - RSS/Atom parsing (sync, run via `asyncio.to_thread`)
  - `trafilatura` - News article extraction (primary, sync via `to_thread`)
  - `newspaper3k` - Fallback extraction (sync via `to_thread`)
  - `cloudscraper` - Anti-bot bypass (optional, sync via `to_thread`)
  - `beautifulsoup4` - HTML parsing for fallback scraping
  - `pyyaml` - Config loading
  - `requests` - HTTP client (sync session for standalone use)

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
