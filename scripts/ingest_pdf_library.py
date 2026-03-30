#!/usr/bin/env python3
"""
Unified PDF Library ingestion pipeline.

Downloads institutional PDFs, processes NLP, and loads to database in a single
pipeline — eliminates the manual 3-command flow of ingest_pdfs.py → process_nlp.py
→ load_to_database.py.

Features:
    - URL-based deduplication (checks articles.link in DB before downloading)
    - Integrated NLP processing (entities, embeddings, chunks)
    - Direct database loading with batch_save()
    - Dry-run mode for testing

Usage:
    python scripts/ingest_pdf_library.py                  # Ingest all from config
    python scripts/ingest_pdf_library.py --dry-run        # Test without saving
    python scripts/ingest_pdf_library.py --source SIPRI   # Single publisher
    python scripts/ingest_pdf_library.py --force           # Re-ingest even if URL exists
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.ingestion.pdf_ingestor import PDFIngestor
from src.nlp.processing import NLPProcessor
from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_url_exists(db: DatabaseManager, url: str) -> bool:
    """Check if a PDF URL already exists in the articles table (dedup by link)."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM articles WHERE link = %s LIMIT 1", (url,))
                return cur.fetchone() is not None
    except Exception as e:
        logger.warning(f"Dedup check failed for {url}: {e}")
        return False


def load_pdf_config(source_filter: Optional[str] = None) -> List[Dict]:
    """Load and optionally filter PDF sources from config."""
    import yaml

    config_path = Path("config/pdf_sources.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return []

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config or 'pdf_sources' not in config:
        logger.error("No pdf_sources found in config")
        return []

    sources = config['pdf_sources']

    if source_filter:
        sources = [s for s in sources if source_filter.lower() in s.get('publisher', '').lower()]
        logger.info(f"Filtered to {len(sources)} sources matching '{source_filter}'")

    return sources


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Unified PDF library ingestion pipeline")
    parser.add_argument('--dry-run', action='store_true', help='Test extraction without saving')
    parser.add_argument('--source', type=str, help='Filter by publisher name (case-insensitive)')
    parser.add_argument('--force', action='store_true', help='Re-ingest even if URL already in DB')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PDF LIBRARY PIPELINE — Unified Ingestion")
    logger.info("=" * 80)

    # ── Step 0: Load config ──────────────────────────────────────────────────
    sources = load_pdf_config(args.source)
    if not sources:
        logger.error("No PDF sources to process")
        return 1

    logger.info(f"\n[CONFIG] {len(sources)} PDF sources loaded")

    # ── Step 1: Initialize services ──────────────────────────────────────────
    logger.info("\n[STEP 1] Initializing services...")

    try:
        ingestor = PDFIngestor()
    except ImportError as e:
        logger.error(f"PDFIngestor init failed: {e}")
        return 1

    db = None
    nlp = None

    if not args.dry_run:
        try:
            db = DatabaseManager()
            logger.info("  ✓ Database connected")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return 1

        try:
            nlp = NLPProcessor()
            logger.info("  ✓ NLP processor initialized")
        except Exception as e:
            logger.error(f"NLP processor init failed: {e}")
            return 1

    # ── Step 2: Deduplication check ──────────────────────────────────────────
    logger.info("\n[STEP 2] Deduplication check...")

    to_process = []
    skipped = []

    for source in sources:
        url = source.get('url', '')
        if not url:
            logger.warning(f"Skipping source without URL: {source.get('title', 'unknown')}")
            skipped.append(('no_url', source.get('title', 'unknown')))
            continue

        if not args.force and db and check_url_exists(db, url):
            logger.info(f"  ⊘ Already in DB: {source['title'][:60]}")
            skipped.append(('duplicate', source['title']))
        else:
            to_process.append(source)

    logger.info(f"  → {len(to_process)} new, {len(skipped)} skipped")

    if not to_process:
        logger.info("\nAll PDFs already ingested. Use --force to re-ingest.")
        return 0

    # ── Step 3: Download & extract PDFs ──────────────────────────────────────
    logger.info(f"\n[STEP 3] Downloading & extracting {len(to_process)} PDFs...")

    articles = asyncio.run(ingestor.run_batch(to_process))
    logger.info(f"  ✓ Extracted {len(articles)} articles from {len(to_process)} sources")

    if not articles:
        logger.warning("No articles extracted successfully")
        return 0

    # ── Dry-run: stop here ───────────────────────────────────────────────────
    if args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("DRY RUN — NO CHANGES MADE")
        logger.info("=" * 80)
        for article in articles:
            text_len = len(article.get('full_content', {}).get('text', ''))
            logger.info(f"  ✓ {article['title'][:60]}... ({text_len:,} chars)")
        return 0

    # ── Step 4: NLP processing (entities, embeddings, chunks) ────────────────
    logger.info(f"\n[STEP 4] NLP processing {len(articles)} articles...")

    processed = 0
    for i, article in enumerate(articles, 1):
        try:
            # NLPProcessor.process_article() adds nlp_data, entities, chunks, embeddings
            nlp.process_article(article)
            processed += 1
            logger.info(f"  [{i}/{len(articles)}] ✓ {article['title'][:50]}...")
        except Exception as e:
            logger.error(f"  [{i}/{len(articles)}] ✗ NLP failed for {article['title'][:50]}: {e}")
            article['nlp_processing'] = {'success': False, 'error': str(e)}

    logger.info(f"  → {processed}/{len(articles)} processed successfully")

    # ── Step 5: Load to database ─────────────────────────────────────────────
    logger.info(f"\n[STEP 5] Loading {processed} articles to database...")

    try:
        save_stats = db.batch_save(articles)
    except Exception as e:
        logger.error(f"Database save failed: {e}")
        # Save to JSON as fallback
        fallback_path = Path("data") / f"pdfs_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fallback_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"  Fallback saved to {fallback_path}")
        return 1

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("PDF LIBRARY PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n  Sources configured:  {len(sources)}")
    logger.info(f"  Skipped (dedup):     {len(skipped)}")
    logger.info(f"  Downloaded:          {len(to_process)}")
    logger.info(f"  Extracted:           {len(articles)}")
    logger.info(f"  NLP processed:       {processed}")
    logger.info(f"  Saved to DB:         {save_stats.get('saved', 0)}")
    logger.info(f"  DB duplicates:       {save_stats.get('skipped', 0)}")
    logger.info(f"  Chunks inserted:     {save_stats.get('total_chunks', 0)}")
    logger.info(f"  Errors:              {save_stats.get('errors', 0)}")

    if db:
        db.close()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
