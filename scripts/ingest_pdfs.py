#!/usr/bin/env python3
"""
PDF ingestion script for institutional documents.

Ingests PDFs from URLs or local files, extracts text, and saves as JSON articles
compatible with the standard NLP pipeline.

Usage:
    python scripts/ingest_pdfs.py --all                    # Ingest all from config
    python scripts/ingest_pdfs.py --url URL                # Single URL
    python scripts/ingest_pdfs.py --file path.pdf          # Local file
    python scripts/ingest_pdfs.py --all --dry-run          # Test without saving
    python scripts/ingest_pdfs.py --all --max-age-days 30  # Skip recently ingested

After ingestion, pass output to:
    python scripts/process_nlp.py
    python scripts/load_to_database.py
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.ingestion.pdf_ingestor import PDFIngestor
from src.utils.logger import get_logger
from scripts.pipeline_manifest import write_step, get_manifest_path

logger = get_logger(__name__)


def find_most_recent_ingested_time(source_name: str) -> datetime:
    """Find the most recent ingestion time for a PDF source."""
    data_dir = Path("data")

    # Look for pdfs_*.json files
    pdf_files = list(data_dir.glob("pdfs_*.json"))
    if not pdf_files:
        return datetime.min

    most_recent = datetime.min
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                for article in articles:
                    if article.get('source', '').startswith(f"pdf:{source_name}"):
                        article_time = datetime.fromisoformat(article.get('published', ''))
                        if article_time > most_recent:
                            most_recent = article_time
        except Exception as e:
            logger.warning(f"Failed to read {pdf_file}: {e}")

    return most_recent


def save_pdf_articles(articles: List[Dict], output_path: Path):
    """Save PDF-extracted articles to JSON file."""
    logger.info(f"Saving {len(articles)} PDF articles to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Saved {len(articles)} articles to {output_path.name}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Ingest PDF documents for intelligence analysis")
    parser.add_argument('--all', action='store_true', help='Ingest all PDFs from config/pdf_sources.yaml')
    parser.add_argument('--url', type=str, help='Single PDF URL to ingest')
    parser.add_argument('--file', type=str, help='Local PDF file to ingest')
    parser.add_argument('--dry-run', action='store_true', help='Test extraction without saving')
    parser.add_argument('--max-age-days', type=int, default=0, help='Skip sources ingested within N days')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PDF INGESTION SCRIPT")
    logger.info("=" * 80)

    # Initialize ingestor
    try:
        ingestor = PDFIngestor()
    except ImportError as e:
        logger.error(f"Failed to initialize PDFIngestor: {e}")
        return 1

    all_articles = []

    # Process all from config
    if args.all:
        logger.info("\n[MODE] Processing all PDFs from config/pdf_sources.yaml")

        try:
            import yaml
            config_path = Path("config/pdf_sources.yaml")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config or 'pdf_sources' not in config:
                logger.error("No pdf_sources found in config")
                return 1

            pdf_configs = config['pdf_sources']
            logger.info(f"Loaded {len(pdf_configs)} PDF sources from config")

            # Filter by max_age_days if specified
            filtered_configs = []
            for pdf_config in pdf_configs:
                if args.max_age_days > 0:
                    last_ingest = find_most_recent_ingested_time(pdf_config.get('publisher', ''))
                    age_days = (datetime.utcnow() - last_ingest).days
                    if age_days < args.max_age_days:
                        logger.info(f"Skipping {pdf_config.get('name')} (ingested {age_days} days ago)")
                        continue

                filtered_configs.append(pdf_config)

            if not filtered_configs:
                logger.info("All PDFs were recently ingested. Use --max-age-days 0 to force re-ingestion.")
                return 0

            logger.info(f"Processing {len(filtered_configs)} PDFs (after filtering by age)...")
            all_articles = ingestor.run_all("config/pdf_sources.yaml")

            if not all_articles:
                logger.warning("No PDFs were successfully ingested")
                return 0

        except Exception as e:
            logger.error(f"Failed to process PDF config: {e}", exc_info=True)
            return 1

    # Process single URL
    elif args.url:
        logger.info(f"\n[MODE] Processing single URL: {args.url}")

        # Parse filename from URL for title
        title = args.url.split('/')[-1].replace('.pdf', '').replace('%20', ' ')

        metadata = {
            'url': args.url,
            'title': title,
            'publisher': 'Manual Upload',
            'category': 'intelligence',
            'subcategory': 'documents'
        }

        import asyncio
        articles = asyncio.run(ingestor.run_batch([metadata]))
        all_articles = articles

        if not articles:
            logger.warning("Failed to ingest PDF from URL")
            return 1

    # Process local file
    elif args.file:
        logger.info(f"\n[MODE] Processing local file: {args.file}")

        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {args.file}")
            return 1

        title = file_path.stem

        metadata = {
            'title': title,
            'publisher': 'Local File',
            'category': 'intelligence',
            'subcategory': 'documents'
        }

        article = ingestor.run_single_file(str(file_path), metadata)
        if article:
            all_articles = [article]
        else:
            logger.warning("Failed to ingest local PDF file")
            return 1

    else:
        parser.print_help()
        logger.error("Please specify --all, --url, or --file")
        return 1

    # Summary
    if args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("DRY RUN - NO FILES SAVED")
        logger.info("=" * 80)
        logger.info(f"\nWould have saved {len(all_articles)} articles")
        for article in all_articles:
            logger.info(f"  - {article['title'][:60]}... ({len(article['full_content']['text'])} chars)")
        return 0

    # Save articles
    logger.info("\n[STEP] Saving PDF articles...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("data") / f"pdfs_{timestamp}.json"

    try:
        save_pdf_articles(all_articles, output_file)
    except Exception as e:
        logger.error(f"Failed to save articles: {e}")
        return 1

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PDF INGESTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_file.name}")
    logger.info(f"Total articles: {len(all_articles)}")

    # Write to manifest (if running inside orchestrated pipeline)
    write_step("pdf_ingestion", {
        "output_file": str(output_file),
        "article_count": len(all_articles),
    })

    logger.info(f"\n✓ Next step: python scripts/process_nlp.py --input {output_file}")

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
