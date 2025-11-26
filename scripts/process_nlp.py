#!/usr/bin/env python3
"""
Process raw articles with NLP: cleaning, entities, embeddings, and chunking.

Usage:
    python scripts/process_nlp.py                              # Process latest raw file
    python scripts/process_nlp.py --input data/articles_*.json # Process specific file
    python scripts/process_nlp.py --chunk-size 1000            # Custom chunk size
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.processing import NLPProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_latest_raw_file() -> Path:
    """Find the most recent raw articles JSON file (not NLP-processed)."""
    data_dir = Path("data")

    # Find all article files that are NOT nlp-processed
    raw_files = [f for f in data_dir.glob("articles_*.json") if "nlp" not in f.name]

    if not raw_files:
        raise FileNotFoundError("No raw article files found in data/ directory")

    latest = max(raw_files, key=lambda p: p.stat().st_mtime)
    return latest


def load_articles(file_path: Path) -> List[Dict]:
    """Load articles from JSON file."""
    logger.info(f"Loading articles from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    if not isinstance(articles, list):
        raise ValueError("JSON file must contain a list of articles")

    logger.info(f"✓ Loaded {len(articles)} articles")
    return articles


def save_processed_articles(articles: List[Dict], output_path: Path):
    """Save processed articles to JSON file."""
    logger.info(f"Saving processed articles to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Saved {len(articles)} articles to {output_path.name}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Process articles with NLP")
    parser.add_argument('--input', type=str, help='Input JSON file path')
    parser.add_argument('--chunk-size', type=int, default=500, help='Words per chunk')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Overlap between chunks')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("NLP PROCESSING SCRIPT")
    logger.info("=" * 80)

    # Find input file
    if args.input:
        input_file = Path(args.input)
        if not input_file.exists():
            logger.error(f"File not found: {input_file}")
            return 1
    else:
        try:
            input_file = find_latest_raw_file()
            logger.info(f"Using latest raw file: {input_file.name}")
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1

    # Load articles
    try:
        articles = load_articles(input_file)
    except Exception as e:
        logger.error(f"Failed to load articles: {e}")
        return 1

    # Initialize NLP processor
    logger.info("\n[STEP 1] Initializing NLP processor...")
    try:
        processor = NLPProcessor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    except Exception as e:
        logger.error(f"Failed to initialize NLP processor: {e}")
        return 1

    # Process articles
    logger.info(f"\n[STEP 2] Processing {len(articles)} articles...")
    logger.info(f"Chunk size: {args.chunk_size} words, overlap: {args.chunk_overlap} words")

    try:
        processed_articles = processor.process_batch(articles, show_progress=True)
    except Exception as e:
        logger.error(f"Failed to process articles: {e}", exc_info=True)
        return 1

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("data") / f"articles_nlp_{timestamp}.json"

    # Save processed articles
    logger.info("\n[STEP 3] Saving processed articles...")
    try:
        save_processed_articles(processed_articles, output_file)
    except Exception as e:
        logger.error(f"Failed to save articles: {e}")
        return 1

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nInput file: {input_file.name}")
    logger.info(f"Output file: {output_file.name}")
    logger.info(f"\nTotal articles: {len(processed_articles)}")

    # Count success/failures
    successful = sum(1 for a in processed_articles if a.get('nlp_processing', {}).get('success', False))
    with_chunks = sum(1 for a in processed_articles if a.get('nlp_data', {}).get('chunks'))

    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Articles with chunks: {with_chunks}")
    logger.info(f"\n✓ All done! Next step: python scripts/load_to_database.py")

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
