"""
Main ingestion pipeline that orchestrates feed parsing and content extraction.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from .feed_parser import FeedParser
from .content_extractor import ContentExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Main pipeline for news data ingestion."""

    def __init__(
        self,
        config_path: str = "config/feeds.yaml",
        output_dir: str = "data",
        extract_full_content: bool = True
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            config_path: Path to feeds configuration
            output_dir: Directory to save extracted data
            extract_full_content: Whether to extract full article content
        """
        self.feed_parser = FeedParser(config_path)
        self.content_extractor = ContentExtractor() if extract_full_content else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Ingestion pipeline initialized")

    def run(
        self,
        category: Optional[str] = None,
        save_output: bool = True,
        extract_content: bool = True
    ) -> List[Dict]:
        """
        Run the complete ingestion pipeline.

        Args:
            category: Optional category filter
            save_output: Whether to save output to file
            extract_content: Whether to extract full content from URLs

        Returns:
            List of processed articles
        """
        logger.info("=" * 80)
        logger.info("Starting news ingestion pipeline")
        logger.info("=" * 80)

        # Step 1: Parse RSS feeds
        logger.info("\n[STEP 1] Parsing RSS feeds...")
        articles = self.feed_parser.parse_all_feeds(category=category)

        if not articles:
            logger.warning("No articles found from RSS feeds")
            return []

        logger.info(f"✓ Parsed {len(articles)} articles from RSS feeds")

        # Step 2: Extract full content (optional)
        if extract_content and self.content_extractor:
            logger.info("\n[STEP 2] Extracting full article content...")
            articles = self.content_extractor.extract_batch(articles)
            success_count = sum(1 for a in articles if a.get('extraction_success'))
            logger.info(f"✓ Extracted full content for {success_count}/{len(articles)} articles")
        else:
            logger.info("\n[STEP 2] Skipping full content extraction")

        # Step 3: Save output
        if save_output:
            logger.info("\n[STEP 3] Saving results...")
            output_file = self._save_output(articles, category)
            logger.info(f"✓ Results saved to: {output_file}")

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline execution completed successfully")
        logger.info("=" * 80)

        return articles

    def _save_output(self, articles: List[Dict], category: Optional[str] = None) -> Path:
        """
        Save articles to JSON file.

        Args:
            articles: List of article dictionaries
            category: Optional category name for filename

        Returns:
            Path to output file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_suffix = f"_{category}" if category else ""
        filename = f"articles{category_suffix}_{timestamp}.json"
        output_file = self.output_dir / filename

        # Convert datetime objects to strings for JSON serialization
        serializable_articles = []
        for article in articles:
            article_copy = article.copy()
            for key, value in article_copy.items():
                if isinstance(value, datetime):
                    article_copy[key] = value.isoformat()
            serializable_articles.append(article_copy)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_articles, f, indent=2, ensure_ascii=False)

        return output_file

    def get_summary(self, articles: List[Dict]) -> Dict:
        """
        Get a summary of the ingested articles.

        Args:
            articles: List of article dictionaries

        Returns:
            Summary statistics dictionary
        """
        total = len(articles)
        by_category = {}
        by_source = {}
        extraction_success = 0

        for article in articles:
            # Category counts
            category = article.get('category', 'unknown')
            by_category[category] = by_category.get(category, 0) + 1

            # Source counts
            source = article.get('source', 'unknown')
            by_source[source] = by_source.get(source, 0) + 1

            # Extraction success
            if article.get('extraction_success'):
                extraction_success += 1

        return {
            'total_articles': total,
            'extraction_success_rate': f"{extraction_success}/{total}" if total > 0 else "0/0",
            'by_category': by_category,
            'by_source': by_source,
            'top_sources': sorted(by_source.items(), key=lambda x: x[1], reverse=True)[:5]
        }


if __name__ == "__main__":
    # Run the pipeline as a test
    pipeline = IngestionPipeline()

    # Run for all feeds (you can also filter by category)
    articles = pipeline.run(
        category=None,  # Use None for all categories, or specify: 'intelligence', 'tech_economy', etc.
        save_output=True,
        extract_content=False  # Set to True to extract full content (slower)
    )

    # Print summary
    summary = pipeline.get_summary(articles)
    print("\n" + "=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal articles: {summary['total_articles']}")
    print(f"Extraction success: {summary['extraction_success_rate']}")
    print(f"\nArticles by category:")
    for category, count in summary['by_category'].items():
        print(f"  {category}: {count}")
    print(f"\nTop 5 sources:")
    for source, count in summary['top_sources']:
        print(f"  {source}: {count}")
