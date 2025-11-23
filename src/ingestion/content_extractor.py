"""
Content Extractor Module

This module extracts full-text content from article URLs using specialized libraries.
It tries multiple extraction methods to get the best quality content.
"""

import requests
from typing import Optional, Dict
from datetime import datetime
import trafilatura
from newspaper import Article as NewspaperArticle  # newspaper4k

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ContentExtractor:
    """Extracts full-text content from article URLs."""

    def __init__(self, timeout: int = 10, user_agent: str = None):
        """
        Initialize the ContentExtractor.

        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def extract_with_trafilatura(self, url: str, html: str = None) -> Optional[Dict]:
        """
        Extract content using Trafilatura (best for news articles).

        Args:
            url: Article URL
            html: Optional pre-fetched HTML content

        Returns:
            Dictionary with extracted content or None
        """
        try:
            if html is None:
                downloaded = trafilatura.fetch_url(url)
            else:
                downloaded = html

            if not downloaded:
                return None

            # Extract with metadata
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                output_format='json',
                with_metadata=True
            )

            if content:
                import json
                content_dict = json.loads(content)
                return {
                    'title': content_dict.get('title'),
                    'author': content_dict.get('author'),
                    'date': content_dict.get('date'),
                    'text': content_dict.get('text'),
                    'description': content_dict.get('description'),
                    'sitename': content_dict.get('sitename'),
                    'extraction_method': 'trafilatura'
                }

        except Exception as e:
            logger.debug(f"Trafilatura extraction failed for {url}: {e}")

        return None

    def extract_with_newspaper(self, url: str) -> Optional[Dict]:
        """
        Extract content using Newspaper3k (good fallback).

        Args:
            url: Article URL

        Returns:
            Dictionary with extracted content or None
        """
        try:
            article = NewspaperArticle(url)
            article.download()
            article.parse()

            if article.text:
                return {
                    'title': article.title,
                    'author': ', '.join(article.authors) if article.authors else None,
                    'date': article.publish_date.isoformat() if article.publish_date else None,
                    'text': article.text,
                    'description': article.meta_description,
                    'sitename': article.source_url,
                    'top_image': article.top_image,
                    'extraction_method': 'newspaper3k'
                }

        except Exception as e:
            logger.debug(f"Newspaper3k extraction failed for {url}: {e}")

        return None

    def extract_content(self, url: str, html: str = None) -> Optional[Dict]:
        """
        Extract full-text content from URL using multiple methods.
        Tries Trafilatura first, then falls back to Newspaper3k.

        Args:
            url: Article URL
            html: Optional pre-fetched HTML content

        Returns:
            Dictionary with extracted content and metadata
        """
        logger.info(f"Extracting content from: {url}")

        # Try Trafilatura first (best for news)
        content = self.extract_with_trafilatura(url, html)
        if content and content.get('text'):
            logger.info(f"Successfully extracted with Trafilatura: {url}")
            return content

        # Fallback to Newspaper3k
        content = self.extract_with_newspaper(url)
        if content and content.get('text'):
            logger.info(f"Successfully extracted with Newspaper3k: {url}")
            return content

        logger.warning(f"Failed to extract content from: {url}")
        return None

    def extract_batch(self, articles: list) -> list:
        """
        Extract full content for a batch of articles.

        Args:
            articles: List of article dictionaries with 'link' key

        Returns:
            List of articles with 'full_content' field added
        """
        results = []
        total = len(articles)

        logger.info(f"Extracting full content for {total} articles...")

        for idx, article in enumerate(articles, 1):
            url = article.get('link')
            if not url:
                logger.warning(f"Article {idx}/{total} has no URL, skipping")
                results.append(article)
                continue

            try:
                full_content = self.extract_content(url)

                # Add full content to article
                article['full_content'] = full_content
                article['extraction_success'] = full_content is not None
                article['extraction_timestamp'] = datetime.now()

                if full_content:
                    logger.info(f"[{idx}/{total}] ✓ Extracted: {article.get('title', 'N/A')[:50]}...")
                else:
                    logger.warning(f"[{idx}/{total}] ✗ Failed: {article.get('title', 'N/A')[:50]}...")

                results.append(article)

            except Exception as e:
                logger.error(f"Error extracting article {idx}/{total}: {e}")
                article['full_content'] = None
                article['extraction_success'] = False
                article['extraction_error'] = str(e)
                results.append(article)

        success_count = sum(1 for a in results if a.get('extraction_success'))
        logger.info(f"Extraction complete: {success_count}/{total} successful")

        return results


if __name__ == "__main__":
    # Test the extractor
    extractor = ContentExtractor()

    # Test with a sample URL
    test_url = "https://www.bbc.com/news"
    content = extractor.extract_content(test_url)

    if content:
        print("\nExtracted content:")
        for key, value in content.items():
            if key == 'text':
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")
    else:
        print("Failed to extract content")
