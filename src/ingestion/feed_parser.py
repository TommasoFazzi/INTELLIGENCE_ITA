"""
RSS Feed Parser Module

This module handles parsing RSS/Atom feeds and extracting article metadata.
"""

import feedparser
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import yaml

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeedParser:
    """Parser for RSS/Atom feeds with support for multiple sources."""

    def __init__(self, config_path: str = "config/feeds.yaml"):
        """
        Initialize the FeedParser with configuration.

        Args:
            config_path: Path to the feeds configuration YAML file
        """
        self.config_path = Path(config_path)
        self.feeds_config = self._load_config()
        logger.info(f"Loaded {len(self.feeds_config)} feeds from configuration")

    def _load_config(self) -> List[Dict]:
        """Load feeds configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('feeds', [])
        except Exception as e:
            logger.error(f"Error loading feed configuration: {e}")
            return []

    def parse_feed(self, feed_url: str, feed_name: str = None) -> List[Dict]:
        """
        Parse a single RSS/Atom feed.

        Args:
            feed_url: URL of the RSS/Atom feed
            feed_name: Optional name for the feed

        Returns:
            List of article dictionaries with metadata
        """
        try:
            logger.info(f"Parsing feed: {feed_name or feed_url}")
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"Feed parse warning for {feed_name}: {feed.bozo_exception}")

            articles = []
            for entry in feed.entries:
                article = self._extract_article_data(entry, feed_name)
                if article:
                    articles.append(article)

            logger.info(f"Extracted {len(articles)} articles from {feed_name or feed_url}")
            return articles

        except Exception as e:
            logger.error(f"Error parsing feed {feed_name or feed_url}: {e}")
            return []

    def _extract_article_data(self, entry, feed_name: str = None) -> Optional[Dict]:
        """
        Extract article data from a feed entry.

        Args:
            entry: Feed entry object from feedparser
            feed_name: Name of the source feed

        Returns:
            Dictionary with article metadata
        """
        try:
            # Extract published date
            published = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])

            # Extract content/summary
            content = ""
            if hasattr(entry, 'content') and entry.content:
                content = entry.content[0].value
            elif hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description

            article = {
                'title': entry.get('title', 'N/A'),
                'link': entry.get('link', ''),
                'published': published,
                'summary': content,
                'source': feed_name or 'Unknown',
                'authors': self._extract_authors(entry),
                'tags': self._extract_tags(entry),
                'fetched_at': datetime.now()
            }

            return article

        except Exception as e:
            logger.error(f"Error extracting article data: {e}")
            return None

    def _extract_authors(self, entry) -> List[str]:
        """Extract author names from entry."""
        authors = []
        if hasattr(entry, 'author'):
            authors.append(entry.author)
        if hasattr(entry, 'authors'):
            authors.extend([a.get('name', '') for a in entry.authors])
        return [a for a in authors if a]

    def _extract_tags(self, entry) -> List[str]:
        """Extract tags/categories from entry."""
        tags = []
        if hasattr(entry, 'tags'):
            tags = [tag.get('term', '') for tag in entry.tags]
        return [t for t in tags if t]

    def parse_all_feeds(self, category: str = None) -> List[Dict]:
        """
        Parse all configured feeds or feeds from a specific category.

        Args:
            category: Optional category filter (e.g., 'intelligence', 'tech_economy')

        Returns:
            List of all articles from all feeds
        """
        all_articles = []
        feeds_to_parse = self.feeds_config

        if category:
            feeds_to_parse = [f for f in self.feeds_config if f.get('category') == category]
            logger.info(f"Filtering feeds by category: {category}")

        logger.info(f"Parsing {len(feeds_to_parse)} feeds...")

        for feed_config in feeds_to_parse:
            feed_name = feed_config.get('name')
            feed_url = feed_config.get('url')
            feed_category = feed_config.get('category')
            feed_subcategory = feed_config.get('subcategory')

            if not feed_url:
                logger.warning(f"No URL found for feed: {feed_name}")
                continue

            articles = self.parse_feed(feed_url, feed_name)

            # Add category and subcategory to each article
            for article in articles:
                article['category'] = feed_category
                article['subcategory'] = feed_subcategory

            all_articles.extend(articles)

        logger.info(f"Total articles extracted: {len(all_articles)}")
        return all_articles

    def get_feeds_by_category(self) -> Dict[str, List[Dict]]:
        """
        Get all feeds organized by category.

        Returns:
            Dictionary with categories as keys and lists of feed configs as values
        """
        categorized = {}
        for feed in self.feeds_config:
            category = feed.get('category', 'uncategorized')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(feed)
        return categorized


if __name__ == "__main__":
    # Test the parser
    parser = FeedParser()
    articles = parser.parse_all_feeds()
    print(f"\nTotal articles fetched: {len(articles)}")

    if articles:
        print("\nFirst article sample:")
        first_article = articles[0]
        for key, value in first_article.items():
            print(f"  {key}: {value}")
