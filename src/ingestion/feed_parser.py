"""
RSS Feed Parser Module

This module handles parsing RSS/Atom feeds and extracting article metadata.
Includes fallback scraping for feeds with broken RSS.
"""

import feedparser
import requests
import random
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from urllib.parse import urljoin
import yaml

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Pool di User-Agent per evitare blocchi
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# Configurazione fallback scraper per feed RSS rotti
# Chiavi: nome feed esatto come in feeds.yaml
# needs_cloudscraper: True per siti con anti-bot protection (403 errors)
FALLBACK_SCRAPERS = {
    'Defense One - All Content': {
        'url': 'https://www.defenseone.com/',
        'selector': 'article.listing-item, div.river-item',
        'title_sel': 'h3 a, h2 a',
        'link_sel': 'h3 a, h2 a',
        'category': 'intelligence',
        'subcategory': 'defense',
        'needs_cloudscraper': False,
    },
    'CSIS - Center for Strategic and International Studies': {
        'url': 'https://www.csis.org/analysis',
        'selector': 'article, div.node--type-commentary',
        'title_sel': 'h2 a, h3 a, .title a',
        'link_sel': 'h2 a, h3 a, .title a',
        'category': 'intelligence',
        'subcategory': 'geopolitics',
        'needs_cloudscraper': False,
    },
    'Council on Foreign Relations': {
        'url': 'https://www.cfr.org/latest',
        # CFR uses link-based selection - articles, blogs, reports
        'selector': 'a[href*="/article/"], a[href*="/blog/"], a[href*="/report/"]',
        'title_sel': None,  # Links are self-contained
        'link_sel': None,   # Links are self-contained
        'category': 'intelligence',
        'subcategory': 'geopolitics',
        'needs_cloudscraper': True,
        'link_based': True,  # Special flag: extract from links directly
    },
    # NOTE: Chatham House ha protezione anti-bot molto aggressiva (WAF)
    # che nemmeno cloudscraper riesce a bypassare. Richiede browser headless.
    # Disabilitato per ora.
    # 'Chatham House': {
    #     'url': 'https://www.chathamhouse.org/publications',
    #     'needs_cloudscraper': True,
    # },
    'European Council on Foreign Relations': {
        'url': 'https://ecfr.eu/publications/',
        'selector': 'article, .publication-item, .card',
        'title_sel': 'h3 a, h2 a, .title a',
        'link_sel': 'h3 a, h2 a, .title a, a',
        'category': 'intelligence',
        'subcategory': 'geopolitics',
        'needs_cloudscraper': False,
    },
    'ISS Africa': {
        'url': 'https://issafrica.org/iss-today',
        'selector': 'a[href*="/iss-today/"]',
        'title_sel': None,
        'link_sel': None,
        'category': 'intelligence',
        'subcategory': 'africa',
        'needs_cloudscraper': False,
        'link_based': True,
    },
    # NOTE: ECB usa JavaScript per caricare i comunicati dinamicamente.
    # Richiede Selenium/Playwright per scraping efficace.
    # 'ECB Press Releases': {
    #     'url': 'https://www.ecb.europa.eu/press/pr/html/index.en.html',
    #     'needs_cloudscraper': False,
    # },
}


class FeedParser:
    """Parser for RSS/Atom feeds with support for multiple sources and fallback scraping."""

    def __init__(self, config_path: str = "config/feeds.yaml"):
        """
        Initialize the FeedParser with configuration.

        Args:
            config_path: Path to the feeds configuration YAML file
        """
        self.config_path = Path(config_path)
        self.feeds_config = self._load_config()
        logger.info(f"Loaded {len(self.feeds_config)} feeds from configuration")

        # Cloudscraper session for anti-bot protected sites
        self.cloudscraper_session = None
        if CLOUDSCRAPER_AVAILABLE:
            try:
                self.cloudscraper_session = cloudscraper.create_scraper(
                    browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
                )
                logger.debug("Cloudscraper session initialized for fallback scraping")
            except Exception as e:
                logger.warning(f"Failed to initialize cloudscraper: {e}")

    def _get_random_ua(self) -> str:
        """Get a random user agent from the pool."""
        return random.choice(USER_AGENTS)

    def _load_config(self) -> List[Dict]:
        """Load feeds configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('feeds', [])
        except Exception as e:
            logger.error(f"Error loading feed configuration: {e}")
            return []

    def scrape_fallback(self, feed_name: str) -> List[Dict]:
        """
        Scrape articles directly from website when RSS feed fails.

        Uses BeautifulSoup to extract article links and titles from
        the source's main page or news section. For sites with anti-bot
        protection (403 errors), uses cloudscraper instead of requests.

        Args:
            feed_name: Name of the feed (must match key in FALLBACK_SCRAPERS)

        Returns:
            List of article dictionaries
        """
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available for fallback scraping")
            return []

        config = FALLBACK_SCRAPERS.get(feed_name)
        if not config:
            logger.debug(f"No fallback scraper configured for: {feed_name}")
            return []

        try:
            logger.info(f"Trying fallback scraper for: {feed_name}")

            # Use cloudscraper for sites that need anti-bot bypass
            needs_cloudscraper = config.get('needs_cloudscraper', False)

            if needs_cloudscraper and self.cloudscraper_session:
                logger.debug(f"Using cloudscraper for {feed_name} (anti-bot protected)")
                response = self.cloudscraper_session.get(
                    config['url'],
                    timeout=20  # Extra time for challenge solving
                )
            else:
                response = requests.get(
                    config['url'],
                    headers={'User-Agent': self._get_random_ua()},
                    timeout=15
                )

            if response.status_code != 200:
                logger.warning(f"Fallback scraper got status {response.status_code} for {feed_name}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')

            articles = []
            seen_links = set()  # Avoid duplicates
            items = soup.select(config['selector'])[:30]  # Limit to 30, filter later

            # Check if this is a link-based scraper (like CFR)
            link_based = config.get('link_based', False)

            for item in items:
                if link_based:
                    # Item IS the link element
                    raw_link = item.get('href')
                    title = item.get_text(strip=True)
                else:
                    # Traditional: item contains title and link sub-elements
                    title_el = item.select_one(config['title_sel'])
                    link_el = item.select_one(config['link_sel'])

                    if not (title_el and link_el):
                        continue

                    raw_link = link_el.get('href')
                    title = title_el.get_text(strip=True)

                if not raw_link:
                    continue

                # Fix relative links
                full_link = urljoin(config['url'], raw_link)

                # Skip duplicates and empty titles
                if full_link in seen_links or not title:
                    continue
                seen_links.add(full_link)

                # Stop at 20 unique articles
                if len(articles) >= 20:
                    break

                articles.append({
                    'title': title,
                    'link': full_link,
                    'source': feed_name,
                    'published': datetime.now(),  # Placeholder date
                    'summary': '',
                    'authors': [],
                    'tags': [],
                    'fetched_at': datetime.now(),
                    'category': config.get('category'),
                    'subcategory': config.get('subcategory'),
                    'extraction_method': 'fallback_scraper',
                })

            logger.info(f"Fallback scraper extracted {len(articles)} articles from {feed_name}")
            return articles

        except Exception as e:
            logger.error(f"Fallback scraping failed for {feed_name}: {e}")
            return []

    def parse_feed(self, feed_url: str, feed_name: str = None) -> List[Dict]:
        """
        Parse a single RSS/Atom feed with fallback scraping support.

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

            # If RSS parsing failed or returned 0 articles, try fallback scraper
            if not articles and feed_name in FALLBACK_SCRAPERS:
                logger.info(f"RSS returned 0 articles, trying fallback scraper for {feed_name}")
                articles = self.scrape_fallback(feed_name)

            logger.info(f"Extracted {len(articles)} articles from {feed_name or feed_url}")
            return articles

        except Exception as e:
            logger.error(f"Error parsing feed {feed_name or feed_url}: {e}")

            # Try fallback on exception too
            if feed_name in FALLBACK_SCRAPERS:
                logger.info(f"RSS parsing failed, trying fallback scraper for {feed_name}")
                return self.scrape_fallback(feed_name)

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
