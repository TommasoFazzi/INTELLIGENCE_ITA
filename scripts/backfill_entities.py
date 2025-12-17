#!/usr/bin/env python3
"""
Entity Backfill Script - Migrate entities from JSONB to entities table

Extracts entities from articles.entities (JSONB) and populates:
- entities table (with UPSERT to increment mention_count)
- entity_mentions junction table (article ↔ entity relationships)

Usage:
    python3 scripts/backfill_entities.py --tier 1  # Last 7 days (Tier 1)
    python3 scripts/backfill_entities.py --tier 2  # Last 30 days (Tier 2)
    python3 scripts/backfill_entities.py --all     # All articles (Tier 3)
"""

import sys
import os
import json
import yaml
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EntityMigrator:
    """Migrates entities from JSONB to relational entities table"""

    def __init__(self, use_blocklist: bool = True):
        self.db = DatabaseManager()
        self.min_length = 3
        self.blocklist = set()
        self.valid_acronyms = set()

        if use_blocklist:
            self._load_blocklist()

    def _load_blocklist(self):
        """Load entity blocklist from YAML (same as clean_entities.py)"""
        blocklist_path = "config/entity_blocklist.yaml"

        if not os.path.exists(blocklist_path):
            logger.warning(f"⚠️  Blocklist not found at {blocklist_path}, using basic filters only")
            return

        with open(blocklist_path, 'r', encoding='utf-8') as f:
            blocklist_data = yaml.safe_load(f)

        # Extract valid acronyms (exceptions)
        if 'valid_acronyms' in blocklist_data:
            self.valid_acronyms = {term.lower() for term in blocklist_data['valid_acronyms']}

        # Flatten all other categories
        for category, terms in blocklist_data.items():
            if category == 'valid_acronyms':
                continue
            if isinstance(terms, list):
                self.blocklist.update(term.lower() for term in terms)

        # Remove valid acronyms from blocklist
        self.blocklist = self.blocklist - self.valid_acronyms

        logger.info(f"✅ Loaded {len(self.blocklist)} blocked terms ({len(self.valid_acronyms)} valid acronyms preserved)")

    def is_valid_entity(self, name: str, entity_type: str) -> bool:
        """
        Filter entities to prevent garbage migration

        Replicates Phase 0 cleanup rules:
        - Too short (< 3 chars) unless valid acronym
        - Pure numbers (years like "2024")
        - Contains URLs/HTML
        - In blocklist (unless valid acronym exception)
        """
        name = name.strip()

        # 1. Length check (with valid acronym exception)
        if len(name) < self.min_length:
            if name.upper() not in self.valid_acronyms:
                return False

        # 2. Pure numbers (years, dates)
        if name.isdigit():
            return False

        # 3. URLs or HTML artifacts
        if "http" in name.lower() or "www." in name.lower() or "href=" in name.lower():
            return False

        # 4. Blocklist check (with valid acronym exception)
        if name.lower() in self.blocklist:
            if entity_type == 'GPE' and name.upper() in self.valid_acronyms:
                return True  # Keep valid GPE acronyms like "US", "EU"
            return False

        # 5. Starts with lowercase (not a proper noun)
        if name[0].islower():
            return False

        return True

    def migrate_batch(self, days_back: int = 7, tier_name: str = "Tier 1"):
        """
        Migrate entities from JSONB to entities table for a time range

        Args:
            days_back: Number of days to look back (7=Tier 1, 30=Tier 2, 9999=Tier 3)
            tier_name: Human-readable tier name for logging
        """
        logger.info("=" * 70)
        logger.info(f"🔄 Starting Entity Migration - {tier_name}")
        logger.info("=" * 70)

        # Calculate cutoff date
        if days_back < 9999:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            date_filter = "published_date >= %s"
            date_params = (cutoff_date,)
            logger.info(f"📅 Date range: Last {days_back} days (since {cutoff_date})")
        else:
            # Tier 3: All articles
            date_filter = "1=1"  # No date filter
            date_params = ()
            logger.info(f"📅 Date range: All articles (full database)")

        # Query: Select unmigrated articles with entities
        query_articles = f"""
            SELECT id, entities, published_date
            FROM articles
            WHERE {date_filter}
            AND entities_migrated_at IS NULL
            AND entities IS NOT NULL
            ORDER BY published_date DESC;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_articles, date_params)
                articles = cur.fetchall()

                logger.info(f"📚 Found {len(articles)} articles to process")

                if len(articles) == 0:
                    logger.info("✅ No articles to migrate (already processed)")
                    return

                # Stats tracking
                stats = {
                    'articles_processed': 0,
                    'entities_created': 0,
                    'entities_updated': 0,
                    'mentions_created': 0,
                    'entities_filtered': 0
                }

                # Process each article
                for article_id, entities_json, pub_date in articles:
                    if not entities_json:
                        continue

                    # Parse entities JSON
                    # Format: {"by_type": {"ORG": [...], "LOC": [...]}, "entities": [...]}
                    if not isinstance(entities_json, dict) or 'by_type' not in entities_json:
                        logger.warning(f"⚠️  Invalid entities format in article {article_id}")
                        continue

                    by_type = entities_json['by_type']

                    # Deduplicate entities within article
                    seen_in_article: Set[Tuple[str, str]] = set()

                    # Process each entity type
                    for entity_type, names in by_type.items():
                        if not isinstance(names, list):
                            continue

                        for name in names:
                            # Validation
                            if not self.is_valid_entity(name, entity_type):
                                stats['entities_filtered'] += 1
                                continue

                            # Deduplicate
                            key = (name, entity_type)
                            if key in seen_in_article:
                                continue
                            seen_in_article.add(key)

                            try:
                                # UPSERT: Create or update entity
                                upsert_entity = """
                                    INSERT INTO entities (name, entity_type, mention_count, geo_status, first_seen, last_seen, created_at)
                                    VALUES (%s, %s, 1, 'PENDING', NOW(), NOW(), NOW())
                                    ON CONFLICT (name, entity_type)
                                    DO UPDATE SET
                                        mention_count = entities.mention_count + 1,
                                        last_seen = NOW()
                                    RETURNING id, (xmax = 0) AS inserted;
                                """
                                cur.execute(upsert_entity, (name, entity_type))
                                entity_id, was_inserted = cur.fetchone()

                                if was_inserted:
                                    stats['entities_created'] += 1
                                else:
                                    stats['entities_updated'] += 1

                                # Create entity mention (article ↔ entity link)
                                insert_mention = """
                                    INSERT INTO entity_mentions (article_id, entity_id)
                                    VALUES (%s, %s)
                                    ON CONFLICT (article_id, entity_id) DO NOTHING;
                                """
                                cur.execute(insert_mention, (article_id, entity_id))
                                if cur.rowcount > 0:
                                    stats['mentions_created'] += 1

                            except Exception as e:
                                logger.error(f"❌ Error processing entity '{name}' ({entity_type}): {e}")
                                continue

                    # Mark article as migrated
                    cur.execute(
                        "UPDATE articles SET entities_migrated_at = NOW() WHERE id = %s",
                        (article_id,)
                    )
                    stats['articles_processed'] += 1

                    # Progress logging
                    if stats['articles_processed'] % 100 == 0:
                        logger.info(f"   Progress: {stats['articles_processed']}/{len(articles)} articles processed...")

        # Final report
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"✅ MIGRATION COMPLETE - {tier_name}")
        logger.info("=" * 70)
        logger.info(f"Articles Processed:    {stats['articles_processed']:,}")
        logger.info(f"Entities Created:      {stats['entities_created']:,}")
        logger.info(f"Entities Updated:      {stats['entities_updated']:,}")
        logger.info(f"Entity Mentions:       {stats['mentions_created']:,}")
        logger.info(f"Entities Filtered:     {stats['entities_filtered']:,} (blocked as garbage)")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Backfill entities from JSONB to entities table')
    parser.add_argument(
        '--tier',
        type=int,
        choices=[1, 2, 3],
        help='Migration tier: 1=Last 7 days, 2=Last 30 days, 3=All articles'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Migrate all articles (Tier 3)'
    )
    parser.add_argument(
        '--no-blocklist',
        action='store_true',
        help='Disable blocklist filtering (not recommended)'
    )

    args = parser.parse_args()

    # Determine tier
    if args.all or args.tier == 3:
        days_back = 9999  # All articles
        tier_name = "Tier 3 (All Articles)"
    elif args.tier == 2:
        days_back = 30
        tier_name = "Tier 2 (Last 30 Days)"
    else:
        # Default: Tier 1
        days_back = 7
        tier_name = "Tier 1 (Last 7 Days)"

    # Initialize migrator
    migrator = EntityMigrator(use_blocklist=not args.no_blocklist)

    # Run migration
    migrator.migrate_batch(days_back=days_back, tier_name=tier_name)

    # Close database
    migrator.db.close()


if __name__ == "__main__":
    main()
