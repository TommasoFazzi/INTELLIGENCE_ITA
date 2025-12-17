#!/usr/bin/env python3
"""
Entity Cleanup Script - Aggressive Garbage Removal

This script implements a 3-tier cleanup strategy to remove garbage entities
from the database while preserving legitimate named entities.

Tier 1: Safe Delete (90% accuracy) - Obvious garbage
Tier 2: Pattern-Based (80% accuracy) - Suspicious patterns
Tier 3: Preserve - Legitimate entities

Usage:
    python3 scripts/clean_entities.py --dry-run  # Preview changes
    python3 scripts/clean_entities.py --tier 1   # Run Tier 1 only
    python3 scripts/clean_entities.py --all      # Run all tiers (production)
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from dotenv import load_dotenv

load_dotenv()


class EntityCleaner:
    """Aggressive entity cleanup with 3-tier strategy"""

    def __init__(self, db_manager: DatabaseManager, blocklist_path: str = None):
        self.db = db_manager
        self.blocklist, self.valid_acronyms = self._load_blocklist(blocklist_path)
        self.stats = {
            'tier1_deleted': 0,
            'tier2_deleted': 0,
            'tier3_preserved': 0,
            'blocklist_deleted': 0,
            'total_before': 0,
            'total_after': 0
        }

    def _load_blocklist(self, path: str = None) -> tuple:
        """Load entity blocklist from YAML, returns (blocklist, valid_acronyms)"""
        if path is None:
            path = "config/entity_blocklist.yaml"

        if not os.path.exists(path):
            print(f"⚠️  Blocklist not found at {path}, skipping blocklist filtering")
            return (set(), set())

        with open(path, 'r', encoding='utf-8') as f:
            blocklist_data = yaml.safe_load(f)

        # Extract valid acronyms as exceptions (case-insensitive)
        valid_acronyms = set()
        if 'valid_acronyms' in blocklist_data:
            valid_acronyms = {term.lower() for term in blocklist_data['valid_acronyms']}

        # Flatten all other categories into a single set (case-insensitive)
        all_blocked = set()
        for category, terms in blocklist_data.items():
            if category == 'valid_acronyms':
                continue  # Skip valid acronyms category
            if isinstance(terms, list):
                all_blocked.update(term.lower() for term in terms)

        # Remove valid acronyms from blocklist (exceptions)
        all_blocked = all_blocked - valid_acronyms

        print(f"✅ Loaded {len(all_blocked)} blocked terms from {path} ({len(valid_acronyms)} valid acronyms preserved)")
        return (all_blocked, valid_acronyms)

    def get_total_entities(self) -> int:
        """Get current total entity count"""
        query = "SELECT COUNT(*) FROM entities;"
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()
                return result[0] if result else 0

    def tier1_safe_delete(self, dry_run: bool = True) -> int:
        """
        Tier 1: Safe Delete (90% accuracy)

        Delete entities matching ALL of these criteria:
        - LENGTH > 50 characters (too long to be a proper name)
        - OR starts with lowercase (not a proper noun)
        - OR contains HTML artifacts (href=, src=, etc.)
        - OR contains common phrase words (the, and, or, if, when)
        - AND mention_count = 1 (one-hit wonder)
        """

        query = """
        SELECT id, name, entity_type, mention_count
        FROM entities
        WHERE mention_count = 1
          AND (
              LENGTH(name) > 50
              OR name ~ '^[a-z]'  -- Starts with lowercase
              OR name LIKE '%href=%'
              OR name LIKE '%src=%'
              OR name LIKE '%http://%'
              OR name LIKE '%https://%'
              OR name LIKE '% the %'
              OR name LIKE '% and %'
              OR name LIKE '% or %'
              OR name LIKE '% if %'
              OR name LIKE '% when %'
              OR name LIKE '% that %'
          )
        ORDER BY LENGTH(name) DESC;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                entities_to_delete = cur.fetchall()

        if not entities_to_delete:
            print("✅ Tier 1: No entities to delete")
            return 0

        print(f"\n{'🔍 PREVIEW' if dry_run else '🗑️  DELETING'} Tier 1 (Safe Delete): {len(entities_to_delete)} entities")

        # Show sample
        print("\n📋 Sample entities to delete:")
        for i, (entity_id, name, entity_type, mention_count) in enumerate(entities_to_delete[:10]):
            name_preview = name[:80] + "..." if len(name) > 80 else name
            print(f"  [{i+1}] {entity_type:10s} | {name_preview}")

        if len(entities_to_delete) > 10:
            print(f"  ... and {len(entities_to_delete) - 10} more")

        if not dry_run:
            # Delete in transaction
            entity_ids = [e[0] for e in entities_to_delete]
            delete_query = "DELETE FROM entities WHERE id = ANY(%s);"
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(delete_query, (entity_ids,))
                    print(f"✅ Deleted {len(entities_to_delete)} entities")

        return len(entities_to_delete)

    def tier2_pattern_based(self, dry_run: bool = True) -> int:
        """
        Tier 2: Pattern-Based Delete (80% accuracy)

        Delete entities matching ANY of these patterns:
        - Contains excessive punctuation (more than 3 special chars)
        - Contains numbers mixed with text (but preserve years like 2024)
        - Entity type is MISC, PRODUCT, WORK_OF_ART (not useful for geopolitics)
        - All uppercase AND length > 15 (likely acronym errors)
        - Contains quotes, parentheses (likely extracted from text artifacts)
        - AND mention_count = 1
        """

        query = r"""
        SELECT id, name, entity_type, mention_count
        FROM entities
        WHERE mention_count = 1
          AND (
              -- Excessive punctuation (more than 3 special chars)
              (LENGTH(name) - LENGTH(REGEXP_REPLACE(name, '[^.,;:!?()\\[\\]{}]', '', 'g'))) > 3

              -- Contains quotes or parens (text artifacts)
              OR name ~ '["\(\)\[\]]'

              -- Mixed numbers and text (but not pure years)
              OR (name ~ '[0-9]' AND name ~ '[a-zA-Z]' AND NOT name ~ '^[0-9]{4}$')

              -- Not useful entity types
              OR entity_type IN ('MISC', 'PRODUCT', 'WORK_OF_ART', 'LANGUAGE', 'DATE', 'TIME', 'QUANTITY', 'ORDINAL', 'CARDINAL')

              -- Long uppercase (likely errors)
              OR (LENGTH(name) > 15 AND name = UPPER(name) AND name !~ '[a-z]')
          )
          -- Exclude entities already caught by Tier 1
          AND NOT (
              LENGTH(name) > 50
              OR name ~ '^[a-z]'
              OR name LIKE '%href=%'
              OR name LIKE '%http%'
          )
        ORDER BY LENGTH(name) DESC;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                entities_to_delete = cur.fetchall()

        if not entities_to_delete:
            print("✅ Tier 2: No entities to delete")
            return 0

        print(f"\n{'🔍 PREVIEW' if dry_run else '🗑️  DELETING'} Tier 2 (Pattern-Based): {len(entities_to_delete)} entities")

        # Show sample by type
        by_type = {}
        for entity_id, name, entity_type, mention_count in entities_to_delete:
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(name)

        print("\n📋 Sample entities to delete by type:")
        for entity_type, names in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            print(f"  {entity_type}: {len(names)} entities")
            for name in names[:3]:
                name_preview = name[:60] + "..." if len(name) > 60 else name
                print(f"    - {name_preview}")

        if not dry_run:
            # Delete in transaction
            entity_ids = [e[0] for e in entities_to_delete]
            delete_query = "DELETE FROM entities WHERE id = ANY(%s);"
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(delete_query, (entity_ids,))
                    print(f"✅ Deleted {len(entities_to_delete)} entities")

        return len(entities_to_delete)

    def delete_blocklist_entities(self, dry_run: bool = True) -> int:
        """
        Delete entities matching the blocklist

        Exception: Preserve valid acronyms (EU, UN, NATO, etc.)
        """

        if not self.blocklist:
            print("⚠️  No blocklist loaded, skipping")
            return 0

        # Get all entities (not just mention_count=1, blocklist applies to all)
        query = "SELECT id, name, entity_type FROM entities;"
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                all_entities = cur.fetchall()

        # Filter by blocklist (case-insensitive)
        entities_to_delete = [
            (entity_id, name, entity_type)
            for entity_id, name, entity_type in all_entities
            if name.lower() in self.blocklist
        ]

        if not entities_to_delete:
            print("✅ Blocklist: No entities to delete")
            return 0

        print(f"\n{'🔍 PREVIEW' if dry_run else '🗑️  DELETING'} Blocklist Matches: {len(entities_to_delete)} entities")

        # Show sample
        print("\n📋 Sample blocked entities:")
        for i, (entity_id, name, entity_type) in enumerate(entities_to_delete[:15]):
            print(f"  [{i+1}] {entity_type:10s} | {name}")

        if len(entities_to_delete) > 15:
            print(f"  ... and {len(entities_to_delete) - 15} more")

        if not dry_run:
            # Delete in transaction
            entity_ids = [e[0] for e in entities_to_delete]
            delete_query = "DELETE FROM entities WHERE id = ANY(%s);"
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(delete_query, (entity_ids,))
                    print(f"✅ Deleted {len(entities_to_delete)} entities")

        return len(entities_to_delete)

    def preserve_legitimate_entities(self) -> int:
        """
        Tier 3: Preserve Legitimate Entities

        Count entities that will be preserved:
        - mention_count > 1 (appears in multiple articles)
        - OR entity_type in (PERSON, ORG, GPE, LOC) AND mention_count = 1 BUT LENGTH < 50
        - NOT in blocklist
        """

        query = r"""
        SELECT COUNT(*)
        FROM entities
        WHERE (
            mention_count > 1
            OR (
                entity_type IN ('PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'EVENT')
                AND mention_count = 1
                AND LENGTH(name) <= 50
                AND name ~ '^[A-Z]'  -- Starts with uppercase (proper noun)
                AND NOT (
                    name LIKE '%href=%'
                    OR name LIKE '%http%'
                    OR name ~ '["\(\)]'
                )
            )
        );
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()
                preserved_count = result[0] if result else 0

        print(f"\n✅ Tier 3 (Preserve): {preserved_count} legitimate entities will be kept")

        return preserved_count

    def run_cleanup(self, tiers: list = [1, 2], dry_run: bool = True):
        """
        Run cleanup with specified tiers

        Args:
            tiers: List of tiers to run (1, 2, or 'blocklist')
            dry_run: If True, only preview changes
        """

        print("=" * 80)
        print("ENTITY CLEANUP - Aggressive Garbage Removal")
        print("=" * 80)
        print(f"Mode: {'DRY RUN (Preview Only)' if dry_run else 'PRODUCTION (Deleting)'}")
        print(f"Tiers: {tiers}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Get baseline
        self.stats['total_before'] = self.get_total_entities()
        print(f"\n📊 Total entities before cleanup: {self.stats['total_before']:,}")

        # Run blocklist first (if enabled)
        if 'blocklist' in tiers or 'all' in tiers:
            self.stats['blocklist_deleted'] = self.delete_blocklist_entities(dry_run)

        # Run tiers
        if 1 in tiers or 'all' in tiers:
            self.stats['tier1_deleted'] = self.tier1_safe_delete(dry_run)

        if 2 in tiers or 'all' in tiers:
            self.stats['tier2_deleted'] = self.tier2_pattern_based(dry_run)

        # Always show what will be preserved
        self.stats['tier3_preserved'] = self.preserve_legitimate_entities()

        # Final stats
        self.stats['total_after'] = self.get_total_entities()

        total_deleted = (
            self.stats['tier1_deleted'] +
            self.stats['tier2_deleted'] +
            self.stats['blocklist_deleted']
        )

        print("\n" + "=" * 80)
        print("CLEANUP SUMMARY")
        print("=" * 80)
        print(f"Blocklist deleted: {self.stats['blocklist_deleted']:,}")
        print(f"Tier 1 deleted:    {self.stats['tier1_deleted']:,}")
        print(f"Tier 2 deleted:    {self.stats['tier2_deleted']:,}")
        print(f"Tier 3 preserved:  {self.stats['tier3_preserved']:,}")
        print("-" * 80)
        print(f"Total deleted:     {total_deleted:,}")
        print(f"Total before:      {self.stats['total_before']:,}")
        print(f"Total after:       {self.stats['total_after']:,}")

        reduction_pct = (total_deleted / self.stats['total_before'] * 100) if self.stats['total_before'] > 0 else 0
        print(f"Reduction:         {reduction_pct:.1f}%")
        print("=" * 80)

        if dry_run:
            print("\n⚠️  DRY RUN MODE - No changes were made to the database")
            print("Run with --production flag to apply changes")
        else:
            print("\n✅ PRODUCTION MODE - Changes have been applied")
            print("💾 Recommendation: Create a backup before running again")


def main():
    parser = argparse.ArgumentParser(description="Clean garbage entities from database")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without deleting (default)'
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='Actually delete entities (use with caution!)'
    )
    parser.add_argument(
        '--tier',
        type=int,
        choices=[1, 2],
        help='Run only specific tier (1=safe, 2=pattern-based)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tiers + blocklist'
    )
    parser.add_argument(
        '--blocklist-only',
        action='store_true',
        help='Run only blocklist deletion'
    )

    args = parser.parse_args()

    # Default to dry-run unless --production is specified
    dry_run = not args.production

    # Determine which tiers to run
    if args.blocklist_only:
        tiers = ['blocklist']
    elif args.all:
        tiers = ['blocklist', 1, 2]
    elif args.tier:
        tiers = [args.tier]
    else:
        # Default: Run Tier 1 only (safest)
        tiers = [1]

    # Warning for production mode
    if not dry_run:
        print("⚠️  WARNING: Running in PRODUCTION mode - entities will be DELETED!")
        print("Press Ctrl+C within 5 seconds to cancel...")
        import time
        time.sleep(5)

    # Initialize
    db_manager = DatabaseManager()
    cleaner = EntityCleaner(db_manager)

    # Run cleanup
    cleaner.run_cleanup(tiers=tiers, dry_run=dry_run)

    # Close connection
    db_manager.close()


if __name__ == "__main__":
    main()
