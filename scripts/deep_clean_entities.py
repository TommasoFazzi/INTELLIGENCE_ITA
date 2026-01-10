#!/usr/bin/env python3
"""
Deep Entity Cleanup Script

Hybrid approach:
1. Level 1: Conservative cleanup (remove obvious garbage)
2. Level 2: Cross-reference with entities table (confidence boost)

Updates articles.entities JSONB with cleaned structure:
{
    "by_type": {...},           # Original (unchanged for backwards compat)
    "clean": {
        "high_confidence": [],   # Entities in validated entities table
        "medium_confidence": [], # Pass filters but not in table
        "all": []               # Union of both
    }
}

Usage:
    python scripts/deep_clean_entities.py [--dry-run] [--limit N] [--verbose]
"""

import sys
import re
import argparse
from pathlib import Path
from collections import Counter

import yaml
from psycopg2.extras import Json

sys.path.insert(0, '.')

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_blocklist() -> set:
    """Load all blocklisted terms from YAML config."""
    blocklist_path = Path("config/entity_blocklist.yaml")

    if not blocklist_path.exists():
        logger.warning(f"Blocklist not found at {blocklist_path}")
        return set()

    with open(blocklist_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Flatten all blocklist categories into a single set (lowercase for matching)
    blocklist = set()
    skip_categories = {'valid_acronyms'}  # Don't block these

    for category, terms in config.items():
        if category in skip_categories:
            continue
        if isinstance(terms, list):
            blocklist.update(t.lower() for t in terms if t)

    logger.info(f"Loaded {len(blocklist)} blocklisted terms")
    return blocklist


def load_valid_acronyms() -> set:
    """Load valid acronyms that should NOT be filtered."""
    blocklist_path = Path("config/entity_blocklist.yaml")

    if not blocklist_path.exists():
        return set()

    with open(blocklist_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    acronyms = config.get('valid_acronyms', [])
    return set(a.upper() for a in acronyms if a)


def load_validated_entities(db: DatabaseManager) -> set:
    """Load all entity names from the validated entities table."""
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT LOWER(name) FROM entities")
            return set(row[0] for row in cur.fetchall())


def is_garbage(entity: str, blocklist: set, valid_acronyms: set) -> bool:
    """
    Check if an entity should be filtered out.

    Returns True if entity is garbage.
    """
    if not entity or not entity.strip():
        return True

    entity = entity.strip()
    entity_lower = entity.lower()

    # Length check (2-50 chars)
    if len(entity) < 2 or len(entity) > 50:
        # Exception for valid acronyms
        if entity.upper() not in valid_acronyms:
            return True

    # Starts with number (unless it's a valid pattern like "G7")
    if re.match(r'^\d', entity) and entity.upper() not in valid_acronyms:
        return True

    # In blocklist
    if entity_lower in blocklist:
        return True

    # Contains only punctuation or whitespace
    if re.match(r'^[\s\W]+$', entity):
        return True

    # Looks like a URL or path
    if '/' in entity or 'http' in entity_lower or '.com' in entity_lower:
        return True

    return False


def clean_entity_list(
    entities: list,
    blocklist: set,
    valid_acronyms: set,
    validated_entities: set
) -> dict:
    """
    Clean a list of entities.

    Returns dict with:
        high_confidence: entities in validated table
        medium_confidence: pass filters but not validated
        all: union of both
    """
    seen = set()
    high_conf = []
    medium_conf = []

    for entity in entities:
        if not entity:
            continue

        entity = entity.strip()

        # Skip duplicates (case-insensitive)
        entity_lower = entity.lower()
        if entity_lower in seen:
            continue
        seen.add(entity_lower)

        # Skip garbage
        if is_garbage(entity, blocklist, valid_acronyms):
            continue

        # Classify by confidence
        if entity_lower in validated_entities:
            high_conf.append(entity)
        else:
            medium_conf.append(entity)

    return {
        'high_confidence': high_conf,
        'medium_confidence': medium_conf,
        'all': high_conf + medium_conf
    }


def process_article_entities(
    entities_json: dict,
    blocklist: set,
    valid_acronyms: set,
    validated_entities: set
) -> dict:
    """
    Process all entities in an article's JSONB.

    Preserves original 'by_type' and adds 'clean' structure.
    """
    by_type = entities_json.get('by_type', {})

    # Collect all entities by priority (GPE > ORG > PERSON > others)
    priority_types = ['GPE', 'ORG', 'PERSON', 'LOC', 'FAC']
    all_entities = []

    for etype in priority_types:
        all_entities.extend(by_type.get(etype, []))

    # Add any other types
    for etype, elist in by_type.items():
        if etype not in priority_types:
            all_entities.extend(elist or [])

    # Clean
    clean_result = clean_entity_list(
        all_entities, blocklist, valid_acronyms, validated_entities
    )

    # Return updated structure
    return {
        'by_type': by_type,  # Preserve original
        'clean': clean_result
    }


def main():
    parser = argparse.ArgumentParser(description="Deep entity cleanup")
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help="Show changes without writing to DB")
    parser.add_argument('--limit', '-n', type=int, default=None,
                        help="Limit number of articles to process")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Show detailed per-article changes")
    args = parser.parse_args()

    print("=" * 70)
    print("DEEP ENTITY CLEANUP")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")

    db = DatabaseManager()

    # Load resources
    blocklist = load_blocklist()
    valid_acronyms = load_valid_acronyms()
    validated_entities = load_validated_entities(db)
    print(f"Validated entities in table: {len(validated_entities)}")

    # Get articles with entities
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            query = "SELECT id, entities FROM articles WHERE entities IS NOT NULL"
            if args.limit:
                query += f" LIMIT {args.limit}"
            cur.execute(query)
            articles = cur.fetchall()

    print(f"\nProcessing {len(articles)} articles...")

    # Stats
    stats = {
        'total_before': 0,
        'total_after': 0,
        'high_confidence': 0,
        'medium_confidence': 0,
        'duplicates_removed': 0,
        'garbage_removed': 0,
    }

    updates = []

    for article_id, entities_json in articles:
        if not entities_json:
            continue

        # Count before
        by_type = entities_json.get('by_type', {})
        before_count = sum(len(v) for v in by_type.values() if v)
        stats['total_before'] += before_count

        # Process
        cleaned = process_article_entities(
            entities_json, blocklist, valid_acronyms, validated_entities
        )

        # Count after
        clean_data = cleaned['clean']
        after_count = len(clean_data['all'])
        stats['total_after'] += after_count
        stats['high_confidence'] += len(clean_data['high_confidence'])
        stats['medium_confidence'] += len(clean_data['medium_confidence'])
        stats['garbage_removed'] += before_count - after_count

        if args.verbose:
            print(f"\n[Article {article_id}]")
            print(f"  Before: {before_count} entities")
            print(f"  After:  {after_count} ({len(clean_data['high_confidence'])} high, "
                  f"{len(clean_data['medium_confidence'])} medium)")
            if clean_data['high_confidence'][:5]:
                print(f"  High:   {clean_data['high_confidence'][:5]}")
            if clean_data['medium_confidence'][:5]:
                print(f"  Medium: {clean_data['medium_confidence'][:5]}")

        updates.append((cleaned, article_id))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Articles processed:     {len(articles)}")
    print(f"Entities before:        {stats['total_before']}")
    print(f"Entities after:         {stats['total_after']}")
    print(f"Garbage removed:        {stats['garbage_removed']} "
          f"({stats['garbage_removed']/max(stats['total_before'],1)*100:.1f}%)")
    print(f"High confidence:        {stats['high_confidence']}")
    print(f"Medium confidence:      {stats['medium_confidence']}")

    # Apply updates
    if not args.dry_run and updates:
        print(f"\nWriting {len(updates)} updates to database...")
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                for cleaned_json, article_id in updates:
                    cur.execute(
                        "UPDATE articles SET entities = %s WHERE id = %s",
                        (Json(cleaned_json), article_id)
                    )
            conn.commit()
        print("Done!")
    elif args.dry_run:
        print("\n[DRY RUN - no changes written]")


if __name__ == "__main__":
    main()
