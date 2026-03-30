#!/usr/bin/env python3
"""
Load OpenSanctions FtM entities into sanctions_registry.

Streams NDJSON (Follow the Money) entities from OpenSanctions export,
transforms to sanctions_registry schema, and bulk-inserts.

Download the dataset first:
    curl -fsSLO https://data.opensanctions.org/datasets/latest/default/entities.ftm.json

Usage:
    python scripts/load_opensanctions.py                         # Load from data/entities.ftm.json
    python scripts/load_opensanctions.py --file /path/to/file    # Custom path
    python scripts/load_opensanctions.py --dry-run               # Count without saving
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# FtM schemas we care about
RELEVANT_SCHEMAS = {'Person', 'Company', 'Organization', 'LegalEntity', 'Vessel', 'Aircraft'}


def transform_entity(raw: dict) -> dict:
    """Transform FtM entity to sanctions_registry row."""
    properties = raw.get('properties', {})

    # Extract fields from FtM properties
    aliases = properties.get('alias', []) + properties.get('weakAlias', [])
    countries = list(set(
        properties.get('country', []) +
        properties.get('nationality', []) +
        properties.get('jurisdiction', [])
    ))

    return {
        'id': raw.get('id', ''),
        'caption': raw.get('caption', ''),
        'schema_type': raw.get('schema', ''),
        'aliases': aliases[:50] if aliases else None,  # Cap alias count
        'countries': [c.upper()[:2] for c in countries if len(c) == 2][:20] or None,
        'datasets': raw.get('datasets', [])[:10] or None,
        'properties': json.dumps(properties),
        'first_seen': raw.get('first_seen'),
        'last_seen': raw.get('last_seen'),
    }


INSERT_SQL = """
    INSERT INTO sanctions_registry (id, caption, schema_type, aliases, countries,
                                    datasets, properties, first_seen, last_seen)
    VALUES (%(id)s, %(caption)s, %(schema_type)s, %(aliases)s, %(countries)s,
            %(datasets)s, %(properties)s::jsonb, %(first_seen)s, %(last_seen)s)
    ON CONFLICT (id) DO UPDATE SET
        caption = EXCLUDED.caption,
        aliases = EXCLUDED.aliases,
        countries = EXCLUDED.countries,
        datasets = EXCLUDED.datasets,
        properties = EXCLUDED.properties,
        last_seen = EXCLUDED.last_seen,
        last_updated = NOW()
"""


def main():
    parser = argparse.ArgumentParser(description="Load OpenSanctions entities")
    parser.add_argument('--file', type=str, default='data/entities.ftm.json',
                        help='Path to FtM NDJSON file')
    parser.add_argument('--dry-run', action='store_true', help='Count without saving')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("OPENSANCTIONS REGISTRY LOADER")
    logger.info("=" * 80)

    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        logger.info("Download with: curl -fsSLO https://data.opensanctions.org/datasets/latest/default/entities.ftm.json")
        return 1

    logger.info(f"Loading from: {file_path}")

    db = None if args.dry_run else DatabaseManager()

    total = 0
    relevant = 0
    saved = 0
    errors = 0
    batch = []
    BATCH_SIZE = 1000

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            try:
                entity = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            # Filter by relevant schema types
            if entity.get('schema') not in RELEVANT_SCHEMAS:
                continue

            relevant += 1

            if args.dry_run:
                continue

            row = transform_entity(entity)
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                saved += _flush_batch(db, batch)
                batch.clear()

            if relevant % 10000 == 0:
                logger.info(f"  ... processed {total:,} entities, {relevant:,} relevant, {saved:,} saved")

    # Final batch
    if batch and not args.dry_run:
        saved += _flush_batch(db, batch)

    if args.dry_run:
        logger.info(f"\n[DRY RUN] Total: {total:,}, Relevant: {relevant:,}")
        return 0

    logger.info(f"\n  ✓ Total scanned: {total:,}")
    logger.info(f"  ✓ Relevant entities: {relevant:,}")
    logger.info(f"  ✓ Saved: {saved:,}")
    logger.info(f"  ✗ Errors: {errors:,}")

    if db:
        db.close()

    logger.info("✓ OpenSanctions loading complete!")
    return 0


def _flush_batch(db: DatabaseManager, batch: list) -> int:
    """Insert a batch of entities. Returns count of successful inserts."""
    saved = 0
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            for row in batch:
                try:
                    cur.execute(INSERT_SQL, row)
                    saved += 1
                except Exception as e:
                    logger.debug(f"Insert error for {row.get('id', '?')}: {e}")
            conn.commit()
    return saved


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
