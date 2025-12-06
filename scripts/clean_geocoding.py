#!/usr/bin/env python3
"""
Geocoding Data Cleaning Script

Fixes:
1. Wrong coordinates (e.g., "Latin America" geocoded to Boston)
2. Non-geographic entities (e.g., "Congress" as a place)
3. Duplicates (e.g., "China" as both GPE and LOC)
4. Suspicious geocoding results
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Entities that should NOT be geocoded (not geographic locations)
NON_GEOGRAPHIC_BLACKLIST = {
    'Congress',      # US legislative body, not a place
    'Parliament',    # Legislative body
    'Senate',
    'Pentagon',      # Building, but often refers to the organization
    'White House',   # Building, but often refers to the administration
    'Kremlin',       # Building, but often refers to the government
    'UN',
    'NATO',
    'EU',
    'World Bank',
    'IMF',
}

# Correct coordinates for commonly mis-geocoded entities
MANUAL_CORRECTIONS = {
    'Latin America': (-8.7832, -55.4915),  # Center of South America (Brazil)
    'DC': (38.9072, -77.0369),             # Washington DC
    'Jenin': (32.4608, 35.2955),           # Jenin, Palestine (not Poland!)
    'Kremlin': (55.7520, 37.6175),         # Kremlin, Moscow
    'America': (37.0902, -95.7129),        # Center of United States
    'Middle East': (29.2985, 42.5510),     # Center of Middle East region
    'Central Asia': (43.2220, 76.8512),    # Center of Central Asia (Kazakhstan)
    'Southeast Asia': (4.5353, 108.1428),  # Center of Southeast Asia
    'Eastern Europe': (50.0000, 26.0000),  # Center of Eastern Europe
    'Palestine': (31.9522, 35.2332),       # Palestine (West Bank + Gaza, not Texas!)
    'Gaza Strip': (31.3547, 34.3088),      # Gaza Strip (will be merged with Gaza)
}

# Entity types to keep (geographic only)
VALID_ENTITY_TYPES = ['GPE', 'LOC', 'FAC']


def remove_non_geographic_entities(db: DatabaseManager):
    """Remove entities that are not geographic locations"""
    logger.info("Removing non-geographic entities...")

    removed = 0
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Remove from blacklist
            for entity_name in NON_GEOGRAPHIC_BLACKLIST:
                cur.execute("""
                    UPDATE entities
                    SET geo_status = 'NOT_FOUND',
                        latitude = NULL,
                        longitude = NULL
                    WHERE name = %s
                      AND geo_status = 'FOUND'
                """, (entity_name,))
                if cur.rowcount > 0:
                    logger.info(f"  ✗ Removed: {entity_name} (not a geographic location)")
                    removed += cur.rowcount

            # Remove entities that are clearly sentence fragments
            # Pattern 1: Contains temporal words (indicates it's a sentence, not a place)
            cur.execute("""
                UPDATE entities
                SET geo_status = 'NOT_FOUND',
                    latitude = NULL,
                    longitude = NULL
                WHERE (
                    name ILIKE '% on %' OR
                    name ILIKE '% monday%' OR
                    name ILIKE '% tuesday%' OR
                    name ILIKE '% wednesday%' OR
                    name ILIKE '% thursday%' OR
                    name ILIKE '% friday%' OR
                    name ILIKE '% saturday%' OR
                    name ILIKE '% sunday%' OR
                    name ILIKE '% meanwhile%' OR
                    name ILIKE '% while%' OR
                    name ILIKE 'from % to %' OR
                    name ILIKE '% this month%' OR
                    name ILIKE '% last week%'
                )
                AND geo_status = 'FOUND'
            """)
            temporal_removed = cur.rowcount
            if temporal_removed > 0:
                logger.info(f"  ✗ Removed {temporal_removed} temporal fragments (e.g., 'Gaza on Monday')")
                removed += temporal_removed

            # Pattern 2: Entity names that are too long (>50 chars = likely full sentences)
            cur.execute("""
                UPDATE entities
                SET geo_status = 'NOT_FOUND',
                    latitude = NULL,
                    longitude = NULL
                WHERE LENGTH(name) > 50
                  AND geo_status = 'FOUND'
            """)
            long_removed = cur.rowcount
            if long_removed > 0:
                logger.info(f"  ✗ Removed {long_removed} overly long names (>50 chars, likely sentences)")
                removed += long_removed

            # Pattern 3: Contains punctuation (quotes, colons, etc.) indicating sentence fragments
            cur.execute("""
                UPDATE entities
                SET geo_status = 'NOT_FOUND',
                    latitude = NULL,
                    longitude = NULL
                WHERE (
                    name LIKE '%:%' OR
                    name LIKE '%"%' OR
                    name LIKE '%''%' OR
                    name LIKE '%.%' OR
                    name LIKE '%|%' OR
                    name LIKE '%-%-%' OR
                    name LIKE '%  %'
                )
                AND geo_status = 'FOUND'
            """)
            punct_removed = cur.rowcount
            if punct_removed > 0:
                logger.info(f"  ✗ Removed {punct_removed} entities with punctuation (likely fragments)")
                removed += punct_removed

            # Pattern 4: Dates and times (e.g., "Mar 2026", "7:00 PM")
            cur.execute("""
                UPDATE entities
                SET geo_status = 'NOT_FOUND',
                    latitude = NULL,
                    longitude = NULL
                WHERE (
                    name ILIKE '%jan %' OR name ILIKE '%feb %' OR name ILIKE '%mar %' OR
                    name ILIKE '%apr %' OR name ILIKE '%may %' OR name ILIKE '%jun %' OR
                    name ILIKE '%jul %' OR name ILIKE '%aug %' OR name ILIKE '%sep %' OR
                    name ILIKE '%oct %' OR name ILIKE '%nov %' OR name ILIKE '%dec %' OR
                    name ILIKE '% 20__' OR name ILIKE '% 19__' OR
                    name ILIKE '%am%' OR name ILIKE '%pm%' OR
                    name ~ '\d{1,2}:\d{2}'
                )
                AND geo_status = 'FOUND'
            """)
            date_removed = cur.rowcount
            if date_removed > 0:
                logger.info(f"  ✗ Removed {date_removed} date/time entities")
                removed += date_removed

            # Pattern 5: Very short names (1-2 chars) that are likely noise
            cur.execute("""
                UPDATE entities
                SET geo_status = 'NOT_FOUND',
                    latitude = NULL,
                    longitude = NULL
                WHERE LENGTH(name) <= 2
                AND geo_status = 'FOUND'
            """)
            short_removed = cur.rowcount
            if short_removed > 0:
                logger.info(f"  ✗ Removed {short_removed} very short names (≤2 chars)")
                removed += short_removed

    logger.info(f"✓ Removed {removed} non-geographic entities total")
    return removed


def apply_manual_corrections(db: DatabaseManager):
    """Apply manual coordinate corrections for known bad geocoding"""
    logger.info("Applying manual coordinate corrections...")

    corrected = 0
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            for entity_name, (lat, lng) in MANUAL_CORRECTIONS.items():
                cur.execute("""
                    UPDATE entities
                    SET latitude = %s,
                        longitude = %s,
                        geocoded_at = %s
                    WHERE name = %s
                      AND geo_status = 'FOUND'
                """, (lat, lng, datetime.now(), entity_name))
                if cur.rowcount > 0:
                    logger.info(f"  ✓ Corrected: {entity_name} → ({lat}, {lng})")
                    corrected += cur.rowcount

    logger.info(f"✓ Corrected {corrected} entities with manual coordinates")
    return corrected


def merge_duplicate_entities(db: DatabaseManager):
    """Merge duplicate entities (same name, different types)"""
    logger.info("Merging duplicate entities...")

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Find duplicates (same name, both geocoded)
            cur.execute("""
                SELECT name, COUNT(*) as count
                FROM entities
                WHERE geo_status = 'FOUND'
                  AND entity_type = ANY(%s)
                GROUP BY name
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
            """, (VALID_ENTITY_TYPES,))

            duplicates = cur.fetchall()
            logger.info(f"Found {len(duplicates)} duplicate entity names")

            merged = 0
            for name, count in duplicates:
                # Get all instances of this entity
                cur.execute("""
                    SELECT id, entity_type, latitude, longitude, mention_count
                    FROM entities
                    WHERE name = %s
                      AND geo_status = 'FOUND'
                    ORDER BY mention_count DESC, entity_type
                """, (name,))

                instances = cur.fetchall()

                # Keep the one with highest mention_count (primary)
                primary_id = instances[0][0]
                primary_type = instances[0][1]
                total_mentions = sum(inst[4] for inst in instances)

                # Merge all others into primary
                for inst_id, inst_type, lat, lng, mentions in instances[1:]:
                    logger.info(f"  Merging: {name} ({inst_type}) → {name} ({primary_type})")

                    # Update entity_mentions to point to primary
                    cur.execute("""
                        UPDATE entity_mentions
                        SET entity_id = %s
                        WHERE entity_id = %s
                    """, (primary_id, inst_id))

                    # Delete duplicate
                    cur.execute("DELETE FROM entities WHERE id = %s", (inst_id,))
                    merged += 1

                # Update primary with total mentions
                cur.execute("""
                    UPDATE entities
                    SET mention_count = %s
                    WHERE id = %s
                """, (total_mentions, primary_id))

    logger.info(f"✓ Merged {merged} duplicate entities")
    return merged


def detect_suspicious_coordinates(db: DatabaseManager):
    """Detect entities with suspicious coordinates"""
    logger.info("Detecting suspicious coordinates...")

    suspicious = []

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Get all geocoded entities
            cur.execute("""
                SELECT name, entity_type, latitude, longitude
                FROM entities
                WHERE geo_status = 'FOUND'
                  AND entity_type = ANY(%s)
                ORDER BY name
            """, (VALID_ENTITY_TYPES,))

            for name, entity_type, lat, lng in cur.fetchall():
                # Check for suspicious patterns
                is_suspicious = False
                reason = ""

                # Regional names geocoded to specific cities
                if any(region in name.lower() for region in ['america', 'asia', 'europe', 'africa']):
                    if 'latin america' not in name.lower():  # Already corrected
                        # Regions should have broad coordinates, not precise city coords
                        if abs(lat) > 60 or abs(lng) > 120:  # Very specific coordinates
                            is_suspicious = True
                            reason = "Regional name with precise coordinates"

                # Entity names that are clearly not places
                non_place_keywords = ['congress', 'parliament', 'government', 'administration']
                if any(keyword in name.lower() for keyword in non_place_keywords):
                    is_suspicious = True
                    reason = "Entity name suggests non-geographic location"

                if is_suspicious:
                    suspicious.append({
                        'name': name,
                        'type': entity_type,
                        'lat': lat,
                        'lng': lng,
                        'reason': reason
                    })

    if suspicious:
        logger.warning(f"Found {len(suspicious)} suspicious entities:")
        for item in suspicious[:20]:  # Show first 20
            logger.warning(f"  ⚠ {item['name']:30s} ({item['lat']:.2f}, {item['lng']:.2f}) - {item['reason']}")
    else:
        logger.info("✓ No suspicious coordinates detected")

    return suspicious


def filter_by_relevance(db: DatabaseManager, min_mentions: int = 2):
    """Remove entities with too few mentions (likely noise)"""
    logger.info(f"Filtering entities with < {min_mentions} mentions...")

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE entities
                SET geo_status = 'NOT_FOUND',
                    latitude = NULL,
                    longitude = NULL
                WHERE mention_count < %s
                  AND geo_status = 'FOUND'
            """, (min_mentions,))

            removed = cur.rowcount
            logger.info(f"✓ Removed {removed} low-relevance entities (< {min_mentions} mentions)")

    return removed


def show_statistics(db: DatabaseManager):
    """Show geocoding statistics"""
    logger.info("\n" + "="*60)
    logger.info("Geocoding Statistics:")

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Overall stats
            cur.execute("""
                SELECT
                    geo_status,
                    COUNT(*) as count
                FROM entities
                WHERE entity_type = ANY(%s)
                GROUP BY geo_status
                ORDER BY count DESC
            """, (VALID_ENTITY_TYPES,))

            for status, count in cur.fetchall():
                logger.info(f"  {status:15s}: {count:5d}")

            # Unique names (after deduplication)
            cur.execute("""
                SELECT COUNT(DISTINCT name)
                FROM entities
                WHERE geo_status = 'FOUND'
                  AND entity_type = ANY(%s)
            """, (VALID_ENTITY_TYPES,))

            unique_count = cur.fetchone()[0]
            logger.info(f"  {'Unique names':15s}: {unique_count:5d}")

            # Show mention distribution
            cur.execute("""
                SELECT mention_count, COUNT(*) as entities
                FROM entities
                WHERE geo_status = 'FOUND'
                  AND entity_type = ANY(%s)
                GROUP BY mention_count
                ORDER BY mention_count DESC
                LIMIT 10
            """, (VALID_ENTITY_TYPES,))

            logger.info("\n  Top mention counts:")
            for mentions, count in cur.fetchall():
                logger.info(f"    {mentions} mentions: {count} entities")

    logger.info("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Clean geocoding data')
    parser.add_argument('--min-mentions', type=int, default=2,
                       help='Minimum mentions to keep entity (default: 2)')
    parser.add_argument('--aggressive', action='store_true',
                       help='Aggressive cleanup (min 3 mentions)')

    args = parser.parse_args()

    min_mentions = 3 if args.aggressive else args.min_mentions

    logger.info("Starting geocoding data cleanup...\n")
    logger.info(f"Configuration: min_mentions={min_mentions}\n")

    db = DatabaseManager()

    # Step 1: Remove non-geographic entities
    removed = remove_non_geographic_entities(db)

    # Step 2: Apply manual corrections
    corrected = apply_manual_corrections(db)

    # Step 3: Merge duplicates
    merged = merge_duplicate_entities(db)

    # Step 4: Filter by relevance (NEW)
    filtered = filter_by_relevance(db, min_mentions=min_mentions)

    # Step 5: Detect remaining suspicious coordinates
    suspicious = detect_suspicious_coordinates(db)

    # Step 6: Show statistics
    show_statistics(db)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Cleanup Summary:")
    logger.info(f"  ✗ Non-geographic removed: {removed}")
    logger.info(f"  ✓ Manually corrected:     {corrected}")
    logger.info(f"  🔀 Duplicates merged:      {merged}")
    logger.info(f"  🧹 Low-relevance filtered: {filtered}")
    logger.info(f"  ⚠ Suspicious remaining:   {len(suspicious)}")
    logger.info("="*60)

    logger.info("\n✓ Cleanup completed!")
