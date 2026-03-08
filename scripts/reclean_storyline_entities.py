#!/usr/bin/env python3
"""
Batch Re-clean Storyline Key Entities

Applies rule-based cleanup to all existing storylines' key_entities,
removing garbage that accumulated from unfiltered spaCy NER:
  - Numeric prefixes (4Trump → Trump)
  - HTML/navigation fragments
  - Article titles masquerading as entities
  - Too-short/too-long strings
  - Trailing punctuation
  - Case-insensitive deduplication

Usage:
    python scripts/reclean_storyline_entities.py               # Live run
    python scripts/reclean_storyline_entities.py --dry-run      # Preview only
    python scripts/reclean_storyline_entities.py --verbose       # Show per-storyline diffs
"""

import sys
import re
import argparse
from collections import Counter

from psycopg2.extras import Json

sys.path.insert(0, '.')

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Garbage Detection ────────────────────────────────────────────────────────

_GARBAGE_PATTERNS = re.compile(
    r'^\d+[a-zA-Z]'                # numeric prefix: "4Trump", "3Hamas"
    r'|^\d+$'                      # pure numbers: "2024", "100"
    r'|\b(?:http|www\.|\.[a-z]{2,4})\b'  # URLs
    r'|(?:Features|Podcasts|Pictures|Investigations|Interactives|Newsletter|Subscribe)'
    r'|(?:Science & Technology|Human Rights|Climate Crisis)'
    r'|^[A-Z]{1,2}$'              # 1-2 letter acronyms
    r'|[\|\[\]{}()]'              # brackets, pipes (HTML)
    r'|^\W+$'                     # only punctuation
    r'|\s{3,}',                   # nav fragments with big gaps
    re.IGNORECASE
)

_VALID_SHORT = {'EU', 'UN', 'US', 'UK', 'G7', 'G20', 'AI', 'ONU', 'UE'}

# Common false positives from spaCy NER
_FALSE_POSITIVES = {
    'not', 'feb', 'jan', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'est', 'gmt', 'cet', 'bst', 'pst', 'edt', 'cdt', 'utc',
    'et', 'pm', 'am', 'dc', 'vs', 'op', 'no', 'ok', 'ad',
    'the', 'this', 'that', 'its', 'his', 'her',
}


def is_garbage(entity: str) -> bool:
    """Check if entity is garbage."""
    if not entity or not isinstance(entity, str):
        return True
    e = entity.strip()
    if len(e) < 2 or len(e) > 60:
        return True
    # Reject entities with too many words (likely article titles)
    if len(e.split()) > 6:
        return True
    # Known false positives
    if e.lower() in _FALSE_POSITIVES:
        return True
    if e.upper() in _VALID_SHORT:
        return False
    if _GARBAGE_PATTERNS.search(e):
        return True
    stripped = e.strip(' -–—.,;:!?/')
    if len(stripped) < 2:
        return True
    return False


def clean_entity(entity: str) -> str:
    """Normalize entity string."""
    e = entity.strip()
    # Strip leading digit prefix
    e = re.sub(r'^\d+', '', e).strip()
    # Strip trailing punctuation
    e = e.strip(' -–—.,;:!?/')    # Strip leading "The " / "Il " / "La " / "L'"
    e = re.sub(r'^(?:The|Il|La|Lo|Le|Gli|L[\u2019\'])\s+', '', e, flags=re.IGNORECASE).strip()    # Collapse whitespace
    e = re.sub(r'\s+', ' ', e).strip()
    return e


def clean_entity_list(entities: list) -> list:
    """Clean a list of entities: normalize, filter garbage, deduplicate."""
    seen = set()
    result = []
    for entity in entities:
        cleaned = clean_entity(entity)
        if is_garbage(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Re-clean storyline key_entities")
    parser.add_argument('--dry-run', action='store_true', help="Preview without writing to DB")
    parser.add_argument('--verbose', action='store_true', help="Show per-storyline diffs")
    parser.add_argument('--status', nargs='+', default=['emerging', 'active'],
                        help="Storyline statuses to process (default: emerging active)")
    args = parser.parse_args()

    db = DatabaseManager()
    status_filter = tuple(args.status)

    print("=" * 70)
    print("STORYLINE KEY ENTITIES RE-CLEAN")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Statuses: {', '.join(args.status)}")
    print("=" * 70)

    # Fetch all storylines with key_entities
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            placeholders = ','.join(['%s'] * len(status_filter))
            cur.execute(f"""
                SELECT id, title, key_entities, momentum_score
                FROM storylines
                WHERE narrative_status IN ({placeholders})
                  AND key_entities IS NOT NULL
                ORDER BY momentum_score DESC
            """, status_filter)
            rows = cur.fetchall()

    print(f"\nFound {len(rows)} storylines to process\n")

    stats = Counter()
    updates = []

    for row in rows:
        sid, title, entities, momentum = row
        if not entities:
            stats['skipped_null'] += 1
            continue

        original = list(entities) if isinstance(entities, list) else []
        cleaned = clean_entity_list(original)

        removed = set(e.lower() for e in original) - set(e.lower() for e in cleaned)
        n_removed = len(original) - len(cleaned)

        if n_removed > 0:
            stats['storylines_modified'] += 1
            stats['entities_removed'] += n_removed
            updates.append((sid, cleaned))

            if args.verbose:
                print(f"  #{sid:4d} ({momentum:.2f}) {title[:50]}")
                print(f"         BEFORE ({len(original)}): {original[:8]}...")
                print(f"         AFTER  ({len(cleaned)}): {cleaned[:8]}...")
                if removed:
                    print(f"         REMOVED: {sorted(removed)[:6]}...")
                print()
        else:
            stats['storylines_clean'] += 1

    # Summary
    print("-" * 70)
    print(f"  Storylines processed:  {len(rows)}")
    print(f"  Already clean:         {stats['storylines_clean']}")
    print(f"  Needing cleanup:       {stats['storylines_modified']}")
    print(f"  Total entities removed: {stats['entities_removed']}")
    print("-" * 70)

    if not updates:
        print("\nNo changes needed.")
        return

    if args.dry_run:
        print(f"\nDRY RUN: {len(updates)} storylines would be updated. Run without --dry-run to apply.")
        return

    # Apply updates
    print(f"\nApplying {len(updates)} updates...")
    applied = 0
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            for sid, cleaned_entities in updates:
                cur.execute("""
                    UPDATE storylines SET key_entities = %s WHERE id = %s
                """, (Json(cleaned_entities), sid))
                applied += 1
        conn.commit()

    print(f"Done! Updated {applied} storylines.")
    print("\nNext steps:")
    print("  1. Rebuild graph edges: python scripts/rebuild_graph_edges.py")
    print("  2. Refresh IDF view:    psql -c 'REFRESH MATERIALIZED VIEW entity_idf;'")
    print("  3. Recompute communities: python scripts/compute_communities.py")


if __name__ == '__main__':
    main()
