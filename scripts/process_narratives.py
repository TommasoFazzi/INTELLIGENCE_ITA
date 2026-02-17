#!/usr/bin/env python3
"""
Narrative Processing Script

Runs the NarrativeProcessor on recent articles to:
1. Micro-cluster near-duplicate articles into unique events
2. Match events to existing storylines (with temporal decay)
3. Discover new storylines via HDBSCAN clustering
4. Evolve storyline summaries with LLM (Gemini)
5. Build graph connections between storylines (entity overlap)
6. Apply decay to inactive storylines

Usage:
    python scripts/process_narratives.py                    # Default: 1 day
    python scripts/process_narratives.py --days 7           # Last 7 days
    python scripts/process_narratives.py --dry-run          # Analyze without DB writes
    python scripts/process_narratives.py --skip-llm         # Skip Gemini calls
    python scripts/process_narratives.py --days 30 --verbose
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.nlp.narrative_processor import NarrativeProcessor
from src.storage.database import DatabaseManager


def main():
    parser = argparse.ArgumentParser(
        description='Process articles into narrative storylines with graph connections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=1,
        help='Time window in days for fetching unassigned articles (default: 1)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze without writing to database'
    )

    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip LLM summary evolution (faster, no API costs)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed processing information'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NARRATIVE PROCESSING")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Days:     {args.days}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"  Skip LLM: {args.skip_llm}")
    print()

    # Initialize
    try:
        db = DatabaseManager()
        processor = NarrativeProcessor(
            db_manager=db,
            skip_llm=args.skip_llm,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize: {e}")
        sys.exit(1)

    # Run processing
    print("--- PROCESSING ---")
    result = processor.process_daily_batch(
        days=args.days,
        dry_run=args.dry_run,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Articles loaded:      {result['articles_loaded']}")
    print(f"Micro-clusters:       {result['micro_clusters']}")
    print(f"Events matched:       {result['events_matched']}")
    print(f"Events orphaned:      {result['events_orphaned']}")
    print(f"New storylines:       {result['new_storylines']}")
    print(f"Summaries evolved:    {result['summaries_evolved']}")
    print(f"Validated on-scope:   {result.get('validated_on_scope', 'N/A')}")
    print(f"Archived off-topic:   {result.get('archived_off_topic', 0)}")
    print(f"Graph edges updated:  {result['graph_edges_updated']}")

    if result['decay_stats']:
        ds = result['decay_stats']
        print(f"Decay - decayed:      {ds.get('decayed', 0)}")
        print(f"Decay - stabilized:   {ds.get('stabilized', 0)}")
        print(f"Decay - archived:     {ds.get('archived', 0)}")

    # Show active storylines summary if verbose
    if args.verbose and not args.dry_run:
        print("\n--- ACTIVE STORYLINES ---")
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, title, narrative_status, article_count,
                               momentum_score, key_entities
                        FROM storylines
                        WHERE narrative_status IN ('emerging', 'active')
                        ORDER BY momentum_score DESC
                        LIMIT 15
                    """)
                    rows = cur.fetchall()

            for row in rows:
                sid, title, status, count, momentum, entities = row
                ent_str = ", ".join((entities or [])[:5])
                print(f"\n  [{sid}] {title[:60]}")
                print(f"      Status: {status} | Articles: {count} | Momentum: {momentum:.2f}")
                print(f"      Entities: {ent_str}")
        except Exception as e:
            print(f"  Error fetching storylines: {e}")

    # Summary
    if args.dry_run:
        print("\n[DRY RUN] No changes were made to the database.")
    else:
        print(f"\nProcessed {result['articles_loaded']} articles into "
              f"{result['events_matched'] + result['new_storylines']} storylines.")

    print("\nDone!")


if __name__ == '__main__':
    main()
