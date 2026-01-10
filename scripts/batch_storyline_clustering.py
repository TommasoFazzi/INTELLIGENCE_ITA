#!/usr/bin/env python3
"""
Batch Storyline Clustering Script

Uses DBSCAN clustering on article embeddings to create storylines.
Generates LLM titles for each cluster using Gemini.

Usage:
    python scripts/batch_storyline_clustering.py                    # Default: 30 days
    python scripts/batch_storyline_clustering.py --days 60          # Custom time window
    python scripts/batch_storyline_clustering.py --dry-run          # Analyze without creating
    python scripts/batch_storyline_clustering.py --reset            # Reset and re-cluster
    python scripts/batch_storyline_clustering.py --eps 0.30         # Custom DBSCAN epsilon
    python scripts/batch_storyline_clustering.py --min-samples 3    # Min articles per cluster

Example:
    # First run: dry run to see clusters
    python scripts/batch_storyline_clustering.py --days 30 --dry-run

    # If clusters look good, run for real
    python scripts/batch_storyline_clustering.py --days 30

    # Reset and re-cluster all
    python scripts/batch_storyline_clustering.py --reset --days 60
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.nlp.story_manager import BatchClusterer
from src.storage.database import DatabaseManager


def main():
    parser = argparse.ArgumentParser(
        description='Batch clustering of articles into storylines using DBSCAN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Time window in days (0 = all articles). Default: 30'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze clusters without creating storylines'
    )

    parser.add_argument(
        '--reset',
        action='store_true',
        help='Delete all existing storylines before clustering'
    )

    parser.add_argument(
        '--eps',
        type=float,
        default=0.28,
        help='DBSCAN epsilon (max cosine distance). Lower = stricter clusters. Default: 0.28'
    )

    parser.add_argument(
        '--min-samples',
        type=int,
        default=2,
        help='Minimum articles to form a cluster. Default: 2'
    )

    parser.add_argument(
        '--include-assigned',
        action='store_true',
        help='Include articles already assigned to storylines'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed cluster information'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BATCH STORYLINE CLUSTERING")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Days: {args.days}")
    print(f"  Epsilon: {args.eps}")
    print(f"  Min samples: {args.min_samples}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Reset: {args.reset}")
    print(f"  Include assigned: {args.include_assigned}")
    print()

    # Initialize
    try:
        db = DatabaseManager()
        clusterer = BatchClusterer(
            db_manager=db,
            eps=args.eps,
            min_samples=args.min_samples
        )
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Run: pip install scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize: {e}")
        sys.exit(1)

    # Reset if requested
    if args.reset:
        print("\n--- RESETTING STORYLINES ---")
        if args.dry_run:
            print("Skipping reset (dry run mode)")
        else:
            confirm = input("This will DELETE all existing storylines. Continue? [y/N]: ")
            if confirm.lower() == 'y':
                result = clusterer.reset_storylines()
                print(f"Deleted {result['storylines_deleted']} storylines, {result['links_deleted']} links")
            else:
                print("Reset cancelled")
                sys.exit(0)

    # Run clustering
    print("\n--- CLUSTERING ---")
    result = clusterer.cluster_articles(
        days=args.days,
        exclude_assigned=not args.include_assigned,
        dry_run=args.dry_run
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Clusters found: {result['clusters_found']}")
    print(f"Storylines created: {result['storylines_created']}")
    print(f"Articles assigned: {result['articles_assigned']}")
    print(f"Noise articles (not clustered): {result['noise_articles']}")

    if args.verbose and result['clusters']:
        print("\n--- CLUSTER DETAILS ---")
        for cluster in result['clusters']:
            print(f"\nCluster {cluster['cluster_id']}: \"{cluster['title']}\"")
            print(f"  Articles ({cluster['article_count']}):")
            for article in cluster['articles'][:5]:  # Limit to 5
                print(f"    - [{article['source']}] {article['title'][:60]}...")
            if cluster['article_count'] > 5:
                print(f"    ... and {cluster['article_count'] - 5} more")

    # Summary
    if args.dry_run:
        print("\n[DRY RUN] No changes were made to the database.")
        print("Run without --dry-run to create storylines.")
    else:
        print(f"\nCreated {result['storylines_created']} storylines with {result['articles_assigned']} articles.")

    print("\nDone!")


if __name__ == '__main__':
    main()
