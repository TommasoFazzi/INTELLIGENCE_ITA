#!/usr/bin/env python3
"""
Test Storyline Clustering

Runs the StoryManager on historical articles to validate:
1. Clustering quality (are related articles grouped?)
2. Storyline count (not too many, not too few)
3. Entity overlap effectiveness

Usage:
    python scripts/test_storyline_clustering.py [--articles N] [--category CAT] [--dry-run]
"""

import sys
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, '.')

from src.storage.database import DatabaseManager
from src.nlp.story_manager import StoryManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_test_articles(db: DatabaseManager, limit: int = 100, category: str = None) -> list:
    """Fetch recent articles with embeddings and entities."""
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT
                    a.id, a.title, a.source, a.published_date, a.category,
                    c.embedding,
                    a.entities
                FROM articles a
                JOIN chunks c ON a.id = c.article_id AND c.chunk_index = 0
                WHERE c.embedding IS NOT NULL
            """
            params = []

            if category:
                query += " AND a.category = %s"
                params.append(category)

            query += " ORDER BY a.published_date DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()

    articles = []
    for row in rows:
        # Use CLEAN entities if available, otherwise fallback to by_type
        entities_json = row[6] or {}
        clean = entities_json.get('clean', {})

        if clean and clean.get('all'):
            # Use cleaned entities (high + medium confidence)
            key_entities = clean.get('all', [])[:15]
        else:
            # Fallback to old format
            by_type = entities_json.get('by_type', {})
            key_entities = (
                by_type.get('GPE', []) +
                by_type.get('ORG', [])[:5] +
                by_type.get('PERSON', [])[:3]
            )

        articles.append({
            'id': row[0],
            'title': row[1],
            'source': row[2],
            'published_date': row[3],
            'category': row[4],
            'embedding': row[5],
            'entities': key_entities
        })

    return articles


def run_clustering_test(
    articles: list,
    story_manager: StoryManager,
    dry_run: bool = False
) -> dict:
    """
    Run clustering on articles and collect stats.

    Args:
        articles: List of article dicts with embeddings
        story_manager: StoryManager instance
        dry_run: If True, don't actually write to DB

    Returns:
        Stats dict
    """
    stats = {
        'total_articles': len(articles),
        'new_storylines': 0,
        'matched_to_existing': 0,
        'storylines_created': [],
        'matches': []
    }

    for i, article in enumerate(articles, 1):
        print(f"\n[{i}/{len(articles)}] Processing: {article['title'][:60]}...")
        print(f"    Entities: {article['entities'][:5]}")

        if dry_run:
            # Just find matches without writing
            matches = story_manager.find_matching_storylines(
                article_embedding=article['embedding'],
                article_entities=article['entities'],
                category=article['category']
            )

            if matches:
                best = matches[0]
                print(f"    → MATCH: '{best['title'][:50]}' (score={best['final_score']:.3f})")
                stats['matched_to_existing'] += 1
                stats['matches'].append({
                    'article_title': article['title'],
                    'storyline_title': best['title'],
                    'score': best['final_score']
                })
            else:
                print(f"    → NEW STORYLINE (would create)")
                stats['new_storylines'] += 1
                stats['storylines_created'].append(article['title'])
        else:
            # Actually run assignment
            assigned_ids = story_manager.assign_article(
                article_id=article['id'],
                article_embedding=article['embedding'],
                article_entities=article['entities'],
                article_title=article['title'],
                category=article['category']
            )

            # Check if it was a new storyline or match
            # (simplified: check if storyline article_count == 1)
            with story_manager.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT article_count FROM storylines WHERE id = %s",
                        (assigned_ids[0],)
                    )
                    count = cur.fetchone()[0]

            if count == 1:
                stats['new_storylines'] += 1
                print(f"    → NEW STORYLINE #{assigned_ids[0]}")
            else:
                stats['matched_to_existing'] += 1
                print(f"    → MATCHED to storyline #{assigned_ids[0]} ({count} articles)")

    return stats


def print_summary(stats: dict, story_manager: StoryManager):
    """Print clustering summary."""
    print("\n" + "=" * 70)
    print("CLUSTERING SUMMARY")
    print("=" * 70)

    print(f"\nArticles processed: {stats['total_articles']}")
    print(f"New storylines created: {stats['new_storylines']}")
    print(f"Matched to existing: {stats['matched_to_existing']}")

    ratio = stats['new_storylines'] / max(stats['total_articles'], 1)
    print(f"\nCluster ratio: {ratio:.1%} new storylines")

    if ratio > 0.7:
        print("⚠️  HIGH ratio - threshold may be too strict")
    elif ratio < 0.2:
        print("⚠️  LOW ratio - threshold may be too loose")
    else:
        print("✓ Ratio looks healthy")

    # Show active storylines
    print("\n" + "-" * 70)
    print("ACTIVE STORYLINES:")
    print("-" * 70)

    storylines = story_manager.get_active_storylines(limit=15)
    for s in storylines:
        print(f"\n  [{s['id']}] {s['title'][:60]}")
        print(f"      Articles: {s['article_count']} | Momentum: {s['momentum_score']:.2f}")
        print(f"      Entities: {', '.join(s['key_entities'][:5])}")
        print(f"      Days active: {s['days_active']} | Last update: {s['days_since_update']}d ago")


def main():
    parser = argparse.ArgumentParser(description="Test storyline clustering")
    parser.add_argument('--articles', '-n', type=int, default=50, help='Number of articles to process')
    parser.add_argument('--category', '-c', type=str, help='Filter by category')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Dry run (no DB writes)')
    parser.add_argument('--reset', '-r', action='store_true', help='Clear all storylines first')
    args = parser.parse_args()

    print("=" * 70)
    print("STORYLINE CLUSTERING TEST")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Articles: {args.articles}")
    print(f"Category: {args.category or 'ALL'}")

    db = DatabaseManager()
    story_manager = StoryManager(db)

    # Optionally reset storylines
    if args.reset and not args.dry_run:
        print("\n⚠️  Resetting all storylines...")
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM article_storylines")
                cur.execute("DELETE FROM storylines")
            conn.commit()
        print("✓ Storylines cleared")

    # Fetch test articles
    print(f"\nFetching {args.articles} articles...")
    articles = get_test_articles(db, limit=args.articles, category=args.category)
    print(f"✓ Found {len(articles)} articles with embeddings")

    if not articles:
        print("No articles found! Make sure embeddings exist.")
        return

    # Run clustering
    stats = run_clustering_test(articles, story_manager, dry_run=args.dry_run)

    # Print summary
    if not args.dry_run:
        print_summary(stats, story_manager)
    else:
        print("\n" + "=" * 70)
        print("DRY RUN SUMMARY")
        print("=" * 70)
        print(f"Would create {stats['new_storylines']} new storylines")
        print(f"Would match {stats['matched_to_existing']} to existing")

        if stats['storylines_created']:
            print("\nNew storylines would be:")
            for title in stats['storylines_created'][:10]:
                print(f"  - {title[:70]}")


if __name__ == "__main__":
    main()
