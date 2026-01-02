#!/usr/bin/env python3
"""
Backfill Report Embeddings

Generates embeddings for existing reports that don't have them.
Required before using The Oracle RAG on reports.

Usage:
    python scripts/backfill_report_embeddings.py

Prerequisites:
    1. Apply migration 006:
       psql -d intelligence_ita -f migrations/006_add_report_embeddings.sql
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer
from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def backfill_embeddings(batch_size: int = 10):
    """
    Generate embeddings for all reports without them.

    Args:
        batch_size: Number of reports to process at a time
    """
    print("=" * 60)
    print("BACKFILL REPORT EMBEDDINGS")
    print("=" * 60)

    # Initialize
    db = DatabaseManager()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print(f"Model loaded: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)")

    # Get reports without embeddings
    reports = db.get_reports_without_embeddings(limit=1000)

    if not reports:
        print("\nNo reports without embeddings found.")
        print("All reports are already indexed for semantic search.")
        return

    print(f"\nFound {len(reports)} reports without embeddings")
    print("-" * 40)

    success_count = 0
    error_count = 0

    for i, report in enumerate(reports, 1):
        report_id = report['id']
        report_date = report.get('report_date', 'N/A')

        # Use final_content if available, otherwise draft_content
        content = report.get('final_content') or report.get('draft_content', '')

        if not content:
            print(f"  [{i}/{len(reports)}] Report #{report_id} - SKIPPED (empty content)")
            error_count += 1
            continue

        try:
            # Generate embedding
            embedding = model.encode(content).tolist()

            # Save to database
            success = db.update_report_embedding(report_id, embedding)

            if success:
                print(f"  [{i}/{len(reports)}] Report #{report_id} ({report_date}) - OK")
                success_count += 1
            else:
                print(f"  [{i}/{len(reports)}] Report #{report_id} - FAILED (db update)")
                error_count += 1

        except Exception as e:
            print(f"  [{i}/{len(reports)}] Report #{report_id} - ERROR: {e}")
            error_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total processed: {len(reports)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print()

    if success_count > 0:
        print("Reports are now ready for semantic search in The Oracle!")


def verify_migration():
    """
    Verify that migration 006 has been applied.
    """
    db = DatabaseManager()

    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'reports'
                    AND column_name = 'content_embedding'
                """)
                result = cur.fetchone()

                if not result:
                    print("ERROR: content_embedding column not found in reports table.")
                    print("Please apply migration 006 first:")
                    print()
                    print("  psql -d intelligence_ita -f migrations/006_add_report_embeddings.sql")
                    print()
                    return False

                return True

    except Exception as e:
        print(f"ERROR: Could not verify migration: {e}")
        return False


if __name__ == "__main__":
    print()

    # Verify migration
    if not verify_migration():
        sys.exit(1)

    print("Migration 006 verified: content_embedding column exists")
    print()

    # Run backfill
    backfill_embeddings()
