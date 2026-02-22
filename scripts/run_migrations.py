#!/usr/bin/env python3
"""
Database migration runner for production deployments.

Applies base schema (via DatabaseManager.init_db) and all incremental
migration SQL files in migrations/ in sorted order. All migrations use
IF NOT EXISTS / ADD COLUMN IF NOT EXISTS, so re-running is safe.

Usage:
    python scripts/run_migrations.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def apply_migration(db: DatabaseManager, migration_path: Path) -> bool:
    """Apply a single SQL migration file. Returns True on success."""
    sql = migration_path.read_text(encoding="utf-8")
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        logger.info(f"  ✓ Applied: {migration_path.name}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed:  {migration_path.name} — {e}")
        return False


def main() -> int:
    logger.info("=" * 60)
    logger.info("DATABASE MIGRATION RUNNER")
    logger.info("=" * 60)

    # Step 1: Connect and create base schema
    logger.info("\n[1/2] Initializing base schema...")
    try:
        db = DatabaseManager()
        db.init_db()
        logger.info("  ✓ Base schema ready")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize base schema: {e}")
        return 1

    # Step 2: Apply incremental migrations (skip rollback files)
    migrations_dir = Path(__file__).parent.parent / "migrations"
    migration_files = sorted(
        f for f in migrations_dir.glob("[0-9]*.sql")
        if "rollback" not in f.name
    )

    if not migration_files:
        logger.info("\n[2/2] No migration files found — skipping")
    else:
        logger.info(f"\n[2/2] Applying {len(migration_files)} migration(s)...")
        for migration_path in migration_files:
            if not apply_migration(db, migration_path):
                logger.error("Migration failed — aborting startup")
                return 1

    logger.info("\n✓ All migrations complete. Database ready.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
