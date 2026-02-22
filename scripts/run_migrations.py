#!/usr/bin/env python3
"""
Database migration runner for production deployments.

Applies base schema (via DatabaseManager.init_db) and all incremental
migration SQL files in migrations/ in sorted order.

Tracks applied migrations in schema_migrations table to ensure each
migration runs exactly once. On first run against an already-migrated
database, seeds the tracking table with all existing files.

Usage:
    python scripts/run_migrations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_migrations_table(db: DatabaseManager) -> bool:
    """
    Create schema_migrations tracking table if it does not exist.
    Returns True if the table was newly created (first time), False if it already existed.
    """
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'schema_migrations'
                )
            """)
            existed = cur.fetchone()[0]
            if not existed:
                cur.execute("""
                    CREATE TABLE schema_migrations (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        applied_at TIMESTAMP DEFAULT NOW()
                    )
                """)
    return not existed


def is_db_already_migrated(db: DatabaseManager) -> bool:
    """
    Heuristic: if the 'storylines' table exists the DB has already had
    migrations applied outside the tracking system (e.g. from a pg_dump).
    """
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'storylines'
                )
            """)
            return cur.fetchone()[0]


def seed_migrations(db: DatabaseManager, migration_files: list) -> None:
    """Mark all migration files as already applied (used on first tracking run)."""
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            for f in migration_files:
                cur.execute(
                    "INSERT INTO schema_migrations (name) VALUES (%s) ON CONFLICT DO NOTHING",
                    (f.name,)
                )
    logger.info(f"  ✓ Seeded {len(migration_files)} migration(s) as already applied")


def apply_migration(db: DatabaseManager, migration_path: Path) -> bool:
    """
    Apply a single SQL migration file and record it in schema_migrations.
    Returns True on success or if already applied, False on error.
    """
    name = migration_path.name

    # Check if already applied
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM schema_migrations WHERE name = %s", (name,))
            if cur.fetchone():
                logger.info(f"  - Already applied: {name}")
                return True

    sql = migration_path.read_text(encoding="utf-8")
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (name) VALUES (%s) ON CONFLICT DO NOTHING",
                    (name,)
                )
        logger.info(f"  ✓ Applied: {name}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed:  {name} — {e}")
        return False


def main() -> int:
    logger.info("=" * 60)
    logger.info("DATABASE MIGRATION RUNNER")
    logger.info("=" * 60)

    # Step 1: Connect and initialize base schema
    logger.info("\n[1/3] Initializing base schema...")
    try:
        db = DatabaseManager()
        db.init_db()
        logger.info("  ✓ Base schema ready")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize base schema: {e}")
        return 1

    # Step 2: Collect migration files (sorted, excluding rollbacks)
    migrations_dir = Path(__file__).parent.parent / "migrations"
    migration_files = sorted(
        f for f in migrations_dir.glob("[0-9]*.sql")
        if "rollback" not in f.name
    )

    if not migration_files:
        logger.info("\n[2/3] No migration files found — skipping")
        logger.info("\n✓ All migrations complete. Database ready.\n")
        return 0

    logger.info(f"\n[2/3] Found {len(migration_files)} migration file(s)")

    # Step 3: Ensure tracking table; seed if upgrading from untracked state
    logger.info("\n[3/3] Applying migrations...")
    is_new_table = ensure_migrations_table(db)

    if is_new_table:
        if is_db_already_migrated(db):
            logger.info(
                "  ℹ First run with migration tracking — "
                "DB already migrated, seeding all as applied"
            )
            seed_migrations(db, migration_files)
        else:
            logger.info("  ✓ Migration tracking table created (fresh database)")

    for migration_path in migration_files:
        if not apply_migration(db, migration_path):
            logger.error("Migration failed — aborting startup")
            return 1

    logger.info("\n✓ All migrations complete. Database ready.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
