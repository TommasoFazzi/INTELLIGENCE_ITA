#!/usr/bin/env python3
"""
Atlas Pipeline — unified orchestrator for reference & spatial data loading.

Runs all data loading tasks (static + dynamic) with centralized error handling,
idempotent static tasks, and post-load VACUUM ANALYZE for PostGIS spatial tables.

Uses individual loader scripts as importable modules.

Usage:
    python scripts/atlas_pipeline.py                       # run all tasks
    python scripts/atlas_pipeline.py --only ucdp,sanctions # run specific tasks
    python scripts/atlas_pipeline.py --static-only          # idempotent tasks only
    python scripts/atlas_pipeline.py --dynamic-only         # update tasks only
    python scripts/atlas_pipeline.py --dry-run              # show what would run
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tables requiring VACUUM ANALYZE after bulk insert (PostGIS spatial tables).
# PostGIS relies heavily on up-to-date statistics to decide between
# GIST index scan vs sequential scan. Without fresh stats after a
# large batch insert, the planner may choose seq scan on 300k+ rows.
SPATIAL_TABLES_VACUUM = {"conflict_events", "strategic_infrastructure", "country_boundaries"}


def _table_count(db: DatabaseManager, table: str) -> int:
    """Count rows in a table (safe, returns 0 on error)."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {table}")
                return cur.fetchone()[0]
    except Exception:
        return 0


def vacuum_analyze_if_spatial(db: DatabaseManager, table_name: str):
    """Run VACUUM ANALYZE on spatial tables after bulk insert."""
    if table_name not in SPATIAL_TABLES_VACUUM:
        return
    logger.info(f"[ATLAS] Running VACUUM ANALYZE {table_name} (PostGIS stats refresh)...")
    try:
        with db.get_connection() as conn:
            conn.set_isolation_level(0)  # AUTOCOMMIT required for VACUUM
            with conn.cursor() as cur:
                cur.execute(f"VACUUM ANALYZE {table_name}")
        logger.info(f"[ATLAS] VACUUM ANALYZE {table_name} completed")
    except Exception as e:
        logger.warning(f"[ATLAS] VACUUM ANALYZE {table_name} failed (non-critical): {e}")


def _run_script(module_name: str, main_fn: str = "main"):
    """Import and run a loader script's main() function."""
    import importlib
    mod = importlib.import_module(module_name)
    fn = getattr(mod, main_fn)
    return fn()


# ── Task Definitions ────────────────────────────────────────────────

# Static tasks are idempotent: skip if table already populated.
STATIC_TASKS = {
    "natural_earth": {
        "table": "country_boundaries",
        "loader": "scripts.load_natural_earth",
        "description": "Natural Earth 50m boundaries (via ogr2ogr shell script)",
        "shell_cmd": ["bash", "scripts/load_natural_earth.sh"],
    },
    "world_bank": {
        "table": "country_profiles",
        "loader": "scripts.load_world_bank",
        "description": "World Bank country profiles (API v2)",
    },
}

# Dynamic tasks run every time (incremental upsert).
DYNAMIC_TASKS = {
    "ucdp": {
        "table": "conflict_events",
        "loader": "scripts.load_ucdp",
        "description": "UCDP GED conflict events (API v24.1)",
        "schedule": "monthly",
    },
    "sanctions": {
        "table": "sanctions_registry",
        "loader": "scripts.load_opensanctions",
        "description": "OpenSanctions FtM entity registry",
        "schedule": "weekly",
    },
}


def run_task(db: DatabaseManager, name: str, config: dict, force: bool = False) -> dict:
    """Execute a single task with centralized try/catch, timing, and post-load VACUUM."""
    logger.info(f"[ATLAS] Starting task: {name} — {config.get('description', '')}")
    start = time.time()
    table = config.get("table", "")

    try:
        # Static tasks: skip if table already has data (unless force)
        if name in STATIC_TASKS and not force:
            count = _table_count(db, table)
            if count > 0:
                logger.info(f"[ATLAS] {name}: table {table} has {count} rows, skipping (idempotent)")
                return {"status": "skipped", "rows": count}

        # Execute the loader
        if config.get("shell_cmd"):
            import subprocess
            result = subprocess.run(config["shell_cmd"], shell=False, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Shell command failed: {result.stderr}")
        else:
            import importlib
            mod = importlib.import_module(config["loader"])
            exit_code = mod.main()
            if exit_code and exit_code != 0:
                raise RuntimeError(f"Loader returned exit code {exit_code}")

        # Post-load: VACUUM ANALYZE spatial tables for GIST index stats
        if table:
            vacuum_analyze_if_spatial(db, table)

        elapsed = time.time() - start
        final_count = _table_count(db, table) if table else 0
        logger.info(f"[ATLAS] {name}: completed in {elapsed:.1f}s ({final_count} rows)")
        return {"status": "ok", "time": elapsed, "rows": final_count}

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[ATLAS] {name}: FAILED in {elapsed:.1f}s — {e}", exc_info=True)
        return {"status": "error", "error": str(e), "time": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Atlas Pipeline — unified data loader")
    parser.add_argument('--only', type=str, help='Comma-separated task names to run')
    parser.add_argument('--static-only', action='store_true', help='Run only static (idempotent) tasks')
    parser.add_argument('--dynamic-only', action='store_true', help='Run only dynamic (update) tasks')
    parser.add_argument('--force', action='store_true', help='Force re-run static tasks even if data exists')
    parser.add_argument('--dry-run', action='store_true', help='Show what would run')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ATLAS PIPELINE — Unified Data Orchestrator")
    logger.info("=" * 80)

    # Build task list
    tasks = {}
    if not args.dynamic_only:
        tasks.update({k: {**v, "_type": "static"} for k, v in STATIC_TASKS.items()})
    if not args.static_only:
        tasks.update({k: {**v, "_type": "dynamic"} for k, v in DYNAMIC_TASKS.items()})

    if args.only:
        selected = {n.strip() for n in args.only.split(",")}
        tasks = {k: v for k, v in tasks.items() if k in selected}

    if not tasks:
        logger.warning("[ATLAS] No tasks to run")
        return 0

    logger.info(f"\n[ATLAS] Tasks to run: {list(tasks.keys())}")

    if args.dry_run:
        for name, config in tasks.items():
            logger.info(f"  → {name} ({config['_type']}): {config.get('description', '')}")
        return 0

    # Initialize DB
    db = DatabaseManager()

    # Execute tasks
    results = {}
    for name, config in tasks.items():
        results[name] = run_task(db, name, config, force=args.force)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ATLAS PIPELINE — SUMMARY")
    logger.info("=" * 80)
    for name, result in results.items():
        status = result["status"]
        icon = {"ok": "✓", "skipped": "⊘", "error": "✗"}.get(status, "?")
        detail = f"{result.get('rows', 0)} rows" if status != "error" else result.get("error", "")[:60]
        logger.info(f"  {icon} {name}: {status} — {detail}")

    db.close()

    # Exit with error if any task failed
    failed = [n for n, r in results.items() if r["status"] == "error"]
    if failed:
        logger.error(f"\n[ATLAS] {len(failed)} task(s) failed: {failed}")
        return 1

    logger.info("\n✓ Atlas Pipeline complete!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
