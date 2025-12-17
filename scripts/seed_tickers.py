#!/usr/bin/env python3
"""
Ticker Mapping Seed Script
Seeds ticker_mappings table with strategic geopolitical market movers

Usage:
    python3 scripts/seed_tickers.py --dry-run   # Preview changes
    python3 scripts/seed_tickers.py --production  # Apply changes
"""

import yaml
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.storage.database import DatabaseManager
from dotenv import load_dotenv

load_dotenv()


def seed_ticker_mappings(dry_run=True):
    """Seed ticker mappings from YAML config"""

    print("🌱 Starting Ticker Mapping Seed...")
    print(f"Mode: {'DRY RUN (Preview Only)' if dry_run else 'PRODUCTION (Inserting)'}")
    print("="*60)

    # Load config
    config_path = 'config/top_50_tickers.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    db = DatabaseManager()
    stats = {'mapped': 0, 'missing': 0, 'skipped': 0}
    missing_entities = []

    for sector, companies in data.items():
        print(f"\n📂 Sector: {sector.upper()}")

        for company in companies:
            ticker = company['ticker']
            exchange = company.get('exchange', 'US')
            aliases = company['aliases']

            # Find entity in database (case-insensitive, ORG/PER/LOC types)
            # Note: Some companies are misclassified by spaCy NER as PER or LOC
            find_query = """
                SELECT id, name, entity_type FROM entities
                WHERE entity_type IN ('ORG', 'PER', 'LOC')
                AND LOWER(name) IN %s
                ORDER BY
                    CASE entity_type
                        WHEN 'ORG' THEN 1
                        WHEN 'PER' THEN 2
                        WHEN 'LOC' THEN 3
                    END
                LIMIT 1;
            """
            aliases_lower = tuple(a.lower() for a in aliases)

            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(find_query, (aliases_lower,))
                    result = cur.fetchone()

                    if result:
                        entity_id, entity_name, entity_type = result

                        if not dry_run:
                            # Insert mapping (with ON CONFLICT to handle duplicates)
                            insert_query = """
                                INSERT INTO ticker_mappings (entity_id, ticker, exchange, confidence)
                                VALUES (%s, %s, %s, 1.0)
                                ON CONFLICT (entity_id, ticker) DO NOTHING
                                RETURNING id;
                            """
                            cur.execute(insert_query, (entity_id, ticker, exchange))

                            if cur.rowcount > 0:
                                print(f"   ✅ Mapped: {entity_name} ({entity_type}) → {ticker} ({exchange})")
                                stats['mapped'] += 1
                            else:
                                print(f"   ⏭️  Exists: {entity_name} ({entity_type}) → {ticker} ({exchange})")
                                stats['skipped'] += 1
                        else:
                            print(f"   ✅ Would map: {entity_name} ({entity_type}) → {ticker} ({exchange})")
                            stats['mapped'] += 1
                    else:
                        print(f"   ⚠️  NOT FOUND: {company['name']} (tried: {', '.join(aliases[:2])}...)")
                        stats['missing'] += 1
                        missing_entities.append({
                            'name': company['name'],
                            'ticker': ticker,
                            'aliases': aliases
                        })

    # Summary
    print("\n" + "="*60)
    print(f"🏁 SEED {'PREVIEW' if dry_run else 'COMPLETE'}")
    print("="*60)
    print(f"✅ Mappings Created: {stats['mapped']}")
    print(f"⏭️  Already Existed: {stats['skipped']}")
    print(f"⚠️  Entities Missing: {stats['missing']}")

    if missing_entities:
        print("\n❌ Missing Entities (NER didn't catch these companies):")
        for i, ent in enumerate(missing_entities[:15], 1):
            print(f"   {i}. {ent['name']} ({ent['ticker']}) - tried: {ent['aliases'][0]}")
        if len(missing_entities) > 15:
            print(f"   ... and {len(missing_entities)-15} more")

        print("\n💡 Possible reasons:")
        print("   - Company name not mentioned in articles yet")
        print("   - Mentioned with different variation (check entity_mentions)")
        print("   - NER classified as different type (not ORG)")

    print("="*60)

    if dry_run:
        print("\n💡 Run with --production to apply changes")
    else:
        print("\n✅ Changes committed to database")

    db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Seed ticker mappings from YAML config')
    parser.add_argument('--production', action='store_true', help='Actually insert mappings (default is dry-run)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without inserting (default)')
    args = parser.parse_args()

    # Default to dry-run unless --production is explicitly specified
    dry_run = not args.production

    seed_ticker_mappings(dry_run=dry_run)
