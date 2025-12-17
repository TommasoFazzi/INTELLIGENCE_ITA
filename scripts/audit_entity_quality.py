#!/usr/bin/env python3
"""
Entity Quality Audit Script - "The Auditor"

Analyzes entity quality in the database and identifies 3 types of garbage entities:
1. Ghost Entities: Too vague or common (e.g., "Today", "The Government", "Reuters")
2. Null Island: Invalid geocoding (0.0, 0.0) or duplicate coordinates
3. Hallucinations: Rare entities (mention_count=1) likely from NER errors

Usage:
    python scripts/audit_entity_quality.py [--output-csv data/audit_report.csv]

Output:
    - Console report with statistics
    - Optional CSV export for further analysis
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import csv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EntityQualityAuditor:
    """Audits entity quality and identifies garbage entities."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def get_total_entity_stats(self) -> Dict[str, any]:
        """Get overall entity statistics."""
        query = """
        SELECT
            COUNT(*) as total_entities,
            COUNT(DISTINCT entity_type) as unique_types,
            SUM(mention_count) as total_mentions,
            COUNT(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 END) as geocoded_count
        FROM entities;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()

        return {
            'total_entities': result[0] or 0,
            'unique_types': result[1] or 0,
            'total_mentions': result[2] or 0,
            'geocoded_count': result[3] or 0
        }

    def get_entity_type_breakdown(self) -> List[Tuple[str, int, float]]:
        """Get entity breakdown by type."""
        query = """
        SELECT
            entity_type,
            COUNT(*) as count,
            (COUNT(*)::FLOAT / (SELECT COUNT(*) FROM entities) * 100) as percentage
        FROM entities
        GROUP BY entity_type
        ORDER BY count DESC;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()

        return results

    def detect_ghost_entities(self) -> Dict[str, any]:
        """
        Detect Type A: Ghost Entities (too vague or common).

        Criteria:
        - Length < 3 characters
        - Appears in > 50% of articles
        - Common media artifacts
        """

        # Query 1: Too short entities
        query_short = """
        SELECT name, entity_type, mention_count
        FROM entities
        WHERE LENGTH(name) < 3
        ORDER BY mention_count DESC
        LIMIT 100;
        """

        # Query 2: Too common entities (appears in >50% of articles)
        query_common = """
        SELECT
            e.name,
            e.entity_type,
            e.mention_count,
            (e.mention_count::FLOAT / (SELECT COUNT(*) FROM articles) * 100) as article_percentage
        FROM entities e
        WHERE e.mention_count > (SELECT COUNT(*) FROM articles) * 0.5
        ORDER BY article_percentage DESC
        LIMIT 100;
        """

        # Query 3: Entities with suspicious patterns
        query_suspicious = """
        SELECT name, entity_type, mention_count
        FROM entities
        WHERE
            -- All uppercase short strings (likely acronyms/errors)
            (LENGTH(name) <= 5 AND name = UPPER(name))
            -- Or contains only numbers
            OR name ~ '^[0-9]+$'
            -- Or temporal words (common false positives)
            OR LOWER(name) IN ('oggi', 'ieri', 'domani', 'today', 'yesterday', 'tomorrow')
        ORDER BY mention_count DESC
        LIMIT 100;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                # Get short entities
                cur.execute(query_short)
                short_entities = cur.fetchall()

                # Get too common entities
                cur.execute(query_common)
                common_entities = cur.fetchall()

                # Get suspicious patterns
                cur.execute(query_suspicious)
                suspicious_entities = cur.fetchall()

        # Calculate total unique ghost entities (may overlap)
        all_ghost_names = set()
        all_ghost_names.update([e[0] for e in short_entities])
        all_ghost_names.update([e[0] for e in common_entities])
        all_ghost_names.update([e[0] for e in suspicious_entities])

        return {
            'total_ghosts': len(all_ghost_names),
            'short_entities': short_entities[:20],  # Top 20
            'common_entities': common_entities[:20],
            'suspicious_entities': suspicious_entities[:20],
            'all_ghost_names': list(all_ghost_names)
        }

    def detect_null_island(self) -> Dict[str, any]:
        """
        Detect Type B: Null Island (invalid geocoding).

        Criteria:
        - Coordinates exactly (0.0, 0.0)
        - Duplicate coordinates for different entities
        - Out of range coordinates
        """

        # Query 1: Null Island entities
        query_null_island = """
        SELECT name, entity_type, latitude, longitude, geo_status, mention_count
        FROM entities
        WHERE latitude = 0.0 AND longitude = 0.0
        ORDER BY mention_count DESC;
        """

        # Query 2: Duplicate coordinates (suspicious)
        query_duplicates = """
        SELECT
            latitude,
            longitude,
            COUNT(*) as entity_count,
            STRING_AGG(name, ', ') as entities
        FROM entities
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        GROUP BY latitude, longitude
        HAVING COUNT(*) > 5
        ORDER BY entity_count DESC
        LIMIT 50;
        """

        # Query 3: High-value entities not geocoded
        query_not_geocoded = """
        SELECT name, entity_type, mention_count, geo_status
        FROM entities
        WHERE
            entity_type IN ('GPE', 'LOC', 'FAC')
            AND mention_count > 10
            AND (geo_status = 'NOT_FOUND' OR geo_status = 'PENDING' OR latitude IS NULL)
        ORDER BY mention_count DESC
        LIMIT 100;
        """

        # Query 4: Out of range coordinates
        query_invalid_coords = """
        SELECT name, entity_type, latitude, longitude, mention_count
        FROM entities
        WHERE
            latitude IS NOT NULL
            AND longitude IS NOT NULL
            AND (
                latitude < -90 OR latitude > 90
                OR longitude < -180 OR longitude > 180
            )
        ORDER BY mention_count DESC;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_null_island)
                null_island_entities = cur.fetchall()

                cur.execute(query_duplicates)
                duplicate_coords = cur.fetchall()

                cur.execute(query_not_geocoded)
                not_geocoded = cur.fetchall()

                cur.execute(query_invalid_coords)
                invalid_coords = cur.fetchall()

        return {
            'null_island_count': len(null_island_entities),
            'null_island_entities': null_island_entities[:50],
            'duplicate_coords': duplicate_coords,
            'not_geocoded': not_geocoded,
            'invalid_coords': invalid_coords
        }

    def detect_hallucinations(self) -> Dict[str, any]:
        """
        Detect Type C: Hallucinations (rare entities, likely NER errors).

        Criteria:
        - mention_count = 1 (appears only once)
        - Contains special characters or numbers
        - All uppercase long strings
        """

        # Query 1: One-hit wonders
        query_one_hit = """
        SELECT name, entity_type, mention_count, first_seen
        FROM entities
        WHERE mention_count = 1
        ORDER BY first_seen DESC;
        """

        # Query 2: Suspicious patterns in one-hit entities
        query_suspicious_patterns = r"""
        SELECT name, entity_type, mention_count
        FROM entities
        WHERE
            mention_count = 1
            AND (
                name ~ '[0-9]'  -- Contains numbers
                OR name ~ '[^a-zA-Z0-9\s\-À-ÿ]'  -- Special chars (allow accents)
                OR (LENGTH(name) > 10 AND name = UPPER(name))  -- Long all-caps
            )
        ORDER BY LENGTH(name) DESC
        LIMIT 200;
        """

        # Query 3: Distribution of mention counts
        query_distribution = """
        SELECT
            CASE
                WHEN mention_count = 1 THEN '1'
                WHEN mention_count BETWEEN 2 AND 5 THEN '2-5'
                WHEN mention_count BETWEEN 6 AND 10 THEN '6-10'
                WHEN mention_count BETWEEN 11 AND 20 THEN '11-20'
                ELSE '20+'
            END as mention_range,
            COUNT(*) as entity_count,
            (COUNT(*)::FLOAT / (SELECT COUNT(*) FROM entities) * 100) as percentage
        FROM entities
        GROUP BY
            CASE
                WHEN mention_count = 1 THEN '1'
                WHEN mention_count BETWEEN 2 AND 5 THEN '2-5'
                WHEN mention_count BETWEEN 6 AND 10 THEN '6-10'
                WHEN mention_count BETWEEN 11 AND 20 THEN '11-20'
                ELSE '20+'
            END
        ORDER BY
            MIN(CASE
                WHEN mention_count = 1 THEN 1
                WHEN mention_count BETWEEN 2 AND 5 THEN 2
                WHEN mention_count BETWEEN 6 AND 10 THEN 3
                WHEN mention_count BETWEEN 11 AND 20 THEN 4
                ELSE 5
            END);
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_one_hit)
                one_hit_entities = cur.fetchall()

                cur.execute(query_suspicious_patterns)
                suspicious_patterns = cur.fetchall()

                cur.execute(query_distribution)
                distribution = cur.fetchall()

        return {
            'total_one_hit': len(one_hit_entities),
            'one_hit_sample': one_hit_entities[:50],
            'suspicious_patterns': suspicious_patterns,
            'mention_distribution': distribution
        }

    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive audit report."""
        logger.info("🔍 Starting entity quality audit...")

        # Gather all metrics
        total_stats = self.get_total_entity_stats()
        type_breakdown = self.get_entity_type_breakdown()
        ghost_data = self.detect_ghost_entities()
        null_island_data = self.detect_null_island()
        hallucination_data = self.detect_hallucinations()

        # Calculate recommended actions
        auto_delete_count = (
            len([e for e in ghost_data['short_entities'] if e[2] <= 5]) +  # Low mention ghosts
            len(null_island_data['null_island_entities']) +
            len([h for h in hallucination_data['suspicious_patterns'] if len(h[0]) < 4])
        )

        review_queue_count = (
            len([e for e in ghost_data['common_entities'] if e[2] > 10]) +
            len(null_island_data['not_geocoded'])
        )

        estimated_clean = total_stats['total_entities'] - len(ghost_data['all_ghost_names']) - hallucination_data['total_one_hit']

        return {
            'timestamp': datetime.now(),
            'total_stats': total_stats,
            'type_breakdown': type_breakdown,
            'ghost_entities': ghost_data,
            'null_island': null_island_data,
            'hallucinations': hallucination_data,
            'recommendations': {
                'auto_delete_count': auto_delete_count,
                'review_queue_count': review_queue_count,
                'estimated_clean_count': max(estimated_clean, 0),
                'reduction_percentage': (1 - (estimated_clean / max(total_stats['total_entities'], 1))) * 100
            }
        }

    def print_report(self, report: Dict[str, any]):
        """Print formatted audit report to console."""
        print("\n" + "="*70)
        print("ENTITY QUALITY AUDIT REPORT")
        print("="*70)
        print(f"Generated: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Total statistics
        stats = report['total_stats']
        print(f"Total Entities: {stats['total_entities']:,}")
        print(f"├─ Unique Types: {stats['unique_types']}")
        print(f"├─ Total Mentions: {stats['total_mentions']:,}")
        print(f"└─ Geocoded: {stats['geocoded_count']:,} ({stats['geocoded_count']/max(stats['total_entities'], 1)*100:.1f}%)")
        print()

        # Entity type breakdown
        print("Entity Breakdown:")
        for entity_type, count, percentage in report['type_breakdown']:
            print(f"├─ {entity_type}: {count:,} ({percentage:.1f}%)")
        print()

        # Quality Issues
        print("Quality Issues Detected:")
        print(f"├─ Ghost Entities (too vague): {report['ghost_entities']['total_ghosts']:,} " +
              f"({report['ghost_entities']['total_ghosts']/max(stats['total_entities'], 1)*100:.1f}%)")
        print(f"│   ├─ Short entities (len<3): {len(report['ghost_entities']['short_entities'])}")
        print(f"│   ├─ Too common (>50% articles): {len(report['ghost_entities']['common_entities'])}")
        print(f"│   └─ Suspicious patterns: {len(report['ghost_entities']['suspicious_entities'])}")

        print(f"├─ Null Island (bad coordinates): {report['null_island']['null_island_count']:,} " +
              f"({report['null_island']['null_island_count']/max(stats['total_entities'], 1)*100:.1f}%)")
        print(f"│   ├─ Coordinates (0.0, 0.0): {report['null_island']['null_island_count']}")
        print(f"│   ├─ Duplicate coordinates: {len(report['null_island']['duplicate_coords'])} locations")
        print(f"│   ├─ Not geocoded (high-value): {len(report['null_island']['not_geocoded'])}")
        print(f"│   └─ Invalid coordinates: {len(report['null_island']['invalid_coords'])}")

        print(f"└─ Hallucinations (mention=1): {report['hallucinations']['total_one_hit']:,} " +
              f"({report['hallucinations']['total_one_hit']/max(stats['total_entities'], 1)*100:.1f}%)")
        print(f"    ├─ One-hit wonders: {report['hallucinations']['total_one_hit']}")
        print(f"    └─ Suspicious patterns: {len(report['hallucinations']['suspicious_patterns'])}")
        print()

        # Mention distribution
        print("Mention Count Distribution:")
        for mention_range, count, percentage in report['hallucinations']['mention_distribution']:
            print(f"├─ {mention_range:>6} mentions: {count:>6,} entities ({percentage:5.1f}%)")
        print()

        # Recommendations
        recs = report['recommendations']
        print("Recommended Actions:")
        print(f"1. Run clean_entities.py --auto-delete")
        print(f"   → Safely removes ~{recs['auto_delete_count']:,} obvious garbage entities")
        print(f"2. Run clean_entities.py --ai-disambiguate")
        print(f"   → Reviews ~{len(report['ghost_entities']['all_ghost_names']):,} ambiguous entities with AI")
        print(f"3. Manual review queue")
        print(f"   → {recs['review_queue_count']:,} high-value entities need human review")
        print()
        print(f"Estimated Clean Database: {recs['estimated_clean_count']:,} entities " +
              f"({100 - recs['reduction_percentage']:.1f}% of current)")
        print(f"Expected Reduction: {recs['reduction_percentage']:.1f}%")
        print()

        # Examples of problematic entities
        print("─" * 70)
        print("SAMPLE PROBLEMATIC ENTITIES")
        print("─" * 70)

        print("\n🔹 Ghost Entities (Short):")
        for name, entity_type, mention_count in report['ghost_entities']['short_entities'][:10]:
            print(f"  - '{name}' ({entity_type}, {mention_count} mentions)")

        print("\n🔹 Null Island (0.0, 0.0):")
        for name, entity_type, lat, lng, geo_status, mention_count in report['null_island']['null_island_entities'][:10]:
            print(f"  - '{name}' ({entity_type}, {mention_count} mentions)")

        print("\n🔹 Hallucinations (Suspicious Patterns):")
        for name, entity_type, mention_count in report['hallucinations']['suspicious_patterns'][:10]:
            print(f"  - '{name}' ({entity_type})")

        print("\n" + "="*70)

    def export_to_csv(self, report: Dict[str, any], output_path: str):
        """Export problematic entities to CSV for review."""
        logger.info(f"📊 Exporting audit results to {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Entity Name', 'Type', 'Mention Count', 'Issue Category', 'Latitude', 'Longitude', 'Geo Status'])

            # Ghost entities
            for name in report['ghost_entities']['all_ghost_names'][:500]:  # Limit to 500
                writer.writerow([name, 'Unknown', 'N/A', 'GHOST_ENTITY', '', '', ''])

            # Null Island
            for name, entity_type, lat, lng, geo_status, mention_count in report['null_island']['null_island_entities']:
                writer.writerow([name, entity_type, mention_count, 'NULL_ISLAND', lat, lng, geo_status])

            # Hallucinations
            for name, entity_type, mention_count in report['hallucinations']['suspicious_patterns']:
                writer.writerow([name, entity_type, mention_count, 'HALLUCINATION', '', '', ''])

        logger.info(f"✓ Exported {report['ghost_entities']['total_ghosts'] + report['null_island']['null_island_count'] + len(report['hallucinations']['suspicious_patterns'])} entities")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Audit entity quality in the database')
    parser.add_argument('--output-csv', type=str, help='Export results to CSV file')
    args = parser.parse_args()

    # Initialize database
    logger.info("🔌 Connecting to database...")
    db_manager = DatabaseManager()

    # Run audit
    auditor = EntityQualityAuditor(db_manager)
    report = auditor.generate_report()

    # Print report
    auditor.print_report(report)

    # Export to CSV if requested
    if args.output_csv:
        auditor.export_to_csv(report, args.output_csv)

    logger.info("✓ Audit complete!")

    # Return exit code based on quality metrics
    ghost_percentage = report['ghost_entities']['total_ghosts'] / max(report['total_stats']['total_entities'], 1) * 100
    hallucination_percentage = report['hallucinations']['total_one_hit'] / max(report['total_stats']['total_entities'], 1) * 100

    if ghost_percentage > 20 or hallucination_percentage > 50:
        logger.warning("⚠️  High garbage entity percentage detected. Cleanup recommended before migration.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
