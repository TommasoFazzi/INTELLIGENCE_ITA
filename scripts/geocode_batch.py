#!/usr/bin/env python3
"""
Batch Geocoding Script - Geocode pending GPE/LOC entities

Uses Nominatim API with 1 req/sec rate limit.
Prioritizes entities by mention_count (most important first).

Usage:
    python3 scripts/geocode_batch.py --batch 200   # Process 200 entities
    python3 scripts/geocode_batch.py --all         # Process all pending
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from dotenv import load_dotenv

# Optional: Install geopy if not already installed
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
except ImportError:
    print("⚠️  geopy not installed. Run: pip install geopy")
    sys.exit(1)

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BatchGeocoder:
    """Batch geocoder with rate limiting and error handling"""

    def __init__(self):
        self.db = DatabaseManager()
        # User agent required by Nominatim
        self.geolocator = Nominatim(user_agent="intelligence_ita_geopolitical_bot_v1.0")
        self.rate_limit_delay = 1.2  # Seconds (Nominatim requires min 1s between requests)

    def get_pending_entities(self, limit: int = 100) -> List[Tuple]:
        """
        Get entities that need geocoding

        Prioritizes:
        1. GPE (Geopolitical Entity) - countries, cities, states
        2. LOC (Location) - mountains, bodies of water, regions
        3. FAC (Facility) - buildings, airports, bridges

        Orders by mention_count DESC (most mentioned = most important)
        """
        query = """
            SELECT id, name, entity_type, mention_count
            FROM entities
            WHERE (geo_status = 'PENDING' OR geo_status IS NULL)
            AND entity_type IN ('GPE', 'LOC', 'FAC')
            ORDER BY mention_count DESC, created_at DESC
            LIMIT %s;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                return cur.fetchall()

    def update_entity_geo(self, entity_id: int, lat: float = None, lon: float = None, status: str = 'NOT_FOUND'):
        """Update entity with geocoding results"""
        query = """
            UPDATE entities
            SET latitude = %s,
                longitude = %s,
                geo_status = %s,
                last_seen = NOW()
            WHERE id = %s;
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (lat, lon, status, entity_id))
                # Connection context manager auto-commits on exit

    def geocode_entity(self, name: str, entity_type: str) -> Tuple[float, float, str]:
        """
        Geocode a single entity with retry logic

        Returns:
            (latitude, longitude, status)
        """
        max_retries = 3
        backoff_delay = 2  # Exponential backoff starting point

        for attempt in range(max_retries):
            try:
                location = self.geolocator.geocode(name, timeout=10)

                if location:
                    return (location.latitude, location.longitude, 'FOUND')
                else:
                    return (None, None, 'NOT_FOUND')

            except GeocoderTimedOut:
                logger.warning(f"⏱️  Timeout on '{name}' (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(backoff_delay * (attempt + 1))  # Exponential backoff
                continue

            except GeocoderServiceError as e:
                logger.error(f"⚠️  Service error on '{name}': {e}")
                return (None, None, 'ERROR')

            except Exception as e:
                logger.error(f"❌ Unexpected error on '{name}': {e}")
                return (None, None, 'ERROR')

        # All retries failed
        return (None, None, 'TIMEOUT')

    def run_batch(self, batch_size: int = 200):
        """Run batch geocoding with progress tracking"""
        logger.info("=" * 70)
        logger.info("🌍 Starting Batch Geocoding")
        logger.info("=" * 70)

        # Get pending entities
        entities = self.get_pending_entities(batch_size)

        if not entities:
            logger.info("✅ No pending entities found. All clean!")
            return

        total = len(entities)
        logger.info(f"📍 Found {total} locations to geocode")
        logger.info(f"⏱️  Estimated time: ~{total * self.rate_limit_delay / 60:.1f} minutes")
        logger.info("")

        # Stats tracking
        stats = {
            'success': 0,
            'not_found': 0,
            'errors': 0,
            'timeout': 0
        }

        start_time = time.time()

        # Process each entity
        for idx, (entity_id, name, entity_type, mention_count) in enumerate(entities, 1):
            # Geocode
            lat, lon, status = self.geocode_entity(name, entity_type)

            # Update database
            self.update_entity_geo(entity_id, lat, lon, status)

            # Track stats
            if status == 'FOUND':
                logger.info(f"   [{idx:3d}/{total}] ✅ {name:40s} → (geocoded) | {mention_count:3d} mentions")
                stats['success'] += 1
            elif status == 'NOT_FOUND':
                logger.info(f"   [{idx:3d}/{total}] ❌ {name:40s} → Not Found | {mention_count:3d} mentions")
                stats['not_found'] += 1
            elif status == 'TIMEOUT':
                logger.warning(f"   [{idx:3d}/{total}] ⏱️  {name:40s} → Timeout")
                stats['timeout'] += 1
            else:  # ERROR
                logger.error(f"   [{idx:3d}/{total}] 🔴 {name:40s} → Error")
                stats['errors'] += 1

            # Rate limiting (only if not last item)
            if idx < total:
                time.sleep(self.rate_limit_delay)

            # Progress checkpoint every 50 entities
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (total - idx) / rate if rate > 0 else 0
                logger.info(f"   Progress: {idx}/{total} | Rate: {rate:.1f} entities/sec | ETA: {remaining/60:.1f} min")

        # Final summary
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info("🏁 BATCH GEOCODING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"✅ Success:     {stats['success']:4d} ({stats['success']/total*100:.1f}%)")
        logger.info(f"❌ Not Found:   {stats['not_found']:4d} ({stats['not_found']/total*100:.1f}%)")
        logger.info(f"⏱️  Timeout:     {stats['timeout']:4d} ({stats['timeout']/total*100:.1f}%)")
        logger.info(f"🔴 Errors:      {stats['errors']:4d} ({stats['errors']/total*100:.1f}%)")
        logger.info("-" * 70)
        logger.info(f"⏱️  Total Time:  {elapsed/60:.1f} minutes")
        logger.info(f"📊 Rate:        {total/elapsed:.2f} entities/sec")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Batch geocode pending entities')
    parser.add_argument(
        '--batch',
        type=int,
        default=200,
        help='Number of entities to process (default: 200)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all pending entities (WARNING: can take hours!)'
    )

    args = parser.parse_args()

    # Determine batch size
    if args.all:
        batch_size = 999999  # Effectively "all"
        logger.warning("⚠️  Processing ALL pending entities. This may take several hours!")
        logger.warning("⚠️  Press Ctrl+C within 5 seconds to cancel...")
        time.sleep(5)
    else:
        batch_size = args.batch

    # Initialize and run
    geocoder = BatchGeocoder()
    geocoder.run_batch(batch_size=batch_size)
    geocoder.db.close()


if __name__ == "__main__":
    main()
