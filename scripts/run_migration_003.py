#!/usr/bin/env python3
"""
Run database migration 003: Add entity coordinates
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager

def run_migration():
    """Execute migration 003"""
    print("Running migration 003: Add entity coordinates...")
    
    db = DatabaseManager()
    migration_file = Path(__file__).parent.parent / 'migrations' / '003_add_entity_coordinates.sql'
    
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(migration_sql)
            conn.commit()
        
        print("✓ Migration 003 completed successfully!")
        print("  - Added latitude, longitude columns")
        print("  - Added geo_status column (PENDING/FOUND/NOT_FOUND/RETRY)")
        print("  - Added geocoded_at timestamp")
        print("  - Created spatial indexes")
        
        return True
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
