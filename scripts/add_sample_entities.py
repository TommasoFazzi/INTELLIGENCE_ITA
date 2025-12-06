#!/usr/bin/env python3
"""
Add sample entities for testing Intelligence Map
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager

# Sample entities with known coordinates
SAMPLE_ENTITIES = [
    ("Taiwan", "GPE", 23.6978, 120.9605),
    ("United States", "GPE", 37.0902, -95.7129),
    ("China", "GPE", 35.8617, 104.1954),
    ("Russia", "GPE", 61.5240, 105.3188),
    ("Ukraine", "GPE", 48.3794, 31.1656),
    ("Israel", "GPE", 31.0461, 34.8516),
    ("Gaza", "LOC", 31.3547, 34.3088),
    ("New York", "GPE", 40.7128, -74.0060),
    ("London", "GPE", 51.5074, -0.1278),
    ("Tokyo", "GPE", 35.6762, 139.6503),
]

def add_sample_entities():
    """Add sample entities to database"""
    db = DatabaseManager()
    
    print("Adding sample entities...")
    
    for name, entity_type, lat, lng in SAMPLE_ENTITIES:
        # Save entity
        entity_id = db.save_entity(name, entity_type, {'sample': True})
        
        if entity_id:
            # Update with coordinates
            db.update_entity_coordinates(entity_id, lat, lng, 'FOUND')
            print(f"✓ Added {name} ({entity_type}): {lat}, {lng}")
    
    print(f"\n✓ Added {len(SAMPLE_ENTITIES)} sample entities")

if __name__ == "__main__":
    add_sample_entities()
