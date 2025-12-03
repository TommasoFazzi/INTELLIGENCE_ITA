# Intelligence Map Intelligence Map - Implementation Guide

## 🎯 Project Overview

**Goal:** Create a Call of Duty-style tactical intelligence map with cinematic interactions, entity-based visualization, and real-time data from PostgreSQL.

**Tech Stack:**
- **Frontend:** Next.js 14 (App Router), React, TypeScript
- **Map Engine:** Mapbox GL JS
- **Animations:** Framer Motion
- **Backend:** FastAPI (Python)
- **Database:** PostgreSQL + pgvector
- **Geocoding:** Nominatim (OpenStreetMap) or Google Geocoding API

---

## 📦 Phase 1: Foundation Setup

### 1.1 Create Next.js Project

```bash
cd /Users/tommasofazzi/INTELLIGENCE_ITA/INTELLIGENCE_ITA
npx create-next-app@latest intelligence-map --typescript --tailwind --app --no-src-dir
cd intelligence-map
```

### 1.2 Install Dependencies

```bash
npm install mapbox-gl react-map-gl framer-motion
npm install @types/mapbox-gl
npm install use-sound  # Optional for sound effects
npm install axios swr  # For API calls
```

### 1.3 Setup Mapbox Account

1. Go to https://www.mapbox.com/
2. Sign up for free account
3. Get API token from dashboard
4. Add to `.env.local`:

```env
NEXT_PUBLIC_MAPBOX_TOKEN=your_token_here
```

### 1.4 Project Structure

```
intelligence-map/
├── app/
│   ├── page.tsx                 # Home/Dashboard
│   ├── intelligence-map/
│   │   └── page.tsx            # Intelligence Map main page
│   └── layout.tsx
├── components/
│   ├── WarRoom/
│   │   ├── TacticalMap.tsx     # Main map component
│   │   ├── DossierPanel.tsx    # Entity detail panel
│   │   ├── HUDOverlay.tsx      # Tactical HUD
│   │   ├── GridOverlay.tsx     # Grid pattern
│   │   └── EntityMarker.tsx    # Custom markers
│   └── ui/                      # Reusable UI components
├── lib/
│   ├── api.ts                   # API client
│   ├── geocoding.ts             # Geocoding service
│   └── types.ts                 # TypeScript types
└── public/
    ├── sounds/                  # Sound effects
    └── images/                  # Icons, textures
```

---

## 🗺️ Phase 2: Geographic Entity Extraction

### 2.1 Geocoding Service

Create a Python service to extract coordinates from entity names:

```python
# scripts/geocode_entities.py
import requests
from src.storage.database import DatabaseManager

def geocode_entity(entity_name: str, entity_type: str):
    """
    Geocode entity using Nominatim (free) or Google Geocoding API
    """
    # Nominatim (OpenStreetMap) - Free
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': entity_name,
        'format': 'json',
        'limit': 1
    }
    
    response = requests.get(url, params=params, headers={'User-Agent': 'INTEL_ITA'})
    data = response.json()
    
    if data:
        return {
            'lat': float(data[0]['lat']),
            'lng': float(data[0]['lon'])
        }
    return None

def backfill_entity_coordinates():
    """
    Add coordinates to existing entities
    """
    db = DatabaseManager()
    
    # Get all GPE (Geo-Political Entity) and LOC (Location) entities
    entities = db.get_entities_by_type(['GPE', 'LOC'])
    
    for entity in entities:
        coords = geocode_entity(entity['name'], entity['type'])
        if coords:
            db.update_entity_coordinates(entity['id'], coords['lat'], coords['lng'])
            print(f"✓ {entity['name']}: {coords}")
        else:
            print(f"✗ {entity['name']}: No coordinates found")

if __name__ == "__main__":
    backfill_entity_coordinates()
```

### 2.2 Database Schema Update

```sql
-- Add coordinates to entities table
ALTER TABLE entities 
ADD COLUMN IF NOT EXISTS latitude DECIMAL(10, 8),
ADD COLUMN IF NOT EXISTS longitude DECIMAL(11, 8);

-- Add spatial index for performance
CREATE INDEX IF NOT EXISTS idx_entities_coordinates 
ON entities(latitude, longitude) 
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
```

---

## 🎨 Phase 3: Intelligence Map UI - Base Map

### 3.1 Custom Mapbox Style

Create dark military style in Mapbox Studio:
- Base color: `#0A1628` (your navy)
- Water: `#0d1b2a`
- Land: `#1a2332`
- Borders: `#FF6B35` (orange, subtle)
- Labels: Minimal, monospace font

### 3.2 TacticalMap Component

```typescript
// components/WarRoom/TacticalMap.tsx
'use client';

import { useState, useCallback } from 'react';
import Map, { Marker, FlyToInterpolator } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

interface ViewState {
  latitude: number;
  longitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

export default function TacticalMap() {
  const [viewState, setViewState] = useState<ViewState>({
    latitude: 41.9028,
    longitude: 12.4964,
    zoom: 3,
    pitch: 0,
    bearing: 0
  });

  return (
    <div className="relative w-full h-screen bg-black">
      <Map
        {...viewState}
        onMove={evt => setViewState(evt.viewState)}
        mapStyle="mapbox://styles/mapbox/dark-v11"
        mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* Grid Overlay */}
      <GridOverlay />
      
      {/* HUD */}
      <HUDOverlay coordinates={viewState} />
    </div>
  );
}
```

---

## 🎯 Phase 4-10: Detailed in Separate Docs

Each phase will have its own implementation guide as we progress.

---

## 🚀 Getting Started

1. **Setup Mapbox account** (5 min)
2. **Create Next.js project** (10 min)
3. **Install dependencies** (5 min)
4. **Create basic map** (30 min)
5. **Test with mock data** (15 min)

**Total time for Phase 1:** ~1 hour

---

## 📝 Notes

- We'll build incrementally, testing each phase
- Mock data first, then real API integration
- Focus on visual polish (this is the wow factor)
- Performance optimization comes later

---

## 🔗 Resources

- [Mapbox GL JS Docs](https://docs.mapbox.com/mapbox-gl-js/)
- [Framer Motion Docs](https://www.framer.com/motion/)
- [Next.js App Router](https://nextjs.org/docs/app)
