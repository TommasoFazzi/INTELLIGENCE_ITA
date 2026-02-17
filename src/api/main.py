"""
FastAPI Backend for Intelligence Map

Provides REST API endpoints for entity visualization on the map.
"""
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from fastapi import FastAPI, HTTPException, Depends, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from src.storage.database import DatabaseManager
from src.utils.logger import get_logger
from src.api.routers import dashboard, reports, stories

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligence ITA API",
    description="REST API for Intelligence ITA platform - Dashboard, Reports, and Map visualization",
    version="1.1.0"
)

# Configure CORS for Next.js frontend (restricted methods and headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002"
    ],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],  # Only allow necessary methods
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],  # Only necessary headers
)

# Include routers
app.include_router(dashboard.router)
app.include_router(reports.router)
app.include_router(stories.router)

# ===================================================================
# API Key Authentication
# ===================================================================

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Load API key from environment (generate one if not set)
API_KEY = os.getenv("INTELLIGENCE_API_KEY")

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key for protected endpoints."""
    if not API_KEY:
        # If no API key is configured, allow access (development mode)
        return "dev_mode"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header."
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return api_key

# Initialize database
db = DatabaseManager()


# ===================================================================
# Pydantic Models
# ===================================================================

class EntityProperties(BaseModel):
    id: int
    name: str
    entity_type: str
    mention_count: int
    metadata: dict = {}


class EntityFeature(BaseModel):
    type: str = "Feature"
    geometry: dict
    properties: EntityProperties


class EntityCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[EntityFeature]


# ===================================================================
# API Endpoints
# ===================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Intelligence ITA API",
        "version": "1.1.0",
        "endpoints": {
            "map": {
                "entities": "/api/v1/map/entities",
                "entity_detail": "/api/v1/map/entities/{id}"
            },
            "dashboard": {
                "stats": "/api/v1/dashboard/stats"
            },
            "reports": {
                "list": "/api/v1/reports",
                "detail": "/api/v1/reports/{id}"
            }
        }
    }


@app.get("/api/v1/map/entities", response_model=EntityCollection)
async def get_entities(
    limit: int = Query(default=5000, ge=1, le=5000, description="Max entities (1-5000)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get all entities with coordinates in GeoJSON format.

    Requires API key authentication via X-API-Key header.

    Args:
        limit: Maximum number of entities to return (1-1000)

    Returns:
        GeoJSON FeatureCollection
    """
    try:
        geojson = db.get_entities_with_coordinates(limit=limit)
        logger.info(f"Returned {len(geojson['features'])} entities")
        return geojson

    except Exception as e:
        logger.error(f"Error fetching entities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/map/entities/{entity_id}")
async def get_entity(
    entity_id: int,
    api_key: str = Depends(verify_api_key)
):
    """
    Get single entity details.

    Requires API key authentication via X-API-Key header.

    Args:
        entity_id: Entity ID

    Returns:
        Entity details with related articles
    """
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Get entity details
                cur.execute("""
                    SELECT 
                        id, name, entity_type, latitude, longitude,
                        mention_count, first_seen, last_seen, metadata
                    FROM entities
                    WHERE id = %s
                """, (entity_id,))
                
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Entity not found")
                
                entity = {
                    'id': row[0],
                    'name': row[1],
                    'entity_type': row[2],
                    'latitude': float(row[3]) if row[3] else None,
                    'longitude': float(row[4]) if row[4] else None,
                    'mention_count': row[5],
                    'first_seen': row[6].isoformat() if row[6] else None,
                    'last_seen': row[7].isoformat() if row[7] else None,
                    'metadata': row[8]
                }
                
                # Get related articles
                cur.execute("""
                    SELECT 
                        a.id, a.title, a.link, a.published_date, a.source
                    FROM articles a
                    JOIN entity_mentions em ON a.id = em.article_id
                    WHERE em.entity_id = %s
                    ORDER BY a.published_date DESC
                    LIMIT 10
                """, (entity_id,))
                
                articles = []
                for article_row in cur.fetchall():
                    articles.append({
                        'id': article_row[0],
                        'title': article_row[1],
                        'link': article_row[2],
                        'published_date': article_row[3].isoformat() if article_row[3] else None,
                        'source': article_row[4]
                    })
                
                entity['related_articles'] = articles
                
                return entity
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
