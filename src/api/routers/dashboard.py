"""Dashboard API router."""
from fastapi import APIRouter, HTTPException
from datetime import datetime

from ..schemas.common import APIResponse
from ..schemas.dashboard import (
    DashboardStats, OverviewStats, ArticleStats,
    EntityStats, QualityStats, SourceCount,
    EntityMention, DateRange
)
from ...storage.database import DatabaseManager

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


def get_db() -> DatabaseManager:
    """Get database connection."""
    return DatabaseManager()


@router.get("/stats", response_model=APIResponse[DashboardStats])
async def get_dashboard_stats():
    """
    Get comprehensive dashboard statistics.

    Returns aggregated metrics for articles, entities, and reports.
    """
    db = get_db()
    try:
        # Get base statistics from existing method
        base_stats = db.get_statistics()

        # Get entity statistics
        entity_stats = _get_entity_stats(db)

        # Get quality metrics
        quality_stats = _get_quality_stats(db)

        # Get date range
        date_range = _get_date_range(db)

        # Build response
        stats = DashboardStats(
            overview=OverviewStats(
                total_articles=base_stats.get('total_articles', 0),
                total_entities=entity_stats['total'],
                total_reports=_count_reports(db),
                geocoded_entities=entity_stats['geocoded'],
                coverage_percentage=_calc_coverage(
                    entity_stats['geocoded'],
                    entity_stats['total']
                )
            ),
            articles=ArticleStats(
                by_category=base_stats.get('by_category', {}),
                by_source=[
                    SourceCount(source=k, count=v)
                    for k, v in base_stats.get('top_sources', {}).items()
                ],
                recent_7d=base_stats.get('recent_articles', 0),
                date_range=date_range
            ),
            entities=EntityStats(
                by_type=entity_stats['by_type'],
                top_mentioned=entity_stats['top_mentioned']
            ),
            quality=quality_stats
        )

        return APIResponse(success=True, data=stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_entity_stats(db: DatabaseManager) -> dict:
    """Get entity statistics from database."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Total and geocoded count
                cur.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(CASE WHEN latitude IS NOT NULL THEN 1 END) as geocoded
                    FROM entities
                """)
                row = cur.fetchone()
                total = row[0] if row else 0
                geocoded = row[1] if row else 0

                # By type distribution
                cur.execute("""
                    SELECT entity_type, COUNT(*) as count
                    FROM entities
                    WHERE entity_type IS NOT NULL
                    GROUP BY entity_type
                    ORDER BY count DESC
                """)
                by_type = {r[0]: r[1] for r in cur.fetchall()}

                # Top mentioned entities
                cur.execute("""
                    SELECT name, entity_type, mention_count
                    FROM entities
                    WHERE mention_count > 0
                    ORDER BY mention_count DESC
                    LIMIT 10
                """)
                top_mentioned = [
                    EntityMention(
                        name=r[0],
                        type=r[1] or "UNKNOWN",
                        mentions=r[2]
                    )
                    for r in cur.fetchall()
                ]

        return {
            'total': total,
            'geocoded': geocoded,
            'by_type': by_type,
            'top_mentioned': top_mentioned
        }
    except Exception:
        return {'total': 0, 'geocoded': 0, 'by_type': {}, 'top_mentioned': []}


def _get_quality_stats(db: DatabaseManager) -> QualityStats:
    """Get report quality statistics."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(DISTINCT r.id) FILTER (WHERE r.status IN ('reviewed', 'approved')) as reviewed,
                        AVG(f.rating) as avg_rating,
                        COUNT(DISTINCT r.id) FILTER (WHERE r.status = 'draft') as pending
                    FROM reports r
                    LEFT JOIN report_feedback f ON r.id = f.report_id
                """)
                row = cur.fetchone()

        return QualityStats(
            reports_reviewed=row[0] if row and row[0] else 0,
            average_rating=round(row[1], 1) if row and row[1] else None,
            pending_review=row[2] if row and row[2] else 0
        )
    except Exception:
        return QualityStats()


def _get_date_range(db: DatabaseManager) -> DateRange:
    """Get article date range."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MIN(published_date), MAX(published_date)
                    FROM articles
                    WHERE published_date IS NOT NULL
                """)
                row = cur.fetchone()

        return DateRange(
            first=row[0] if row else None,
            last=row[1] if row else None
        )
    except Exception:
        return DateRange()


def _count_reports(db: DatabaseManager) -> int:
    """Count total reports."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM reports")
                row = cur.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def _calc_coverage(geocoded: int, total: int) -> float:
    """Calculate geocoding coverage percentage."""
    if total == 0:
        return 0.0
    return round((geocoded / total) * 100, 1)
