"""Reports API router."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, date
from typing import Optional

from ..schemas.common import APIResponse, PaginationMeta
from ..schemas.reports import (
    ReportListItem, ReportDetail, ReportFilters,
    ReportContent, ReportSource, ReportFeedback,
    ReportMetadata
)
from ...storage.database import DatabaseManager
from ..auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])


def get_db() -> DatabaseManager:
    """Get database connection."""
    return DatabaseManager()


@router.get("")
async def list_reports(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status (draft, reviewed, approved)"),
    report_type: Optional[str] = Query(None, description="Filter by type (daily, weekly)"),
    date_from: Optional[date] = Query(None, description="Filter reports from this date"),
    date_to: Optional[date] = Query(None, description="Filter reports until this date"),
    api_key: str = Depends(verify_api_key),
):
    """
    List reports with pagination and filters.

    - **page**: Page number (default: 1)
    - **per_page**: Items per page (default: 20, max: 100)
    - **status**: Filter by status (draft, reviewed, approved)
    - **report_type**: Filter by type (daily, weekly)
    - **date_from**: Filter reports from this date
    - **date_to**: Filter reports until this date
    """
    db = get_db()
    try:
        # Build query
        conditions = []
        params = []

        if status:
            conditions.append("status = %s")
            params.append(status)
        if report_type:
            conditions.append("report_type = %s")
            params.append(report_type)
        if date_from:
            conditions.append("report_date >= %s")
            params.append(date_from)
        if date_to:
            conditions.append("report_date <= %s")
            params.append(date_to)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Get total count
                cur.execute(f"SELECT COUNT(*) FROM reports WHERE {where_clause}", params)
                total = cur.fetchone()[0]

                # Get paginated results
                offset = (page - 1) * per_page
                cur.execute(f"""
                    SELECT
                        id, report_date, report_type, status,
                        COALESCE(
                            metadata->>'title',
                            SUBSTRING(COALESCE(final_content, draft_content), 1, 80)
                        ) as title,
                        COALESCE(
                            metadata->>'category',
                            UPPER(metadata->'focus_areas'->>0)
                        ) as category,
                        SUBSTRING(COALESCE(final_content, draft_content), 1, 300) as summary,
                        COALESCE(
                            (metadata->>'article_count')::int,
                            (metadata->>'recent_articles_count')::int,
                            0
                        ) as article_count,
                        generated_at,
                        human_reviewed_at,
                        human_reviewer
                    FROM reports
                    WHERE {where_clause}
                    ORDER BY report_date DESC, generated_at DESC
                    LIMIT %s OFFSET %s
                """, params + [per_page, offset])

                reports = [
                    ReportListItem(
                        id=r[0],
                        report_date=r[1],
                        report_type=r[2] or "daily",
                        status=r[3] or "draft",
                        title=r[4],
                        category=r[5],
                        executive_summary=r[6],
                        article_count=r[7] or 0,
                        generated_at=r[8],
                        reviewed_at=r[9],
                        reviewer=r[10]
                    )
                    for r in cur.fetchall()
                ]

        return {
            "success": True,
            "data": {
                "reports": [r.model_dump() for r in reports],
                "pagination": PaginationMeta.calculate(total, page, per_page).model_dump(),
                "filters_applied": ReportFilters(
                    status=status,
                    report_type=report_type,
                    date_from=date_from,
                    date_to=date_to
                ).model_dump()
            },
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("List reports error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{report_id}")
async def get_report(report_id: int, api_key: str = Depends(verify_api_key)):
    """
    Get detailed report by ID.

    Returns full report content, sources, and feedback.
    """
    db = get_db()
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Get report
                cur.execute("""
                    SELECT
                        id, report_date, report_type, status,
                        model_used, draft_content, final_content,
                        metadata, sources, generated_at
                    FROM reports
                    WHERE id = %s
                """, [report_id])
                row = cur.fetchone()

                if not row:
                    raise HTTPException(status_code=404, detail="Report not found")

                # Get feedback
                cur.execute("""
                    SELECT section_name, rating, comment
                    FROM report_feedback
                    WHERE report_id = %s
                """, [report_id])
                feedback_rows = cur.fetchall()

        # Parse content
        content_text = row[6] or row[5] or ""
        metadata = row[7] or {}
        sources_data = row[8] or []

        # Build sources list safely
        # Sources can be a flat list or a dict with recent_articles/historical_context
        sources = []
        source_items = []
        if isinstance(sources_data, dict):
            source_items = sources_data.get('recent_articles', [])
            source_items += sources_data.get('historical_context', [])
        elif isinstance(sources_data, list):
            source_items = sources_data

        for s in source_items:
            if isinstance(s, dict):
                sources.append(ReportSource(
                    article_id=s.get('article_id', 0),
                    title=s.get('title', ''),
                    link=s.get('link', ''),
                    relevance_score=s.get('relevance_score') or s.get('similarity')
                ))

        # Derive title: metadata title > first line of content > fallback
        title = metadata.get('title')
        if not title and content_text:
            first_line = content_text.strip().split('\n')[0]
            # Strip markdown headers
            title = first_line.lstrip('#').strip()[:120] or f"Report {row[1]}"
        elif not title:
            title = f"Report {row[1]}"

        report = ReportDetail(
            id=row[0],
            report_date=row[1],
            report_type=row[2] or "daily",
            status=row[3] or "draft",
            model_used=row[4],
            content=ReportContent(
                title=title,
                executive_summary=content_text[:500] if content_text else "",
                full_text=content_text,
                sections=[]
            ),
            sources=sources,
            feedback=[
                ReportFeedback(
                    section=f[0] or "general",
                    rating=f[1],
                    comment=f[2]
                )
                for f in feedback_rows
            ],
            metadata=ReportMetadata(
                processing_time_ms=metadata.get('processing_time_ms'),
                token_count=metadata.get('token_count')
            )
        )

        return {
            "success": True,
            "data": report.model_dump(),
            "generated_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get report %s error: %s", report_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
