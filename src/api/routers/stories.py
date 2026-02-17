"""Stories & Graph API router."""
import json
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import Optional

from ..schemas.common import APIResponse, PaginationMeta
from ..schemas.stories import (
    StorylineNode, StorylineEdge, GraphStats, GraphNetwork,
    StorylineDetail, RelatedStoryline, LinkedArticle,
)
from ...storage.database import DatabaseManager

router = APIRouter(prefix="/api/v1/stories", tags=["Stories"])


def get_db() -> DatabaseManager:
    """Get database connection."""
    return DatabaseManager()


@router.get("/graph")
async def get_graph_network():
    """
    Get the full narrative graph: active storyline nodes + edges.

    Returns data structured for react-force-graph (nodes + links).
    """
    db = get_db()
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Nodes: active storylines from the view
                cur.execute("""
                    SELECT id, title, summary, narrative_status,
                           category, article_count, momentum_score,
                           key_entities, start_date, last_update,
                           days_active
                    FROM v_active_storylines
                """)
                node_rows = cur.fetchall()

                # Edges: from the graph view
                cur.execute("""
                    SELECT source_story_id, target_story_id,
                           weight, relation_type
                    FROM v_storyline_graph
                """)
                edge_rows = cur.fetchall()

        nodes = []
        momentum_sum = 0.0
        for r in node_rows:
            entities = r[7] or []
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                except (json.JSONDecodeError, TypeError):
                    entities = []

            node = StorylineNode(
                id=r[0],
                title=r[1],
                summary=r[2],
                narrative_status=r[3] or "active",
                category=r[4],
                article_count=r[5] or 0,
                momentum_score=round(r[6] or 0.0, 3),
                key_entities=entities if isinstance(entities, list) else [],
                start_date=r[8].isoformat() if r[8] else None,
                last_update=r[9].isoformat() if r[9] else None,
                days_active=r[10],
            )
            nodes.append(node)
            momentum_sum += node.momentum_score

        links = [
            StorylineEdge(
                source=r[0],
                target=r[1],
                weight=round(r[2] or 0.0, 3),
                relation_type=r[3] or "relates_to",
            )
            for r in edge_rows
        ]

        avg_momentum = round(momentum_sum / len(nodes), 3) if nodes else 0.0

        graph = GraphNetwork(
            nodes=nodes,
            links=links,
            stats=GraphStats(
                total_nodes=len(nodes),
                total_edges=len(links),
                avg_momentum=avg_momentum,
            ),
        )

        return {
            "success": True,
            "data": graph.model_dump(),
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_storylines(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(
        None,
        description="Filter by narrative_status (emerging, active, stabilized, archived)",
    ),
):
    """
    List storylines with pagination, ordered by momentum_score DESC.
    """
    db = get_db()
    try:
        conditions = ["1=1"]
        params: list = []

        if status:
            conditions.append("narrative_status = %s")
            params.append(status)
        else:
            # Default: only active storylines
            conditions.append("narrative_status IN ('emerging', 'active')")

        where_clause = " AND ".join(conditions)

        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM storylines WHERE {where_clause}",
                    params,
                )
                total = cur.fetchone()[0]

                offset = (page - 1) * per_page
                cur.execute(f"""
                    SELECT id, title, summary, narrative_status,
                           category, article_count, momentum_score,
                           key_entities, start_date, last_update,
                           EXTRACT(DAY FROM NOW() - start_date)::INTEGER AS days_active
                    FROM storylines
                    WHERE {where_clause}
                    ORDER BY momentum_score DESC, last_update DESC
                    LIMIT %s OFFSET %s
                """, params + [per_page, offset])

                rows = cur.fetchall()

        storylines = []
        for r in rows:
            entities = r[7] or []
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                except (json.JSONDecodeError, TypeError):
                    entities = []

            storylines.append(StorylineNode(
                id=r[0],
                title=r[1],
                summary=r[2],
                narrative_status=r[3] or "active",
                category=r[4],
                article_count=r[5] or 0,
                momentum_score=round(r[6] or 0.0, 3),
                key_entities=entities if isinstance(entities, list) else [],
                start_date=r[8].isoformat() if r[8] else None,
                last_update=r[9].isoformat() if r[9] else None,
                days_active=r[10],
            ).model_dump())

        return {
            "success": True,
            "data": {
                "storylines": storylines,
                "pagination": PaginationMeta.calculate(total, page, per_page).model_dump(),
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{storyline_id}")
async def get_storyline_detail(storyline_id: int):
    """
    Get detailed storyline with related storylines and recent articles.
    """
    db = get_db()
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Storyline base data
                cur.execute("""
                    SELECT id, title, summary, narrative_status,
                           category, article_count, momentum_score,
                           key_entities, start_date, last_update,
                           EXTRACT(DAY FROM NOW() - start_date)::INTEGER AS days_active
                    FROM storylines
                    WHERE id = %s
                """, [storyline_id])
                row = cur.fetchone()

                if not row:
                    raise HTTPException(status_code=404, detail="Storyline not found")

                # Related storylines via edges (both directions)
                cur.execute("""
                    SELECT s.id, s.title, e.weight, e.relation_type
                    FROM storyline_edges e
                    JOIN storylines s ON (
                        CASE WHEN e.source_story_id = %s
                             THEN e.target_story_id
                             ELSE e.source_story_id
                        END = s.id
                    )
                    WHERE (e.source_story_id = %s OR e.target_story_id = %s)
                      AND s.narrative_status IN ('emerging', 'active')
                    ORDER BY e.weight DESC
                    LIMIT 10
                """, [storyline_id, storyline_id, storyline_id])
                related_rows = cur.fetchall()

                # Recent articles (last 10)
                cur.execute("""
                    SELECT a.id, a.title, a.source, a.published_date
                    FROM article_storylines als
                    JOIN articles a ON als.article_id = a.id
                    WHERE als.storyline_id = %s
                    ORDER BY a.published_date DESC
                    LIMIT 10
                """, [storyline_id])
                article_rows = cur.fetchall()

        entities = row[7] or []
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except (json.JSONDecodeError, TypeError):
                entities = []

        storyline_node = StorylineNode(
            id=row[0],
            title=row[1],
            summary=row[2],
            narrative_status=row[3] or "active",
            category=row[4],
            article_count=row[5] or 0,
            momentum_score=round(row[6] or 0.0, 3),
            key_entities=entities if isinstance(entities, list) else [],
            start_date=row[8].isoformat() if row[8] else None,
            last_update=row[9].isoformat() if row[9] else None,
            days_active=row[10],
        )

        detail = StorylineDetail(
            storyline=storyline_node,
            related_storylines=[
                RelatedStoryline(
                    id=r[0], title=r[1],
                    weight=round(r[2] or 0.0, 3),
                    relation_type=r[3] or "relates_to",
                )
                for r in related_rows
            ],
            recent_articles=[
                LinkedArticle(
                    id=r[0], title=r[1],
                    source=r[2],
                    published_date=r[3].isoformat() if r[3] else None,
                )
                for r in article_rows
            ],
        )

        return {
            "success": True,
            "data": detail.model_dump(),
            "generated_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
