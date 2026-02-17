"""Pydantic schemas for Storyline & Graph API responses."""
from typing import Optional
from pydantic import BaseModel


class StorylineNode(BaseModel):
    """A storyline node in the narrative graph."""
    id: int
    title: str
    summary: Optional[str] = None
    category: Optional[str] = None
    narrative_status: str  # emerging, active, stabilized
    momentum_score: float
    article_count: int
    key_entities: list[str] = []
    start_date: Optional[str] = None
    last_update: Optional[str] = None
    days_active: Optional[int] = None


class StorylineEdge(BaseModel):
    """An edge between two storylines."""
    source: int
    target: int
    weight: float
    relation_type: str = "relates_to"


class GraphStats(BaseModel):
    """Aggregate stats for the graph."""
    total_nodes: int
    total_edges: int
    avg_momentum: float


class GraphNetwork(BaseModel):
    """Full graph network: nodes + links + stats."""
    nodes: list[StorylineNode]
    links: list[StorylineEdge]
    stats: GraphStats


class RelatedStoryline(BaseModel):
    """A related storyline reference."""
    id: int
    title: str
    weight: float
    relation_type: str


class LinkedArticle(BaseModel):
    """An article linked to a storyline."""
    id: int
    title: str
    source: Optional[str] = None
    published_date: Optional[str] = None


class StorylineDetail(BaseModel):
    """Detailed storyline with relations and articles."""
    storyline: StorylineNode
    related_storylines: list[RelatedStoryline] = []
    recent_articles: list[LinkedArticle] = []
