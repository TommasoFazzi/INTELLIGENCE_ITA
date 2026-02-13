"""Dashboard statistics Pydantic schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class OverviewStats(BaseModel):
    """High-level platform statistics."""
    total_articles: int = 0
    total_entities: int = 0
    total_reports: int = 0
    geocoded_entities: int = 0
    coverage_percentage: float = Field(default=0.0, ge=0, le=100)


class DateRange(BaseModel):
    """Date range for articles."""
    first: Optional[datetime] = None
    last: Optional[datetime] = None


class SourceCount(BaseModel):
    """Article count per source."""
    source: str
    count: int


class ArticleStats(BaseModel):
    """Article-specific statistics."""
    by_category: dict[str, int] = Field(default_factory=dict)
    by_source: list[SourceCount] = Field(default_factory=list)
    recent_7d: int = 0
    date_range: DateRange = Field(default_factory=DateRange)


class EntityMention(BaseModel):
    """Top mentioned entity."""
    name: str
    type: str = "UNKNOWN"
    mentions: int = 0


class EntityStats(BaseModel):
    """Entity-specific statistics."""
    by_type: dict[str, int] = Field(default_factory=dict)
    top_mentioned: list[EntityMention] = Field(default_factory=list)


class QualityStats(BaseModel):
    """Report quality metrics."""
    reports_reviewed: int = 0
    average_rating: Optional[float] = Field(default=None, ge=1, le=5)
    pending_review: int = 0


class DashboardStats(BaseModel):
    """Complete dashboard statistics response."""
    overview: OverviewStats
    articles: ArticleStats
    entities: EntityStats
    quality: QualityStats
