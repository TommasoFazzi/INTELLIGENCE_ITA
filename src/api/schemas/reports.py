"""Reports Pydantic schemas."""
from datetime import datetime, date
from typing import Optional, Literal
from pydantic import BaseModel, Field

ReportStatus = Literal["draft", "reviewed", "approved"]
ReportType = Literal["daily", "weekly"]
Category = Literal["GEOPOLITICS", "DEFENSE", "ECONOMY", "CYBER", "ENERGY", "OTHER"]


class ReportSource(BaseModel):
    """Source article reference."""
    article_id: int
    title: str
    link: str
    relevance_score: Optional[float] = None


class ReportFeedback(BaseModel):
    """Report section feedback."""
    section: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    comment: Optional[str] = None


class ReportListItem(BaseModel):
    """Report item for list view."""
    id: int
    report_date: date
    report_type: str = "daily"
    status: str = "draft"
    title: Optional[str] = None
    category: Optional[str] = None
    executive_summary: Optional[str] = None
    article_count: int = 0
    generated_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    reviewer: Optional[str] = None


class ReportSection(BaseModel):
    """Report content section."""
    category: str
    content: str
    entities: list[str] = Field(default_factory=list)


class ReportContent(BaseModel):
    """Full report content structure."""
    title: str
    executive_summary: str
    full_text: str = ""
    sections: list[ReportSection] = Field(default_factory=list)


class ReportMetadata(BaseModel):
    """Report processing metadata."""
    processing_time_ms: Optional[int] = None
    token_count: Optional[int] = None


class ReportDetail(BaseModel):
    """Complete report detail response."""
    id: int
    report_date: date
    report_type: str = "daily"
    status: str = "draft"
    model_used: Optional[str] = None
    content: ReportContent
    sources: list[ReportSource] = Field(default_factory=list)
    feedback: list[ReportFeedback] = Field(default_factory=list)
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)


class ReportFilters(BaseModel):
    """Applied filters for report listing."""
    status: Optional[str] = None
    report_type: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
