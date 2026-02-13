"""Common Pydantic schemas for API responses."""
from datetime import datetime
from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""
    success: bool = True
    data: T
    error: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class PaginationMeta(BaseModel):
    """Pagination metadata."""
    page: int = 1
    per_page: int = 20
    total: int
    pages: int

    @classmethod
    def calculate(cls, total: int, page: int = 1, per_page: int = 20) -> "PaginationMeta":
        """Calculate pagination metadata from total count."""
        return cls(
            page=page,
            per_page=per_page,
            total=total,
            pages=max(1, (total + per_page - 1) // per_page)
        )
