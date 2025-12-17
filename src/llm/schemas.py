"""
Pydantic schemas for structured LLM output validation

Sprint 2.1 MVP: Simplified schema for initial testing
Sprint 2.2: Full schema with nested models (trade signals, impact scores)
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional


class IntelligenceReportMVP(BaseModel):
    """
    Minimal schema for Sprint 2.1 MVP validation

    Focus: Validate JSON mode works reliably before expanding to complex nested models
    Success criteria: 95%+ validation success rate on real articles
    """

    title: str = Field(
        ...,
        description="Concise article title (5-15 words)",
        min_length=10,
        max_length=200
    )

    category: Literal["GEOPOLITICS", "DEFENSE", "ECONOMY", "CYBER", "ENERGY", "OTHER"] = Field(
        ...,
        description="Primary category for article classification"
    )

    executive_summary: str = Field(
        ...,
        description="BLUF-style summary: Bottom Line Up Front (100-300 words)",
        min_length=100,
        max_length=1500
    )

    sentiment_label: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"] = Field(
        ...,
        description="Overall sentiment towards investment/security outlook"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Analyst confidence in the assessment (0.0 = low, 1.0 = high)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "China Expands Military Presence in South China Sea",
                "category": "GEOPOLITICS",
                "executive_summary": "BLUF: China deployed 3 additional Type 055 destroyers to Hainan Naval Base...",
                "sentiment_label": "NEGATIVE",
                "confidence_score": 0.85
            }
        }


# ============================================================================
# SPRINT 2.2 - FULL SCHEMA (Implement AFTER MVP validates)
# ============================================================================
# Uncomment and test after MVP achieves 95%+ success rate

class ImpactScore(BaseModel):
    """Nested model for impact assessment"""
    score: int = Field(..., ge=0, le=10, description="Impact severity (0-10 scale)")
    reasoning: str = Field(..., description="Justification for impact score")


class SentimentAnalysis(BaseModel):
    """Enhanced sentiment with numeric score"""
    label: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment polarity (-1.0 to +1.0)")


class TradeSignal(BaseModel):
    """Trade recommendation with context"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'LMT', 'NVDA')")
    signal: Literal["BULLISH", "BEARISH", "NEUTRAL", "WATCHLIST"]
    timeframe: Literal["SHORT_TERM", "MEDIUM_TERM", "LONG_TERM"]
    rationale: str = Field(..., description="Specific catalyst driving the signal")


class IntelligenceReport(BaseModel):
    """
    Full schema with nested models (Sprint 2.2)

    WARNING: Complex schema - test AFTER MVP validation succeeds
    """
    title: str
    category: Literal["GEOPOLITICS", "DEFENSE", "ECONOMY", "CYBER", "ENERGY"]
    impact: ImpactScore
    sentiment: SentimentAnalysis
    key_entities: list[str] = Field(..., description="Top 5-10 entities mentioned")
    related_tickers: list[TradeSignal] = Field(..., description="Trade signals with tickers")
    executive_summary: str
    analysis_content: str = Field(..., description="Full markdown analysis")
    confidence_score: float = Field(ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Taiwan Semiconductor Reports Record Q4 Earnings",
                "category": "ECONOMY",
                "impact": {
                    "score": 8,
                    "reasoning": "TSMC is critical supplier for 90% of advanced chips globally"
                },
                "sentiment": {
                    "label": "POSITIVE",
                    "score": 0.75
                },
                "key_entities": ["Taiwan Semiconductor", "NVIDIA", "Apple", "China"],
                "related_tickers": [
                    {
                        "ticker": "TSM",
                        "signal": "BULLISH",
                        "timeframe": "MEDIUM_TERM",
                        "rationale": "Q4 revenue up 32% YoY driven by AI chip demand"
                    }
                ],
                "executive_summary": "BLUF: TSMC reported Q4 2024 revenue of $23.8B...",
                "analysis_content": "## Key Developments\n\n...",
                "confidence_score": 0.90
            }
        }
