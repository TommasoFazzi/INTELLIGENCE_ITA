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


# ============================================================================
# MACRO-FIRST PIPELINE SCHEMAS
# ============================================================================
# Used by the serialized pipeline (--macro-first flag) where:
# 1. Macro report is generated first
# 2. Context is condensed for efficiency
# 3. Trade signals are extracted with macro alignment check


class MacroCondensedContext(BaseModel):
    """
    Token-efficient condensation of macro report (~500 tokens).

    Used as context for article-level signal extraction instead of
    passing the full report (5000+ tokens), reducing API costs by ~90%.
    """
    key_themes: list[str] = Field(
        ...,
        description="Top 5-7 strategic themes from the macro report",
        min_length=3,
        max_length=10
    )

    dominant_sentiment: Literal["RISK_ON", "RISK_OFF", "MIXED"] = Field(
        ...,
        description="Overall market sentiment from macro analysis"
    )

    priority_sectors: list[str] = Field(
        ...,
        description="Sectors most affected by current events (e.g., 'Defense', 'Semiconductors')",
        max_length=5
    )

    tickers_mentioned: list[str] = Field(
        ...,
        description="Tickers explicitly mentioned in the macro report",
        max_length=20
    )

    geopolitical_hotspots: list[str] = Field(
        ...,
        description="Active geopolitical regions (e.g., 'Taiwan Strait', 'Middle East')",
        max_length=5
    )

    time_horizon_focus: Literal["IMMEDIATE", "SHORT_TERM", "MEDIUM_TERM"] = Field(
        ...,
        description="Primary time horizon of macro concerns"
    )


class ReportLevelSignal(BaseModel):
    """
    Trade signal extracted at macro report level.

    These are HIGH-CONVICTION signals derived from the synthesis of
    multiple articles, not individual article events.
    """
    ticker: str = Field(..., description="Stock ticker symbol from whitelist")
    signal: Literal["BULLISH", "BEARISH", "NEUTRAL", "WATCHLIST"]
    timeframe: Literal["SHORT_TERM", "MEDIUM_TERM", "LONG_TERM"]
    rationale: str = Field(
        ...,
        description="Macro-level rationale spanning multiple articles/themes"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this signal (0.7+ for macro signals)"
    )
    supporting_themes: list[str] = Field(
        ...,
        description="Which macro themes support this signal"
    )


class ArticleLevelSignal(BaseModel):
    """
    Trade signal extracted from individual article WITH macro alignment check.

    Includes alignment_score indicating how well the signal aligns with
    the broader macro narrative. Low alignment may indicate contrarian signal.
    """
    ticker: str = Field(..., description="Stock ticker symbol from whitelist")
    signal: Literal["BULLISH", "BEARISH", "NEUTRAL", "WATCHLIST"]
    timeframe: Literal["SHORT_TERM", "MEDIUM_TERM", "LONG_TERM"]
    rationale: str = Field(..., description="Article-specific catalyst")
    confidence: float = Field(..., ge=0.0, le=1.0)
    alignment_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Alignment with macro narrative (1.0 = perfect, 0.0 = contrarian)"
    )
    alignment_reasoning: str = Field(
        ...,
        description="Explanation of alignment or divergence from macro themes"
    )


class MacroSignalsResult(BaseModel):
    """
    Complete output from report-level signal extraction.
    """
    condensed_context: MacroCondensedContext
    report_signals: list[ReportLevelSignal]
    extraction_timestamp: str


class ArticleSignalsResult(BaseModel):
    """
    Output from article-level signal extraction with macro context.
    """
    article_id: int
    article_title: str
    signals: list[ArticleLevelSignal]
    macro_alignment_summary: str = Field(
        ...,
        description="Brief summary of how this article fits the macro narrative"
    )


# ============================================================================
# QUERY ANALYZER SCHEMA (Query pre-processing for Oracle)
# ============================================================================

class ExtractedFilters(BaseModel):
    """
    Structured filters extracted from natural language query.

    Used by QueryAnalyzer to enable temporal/categorical filtering in RAG.
    Solves the problem of vector search not understanding date constraints.

    Example:
        Input: "Cosa è successo a Taiwan negli ultimi 7 giorni?"
        Output: ExtractedFilters(
            gpe_filter=['Taiwan'],
            start_date='2024-12-26',
            end_date='2025-01-02',
            semantic_query='Taiwan events developments'
        )
    """

    start_date: Optional[str] = Field(
        None,
        description="ISO date (YYYY-MM-DD) for range start. "
                    "Extract from 'ultimi X giorni', 'da settembre', 'il 15 dicembre'."
    )
    end_date: Optional[str] = Field(
        None,
        description="ISO date (YYYY-MM-DD) for range end. Usually today unless specified."
    )
    categories: Optional[list[Literal["GEOPOLITICS", "DEFENSE", "ECONOMY", "CYBER", "ENERGY"]]] = Field(
        None,
        description="Inferred categories. 'cyber attack' -> CYBER, 'difesa' -> DEFENSE"
    )
    gpe_filter: Optional[list[str]] = Field(
        None,
        description="Geographic entities normalized to English: 'Cina' -> 'China'"
    )
    sources: Optional[list[str]] = Field(
        None,
        description="News sources if explicitly mentioned: 'Reuters', 'Bloomberg'"
    )
    semantic_query: str = Field(
        ...,
        description="Query optimized for semantic search - temporal expressions removed"
    )
    extraction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction accuracy (0.0-1.0)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2024-12-26",
                "end_date": "2025-01-02",
                "categories": ["CYBER"],
                "gpe_filter": ["Russia", "Ukraine"],
                "sources": None,
                "semantic_query": "cyber attacks critical infrastructure",
                "extraction_confidence": 0.85
            }
        }
