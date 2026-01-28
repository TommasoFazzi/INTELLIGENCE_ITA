"""
Scoring functions for Financial Intelligence v2.

Calculates penalties and bonuses based on market metrics,
then combines with LLM confidence to produce intelligence_score.
"""

from typing import Tuple, Optional
from .types import TickerMetrics
from .constants import THRESHOLDS


def calculate_technical_penalty(metrics: TickerMetrics) -> int:
    """
    Calculate penalty based on SMA200 deviation.

    Uses non-linear scaling for extreme deviations to penalize
    chasing extended stocks.

    Penalty formula for deviation > 30%:
        z_proxy = deviation / 20.0  (approximates standard deviations)
        penalty = ((z_proxy - 1.5) ^ 1.5) * 8

    Args:
        metrics: TickerMetrics with sma_200_deviation_pct

    Returns:
        Penalty points (0-40, higher = worse)
    """
    if metrics.sma_200_deviation_pct is None:
        return 0  # No penalty if no data

    deviation = abs(metrics.sma_200_deviation_pct)

    if deviation <= THRESHOLDS["SMA_MINOR_DEVIATION"]:
        # Minor deviation (<15%): no penalty
        return 0

    if deviation <= THRESHOLDS["SMA_MODERATE_DEVIATION"]:
        # Moderate deviation (15-30%): linear penalty
        # 1 point per 5% over threshold
        excess = deviation - THRESHOLDS["SMA_MINOR_DEVIATION"]
        return int(excess / 5)

    # Extreme deviation (>30%): non-linear penalty
    z_proxy = deviation / 20.0  # Approximate z-score (20% ~ 1 std dev)
    penalty = ((z_proxy - 1.5) ** 1.5) * 8

    return min(int(penalty), THRESHOLDS["SMA_MAX_PENALTY"])


def calculate_fundamental_score(metrics: TickerMetrics) -> int:
    """
    Calculate bonus/penalty based on P/E valuation.

    Rules:
    - Loss-making (PE < 0): -15 points
    - Bubble (PE > 2x sector): -20 points
    - Overvalued (PE > 1.5x sector): -10 points
    - Fair (PE 0.8-1.5x sector): 0 points
    - Undervalued (PE < 0.8x sector): +10 points

    Args:
        metrics: TickerMetrics with pe_rel_valuation

    Returns:
        Score adjustment (-20 to +10)
    """
    # Handle loss-making companies first
    if metrics.is_loss_making:
        return THRESHOLDS["LOSS_MAKING_PENALTY"]

    # No adjustment if PE data unavailable
    if metrics.pe_rel_valuation is None:
        return 0

    rel_val = metrics.pe_rel_valuation

    if rel_val < THRESHOLDS["PE_UNDERVALUED"]:
        return THRESHOLDS["UNDERVALUED_BONUS"]
    elif rel_val >= THRESHOLDS["PE_BUBBLE"]:
        return THRESHOLDS["BUBBLE_PENALTY"]
    elif rel_val >= THRESHOLDS["PE_OVERVALUED"]:
        return THRESHOLDS["OVERVALUED_PENALTY"]
    else:
        return 0  # Fair valuation


def calculate_intelligence_score(
    llm_confidence: float,
    metrics: TickerMetrics
) -> Tuple[int, str]:
    """
    Calculate final intelligence score (0-100).

    Formula:
        base = llm_confidence * 100
        score = base - technical_penalty + fundamental_score

    With HARD CAP:
        If SMA deviation > 50%, max score = 50

    This ensures that even high-confidence LLM signals are capped
    when the stock is in bubble territory.

    Args:
        llm_confidence: LLM signal confidence (0.0-1.0)
        metrics: TickerMetrics with all available data

    Returns:
        Tuple of (intelligence_score, data_quality)
    """
    # Start with LLM confidence as base (0-100)
    base_score = llm_confidence * 100

    # Calculate adjustments
    tech_penalty = calculate_technical_penalty(metrics)
    fund_score = calculate_fundamental_score(metrics)

    # Apply adjustments
    final_score = base_score - tech_penalty + fund_score

    # Apply hard cap for extreme SMA deviation
    if metrics.is_bubble_territory:
        final_score = min(final_score, THRESHOLDS["BUBBLE_SCORE_CAP"])

    # Clamp to valid range
    final_score = max(0, min(100, int(final_score)))

    return final_score, metrics.data_quality


def get_valuation_rating(metrics: TickerMetrics) -> str:
    """
    Determine valuation rating based on metrics.

    Returns:
        UNDERVALUED, FAIR, OVERVALUED, BUBBLE, LOSS_MAKING, or UNKNOWN
    """
    if metrics.is_loss_making:
        return "LOSS_MAKING"

    if metrics.pe_rel_valuation is None:
        return "UNKNOWN"

    rel_val = metrics.pe_rel_valuation

    if rel_val < THRESHOLDS["PE_UNDERVALUED"]:
        return "UNDERVALUED"
    elif rel_val < THRESHOLDS["PE_OVERVALUED"]:
        return "FAIR"
    elif rel_val < THRESHOLDS["PE_BUBBLE"]:
        return "OVERVALUED"
    else:
        return "BUBBLE"


def enrich_signal_with_intelligence(
    signal_dict: dict,
    metrics: TickerMetrics,
    llm_confidence: Optional[float] = None
) -> dict:
    """
    Enrich a signal dictionary with intelligence scoring.

    Used in the pipeline between JSON parse and Pydantic validation.
    Adds intelligence_score and related fields to the signal.

    Args:
        signal_dict: Raw signal from LLM (dict)
        metrics: TickerMetrics for the signal's ticker
        llm_confidence: LLM confidence (from signal or default 0.8)

    Returns:
        Enriched signal dict with:
        - intelligence_score
        - sma_200_deviation
        - pe_rel_valuation
        - valuation_rating
        - data_quality
    """
    # Get confidence from signal or use default
    # NOTE: Default 0.5 (neutral) - non assumere alta confidence quando mancante
    confidence = llm_confidence
    if confidence is None:
        confidence = signal_dict.get("confidence", 0.5)
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = 0.5

    # Calculate intelligence score
    intelligence_score, data_quality = calculate_intelligence_score(
        confidence, metrics
    )

    # Get valuation rating
    valuation_rating = get_valuation_rating(metrics)

    # Enrich signal
    signal_dict["intelligence_score"] = intelligence_score
    signal_dict["sma_200_deviation"] = (
        round(metrics.sma_200_deviation_pct, 2)
        if metrics.sma_200_deviation_pct is not None
        else None
    )
    signal_dict["pe_rel_valuation"] = (
        round(metrics.pe_rel_valuation, 2)
        if metrics.pe_rel_valuation is not None
        else None
    )
    signal_dict["valuation_rating"] = valuation_rating
    signal_dict["data_quality"] = data_quality

    # Audit trail - traccia provenienza dati per trasparenza UI
    signal_dict["price_source"] = metrics.price_source
    signal_dict["sma_source"] = metrics.sma_source
    signal_dict["pe_source"] = metrics.pe_source
    signal_dict["sector_pe_source"] = metrics.sector_pe_source
    signal_dict["fetched_at"] = (
        metrics.fetched_at.isoformat() if metrics.fetched_at else None
    )
    signal_dict["days_of_history"] = metrics.days_of_history

    return signal_dict
