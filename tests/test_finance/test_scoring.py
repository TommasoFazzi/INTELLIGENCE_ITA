"""
Tests for Financial Intelligence v2 scoring module.

Tests cover:
- Technical penalty calculations (SMA200 deviation)
- Fundamental score calculations (PE valuation)
- Intelligence score calculations with hard caps
- Signal enrichment
"""

import pytest
from src.finance.types import TickerMetrics
from src.finance.scoring import (
    calculate_technical_penalty,
    calculate_fundamental_score,
    calculate_intelligence_score,
    get_valuation_rating,
    enrich_signal_with_intelligence,
)
from src.finance.constants import THRESHOLDS


class TestTechnicalPenalty:
    """Tests for SMA200 deviation penalty calculation."""

    def test_no_penalty_for_minor_deviation(self):
        """Deviation <15% should have no penalty."""
        metrics = TickerMetrics(
            ticker="LMT",
            price=500.0,
            sma_200=480.0,
            sma_200_deviation_pct=4.17  # (500-480)/480 * 100
        )
        assert calculate_technical_penalty(metrics) == 0

    def test_no_penalty_for_negative_deviation(self):
        """Price below SMA200 should have no penalty (we're looking for overextension)."""
        metrics = TickerMetrics(
            ticker="BA",
            price=150.0,
            sma_200=180.0,
            sma_200_deviation_pct=-16.67
        )
        assert calculate_technical_penalty(metrics) == 0

    def test_linear_penalty_for_moderate_deviation(self):
        """Deviation 15-30% should have linear penalty."""
        metrics = TickerMetrics(
            ticker="RTX",
            price=120.0,
            sma_200=100.0,
            sma_200_deviation_pct=20.0  # 5% over threshold
        )
        # (20 - 15) / 5 = 1 point
        assert calculate_technical_penalty(metrics) == 1

    def test_nonlinear_penalty_for_extreme_deviation(self):
        """Deviation >30% should have non-linear penalty."""
        metrics = TickerMetrics(
            ticker="NVDA",
            price=800.0,
            sma_200=500.0,
            sma_200_deviation_pct=60.0
        )
        # z_proxy = 60/20 = 3.0
        # penalty = (3.0 - 1.5)^1.5 * 8 = 1.5^1.5 * 8 = 1.837 * 8 = ~14.7
        penalty = calculate_technical_penalty(metrics)
        assert 10 < penalty < 20  # Reasonable range

    def test_max_penalty_cap(self):
        """Penalty should be capped at MAX_PENALTY."""
        metrics = TickerMetrics(
            ticker="MEME",
            price=200.0,
            sma_200=50.0,
            sma_200_deviation_pct=300.0  # Extreme meme stock
        )
        penalty = calculate_technical_penalty(metrics)
        assert penalty == THRESHOLDS["SMA_MAX_PENALTY"]  # 40

    def test_no_penalty_when_data_missing(self):
        """No penalty if SMA200 deviation is None."""
        metrics = TickerMetrics(
            ticker="NEW_IPO",
            price=50.0,
            sma_200=None,
            sma_200_deviation_pct=None,
            data_quality="INSUFFICIENT"
        )
        assert calculate_technical_penalty(metrics) == 0


class TestFundamentalScore:
    """Tests for PE valuation scoring."""

    def test_undervalued_bonus(self):
        """PE < 0.8x sector should get bonus."""
        metrics = TickerMetrics(
            ticker="VALUE",
            price=100.0,
            pe_ratio=10.0,
            pe_sector_median=15.0,
            pe_rel_valuation=0.67  # 10/15 = 0.67
        )
        assert calculate_fundamental_score(metrics) == THRESHOLDS["UNDERVALUED_BONUS"]  # +10

    def test_fair_valuation_no_adjustment(self):
        """PE 0.8-1.5x sector should have no adjustment."""
        metrics = TickerMetrics(
            ticker="FAIR",
            price=100.0,
            pe_ratio=18.0,
            pe_sector_median=16.0,
            pe_rel_valuation=1.125
        )
        assert calculate_fundamental_score(metrics) == 0

    def test_overvalued_penalty(self):
        """PE 1.5-2x sector should get moderate penalty."""
        metrics = TickerMetrics(
            ticker="GROWTH",
            price=100.0,
            pe_ratio=28.0,
            pe_sector_median=16.0,
            pe_rel_valuation=1.75
        )
        assert calculate_fundamental_score(metrics) == THRESHOLDS["OVERVALUED_PENALTY"]  # -10

    def test_bubble_penalty(self):
        """PE > 2x sector should get severe penalty."""
        metrics = TickerMetrics(
            ticker="HYPE",
            price=100.0,
            pe_ratio=50.0,
            pe_sector_median=16.0,
            pe_rel_valuation=3.125
        )
        assert calculate_fundamental_score(metrics) == THRESHOLDS["BUBBLE_PENALTY"]  # -20

    def test_loss_making_penalty(self):
        """Negative PE should get loss-making penalty."""
        metrics = TickerMetrics(
            ticker="RIVN",
            price=15.0,
            pe_ratio=-25.0,
            pe_sector_median=16.0,
            pe_rel_valuation=None  # Can't calculate for negative PE
        )
        assert calculate_fundamental_score(metrics) == THRESHOLDS["LOSS_MAKING_PENALTY"]  # -15

    def test_no_adjustment_when_data_missing(self):
        """No adjustment if PE data unavailable."""
        metrics = TickerMetrics(
            ticker="ETF",
            price=100.0,
            pe_ratio=None,
            pe_sector_median=None,
            pe_rel_valuation=None
        )
        assert calculate_fundamental_score(metrics) == 0


class TestIntelligenceScore:
    """Tests for combined intelligence score calculation."""

    def test_full_score_for_undervalued_near_sma(self):
        """High confidence + undervalued + near SMA = high score."""
        metrics = TickerMetrics(
            ticker="BARGAIN",
            price=100.0,
            sma_200=98.0,
            sma_200_deviation_pct=2.0,
            pe_ratio=10.0,
            pe_sector_median=15.0,
            pe_rel_valuation=0.67,
            data_quality="FULL"
        )
        score, quality = calculate_intelligence_score(0.9, metrics)
        # 90 base + 10 undervalued bonus - 0 technical = 100 (capped)
        assert score == 100
        assert quality == "FULL"

    def test_reduced_score_for_extended_stock(self):
        """Good signal but extended stock = reduced score."""
        metrics = TickerMetrics(
            ticker="HOT",
            price=130.0,
            sma_200=100.0,
            sma_200_deviation_pct=30.0,
            pe_ratio=20.0,
            pe_sector_median=18.0,
            pe_rel_valuation=1.11,
            data_quality="FULL"
        )
        score, _ = calculate_intelligence_score(0.85, metrics)
        # 85 base - moderate penalty - 0 fundamental = ~75-80
        assert 70 < score < 85

    def test_hard_cap_for_bubble_territory(self):
        """Score capped at 50 when >50% above SMA200."""
        metrics = TickerMetrics(
            ticker="BUBBLE",
            price=180.0,
            sma_200=100.0,
            sma_200_deviation_pct=80.0,  # Way over 50%
            pe_ratio=40.0,
            pe_sector_median=16.0,
            pe_rel_valuation=2.5,
            data_quality="FULL"
        )
        score, _ = calculate_intelligence_score(1.0, metrics)  # Max confidence
        assert score <= THRESHOLDS["BUBBLE_SCORE_CAP"]  # 50

    def test_minimum_score_zero(self):
        """Score should never go below 0."""
        metrics = TickerMetrics(
            ticker="TERRIBLE",
            price=200.0,
            sma_200=100.0,
            sma_200_deviation_pct=100.0,
            pe_ratio=-10.0,  # Loss making
            data_quality="FULL"
        )
        score, _ = calculate_intelligence_score(0.3, metrics)  # Low confidence
        assert score >= 0


class TestValuationRating:
    """Tests for valuation rating determination."""

    def test_undervalued_rating(self):
        metrics = TickerMetrics(ticker="A", price=100, pe_rel_valuation=0.7)
        assert get_valuation_rating(metrics) == "UNDERVALUED"

    def test_fair_rating(self):
        metrics = TickerMetrics(ticker="B", price=100, pe_rel_valuation=1.2)
        assert get_valuation_rating(metrics) == "FAIR"

    def test_overvalued_rating(self):
        metrics = TickerMetrics(ticker="C", price=100, pe_rel_valuation=1.7)
        assert get_valuation_rating(metrics) == "OVERVALUED"

    def test_bubble_rating(self):
        metrics = TickerMetrics(ticker="D", price=100, pe_rel_valuation=2.5)
        assert get_valuation_rating(metrics) == "BUBBLE"

    def test_loss_making_rating(self):
        metrics = TickerMetrics(ticker="E", price=100, pe_ratio=-10.0)
        assert get_valuation_rating(metrics) == "LOSS_MAKING"

    def test_unknown_rating(self):
        metrics = TickerMetrics(ticker="F", price=100, pe_rel_valuation=None)
        assert get_valuation_rating(metrics) == "UNKNOWN"


class TestSignalEnrichment:
    """Tests for signal enrichment function."""

    def test_enriches_signal_with_all_fields(self):
        """Signal dict should have all intelligence fields after enrichment."""
        metrics = TickerMetrics(
            ticker="LMT",
            price=500.0,
            sma_200=480.0,
            sma_200_deviation_pct=4.17,
            pe_ratio=18.0,
            pe_sector_median=16.0,
            pe_rel_valuation=1.125,
            data_quality="FULL"
        )
        signal = {
            "ticker": "LMT",
            "signal": "BULLISH",
            "timeframe": "MEDIUM_TERM",
            "rationale": "Defense spending increase",
            "confidence": 0.85
        }

        enriched = enrich_signal_with_intelligence(signal, metrics)

        assert "intelligence_score" in enriched
        assert "sma_200_deviation" in enriched
        assert "pe_rel_valuation" in enriched
        assert "valuation_rating" in enriched
        assert "data_quality" in enriched
        assert enriched["intelligence_score"] > 0
        assert enriched["valuation_rating"] == "FAIR"

    def test_handles_missing_confidence(self):
        """Should use default confidence if not in signal."""
        metrics = TickerMetrics(ticker="TEST", price=100.0, data_quality="PARTIAL")
        signal = {"ticker": "TEST", "signal": "BULLISH"}

        enriched = enrich_signal_with_intelligence(signal, metrics)

        assert "intelligence_score" in enriched
        # Default confidence 0.8 * 100 = 80 base score
        assert enriched["intelligence_score"] <= 80

    def test_rounds_numeric_values(self):
        """Numeric values should be rounded for cleaner output."""
        metrics = TickerMetrics(
            ticker="TEST",
            price=100.0,
            sma_200_deviation_pct=12.3456789,
            pe_rel_valuation=1.23456789,
            data_quality="FULL"
        )
        signal = {"ticker": "TEST", "confidence": 0.85}

        enriched = enrich_signal_with_intelligence(signal, metrics)

        assert enriched["sma_200_deviation"] == 12.35  # Rounded to 2 decimal
        assert enriched["pe_rel_valuation"] == 1.23


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_region_detection_us(self):
        """US tickers should be detected correctly."""
        from src.finance.constants import get_region
        assert get_region("LMT") == "US"
        assert get_region("AAPL") == "US"

    def test_region_detection_eu(self):
        """EU tickers should be detected correctly."""
        from src.finance.constants import get_region
        assert get_region("RHM.DE") == "EU"
        assert get_region("BA.L") == "EU"
        assert get_region("AIR.PA") == "EU"

    def test_region_detection_asia(self):
        """Asian tickers should be detected correctly."""
        from src.finance.constants import get_region
        assert get_region("005930.KS") == "ASIA"
        assert get_region("9984.T") == "ASIA"

    def test_ticker_metrics_properties(self):
        """TickerMetrics properties should work correctly."""
        # Loss making
        loss_metrics = TickerMetrics(ticker="A", price=10, pe_ratio=-5)
        assert loss_metrics.is_loss_making is True

        # Profitable
        profit_metrics = TickerMetrics(ticker="B", price=10, pe_ratio=15)
        assert profit_metrics.is_loss_making is False

        # Bubble territory
        bubble_metrics = TickerMetrics(ticker="C", price=10, sma_200_deviation_pct=60)
        assert bubble_metrics.is_bubble_territory is True

        # Normal
        normal_metrics = TickerMetrics(ticker="D", price=10, sma_200_deviation_pct=10)
        assert normal_metrics.is_bubble_territory is False
