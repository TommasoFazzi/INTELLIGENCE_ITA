# Test Finance — Context

## Purpose
Unit tests for the `src/finance/` scoring module. Validates the Financial Intelligence v2 scoring pipeline that enriches LLM trade signals with quantitative market metrics. Tests cover the penalty and bonus calculation functions, the combined intelligence score with hard caps, valuation rating classification, signal enrichment, and ticker region detection.

## Test Coverage

### `test_scoring.py`
All tests are class-based (no `@pytest.mark` decorators) and import directly from `src.finance.scoring`, `src.finance.types.TickerMetrics`, and `src.finance.constants.THRESHOLDS`.

**`TestTechnicalPenalty`** — Tests for `calculate_technical_penalty(metrics: TickerMetrics) -> int`

Validates the SMA200 deviation penalty using a 3-tier formula:
- `test_no_penalty_for_minor_deviation` — deviation < 15% returns 0
- `test_no_penalty_for_negative_deviation` — price below SMA200 (negative deviation) returns 0 (only overextension is penalized)
- `test_linear_penalty_for_moderate_deviation` — 15–30% deviation: linear 1pt per 5% over threshold (e.g. 20% deviation → 1pt)
- `test_nonlinear_penalty_for_extreme_deviation` — deviation > 30%: non-linear formula `((deviation/20 - 1.5)^1.5) * 8`; 60% deviation → 10–20pt range
- `test_max_penalty_cap` — extreme deviation (300%) is capped at `THRESHOLDS["SMA_MAX_PENALTY"]` (40)
- `test_no_penalty_when_data_missing` — `sma_200_deviation_pct=None` returns 0

**`TestFundamentalScore`** — Tests for `calculate_fundamental_score(metrics: TickerMetrics) -> int`

Validates P/E valuation scoring against sector median:
- `test_undervalued_bonus` — PE < 0.8x sector → `+THRESHOLDS["UNDERVALUED_BONUS"]` (+10)
- `test_fair_valuation_no_adjustment` — PE 0.8–1.5x sector → 0
- `test_overvalued_penalty` — PE 1.5–2x sector → `THRESHOLDS["OVERVALUED_PENALTY"]` (-10)
- `test_bubble_penalty` — PE > 2x sector → `THRESHOLDS["BUBBLE_PENALTY"]` (-20)
- `test_loss_making_penalty` — negative PE → `THRESHOLDS["LOSS_MAKING_PENALTY"]` (-15)
- `test_no_adjustment_when_data_missing` — `pe_ratio=None`, `pe_rel_valuation=None` → 0

**`TestIntelligenceScore`** — Tests for `calculate_intelligence_score(confidence: float, metrics: TickerMetrics) -> Tuple[int, str]`

Validates the combined score: `base_score = confidence * 100`, adjusted by fundamental bonus and technical penalty, then capped:
- `test_full_score_for_undervalued_near_sma` — confidence 0.9 + undervalued bonus + no technical penalty → score capped at 100
- `test_reduced_score_for_extended_stock` — confidence 0.85, 30% SMA deviation → score in 70–85 range
- `test_hard_cap_for_bubble_territory` — SMA deviation > 50% triggers hard cap at `THRESHOLDS["BUBBLE_SCORE_CAP"]` (50), even at max confidence
- `test_minimum_score_zero` — heavy penalties on low-confidence signal still produce score ≥ 0

**`TestValuationRating`** — Tests for `get_valuation_rating(metrics: TickerMetrics) -> str`

Validates rating string returned from `pe_rel_valuation`:
- `test_undervalued_rating` — `pe_rel_valuation=0.7` → `"UNDERVALUED"`
- `test_fair_rating` — `pe_rel_valuation=1.2` → `"FAIR"`
- `test_overvalued_rating` — `pe_rel_valuation=1.7` → `"OVERVALUED"`
- `test_bubble_rating` — `pe_rel_valuation=2.5` → `"BUBBLE"`
- `test_loss_making_rating` — `pe_ratio=-10.0` → `"LOSS_MAKING"`
- `test_unknown_rating` — `pe_rel_valuation=None` → `"UNKNOWN"`

**`TestSignalEnrichment`** — Tests for `enrich_signal_with_intelligence(signal: dict, metrics: TickerMetrics) -> dict`

Validates that a raw LLM trade signal dict is enriched with quantitative fields:
- `test_enriches_signal_with_all_fields` — output has `intelligence_score`, `sma_200_deviation`, `pe_rel_valuation`, `valuation_rating`, `data_quality`; `intelligence_score > 0`; `valuation_rating == "FAIR"` for in-range PE
- `test_handles_missing_confidence` — signal without `confidence` key uses default (0.8); resulting score ≤ 80
- `test_rounds_numeric_values` — `sma_200_deviation` rounded to 2 decimal places; `pe_rel_valuation` rounded to 2 decimal places

**`TestEdgeCases`** — Tests for region detection (`src.finance.constants.get_region`) and `TickerMetrics` properties:
- `test_region_detection_us` — bare tickers like `"LMT"`, `"AAPL"` → `"US"`
- `test_region_detection_eu` — `.DE`, `.L`, `.PA` suffix tickers → `"EU"`
- `test_region_detection_asia` — `.KS`, `.T` suffix tickers → `"ASIA"`
- `test_ticker_metrics_properties` — `TickerMetrics.is_loss_making` (True when `pe_ratio < 0`); `TickerMetrics.is_bubble_territory` (True when `sma_200_deviation_pct > 50`)

## Key Test Patterns

- **Class-based test organization:** All test classes group related assertions by function under test (`TestTechnicalPenalty`, `TestFundamentalScore`, etc.), with no `@pytest.mark` decorators applied at the function level.
- **`TickerMetrics` as test data builder:** Every test constructs a `TickerMetrics` dataclass directly with only the fields relevant to the function under test; other fields use their defaults. This makes each test self-documenting about which inputs drive the assertion.
- **Threshold constants imported from source:** Assertions reference `THRESHOLDS["SMA_MAX_PENALTY"]` rather than hardcoding `40`, so tests stay in sync if threshold values change.
- **No mocking required:** All tested functions are pure (no I/O, no DB calls, no LLM calls), so tests run without any `patch()` or `Mock` usage.
- **Boundary value testing:** Tests are written at exact threshold boundaries (e.g. 15%, 30%, 50% SMA deviation; 0.8x, 1.5x, 2x PE ratio) to verify branching logic.

## Dependencies

- **Internal:**
  - `src.finance.scoring` — `calculate_technical_penalty`, `calculate_fundamental_score`, `calculate_intelligence_score`, `get_valuation_rating`, `enrich_signal_with_intelligence`
  - `src.finance.types.TickerMetrics` — Pydantic/dataclass model for market metrics
  - `src.finance.constants` — `THRESHOLDS` dict, `get_region` function
- **External:**
  - `pytest` — test framework
- **No fixtures from `conftest.py` are used** — all test data is constructed inline.
- **No network, database, or LLM calls** — fully isolated, deterministic unit tests.

## Known Gotchas

- **`calculate_technical_penalty` uses `abs()`:** The function penalizes only positive overextension relative to SMA200; negative deviations (price below SMA) are converted to the absolute value internally, but the formula starts at 0 for values ≤ `SMA_MINOR_DEVIATION`. The test `test_no_penalty_for_negative_deviation` documents this intentional asymmetry.
- **Non-linear penalty formula precision:** The test for extreme deviation (`test_nonlinear_penalty_for_extreme_deviation`) uses a range assertion (`10 < penalty < 20`) rather than an exact value, because floating-point exponentiation and `int()` truncation mean small input variations produce different integer results. Do not tighten this to an exact equality.
- **Hard cap at bubble territory:** `test_hard_cap_for_bubble_territory` uses `confidence=1.0` (100 base) but expects `score <= 50`. This cap only triggers when `sma_200_deviation_pct > 50`; the cap is applied after all other adjustments.
- **`TickerMetrics` default fields:** Many fields default to `None`. Passing only the relevant fields in tests is intentional; importing `TickerMetrics` will fail if `src.finance.types` is not importable (i.e. if `pydantic` or the `finance` package is not installed).
- **No `@pytest.mark.unit` applied:** Unlike `test_nlp/`, these tests do not use pytest markers. Run them with `pytest tests/test_finance/ -v` or they will be included in the default test run.
