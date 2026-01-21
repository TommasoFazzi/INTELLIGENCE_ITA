# Test LLM Context

## Purpose
Unit tests for the `src/llm/` module. Validates query analysis, query expansion, and report generation logic.

## Architecture Role
Test layer ensuring LLM integration reliability. Tests focus on structured output validation, filter extraction accuracy, and query expansion quality.

## Key Files

- `test_query_analyzer.py` - Query filter extraction tests
  - Date extraction (relative: "ultimi 7 giorni", absolute: "15 dicembre")
  - GPE filter extraction ("Taiwan", "Russia")
  - Category filter extraction
  - Source filter extraction
  - Semantic query cleaning
  - Confidence score validation
  - Fallback behavior when extraction fails

- `test_query_expansion.py` - Query expansion tests
  - Multi-variant generation (2+ variants per focus area)
  - Semantic diversity of expanded queries
  - Deduplication of similar expansions
  - Focus area coverage (key entities, themes)

## Dependencies

- **Internal**: `src/llm/query_analyzer`, `src/llm/schemas`
- **External**:
  - `pytest` - Test framework
  - `pytest-mock` - Mocking Gemini API responses
  - `pydantic` - Schema validation

## Test Data

- Sample user queries in Italian/English
- Expected filter outputs
- Mock Gemini API responses

## Running

```bash
pytest tests/test_llm/ -v
```

## Notes

- Tests may require `GEMINI_API_KEY` for integration tests
- Unit tests use mocked API responses for isolation
