# Test HITL — Context

## Purpose
Tests for the HITL (Human-In-The-Loop) dashboard module located in `src/hitl/` and driven by `Home.py` (Streamlit entry point). This directory validates the Streamlit utility functions that support the review, editing, rating, and feedback workflow used by analysts to inspect and approve generated intelligence reports before they enter production.

## Test Coverage

This directory currently contains only `__init__.py` — no test files have been written yet. The module is reserved for future tests of Streamlit utilities and HITL workflow logic. Based on the root `tests/context.md`, expected coverage includes:

- **Streamlit utility functions** (`src/hitl/` helpers): data formatting, state management helpers, rating submission, report editing utilities.
- **Report display logic:** Verifying that report fields are rendered in the expected format for analyst consumption.
- **Feedback loop helpers:** Functions that write analyst ratings and edits back to the database.
- **Session state management:** Streamlit `st.session_state` interactions that persist review progress across page reloads.

## Key Test Patterns

When implemented, tests in this directory are expected to follow these patterns:

- **Mock `streamlit`:** Since Streamlit components cannot run outside a Streamlit server context, all tests will need to mock `streamlit` (e.g. `st.session_state`, `st.write`, `st.button`) using `unittest.mock.patch` or a dedicated Streamlit testing library (`streamlit.testing.v1.AppTest` if available).
- **No `@pytest.mark.e2e`** — HITL utility tests should remain unit-level by mocking all I/O (DB calls, Streamlit rendering) and be marked `@pytest.mark.unit`.
- **DB interactions mocked:** Any helper that reads from or writes to PostgreSQL (e.g. saving analyst ratings) should be patched via `unittest.mock.patch` on `src.storage.database.DatabaseManager`.
- **Fixtures from `conftest.py`:** `sample_article` and `sample_articles_batch` provide realistic article data for rendering tests.

## Dependencies

- **Internal:** `src/hitl/` utility modules, `Home.py` Streamlit app, `src/storage/database.DatabaseManager`
- **External:**
  - `pytest` — test framework
  - `unittest.mock` — `patch`, `MagicMock` for Streamlit and DB mocking
  - `streamlit` — must be installed; components need mocking for headless testing
- **Fixtures from `conftest.py`:** `sample_article`, `sample_articles_batch`, `test_db_url`

## Known Gotchas

- **No test files exist yet:** The directory contains only `__init__.py`. All items above describe intended future coverage.
- **Streamlit is not headlessly testable by default:** Calling Streamlit functions outside a running server raises errors. Tests must either mock the entire `streamlit` module or use `streamlit.testing.v1.AppTest` (available since Streamlit 1.18). Choose the approach before writing tests to avoid inconsistent patterns across files.
- **`Home.py` is the Streamlit entry point:** It lives at the root of the inner `INTELLIGENCE_ITA/` directory, not inside `src/`. Tests targeting the full app must import from `Home` directly or use `AppTest.from_file("Home.py")`.
- **Session state side effects:** Streamlit `st.session_state` is a global singleton. Tests that manipulate session state must reset it between test cases to avoid state leaking across tests.
- **Database dependency for integration-level HITL tests:** Any test that exercises the full review-to-DB cycle requires a test PostgreSQL instance (`TEST_DATABASE_URL`) with the HITL-related tables (`reports`, `report_ratings`, etc.) already present.
