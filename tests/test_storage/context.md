# Test Storage Context

## Purpose
Unit tests for the `src/storage/` module. Validates database operations, connection pooling, and deduplication logic.

## Architecture Role
Test layer ensuring data persistence reliability. Tests cover connection management, CRUD operations, semantic search, and content-hash deduplication.

## Key Files

- `test_database_logic.py` - Core database operation tests
  - Connection pool management
  - Article save/retrieve operations
  - Batch insert handling
  - Semantic search accuracy
  - Report save/update lifecycle
  - Feedback storage
  - Entity coordinate retrieval (GeoJSON)

- `test_content_hash_dedup.py` - Content deduplication tests
  - MD5 hash generation
  - Duplicate detection on insert
  - Near-duplicate handling
  - Hash collision edge cases
  - Batch save with dedup

## Dependencies

- **Internal**: `src/storage/database`
- **External**:
  - `pytest` - Test framework
  - `psycopg2` - PostgreSQL adapter
  - Test database (separate from production)

## Test Data

- Sample articles with embeddings
- Duplicate article pairs
- Test embedding vectors (384-dim)

## Database Fixtures

Tests use pytest fixtures for:
- Test database connection
- Sample article data
- Cleanup after tests

## Running

```bash
# Requires test database configured
pytest tests/test_storage/ -v

# With coverage
pytest tests/test_storage/ --cov=src/storage
```

## Notes

- Tests may modify database state
- Use separate test database (`intelligence_ita_test`)
- Fixtures handle setup/teardown
