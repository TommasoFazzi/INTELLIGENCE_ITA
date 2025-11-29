# Database Migrations

This directory contains SQL migrations for the INTELLIGENCE_ITA database.

## How to Apply Migrations

### Using psql
```bash
psql -U your_user -d your_database -f migrations/001_add_content_hash.sql
```

### Using Python
```python
from src.storage.database import DatabaseManager

db = DatabaseManager()
with db.get_connection() as conn:
    with conn.cursor() as cur:
        with open('migrations/001_add_content_hash.sql', 'r') as f:
            cur.execute(f.read())
    conn.commit()
```

## Rollback

To rollback a migration:
```bash
psql -U your_user -d your_database -f migrations/001_add_content_hash_rollback.sql
```

## Available Migrations

### 001_add_content_hash.sql
**Purpose**: Add content-based deduplication (Phase 2)

**Changes**:
- Adds `content_hash` VARCHAR(32) column to `articles` table
- Creates index on `content_hash` for fast duplicate detection
- Creates index on `published_date` for time-windowed queries

**Rollback**: `001_add_content_hash_rollback.sql`

## Migration Best Practices

1. **Backup First**: Always backup your database before applying migrations
2. **Test on Staging**: Apply migrations to staging environment first
3. **Check Existing Data**: Some migrations may need to backfill data
4. **Monitor Performance**: Large tables may require careful index creation

## Verification

After applying migration 001:
```sql
-- Check that column was added
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'articles' AND column_name = 'content_hash';

-- Check indexes
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'articles'
AND indexname IN ('idx_articles_content_hash', 'idx_articles_published_date');

-- Check data
SELECT COUNT(*) as total,
       COUNT(content_hash) as with_hash,
       COUNT(*) - COUNT(content_hash) as without_hash
FROM articles;
```
