# Phase 4: LLM Report Generation with RAG

## Overview

Phase 4 implements intelligent report generation using Google Gemini LLM with Retrieval-Augmented Generation (RAG). The system combines:
- Fresh articles from the last 24 hours
- Historical context via semantic search
- LLM analysis for comprehensive intelligence briefings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Report Generator                          │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐    ┌─────────────┐  │
│  │  Database    │────▶│     RAG      │───▶│   Gemini    │  │
│  │  (Recent     │     │  (Semantic   │    │    LLM      │  │
│  │   Articles)  │     │   Search)    │    │             │  │
│  └──────────────┘     └──────────────┘    └─────────────┘  │
│         │                     │                    │        │
│         ▼                     ▼                    ▼        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Combined Prompt with Context                │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│                  ┌──────────────────┐                       │
│                  │  Intelligence    │                       │
│                  │     Report       │                       │
│                  └──────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. ReportGenerator Class

**Location:** [src/llm/report_generator.py](../src/llm/report_generator.py)

**Key Methods:**

#### `__init__(db_manager, nlp_processor, gemini_api_key, model_name)`
Initializes the report generator with:
- Database connection for article retrieval
- NLP processor for query embeddings
- Gemini API configuration (default: gemini-1.5-flash)

#### `get_rag_context(query, top_k=10, category=None)`
Performs semantic search for historical context:
1. Encodes query into 384-dim embedding
2. Searches database using pgvector cosine similarity
3. Returns top-k most relevant historical chunks

**Example:**
```python
context = generator.get_rag_context(
    query="cybersecurity threats in Asia",
    top_k=5
)
# Returns: List of {chunk_id, content, title, link, similarity, ...}
```

#### `generate_report(focus_areas, days=1, rag_queries=None, rag_top_k=5)`
Main report generation pipeline:
1. **Fetch recent articles** from last N days
2. **Retrieve RAG context** for each focus area
3. **Construct prompt** combining fresh + historical data
4. **Generate report** using Gemini LLM
5. **Return structured output** with metadata and sources

**Parameters:**
- `focus_areas`: List of topics (e.g., ["cybersecurity", "geopolitics"])
- `days`: How many days back to fetch articles (default: 1)
- `rag_queries`: Custom RAG search queries (auto-generated if None)
- `rag_top_k`: Number of historical chunks per query (default: 5)

**Returns:**
```python
{
    'success': True,
    'timestamp': '2025-11-23T10:30:00',
    'report_text': '# Executive Summary\n\n...',
    'metadata': {
        'focus_areas': [...],
        'recent_articles_count': 134,
        'historical_chunks_count': 15,
        'days_covered': 1,
        'model_used': 'gemini-1.5-flash'
    },
    'sources': {
        'recent_articles': [{title, link, source, published_date}, ...],
        'historical_context': [{title, link, similarity}, ...]
    }
}
```

#### `save_report(report, output_dir="reports")`
Saves report in two formats:
- **JSON** (complete structured data)
- **Markdown** (human-readable report)

Filenames: `intelligence_report_YYYYMMDD_HHMMSS.{json,md}`

#### `run_daily_report(focus_areas=None, save=True, output_dir="reports")`
One-command daily report generation:
1. Generates report with default/custom focus areas
2. Saves to file (if save=True)
3. Prints summary statistics

### 2. Report Generation Script

**Location:** [scripts/generate_report.py](../scripts/generate_report.py)

**Usage:**
```bash
# Basic usage (last 24 hours)
python scripts/generate_report.py

# Include last 3 days
python scripts/generate_report.py --days 3

# Don't save to file (print only)
python scripts/generate_report.py --no-save

# Use different model
python scripts/generate_report.py --model gemini-1.5-pro

# Custom output directory
python scripts/generate_report.py --output-dir /path/to/reports
```

**Command-line Arguments:**
- `--days N`: Number of days to look back (default: 1)
- `--no-save`: Don't save report to file
- `--output-dir DIR`: Output directory (default: reports/)
- `--model NAME`: Gemini model (default: gemini-1.5-flash)

## RAG Implementation Details

### How RAG Works

1. **Query Generation**
   - Focus areas define what historical context is relevant
   - Default focus: cybersecurity, geopolitics, tech/economy
   - Each focus area becomes a semantic search query

2. **Embedding Generation**
   - Query text → NLP processor → 384-dim vector
   - Uses same model as article chunking (paraphrase-multilingual-MiniLM-L12-v2)
   - Ensures semantic alignment between queries and stored chunks

3. **Vector Search**
   - Database query: `ORDER BY embedding <=> query_embedding`
   - Uses HNSW index for fast approximate nearest neighbor
   - Returns chunks with similarity scores (0-1 range)

4. **Context Formatting**
   - Combines top-k chunks from all focus areas
   - Removes duplicates (same chunk_id)
   - Formats with article metadata (title, source, date, link)

### Prompt Engineering

The LLM receives a structured prompt with:

**1. Task Definition**
```
You are an intelligence analyst generating a daily intelligence briefing.
```

**2. Focus Areas**
- Cybersecurity & Technology
- Geopolitical Events  
- Economic Trends

**3. Report Structure**
- Executive Summary (2-3 paragraphs)
- Key Developments by Category
- Trend Analysis (connections with historical context)
- Actionable Insights

**4. Guidelines**
- Concise but comprehensive
- Actionable intelligence over general news
- Cite articles with [Article N] references
- Connect current events with historical patterns
- Highlight emerging threats/opportunities
- Professional, analytical tone

**5. Context Data**
```
=== TODAY'S NEWS ARTICLES ===
[Article 1]
Title: ...
Source: ... | Date: ... | Category: ...
Summary: ...
Key entities: PERSON: ... | ORG: ... | GPE: ...
Full text: ...
Link: ...

=== RELEVANT HISTORICAL CONTEXT ===
[1] Previous Article Title
Source: ... | Date: ... | Category: ... | Similarity: 0.875
Relevant excerpt: ...
Link: ...
```

## Configuration

### Environment Variables

Add to `.env` file:

```bash
# Gemini API Key (required)
GEMINI_API_KEY=your_gemini_api_key_here

# Database connection (already configured in Phase 3)
DATABASE_URL=postgresql://user:password@localhost:5432/intelligence_ita

# Optional: Report settings
REPORT_OUTPUT_DIR=./reports
LOG_LEVEL=INFO
```

### Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy key to `.env` file

**Note:** Gemini 1.5 Flash has generous free tier:
- 15 requests per minute
- 1 million tokens per day
- Perfect for daily intelligence reports

## Usage Examples

### Basic Report Generation

```python
from src.llm.report_generator import ReportGenerator

# Initialize
generator = ReportGenerator()

# Generate report
report = generator.run_daily_report(
    focus_areas=[
        "cybersecurity threats and vulnerabilities",
        "geopolitical developments in Asia",
        "technology and economic trends"
    ],
    save=True
)

# Print report
print(report['report_text'])
```

### Custom RAG Queries

```python
# Generate report with specific RAG queries
report = generator.generate_report(
    focus_areas=["AI developments", "data privacy"],
    days=1,
    rag_queries=[
        "artificial intelligence regulations and policies",
        "data protection breaches and privacy laws",
        "machine learning security vulnerabilities"
    ],
    rag_top_k=10  # Get top 10 historical chunks per query
)
```

### Multi-Day Analysis

```python
# Analyze last 3 days of news
report = generator.generate_report(
    focus_areas=["cybersecurity", "geopolitics"],
    days=3
)

print(f"Analyzed {report['metadata']['recent_articles_count']} articles")
print(f"With {report['metadata']['historical_chunks_count']} historical chunks")
```

## Report Output Format

### JSON Structure

```json
{
  "success": true,
  "timestamp": "2025-11-23T10:30:00",
  "report_text": "# Executive Summary\n\nToday's intelligence highlights...",
  "metadata": {
    "focus_areas": ["cybersecurity", "geopolitics", "economy"],
    "recent_articles_count": 134,
    "historical_chunks_count": 15,
    "days_covered": 1,
    "model_used": "gemini-1.5-flash"
  },
  "sources": {
    "recent_articles": [
      {
        "title": "China's New AI Regulations...",
        "link": "https://...",
        "source": "Asia Times",
        "published_date": "2025-11-23T08:15:00"
      }
    ],
    "historical_context": [
      {
        "title": "Previous AI Policy Analysis",
        "link": "https://...",
        "similarity": 0.875
      }
    ]
  }
}
```

### Markdown Structure

```markdown
# Intelligence Report - 2025-11-23 10:30:00

## Executive Summary

[LLM-generated summary of critical developments]

## Key Developments by Category

### Cybersecurity & Technology
[Analysis of cyber threats, breaches, vulnerabilities]

### Geopolitical Events
[Analysis of conflicts, tensions, diplomatic developments]

### Economic Trends
[Analysis of policy changes, market movements, sanctions]

## Trend Analysis

[Connections between current events and historical patterns from RAG context]

## Actionable Insights

[What decision-makers should know and potential actions]

---

**Generated by:** gemini-1.5-flash
**Sources:** 134 recent articles, 15 historical chunks
```

## Performance Considerations

### Speed
- **Database queries**: ~50ms (with HNSW index)
- **Embedding generation**: ~100ms per query
- **LLM generation**: ~5-15 seconds (depends on Gemini load)
- **Total**: Typically 10-20 seconds per report

### Cost
- **Gemini 1.5 Flash**: Free tier (15 RPM, 1M tokens/day)
- **Embeddings**: Free (local Sentence Transformers)
- **Database**: Free (local PostgreSQL)

### Token Usage
- **Input tokens**: ~3,000-10,000 (depends on article count)
- **Output tokens**: ~1,000-3,000 (typical report length)
- **Daily usage**: ~10,000-40,000 tokens (well within free tier)

## Troubleshooting

### "GEMINI_API_KEY not found"
```bash
# Add to .env file
echo "GEMINI_API_KEY=your_key_here" >> .env
```

### "No recent articles found"
```bash
# Run ingestion pipeline first
python -m src.ingestion.pipeline

# Process with NLP
python scripts/process_nlp.py

# Load to database
python scripts/load_to_database.py
```

### "Database connection failed"
```bash
# Check PostgreSQL is running
psql -d intelligence_ita -c "SELECT COUNT(*) FROM articles;"

# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL
```

### Low-quality reports
- **Increase RAG context**: Set `rag_top_k=10` or higher
- **Expand time window**: Set `days=3` for more articles
- **Refine focus areas**: Be more specific in focus_areas
- **Use better model**: Try `model_name="gemini-1.5-pro"` (slower but higher quality)

## Next Steps

### Phase 5: Human-in-the-Loop (Planned)
- Web interface for reviewing reports
- Feedback mechanism (thumbs up/down on sections)
- Manual editing and annotation
- Report versioning and history

### Phase 6: Automation (Planned)
- Scheduled daily execution (cron/systemd timer)
- Email/Slack notifications
- Report archiving and indexing
- Performance monitoring and alerts

## API Reference

See [src/llm/report_generator.py](../src/llm/report_generator.py:23) for complete API documentation.

## Testing

```bash
# Test with sample data
python scripts/generate_report.py --no-save

# Test with different time windows
python scripts/generate_report.py --days 7

# Test with different models
python scripts/generate_report.py --model gemini-1.5-pro
```

## License

MIT License - see [LICENSE](../LICENSE) for details.
