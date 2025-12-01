# INTELLIGENCE_ITA - Project Summary

## Overview

Sistema completo di intelligence news analysis con pipeline end-to-end:
**RSS Ingestion → NLP Processing → Vector Database → LLM Report Generation → Human Review**

**Status**: ✅ Fasi 1-5 completate (pronto per produzione)

## System Architecture

```
                    INTELLIGENCE_ITA PIPELINE
                    
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  FASE 1: DATA INGESTION                                         │
│  ┌────────────┐      ┌──────────────┐      ┌───────────────┐  │
│  │  23 RSS    │─────▶│  Trafilatura │─────▶│  JSON Export  │  │
│  │  Feeds     │      │  + Newspaper │      │  (134 art.)   │  │
│  └────────────┘      └──────────────┘      └───────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FASE 2: NLP PROCESSING                                         │
│  ┌────────────┐      ┌──────────────┐      ┌───────────────┐  │
│  │  spaCy     │─────▶│  Chunking    │─────▶│  Embeddings   │  │
│  │  (NER)     │      │  (500w+50w)  │      │  (384-dim)    │  │
│  └────────────┘      └──────────────┘      └───────────────┘  │
│                                                   │             │
│                                           183 chunks            │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FASE 3: VECTOR DATABASE                                        │
│  ┌────────────┐      ┌──────────────┐      ┌───────────────┐  │
│  │ PostgreSQL │◀────│  Batch Load  │◀────│  Connection   │  │
│  │ + pgvector │      │  (execute_   │      │  Pooling      │  │
│  │            │      │   batch)     │      │               │  │
│  └──────┬─────┘      └──────────────┘      └───────────────┘  │
│         │                                                       │
│         │ HNSW Index for fast semantic search                  │
│         │                                                       │
├─────────┴───────────────────────────────────────────────────────┤
│                                                                  │
│  FASE 4: LLM REPORT GENERATION                                  │
│  ┌────────────┐      ┌──────────────┐      ┌───────────────┐  │
│  │  Recent    │      │     RAG      │      │    Gemini     │  │
│  │  Articles  │─────▶│  (Semantic   │─────▶│   1.5 Flash   │  │
│  │  (24h)     │      │   Search)    │      │               │  │
│  └────────────┘      └──────────────┘      └───────┬───────┘  │
│                                                     │           │
│                                           Intelligence Report   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FASE 5: HUMAN-IN-THE-LOOP                                      │
│  ┌────────────┐      ┌──────────────┐      ┌───────────────┐  │
│  │ Streamlit  │      │    Human     │      │   Feedback    │  │
│  │ Dashboard  │─────▶│   Review &   │─────▶│   Database    │  │
│  │            │      │    Edit      │      │               │  │
│  └────────────┘      └──────────────┘      └───────┬───────┘  │
│                                                     │           │
│                                           Approved Report       │
│                                                     │           │
│         ┌───────────────────────────────────────────┘           │
│         │                                                       │
│         └──────────▶ Feedback Loop (Prompt Improvement)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Language** | Python | 3.12 | Main programming language |
| **Web Scraping** | Trafilatura | 1.7.0 | Article text extraction |
| **Web Scraping** | Newspaper3k | 0.2.8 | Fallback extraction |
| **NLP** | spaCy | 3.7.2 | NER, tokenization |
| **Embeddings** | Sentence Transformers | 2.3.1 | Semantic vectors (384-dim) |
| **Database** | PostgreSQL | 14+ | Relational storage |
| **Vector Search** | pgvector | 0.2.4 | Semantic similarity search |
| **LLM** | Google Gemini | 2.5 Flash | Report generation |
| **UI** | Streamlit | 1.51.0 | HITL dashboard |
| **Orchestration** | Custom Python | - | Pipeline coordination |

## Key Metrics

### Data Pipeline
- **Sources**: 23 RSS feeds across 3 categories
- **Articles/day**: ~100-150 (filtered to 24h)
- **Extraction success**: 62.7% (84/134)
- **NLP success**: 100% (134/134)
- **Entities extracted**: 4,991 (PERSON, ORG, GPE, DATE)
- **Chunks created**: 183 (avg 1.4 per article)
- **Embedding dimension**: 384

### Performance
- **Ingestion**: ~2-3 minutes (23 feeds)
- **NLP Processing**: ~3-4 minutes (134 articles)
- **Database Load**: <1 second (batch insert)
- **Report Generation**: ~10-15 seconds (Gemini API)
- **Total automation**: ~6-7 minutes

### Database (esempio da test run)
- **Articles stored**: 134
- **Chunks stored**: 183
- **Vector index**: HNSW (O(log n) search)
- **Query latency**: ~50ms (semantic search)

### Costs
- **Gemini API**: $0 (free tier, <1M tokens/month)
- **NLP Models**: $0 (local inference)
- **Database**: $0 (PostgreSQL local)
- **Total**: Completamente gratuito

## Project Structure

```
INTELLIGENCE_ITA/
├── config/
│   └── feeds.yaml                     # 23 RSS feed configurations
│
├── src/
│   ├── ingestion/
│   │   ├── feed_parser.py            # RSS/Atom parsing
│   │   ├── content_extractor.py      # Full-text extraction
│   │   └── pipeline.py               # Orchestration
│   │
│   ├── nlp/
│   │   └── processing.py             # Hybrid NLP processor (577 lines)
│   │                                  # - Text cleaning
│   │                                  # - Chunking with overlap
│   │                                  # - NER (spaCy)
│   │                                  # - Embeddings (SentenceTransformer)
│   │
│   ├── storage/
│   │   └── database.py               # PostgreSQL + pgvector (700+ lines)
│   │                                  # - Connection pooling
│   │                                  # - Semantic search
│   │                                  # - Report storage
│   │                                  # - Feedback tracking
│   │
│   ├── llm/
│   │   └── report_generator.py       # Gemini integration (450 lines)
│   │                                  # - RAG context retrieval
│   │                                  # - Prompt engineering
│   │                                  # - Report generation
│   │
│   ├── hitl/
│   │   └── dashboard.py              # Streamlit UI (350 lines)
│   │                                  # - Report viewer/editor
│   │                                  # - Rating system
│   │                                  # - Feedback collection
│   │
│   └── utils/
│       └── logger.py                 # Logging utilities
│
├── scripts/
│   ├── process_nlp.py                # NLP processing CLI
│   ├── load_to_database.py           # Database loader CLI
│   ├── generate_report.py            # Report generator CLI
│   ├── run_dashboard.sh              # Dashboard launcher
│   └── check_setup.py                # System health check
│
├── docs/
│   ├── QUICKSTART.md                 # Getting started guide
│   ├── PHASE4_REPORT_GENERATION.md   # LLM + RAG docs
│   ├── PHASE5_HITL.md                # Dashboard docs
│   └── HITL_FEEDBACK_LOOP.md         # Improvement process
│
├── data/                             # Generated data (gitignored)
├── reports/                          # Generated reports (gitignored)
├── logs/                             # Application logs (gitignored)
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment template
└── README.md                         # Main documentation
```

## Features Completate

### ✅ Phase 1: Data Ingestion
- Multi-source RSS aggregation (23 feeds)
- Dual extraction (Trafilatura + Newspaper3k)
- Date filtering (only last 24h articles)
- Category classification
- JSON export with full metadata

### ✅ Phase 2: NLP Processing
- Text cleaning (remove boilerplate)
- Semantic chunking (500 words + 50 overlap)
- Named Entity Recognition (4 entity types)
- Multilingual embeddings (384-dim vectors)
- Batch processing with progress tracking

### ✅ Phase 3: Vector Database
- PostgreSQL schema with pgvector
- HNSW index for fast similarity search
- Connection pooling (1-10 connections)
- Batch inserts with execute_batch()
- Comprehensive statistics API

### ✅ Phase 4: LLM Reports
- Google Gemini integration (1.5 Flash)
- RAG context retrieval (top-k semantic search)
- Structured prompts (Executive Summary, Developments, Analysis, Insights)
- Multi-format export (JSON + Markdown)
- Source attribution

### ✅ Phase 5: HITL Dashboard
- Streamlit web interface
- Side-by-side draft/final comparison
- Interactive text editor
- Rating system (1-5 stars)
- Feedback database (corrections, additions, removals)
- Status workflow (Draft → Reviewed → Approved)
- Source links and RAG context viewer

## Usage Examples

### Daily Pipeline (Manual)

```bash
# Step-by-step execution
python -m src.ingestion.pipeline           # ~3 min
python scripts/process_nlp.py              # ~4 min
python scripts/load_to_database.py         # ~1 sec
python scripts/generate_report.py          # ~15 sec
./scripts/run_dashboard.sh                 # Opens browser
```

### Dashboard Workflow

```bash
# 1. Start dashboard
./scripts/run_dashboard.sh

# 2. In browser (http://localhost:8501)
#    - Click "➕ Genera Nuovo Report"
#    - Wait 10-20 seconds
#    - Review in "Bozza LLM" tab
#    - Edit in "Versione Finale" tab
#    - Rate 1-5 stars
#    - Add comments
#    - Click "✅ Approva"

# 3. Approved report saved to database
```

### Programmatic Usage

```python
from src.llm.report_generator import ReportGenerator

# Generate report
generator = ReportGenerator()
report = generator.run_daily_report(
    focus_areas=[
        "cybersecurity threats",
        "geopolitical developments",
        "economic trends"
    ],
    save=True,           # Save to reports/
    save_to_db=True      # Save to database for HITL
)

# Report available at:
# - reports/intelligence_report_YYYYMMDD_HHMMSS.md
# - Database with ID: report['report_id']
```

## Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/intelligence_ita

# LLM API
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
LOG_LEVEL=INFO
REPORT_OUTPUT_DIR=./reports
```

### Feed Configuration (config/feeds.yaml)

```yaml
breaking_news:
  - url: https://www.themoscowtimes.com/rss/news
    source: "The Moscow Times"
    subcategory: "breaking_news"

intelligence:
  - url: https://www.aseanbeat.com/feed/
    source: "ASEAN Beat"
    subcategory: "geopolitics_asia"
  # ... 21 more feeds
```

## Database Schema

### Core Tables
- **articles** (134 rows): Title, link, published_date, full_text, entities, embedding
- **chunks** (183 rows): Article chunks with embeddings for RAG
- **reports** (N rows): LLM-generated + human-edited reports
- **report_feedback** (N rows): Human corrections and ratings

### Key Indexes
- HNSW on embeddings (vector similarity)
- B-tree on published_date (time-range queries)
- B-tree on report_date (recent reports)

## API Reference

### ReportGenerator

```python
from src.llm.report_generator import ReportGenerator

generator = ReportGenerator(
    db_manager=None,           # Auto-creates if None
    nlp_processor=None,        # Auto-creates if None
    gemini_api_key=None,       # Reads from env if None
    model_name="gemini-2.5-flash"
)

# Generate with custom focus
report = generator.generate_report(
    focus_areas=["cybersecurity", "AI policy"],
    days=1,
    rag_top_k=10
)

# Get RAG context
context = generator.get_rag_context(
    query="AI regulations in Europe",
    top_k=5
)

# Full pipeline
report = generator.run_daily_report(
    focus_areas=None,      # Uses defaults
    save=True,             # Save to files
    save_to_db=True        # Save to DB for HITL
)
```

### DatabaseManager

```python
from src.storage.database import DatabaseManager

db = DatabaseManager()  # Auto-connects via .env

# Init schema
db.init_db()

# Semantic search
results = db.semantic_search(
    query_embedding=[0.1, 0.2, ...],  # 384-dim vector
    top_k=10,
    category="intelligence"  # Optional filter
)

# Get recent articles
articles = db.get_recent_articles(days=1, category=None)

# Report management
report_id = db.save_report(report_dict)
report = db.get_report(report_id)
db.update_report(report_id, final_content, status='approved')

# Feedback
feedback_id = db.save_feedback(
    report_id, section_name, feedback_type,
    original_text, corrected_text, comment, rating
)
feedback_list = db.get_report_feedback(report_id)

# Stats
stats = db.get_statistics()
```

### NLPProcessor

```python
from src.nlp.processing import NLPProcessor

nlp = NLPProcessor(
    spacy_model="en_core_web_sm",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size=500,
    chunk_overlap=50
)

# Process single article
result = nlp.process_article(article_dict)

# Batch process
results = nlp.batch_process(articles_list, max_workers=4)
```

## Key Design Decisions

### 1. Hybrid NLP Approach

**Decision**: Combine text cleaning + chunking + NER + embeddings

**Why**: 
- Text cleaning removes scraping noise
- Chunking preserves context for RAG
- NER for structured metadata
- Embeddings for semantic search

**Alternative rejected**: Pure embedding approach (lost structured metadata)

### 2. 24-Hour Article Window

**Decision**: Only ingest articles from last 24h (`max_age_days=1`)

**Why**:
- Daily execution → only need recent news
- Reduced processing time (587 → 134 articles)
- More focused reports

**Alternative rejected**: All available RSS articles (too many old/irrelevant)

### 3. Gemini 1.5 Flash over GPT-4

**Decision**: Default to Gemini 2.5 Flash

**Why**:
- Generous free tier (15 req/min, 1M tokens/day)
- Fast response (~5-10 sec)
- Good quality for structured tasks
- No credit card required
- Improved over 1.5 Flash

**Alternative**: GPT-4 (better quality but paid, slower)

### 4. Streamlit over Custom React

**Decision**: Use Streamlit for HITL dashboard

**Why**:
- Pure Python (no JS/HTML/CSS)
- Rapid development (1 day vs 1 week)
- Built-in state management
- Instant UI updates

**Alternative rejected**: React + Flask (overkill for internal tool)

### 5. PostgreSQL + pgvector over Vector DBs

**Decision**: Use PostgreSQL with pgvector extension

**Why**:
- Single database for relational + vector data
- No need for separate vector DB (Pinecone, Weaviate)
- ACID transactions
- Familiar SQL queries

**Alternative rejected**: Dedicated vector DB (added complexity)

## Performance Optimization

### Implemented
1. **Connection pooling** - Reuse DB connections (1-10 pool)
2. **Batch inserts** - execute_batch() for chunks (10x faster)
3. **HNSW indexing** - Approximate nearest neighbor (O(log n))
4. **Parallel processing** - Multi-threaded NLP (4 workers)
5. **Caching** - Session state in Streamlit

### Potential Future
1. **Redis cache** - Cache embeddings and RAG results
2. **Async pipeline** - Use asyncio for I/O operations
3. **Model quantization** - Smaller embedding models
4. **Incremental updates** - Only process new articles

## Security Considerations

### Current
- ✅ Environment variables for secrets (.env)
- ✅ SQL injection protection (parameterized queries)
- ✅ HTTPS for RSS feeds
- ✅ Input validation (article metadata)

### TODO for Production
- [ ] Database SSL/TLS connections
- [ ] Streamlit authentication (streamlit-authenticator)
- [ ] API rate limiting
- [ ] Audit logging for report approvals
- [ ] Backup automation

## Known Limitations

1. **Language**: Currently English-only (can extend to Italian/multilingual)
2. **Scale**: Optimized for ~100-200 articles/day (can handle more)
3. **Real-time**: Pipeline runs in batches (can add streaming)
4. **LLM Quality**: Dependent on Gemini API (can fine-tune)
5. **Single-user**: Dashboard not multi-user ready (can add auth)

## Future Roadmap

### Phase 6: Automation (Next)
- Cron job for daily execution
- Email distribution of approved reports
- Slack/Discord notifications
- Error alerting and monitoring

### Phase 7: Advanced Analytics
- Trend analysis over time
- Entity relationship graphs
- Topic modeling on historical reports
- Anomaly detection (breaking news)

### Phase 8: Multi-Language
- Italian report generation
- Multilingual article processing
- Auto-translation for sources

### Phase 9: API & Integration
- REST API for programmatic access
- Webhooks for real-time alerts
- Integration with Slack/Teams
- Export to various formats (PDF, DOCX)

## Success Criteria

### Functional ✅
- [x] Ingest 100+ articles daily
- [x] Extract full text (>60% success rate)
- [x] Generate embeddings for all articles
- [x] Store in vector database
- [x] Generate daily report with LLM
- [x] Human review interface
- [x] Feedback collection

### Quality ✅
- [x] Report generation < 20 seconds
- [x] Semantic search < 100ms
- [x] NLP processing < 5 minutes
- [x] No data loss (all transactions ACID)

### Usability ✅
- [x] One-command pipeline execution
- [x] Web-based dashboard (no CLI needed)
- [x] Clear documentation
- [x] Error handling and logging

## Contributors

- **Tommaso Fazzi** - Project lead & development

## License

MIT License

---

**Last Updated**: 2025-11-25
**Version**: 1.0.0
**Status**: Production Ready
