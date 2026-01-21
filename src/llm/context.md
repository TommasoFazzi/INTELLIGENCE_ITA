# LLM Context

## Purpose
Large Language Model integration layer for intelligence report generation, RAG-based Q&A, and query analysis. Uses Google Gemini for text generation with structured output validation via Pydantic schemas.

## Architecture Role
Intelligence synthesis layer that consumes context from the vector database and produces human-readable reports. Sits between `src/storage/` (RAG retrieval) and `src/hitl/` (human review). Also provides conversational interface via "The Oracle" engine.

## Key Files

- `report_generator.py` - Daily/weekly report generation (~1000+ lines)
  - `ReportGenerator` class - Main report generation engine
  - RAG pipeline: Query expansion → Semantic search → Reranking → LLM synthesis
  - Query expansion with Gemini to generate multiple search variants
  - Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) for precision
  - Chunk deduplication by cosine similarity (default: 0.98 threshold)
  - Trade signal extraction with ticker whitelist from `config/top_50_tickers.yaml`
  - Macro-first pipeline (`--macro-first` flag): generates macro report first, then article signals
  - Output: Markdown reports saved to `reports/` directory

- `query_analyzer.py` - Pre-search filter extraction
  - `QueryAnalyzer` class - Extracts structured filters from natural language
  - Solves temporal constraint problem: "What happened on Dec 15th?" → `start_date`
  - Extracts: `gpe_filter`, `start_date`, `end_date`, `category_filter`, `source_filter`
  - Uses Gemini Flash for low latency (<500ms)
  - `ExtractedFilters` Pydantic schema for validation

- `oracle_engine.py` - Hybrid RAG chat engine
  - `OracleEngine` class - Interactive Q&A over intelligence database
  - Three search modes: `both` (default), `factual` (articles only), `strategic` (reports only)
  - Searches both chunks (articles) and reports (generated intelligence)
  - XML-like context injection for anti-hallucination
  - Freshness indicators for source dating
  - Integration with query analyzer for filter extraction

- `schemas.py` - Pydantic schemas for structured LLM output
  - `IntelligenceReportMVP` - Minimal schema (title, category, summary, sentiment, confidence)
  - `IntelligenceReport` - Full schema with nested models
  - `TradeSignal` - Trade recommendation (ticker, signal, timeframe, rationale)
  - `ImpactScore` - Impact assessment (score 0-10, reasoning)
  - `MacroCondensedContext` - Token-efficient macro context for signal extraction
  - `MacroDashboardItem` - Dashboard display schema
  - `ExtractedFilters` - Query analyzer output schema

## Dependencies

- **Internal**: `src/storage/database`, `src/nlp/processing`, `src/utils/logger`, `src/finance/`
- **External**:
  - `google-generativeai` - Gemini API (gemini-2.5-flash default)
  - `pydantic` - Structured output validation
  - `sentence-transformers` - Embeddings and Cross-Encoder reranking
  - `numpy` - Vector operations
  - `pyyaml` - Ticker config loading

## Data Flow

- **Input**:
  - Recent articles from database (last 24h-7d)
  - RAG context from semantic search on chunks
  - Historical reports for Oracle context
  - User queries for Q&A

- **Output**:
  - `reports/intelligence_report_{timestamp}.json` - Structured report
  - `reports/intelligence_report_{timestamp}.md` - Markdown report
  - `reports/WEEKLY_REPORT_{date}.md` - Weekly meta-analysis
  - Trade signals with intelligence scores
  - Oracle chat responses with cited sources
