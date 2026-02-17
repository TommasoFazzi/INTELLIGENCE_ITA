# LLM Context

## Purpose
Large Language Model integration layer for intelligence report generation, RAG-based Q&A, and query analysis. Uses Google Gemini for text generation with structured output validation via Pydantic schemas. Reports are enriched with **narrative storyline context** from the Narrative Engine.

## Architecture Role
Intelligence synthesis layer that consumes context from the vector database and narrative graph, then produces human-readable reports. Sits between `src/storage/` (RAG retrieval), `src/nlp/narrative_processor.py` (storyline context), and `src/hitl/` (human review).

## Key Files

- `report_generator.py` - Daily/weekly report generation (~2700 lines)
  - `ReportGenerator` class - Main report generation engine
  - RAG pipeline: Query expansion → Semantic search → Reranking → LLM synthesis
  - **Narrative context** (`_get_narrative_context()`): Fetches top 10 storylines by momentum, their graph edges, and recent linked articles
  - **XML formatting** (`_format_narrative_xml()`): Formats storyline data as structured XML for the LLM prompt
  - Report structure includes 5 sections: Executive Summary, Key Developments, Trend Analysis, Investment Implications, **Strategic Storyline Tracker**
  - Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) for precision
  - Trade signal extraction with ticker whitelist
  - Macro-first pipeline (`--macro-first` flag)
  - Output metadata includes `narrative_context` (storylines used, edges count)

- `query_analyzer.py` - Pre-search filter extraction
  - `QueryAnalyzer` class - Extracts structured filters from natural language
  - Temporal constraints, entity filters, category filters
  - Uses Gemini Flash for low latency (<500ms)

- `oracle_engine.py` - Hybrid RAG chat engine
  - `OracleEngine` class - Interactive Q&A over intelligence database
  - Three search modes: `both`, `factual`, `strategic`
  - XML-like context injection for anti-hallucination

- `schemas.py` - Pydantic schemas for structured LLM output
  - `IntelligenceReportMVP`, `IntelligenceReport`, `TradeSignal`, `ImpactScore`
  - `MacroCondensedContext`, `MacroDashboardItem`, `ExtractedFilters`

## Dependencies

- **Internal**: `src/storage/database`, `src/nlp/processing`, `src/utils/logger`, `src/finance/`
- **External**:
  - `google-generativeai` - Gemini API (gemini-2.5-flash)
  - `pydantic` - Structured output validation
  - `sentence-transformers` - Embeddings and Cross-Encoder reranking
  - `numpy` - Vector operations

## Data Flow

- **Input**:
  - Recent articles from database (last 24h-7d)
  - RAG context from semantic search on chunks
  - **Narrative storyline context** from `v_active_storylines` and `v_storyline_graph`
  - Historical reports for Oracle context

- **Output**:
  - `reports/intelligence_report_{timestamp}.json` - Structured report
  - `reports/intelligence_report_{timestamp}.md` - Markdown report (now includes Storyline Tracker section)
  - `reports/WEEKLY_REPORT_{date}.md` - Weekly meta-analysis
  - Trade signals with intelligence scores
