# ORACLE 2.0 - Piano di Sviluppo Completo

**Obiettivo:** Trasformare l'Oracle esistente in un sistema avanzato di analisi intelligence che possa gestire qualsiasi tipo di query attraverso tool calling, query routing, e reasoning multi-step.

---

## 📊 ARCHITETTURA ATTUALE (Baseline)

### Componenti
```
OracleEngine (oracle_engine.py) - 567 lines
├── Hybrid Search (vector + BM25 + keyword)
├── QueryAnalyzer (estrazione filtri NL → structured)
├── 3 modalità: Hybrid/Investigative/Strategic
├── Context formatting (XML delimiters)
└── Gemini LLM (gemini-2.5-flash)

QueryAnalyzer (query_analyzer.py) - 306 lines
├── Date extraction (relative + absolute)
├── Category inference
├── GPE normalization
├── Semantic query optimization
└── Confidence scoring

Streamlit UI (2_The_Oracle.py) - 423 lines
├── Chat interface con history
├── Source citations
├── Freshness indicators
└── Filter UI (date, category, geography)
```

### Capacità Attuali
- ✅ Ricerca semantica + keyword hybrid
- ✅ Estrazione filtri da linguaggio naturale
- ✅ Context-aware LLM responses
- ✅ Source citation con freshness
- ✅ Modalità ibrid/factual/strategic

### Limiti Identificati
1. **No conversational memory** → Ogni query è standalone, non ricorda il contesto
2. **No tool calling** → Non può eseguire SQL, calcoli, aggregazioni dinamiche
3. **No query routing** → Tratta tutte le query allo stesso modo (always RAG)
4. **No multi-document reasoning** → Non confronta o sintetizza tra storylines/reports
5. **No deep-dive su relazioni** → Non esplora entity_mentions, storyline_edges, trade_signals
6. **Integrazione parziale con dati strutturati** → Usa solo articles/reports, ignora macro_indicators, market_data

---

## 🎯 ORACLE 2.0 - Nuova Architettura

### Principi di Design
1. **Modularità**: Tool registry con plugin system
2. **Intelligenza di routing**: Query classifier che decide quale strategia usare
3. **Multi-step reasoning**: Chain-of-thought per query complesse
4. **Memoria conversazionale**: Context buffer per follow-up
5. **Osservabilità**: Logging dettagliato di ogni step decisionale

---

## 🏗️ COMPONENTI DA IMPLEMENTARE

### **1. Query Router** (CORE)
**File:** `src/llm/query_router.py`

**Responsabilità:**
- Classifica l'intent della query (5 categorie)
- Valuta la complessità (Simple/Medium/Complex)
- Genera un piano di esecuzione (query plan) con tools da chiamare
- Decide se serve decomposition in sub-queries

**Intent Categories:**
```python
class QueryIntent(Enum):
    FACTUAL = "factual"           # "Cosa è successo a Taiwan il 15 febbraio?"
    ANALYTICAL = "analytical"     # "Mostrami il trend degli attacchi cyber negli ultimi 3 mesi"
    NARRATIVE = "narrative"       # "Qual è lo stato della storyline Russia-Ucraina?"
    MARKET = "market"             # "Quali sono i trade signals bullish oggi?"
    COMPARATIVE = "comparative"   # "Confronta la postura militare della Cina vs. 6 mesi fa"
```

**Complexity Levels:**
```python
class QueryComplexity(Enum):
    SIMPLE = "simple"       # Single RAG call, 1 tool
    MEDIUM = "medium"       # Multi-tool, 2-3 steps
    COMPLEX = "complex"     # Decomposition, chain-of-thought, 4+ steps
```

**Output: QueryPlan**
```python
@dataclass
class QueryPlan:
    intent: QueryIntent
    complexity: QueryComplexity
    tools: List[str]                    # ['rag_tool', 'sql_aggregation', 'graph_tool']
    execution_steps: List[ExecutionStep]
    estimated_time: float               # in seconds
    requires_decomposition: bool
    sub_queries: Optional[List[str]] = None
```

**Implementazione:**
```python
class QueryRouter:
    def __init__(self, llm: GenerativeModel):
        self.llm = llm
        self.tool_registry = ToolRegistry()
    
    def route(self, query: str, context: ConversationContext = None) -> QueryPlan:
        """
        Classifica query e genera piano esecuzione.
        
        Flow:
        1. Intent classification (Gemini + few-shot examples)
        2. Complexity assessment (query length, keywords, entities count)
        3. Tool selection (based on intent + complexity)
        4. Decomposition (se necessario)
        5. QueryPlan assembly
        """
        # Step 1: LLM classification
        classification_prompt = self._build_classification_prompt(query, context)
        classification = self._classify_with_llm(classification_prompt)
        
        # Step 2: Complexity heuristic
        complexity = self._assess_complexity(query, classification)
        
        # Step 3: Tool mapping
        tools = self._select_tools(classification, complexity)
        
        # Step 4: Build execution plan
        steps = self._build_execution_steps(classification, tools, complexity)
        
        # Step 5: Decomposition (if COMPLEX)
        sub_queries = None
        if complexity == QueryComplexity.COMPLEX:
            sub_queries = self._decompose_query(query, classification)
        
        return QueryPlan(
            intent=classification.intent,
            complexity=complexity,
            tools=[t.name for t in tools],
            execution_steps=steps,
            estimated_time=self._estimate_time(steps),
            requires_decomposition=(sub_queries is not None),
            sub_queries=sub_queries
        )
```

---

### **2. Tool Registry** (CORE)
**File:** `src/llm/tools/registry.py`

**Responsabilità:**
- Catalogo di tutti i tools disponibili
- Lazy loading dei tools (init solo quando necessari)
- Dependency injection per DB/LLM
- Validation input/output

**Base Tool Interface:**
```python
# src/llm/tools/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Standardized tool output."""
    success: bool
    data: Any
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    execution_time: float = 0.0

class BaseTool(ABC):
    """Base class for all Oracle tools."""
    
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for validation
    
    def __init__(self, db: DatabaseManager, llm: Optional[GenerativeModel] = None):
        self.db = db
        self.llm = llm
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute tool logic. Must be implemented by subclasses."""
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input against parameters schema."""
        # Pydantic validation logic
        pass
    
    def format_for_llm(self, result: ToolResult) -> str:
        """Format result for LLM consumption (markdown/XML)."""
        if not result.success:
            return f"❌ Tool failed: {result.error}"
        return self._format_success(result.data, result.metadata)
    
    @abstractmethod
    def _format_success(self, data: Any, metadata: Dict) -> str:
        """Subclass-specific formatting."""
        pass
```

**Tool Registry:**
```python
class ToolRegistry:
    """Central registry of all Oracle tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False
    
    def register(self, tool_class: type[BaseTool], **init_kwargs):
        """Register a tool (lazy init)."""
        tool_name = tool_class.name
        self._tools[tool_name] = (tool_class, init_kwargs)
    
    def get_tool(self, name: str) -> BaseTool:
        """Get initialized tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not registered")
        
        tool_class, init_kwargs = self._tools[name]
        return tool_class(**init_kwargs)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools with metadata (for LLM tool selection)."""
        return [
            {
                "name": tool_class.name,
                "description": tool_class.description,
                "parameters": tool_class.parameters
            }
            for tool_class, _ in self._tools.values()
        ]
```

---

### **3. Concrete Tools** (IMPLEMENTATION)

#### **3.1 RAGTool** (refactor esistente)
**File:** `src/llm/tools/rag_tool.py`

```python
class RAGTool(BaseTool):
    """
    Semantic + keyword search su articles/reports.
    Refactored dall'OracleEngine esistente.
    """
    
    name = "rag_search"
    description = "Search articles and reports using semantic vector search + BM25"
    parameters = {
        "query": {"type": "string", "required": True},
        "mode": {"type": "string", "enum": ["both", "factual", "strategic"]},
        "top_k": {"type": "integer", "min": 1, "max": 50},
        "filters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "categories": {"type": "array", "items": {"type": "string"}},
                "gpe_filter": {"type": "array", "items": {"type": "string"}},
                "sources": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
    
    def execute(self, query: str, mode: str = "both", top_k: int = 10, filters: Dict = None) -> ToolResult:
        """Execute hybrid search."""
        start_time = time.time()
        try:
            # Use existing OracleEngine logic
            embedding = self._get_embedding(query)
            
            chunks = []
            reports = []
            
            if mode in ("both", "factual"):
                chunks = self.db.hybrid_search(
                    query=query,
                    query_embedding=embedding,
                    top_k=top_k,
                    **(filters or {})
                )
            
            if mode in ("both", "strategic"):
                reports = self.db.semantic_search_reports(
                    query_embedding=embedding,
                    top_k=top_k,
                    **(filters or {})
                )
            
            return ToolResult(
                success=True,
                data={"chunks": chunks, "reports": reports},
                metadata={
                    "chunks_count": len(chunks),
                    "reports_count": len(reports),
                    "mode": mode
                },
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), execution_time=time.time() - start_time)
    
    def _format_success(self, data: Dict, metadata: Dict) -> str:
        """Format RAG results as XML context."""
        # Usa format_context_for_llm esistente
        return self._format_context_xml(data['reports'], data['chunks'])
```

---

#### **3.2 SQLTool** (NEW)
**File:** `src/llm/tools/sql_tool.py`

**Capacità:**
- Esegue query SQL dinamiche per aggregazioni
- Safe: whitelist di tabelle + prepared statements
- Read-only mode (no INSERT/UPDATE/DELETE)

```python
class SQLTool(BaseTool):
    """
    Execute read-only SQL queries for structured data analysis.
    
    Use cases:
    - "Quanti articoli su Taiwan negli ultimi 30 giorni?"
    - "Quali sono le top 5 fonti per categoria DEFENSE?"
    - "Media di mention_count per entity_type GPE?"
    """
    
    name = "sql_query"
    description = "Execute read-only SQL queries on intelligence database"
    parameters = {
        "query": {"type": "string", "required": True},
        "safety_check": {"type": "boolean", "default": True}
    }
    
    # Whitelist di tabelle accessibili
    ALLOWED_TABLES = [
        "articles", "chunks", "reports", "storylines", "entities",
        "entity_mentions", "trade_signals", "macro_indicators",
        "market_data", "article_storylines", "storyline_edges"
    ]
    
    # Blacklist di keywords pericolose
    FORBIDDEN_KEYWORDS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "GRANT", "REVOKE"
    ]
    
    def execute(self, query: str, safety_check: bool = True) -> ToolResult:
        """Execute SQL query with safety validation."""
        start_time = time.time()
        
        # Safety checks
        if safety_check:
            if not self._is_safe_query(query):
                return ToolResult(
                    success=False,
                    error="Query contains forbidden keywords or accesses unauthorized tables",
                    execution_time=time.time() - start_time
                )
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    
                    # Convert to list of dicts
                    results = [dict(zip(columns, row)) for row in rows]
                    
                    return ToolResult(
                        success=True,
                        data={"results": results, "columns": columns},
                        metadata={"row_count": len(results), "query": query},
                        execution_time=time.time() - start_time
                    )
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return ToolResult(success=False, error=str(e), execution_time=time.time() - start_time)
    
    def _is_safe_query(self, query: str) -> bool:
        """Validate query safety."""
        query_upper = query.upper()
        
        # Check forbidden keywords
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in query_upper:
                return False
        
        # Check table whitelist (extract FROM clauses)
        import re
        from_clauses = re.findall(r'FROM\s+(\w+)', query_upper)
        for table in from_clauses:
            if table.lower() not in self.ALLOWED_TABLES:
                return False
        
        return True
    
    def _format_success(self, data: Dict, metadata: Dict) -> str:
        """Format SQL results as markdown table."""
        results = data['results']
        columns = data['columns']
        
        if not results:
            return f"Query returned 0 results.\n\nQuery: `{metadata['query']}`"
        
        # Build markdown table
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = []
        for row in results[:20]:  # Limit to 20 rows for LLM context
            row_str = "| " + " | ".join(str(row[col]) for col in columns) + " |"
            rows.append(row_str)
        
        table = "\n".join([header, separator] + rows)
        
        if len(results) > 20:
            table += f"\n\n... and {len(results) - 20} more rows (total: {len(results)})"
        
        return f"**SQL Query Results** ({metadata['row_count']} rows):\n\n{table}\n\nQuery: `{metadata['query']}`"
```

---

#### **3.3 AggregationTool** (NEW)
**File:** `src/llm/tools/aggregation_tool.py`

**Capacità:**
- Trend temporali (count per giorno/settimana/mese)
- Top-N rankings (top fonti, top entities, top categories)
- Statistiche descrittive (media, mediana, percentili)

```python
class AggregationTool(BaseTool):
    """
    High-level aggregations and statistical analysis.
    
    Use cases:
    - "Trend degli articoli su cybersecurity negli ultimi 3 mesi"
    - "Top 10 entità GPE più menzionate"
    - "Distribuzione degli articoli per categoria"
    """
    
    name = "aggregation"
    description = "Compute aggregations, trends, and statistics"
    parameters = {
        "aggregation_type": {
            "type": "string",
            "enum": ["trend_over_time", "top_n", "distribution", "statistics"],
            "required": True
        },
        "target": {
            "type": "string",
            "description": "What to aggregate (e.g., 'articles', 'entities', 'trade_signals')"
        },
        "filters": {"type": "object"},
        "time_bucket": {
            "type": "string",
            "enum": ["day", "week", "month"],
            "default": "day"
        },
        "limit": {"type": "integer", "default": 10}
    }
    
    def execute(
        self,
        aggregation_type: str,
        target: str,
        filters: Dict = None,
        time_bucket: str = "day",
        limit: int = 10
    ) -> ToolResult:
        """Execute aggregation query."""
        start_time = time.time()
        
        try:
            if aggregation_type == "trend_over_time":
                data = self._compute_trend(target, filters, time_bucket)
            elif aggregation_type == "top_n":
                data = self._compute_top_n(target, filters, limit)
            elif aggregation_type == "distribution":
                data = self._compute_distribution(target, filters)
            elif aggregation_type == "statistics":
                data = self._compute_statistics(target, filters)
            else:
                raise ValueError(f"Unknown aggregation type: {aggregation_type}")
            
            return ToolResult(
                success=True,
                data=data,
                metadata={
                    "aggregation_type": aggregation_type,
                    "target": target,
                    "record_count": len(data.get('records', []))
                },
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), execution_time=time.time() - start_time)
    
    def _compute_trend(self, target: str, filters: Dict, time_bucket: str) -> Dict:
        """Compute time series trend."""
        # SQL query con GROUP BY date trunc
        query = self._build_trend_query(target, filters, time_bucket)
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                
                return {
                    "records": [{"date": row[0], "count": row[1]} for row in rows],
                    "time_bucket": time_bucket
                }
    
    def _compute_top_n(self, target: str, filters: Dict, limit: int) -> Dict:
        """Compute top N ranking."""
        # Example: top entities by mention_count
        if target == "entities":
            query = """
                SELECT e.name, e.entity_type, COUNT(em.id) as mentions
                FROM entities e
                JOIN entity_mentions em ON e.id = em.entity_id
                WHERE 1=1
            """
            # Add filters dynamically
            # ...GROUP BY e.name, e.entity_type ORDER BY mentions DESC LIMIT %s
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (limit,))
                    rows = cur.fetchall()
                    
                    return {
                        "records": [
                            {"name": row[0], "type": row[1], "count": row[2]}
                            for row in rows
                        ]
                    }
        # ... altri target types
    
    def _format_success(self, data: Dict, metadata: Dict) -> str:
        """Format aggregation as chart data or table."""
        agg_type = metadata['aggregation_type']
        
        if agg_type == "trend_over_time":
            # Format as time series
            records = data['records']
            lines = [f"- {r['date']}: {r['count']}" for r in records]
            return f"**Trend over time** ({data['time_bucket']} buckets):\n\n" + "\n".join(lines)
        
        elif agg_type == "top_n":
            # Format as ranking
            records = data['records']
            lines = [f"{i+1}. {r['name']} ({r.get('type', 'N/A')}): {r['count']}" for i, r in enumerate(records)]
            return f"**Top {len(records)} ranking**:\n\n" + "\n".join(lines)
        
        # ... altri format
```

---

#### **3.4 GraphTool** (NEW)
**File:** `src/llm/tools/graph_tool.py`

**Capacità:**
- Naviga storyline_edges (connected storylines)
- Trova percorsi tra entities via entity_mentions
- Analizza cluster di storylines correlate

```python
class GraphTool(BaseTool):
    """
    Navigate narrative graph (storylines + entities).
    
    Use cases:
    - "Quali storylines sono collegate alla Russia-Ucraina?"
    - "Trova il percorso tra Cina e semiconduttori"
    - "Cluster di storylines nel dominio DEFENSE"
    """
    
    name = "graph_navigation"
    description = "Navigate storyline and entity relationship graphs"
    parameters = {
        "operation": {
            "type": "string",
            "enum": ["connected_storylines", "entity_path", "storyline_cluster"],
            "required": True
        },
        "source": {"type": "string"},
        "target": {"type": "string"},
        "max_depth": {"type": "integer", "default": 3}
    }
    
    def execute(self, operation: str, source: str = None, target: str = None, max_depth: int = 3) -> ToolResult:
        """Execute graph query."""
        start_time = time.time()
        
        try:
            if operation == "connected_storylines":
                data = self._get_connected_storylines(source, max_depth)
            elif operation == "entity_path":
                data = self._find_entity_path(source, target, max_depth)
            elif operation == "storyline_cluster":
                data = self._get_storyline_cluster(source)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return ToolResult(
                success=True,
                data=data,
                metadata={"operation": operation, "node_count": len(data.get('nodes', []))},
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), execution_time=time.time() - start_time)
    
    def _get_connected_storylines(self, storyline_id: str, max_depth: int) -> Dict:
        """Find all storylines connected to source within max_depth."""
        query = """
            WITH RECURSIVE storyline_graph AS (
                -- Base case: source storyline
                SELECT 
                    source_id, target_id, edge_type, weight, 1 as depth
                FROM storyline_edges
                WHERE source_id = %s
                
                UNION ALL
                
                -- Recursive case: expand graph
                SELECT 
                    se.source_id, se.target_id, se.edge_type, se.weight, sg.depth + 1
                FROM storyline_edges se
                JOIN storyline_graph sg ON se.source_id = sg.target_id
                WHERE sg.depth < %s
            )
            SELECT DISTINCT
                s.id, s.title, s.narrative_status, s.momentum_score,
                sg.edge_type, sg.weight, sg.depth
            FROM storyline_graph sg
            JOIN storylines s ON s.id = sg.target_id
            ORDER BY sg.depth, sg.weight DESC
        """
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (storyline_id, max_depth))
                rows = cur.fetchall()
                
                return {
                    "nodes": [
                        {
                            "id": row[0],
                            "title": row[1],
                            "status": row[2],
                            "momentum": row[3],
                            "edge_type": row[4],
                            "weight": row[5],
                            "depth": row[6]
                        }
                        for row in rows
                    ]
                }
    
    def _format_success(self, data: Dict, metadata: Dict) -> str:
        """Format graph as hierarchical list."""
        nodes = data['nodes']
        
        # Group by depth
        by_depth = {}
        for node in nodes:
            depth = node['depth']
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(node)
        
        lines = [f"**Connected Storylines** ({len(nodes)} nodes found):"]
        for depth in sorted(by_depth.keys()):
            lines.append(f"\n**Depth {depth}:**")
            for node in by_depth[depth]:
                lines.append(
                    f"  - [{node['id']}] {node['title']} "
                    f"(status: {node['status']}, momentum: {node['momentum']:.2f}, "
                    f"edge: {node['edge_type']}, weight: {node['weight']:.2f})"
                )
        
        return "\n".join(lines)
```

---

#### **3.5 MarketTool** (NEW)
**File:** `src/llm/tools/market_tool.py`

**Capacità:**
- Analizza trade_signals con filtri avanzati
- Correla macro_indicators con eventi geopolitici
- Calcola momentum e valuation context automaticamente

```python
class MarketTool(BaseTool):
    """
    Analyze trade signals and macro indicators.
    
    Use cases:
    - "Mostrami i trade signals bullish oggi per il settore defense"
    - "Come si correla il VIX con gli articoli su crisi geopolitiche?"
    - "Quali ticker hanno PE undervalued + high intelligence score?"
    """
    
    name = "market_analysis"
    description = "Analyze trade signals, macro indicators, and market correlations"
    parameters = {
        "analysis_type": {
            "type": "string",
            "enum": ["signals_filter", "macro_correlation", "valuation_screen"],
            "required": True
        },
        "filters": {"type": "object"},
        "timeframe": {"type": "string", "default": "SHORT_TERM"}
    }
    
    def execute(self, analysis_type: str, filters: Dict = None, timeframe: str = "SHORT_TERM") -> ToolResult:
        """Execute market analysis."""
        start_time = time.time()
        
        try:
            if analysis_type == "signals_filter":
                data = self._filter_trade_signals(filters, timeframe)
            elif analysis_type == "macro_correlation":
                data = self._correlate_macro_events(filters)
            elif analysis_type == "valuation_screen":
                data = self._screen_valuation(filters)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            return ToolResult(
                success=True,
                data=data,
                metadata={"analysis_type": analysis_type, "signal_count": len(data.get('signals', []))},
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), execution_time=time.time() - start_time)
    
    def _filter_trade_signals(self, filters: Dict, timeframe: str) -> Dict:
        """Filter trade signals with advanced criteria."""
        query = """
            SELECT 
                ts.ticker, ts.signal, ts.timeframe, ts.confidence,
                ts.intelligence_score, ts.valuation_rating,
                ts.sma_200_deviation, ts.pe_rel_valuation,
                STRING_AGG(DISTINCT ts.rationale, ' | ') as rationales,
                COUNT(*) as signal_count
            FROM trade_signals ts
            WHERE ts.timeframe = %s
        """
        
        params = [timeframe]
        
        # Add dynamic filters
        if filters:
            if 'signal' in filters:
                query += " AND ts.signal = %s"
                params.append(filters['signal'])
            if 'min_confidence' in filters:
                query += " AND ts.confidence >= %s"
                params.append(filters['min_confidence'])
            if 'min_intelligence_score' in filters:
                query += " AND ts.intelligence_score >= %s"
                params.append(filters['min_intelligence_score'])
            if 'category' in filters:
                query += " AND ts.category = %s"
                params.append(filters['category'])
        
        query += " GROUP BY ts.ticker, ts.signal, ts.timeframe, ts.confidence, ts.intelligence_score, ts.valuation_rating, ts.sma_200_deviation, ts.pe_rel_valuation"
        query += " ORDER BY ts.intelligence_score DESC, ts.confidence DESC"
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                
                return {
                    "signals": [
                        {
                            "ticker": row[0],
                            "signal": row[1],
                            "timeframe": row[2],
                            "confidence": float(row[3]) if row[3] else None,
                            "intelligence_score": row[4],
                            "valuation_rating": row[5],
                            "sma_200_deviation": float(row[6]) if row[6] else None,
                            "pe_rel_valuation": float(row[7]) if row[7] else None,
                            "rationales": row[8],
                            "signal_count": row[9]
                        }
                        for row in rows
                    ]
                }
    
    def _format_success(self, data: Dict, metadata: Dict) -> str:
        """Format market signals as detailed table."""
        signals = data['signals']
        
        if not signals:
            return f"No trade signals found matching filters."
        
        lines = [f"**Trade Signals Analysis** ({len(signals)} tickers):"]
        for sig in signals:
            lines.append(
                f"\n**{sig['ticker']}** - {sig['signal']} ({sig['timeframe']})\n"
                f"  - Confidence: {sig['confidence']:.0%}\n"
                f"  - Intelligence Score: {sig['intelligence_score']}/100\n"
                f"  - Valuation: {sig['valuation_rating']} (PE rel: {sig['pe_rel_valuation']:.1f}%, SMA200: {sig['sma_200_deviation']:.1f}%)\n"
                f"  - Rationales ({sig['signal_count']} signals): {sig['rationales'][:200]}..."
            )
        
        return "\n".join(lines)
```

---

### **4. Conversation Memory** (NEW)
**File:** `src/llm/conversation_memory.py`

**Capacità:**
- Buffer degli ultimi N messaggi
- Compression summaries per long-term context
- Entity tracking cross-query
- Follow-up detection

```python
from collections import deque
from typing import List, Dict, Optional

class ConversationContext:
    """Track conversation state across multiple queries."""
    
    def __init__(self, max_buffer_size: int = 10, summary_threshold: int = 20):
        self.messages: deque = deque(maxlen=max_buffer_size)
        self.compressed_history: List[str] = []  # Summarized old messages
        self.entity_tracker: Dict[str, int] = {}  # Track mentioned entities
        self.last_query_plan: Optional[QueryPlan] = None
        self.message_count = 0
        self.summary_threshold = summary_threshold
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to buffer."""
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        self.message_count += 1
        
        # Trigger compression if threshold reached
        if self.message_count % self.summary_threshold == 0:
            self._compress_history()
    
    def _compress_history(self):
        """Compress old messages into summaries (LLM-based)."""
        # Extract oldest N messages, summarize with LLM, store summary
        pass
    
    def get_context_for_llm(self) -> str:
        """Format context for LLM prompt."""
        context_parts = []
        
        # Compressed history (if any)
        if self.compressed_history:
            context_parts.append("**Previous Conversation Summary:**\n" + "\n".join(self.compressed_history))
        
        # Recent messages
        context_parts.append("\n**Recent Messages:**")
        for msg in self.messages:
            context_parts.append(f"[{msg['role'].upper()}]: {msg['content']}")
        
        # Entity tracking
        if self.entity_tracker:
            top_entities = sorted(self.entity_tracker.items(), key=lambda x: x[1], reverse=True)[:5]
            context_parts.append(f"\n**Tracked Entities:** {', '.join([e[0] for e in top_entities])}")
        
        return "\n".join(context_parts)
    
    def detect_follow_up(self, query: str) -> bool:
        """Detect if query is a follow-up (uses pronouns, refers to previous context)."""
        follow_up_indicators = [
            "anche", "inoltre", "e", "rispetto a", "confronta con", "invece",
            "stesso", "quella", "questo", "questi", "approfondisci"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in follow_up_indicators)
```

---

### **5. Orchestrator** (CORE LOGIC)
**File:** `src/llm/oracle_orchestrator.py`

**Responsabilità:**
- Entry point per tutte le query
- Coordina Router → Tools → LLM → Response
- Gestisce multi-step execution
- Implementa Chain-of-Thought per query complesse

```python
class OracleOrchestrator:
    """
    Main orchestrator for Oracle 2.0.
    Coordinates query routing, tool execution, and LLM synthesis.
    """
    
    def __init__(self, db: DatabaseManager, llm: GenerativeModel):
        self.db = db
        self.llm = llm
        self.router = QueryRouter(llm)
        self.tool_registry = ToolRegistry()
        self.conversation_memory = ConversationContext()
        
        # Register all tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        self.tool_registry.register(RAGTool, db=self.db)
        self.tool_registry.register(SQLTool, db=self.db)
        self.tool_registry.register(AggregationTool, db=self.db)
        self.tool_registry.register(GraphTool, db=self.db)
        self.tool_registry.register(MarketTool, db=self.db)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for query processing.
        
        Flow:
        1. Check conversation context (follow-up detection)
        2. Route query → QueryPlan
        3. Execute tools sequentially or in parallel
        4. Synthesize results with LLM
        5. Update conversation memory
        6. Return response
        """
        start_time = time.time()
        
        # Step 1: Context check
        is_follow_up = self.conversation_memory.detect_follow_up(query)
        context = self.conversation_memory if is_follow_up else None
        
        # Step 2: Routing
        logger.info(f"Routing query: {query}")
        query_plan = self.router.route(query, context)
        logger.info(f"Query plan: intent={query_plan.intent}, complexity={query_plan.complexity}, tools={query_plan.tools}")
        
        # Step 3: Tool execution
        tool_results = self._execute_tools(query_plan)
        
        # Step 4: LLM synthesis
        response_text = self._synthesize_response(query, query_plan, tool_results, context)
        
        # Step 5: Update memory
        self.conversation_memory.add_message("user", query)
        self.conversation_memory.add_message("assistant", response_text, metadata={
            "query_plan": query_plan.model_dump(),
            "tool_results_count": len(tool_results)
        })
        self.conversation_memory.last_query_plan = query_plan
        
        # Step 6: Response assembly
        return {
            "answer": response_text,
            "query_plan": query_plan.model_dump(),
            "tool_results": [r.model_dump() for r in tool_results],
            "metadata": {
                "query": query,
                "is_follow_up": is_follow_up,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _execute_tools(self, query_plan: QueryPlan) -> List[ToolResult]:
        """Execute tools according to query plan."""
        results = []
        
        for step in query_plan.execution_steps:
            tool_name = step.tool_name
            tool = self.tool_registry.get_tool(tool_name)
            
            logger.info(f"Executing tool: {tool_name} with params: {step.parameters}")
            result = tool.execute(**step.parameters)
            results.append(result)
            
            # Early exit if critical tool fails
            if not result.success and step.is_critical:
                logger.error(f"Critical tool {tool_name} failed: {result.error}")
                break
        
        return results
    
    def _synthesize_response(
        self,
        query: str,
        query_plan: QueryPlan,
        tool_results: List[ToolResult],
        context: Optional[ConversationContext]
    ) -> str:
        """Synthesize final response using LLM."""
        
        # Build synthesis prompt
        prompt_parts = []
        
        # Conversation context (if follow-up)
        if context:
            prompt_parts.append(context.get_context_for_llm())
        
        # Original query
        prompt_parts.append(f"\n**USER QUERY:** {query}")
        
        # Tool results (formatted for LLM)
        prompt_parts.append("\n**TOOL RESULTS:**")
        for i, result in enumerate(tool_results, 1):
            tool_name = query_plan.execution_steps[i-1].tool_name
            tool = self.tool_registry.get_tool(tool_name)
            formatted = tool.format_for_llm(result)
            prompt_parts.append(f"\n[TOOL {i}: {tool_name}]\n{formatted}\n")
        
        # Instructions
        prompt_parts.append("""
\n**TASK:** Synthesize a comprehensive answer to the user's query based on the tool results above.

**REQUIREMENTS:**
1. Be analytical and precise
2. Cite tool results explicitly: [Tool 1: rag_search], [Tool 2: aggregation]
3. If tool results are empty or failed, acknowledge limitations
4. Provide actionable insights, not just data regurgitation
5. Format: Executive Summary + Detailed Analysis + Implications
6. Length: 3-5 paragraphs minimum

**ANSWER:**
""")
        
        full_prompt = "\n".join(prompt_parts)
        
        # LLM call
        response = self.llm.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=4096
            )
        )
        
        return response.text
```

---

## 📅 ROADMAP DI IMPLEMENTAZIONE

### **FASE 1: Foundation (Week 1-2)** ✅
**Obiettivo:** Setup infrastruttura base + tool registry

- [ ] Creare `src/llm/tools/` directory structure
- [ ] Implementare `BaseTool` abstract class con `ToolResult`
- [ ] Implementare `ToolRegistry` con lazy loading
- [ ] Refactor `OracleEngine` → `RAGTool` (mantenere backward compatibility)
- [ ] Test unitari per tool system

**Deliverables:**
- `src/llm/tools/base.py`
- `src/llm/tools/registry.py`
- `src/llm/tools/rag_tool.py` (refactored)
- Test: `tests/test_llm/test_tool_registry.py`

---

### **FASE 2: Query Router (Week 3-4)**
**Obiettivo:** Intelligent query classification + execution planning

- [ ] Implementare `QueryRouter` con intent classification
- [ ] Few-shot examples per ogni intent category
- [ ] Complexity assessment heuristics
- [ ] Tool-to-intent mapping logic
- [ ] Query decomposition per COMPLEX queries
- [ ] `QueryPlan` serialization/deserialization

**Deliverables:**
- `src/llm/query_router.py`
- `src/llm/schemas.py` (QueryPlan, QueryIntent, ExecutionStep)
- Test: `tests/test_llm/test_query_router.py`

---

### **FASE 3: Core Tools (Week 5-8)**
**Obiettivo:** Implementare 5 tools principali

**Week 5:**
- [ ] `SQLTool`: safe SQL execution, whitelist enforcement, query validation
- [ ] Test: injection attacks, forbidden keywords, table access control

**Week 6:**
- [ ] `AggregationTool`: trend_over_time, top_n, distribution, statistics
- [ ] Test: query correctness, performance benchmarks

**Week 7:**
- [ ] `GraphTool`: connected_storylines, entity_path, storyline_cluster
- [ ] Test: graph traversal correctness, max_depth limits

**Week 8:**
- [ ] `MarketTool`: signals_filter, macro_correlation, valuation_screen
- [ ] Test: correlation accuracy, signal ranking

**Deliverables:**
- `src/llm/tools/sql_tool.py`
- `src/llm/tools/aggregation_tool.py`
- `src/llm/tools/graph_tool.py`
- `src/llm/tools/market_tool.py`
- Test suite completo per ogni tool

---

### **FASE 4: Orchestrator + Memory (Week 9-10)**
**Obiettivo:** Coordinazione end-to-end + conversation tracking

**Week 9:**
- [ ] `OracleOrchestrator`: process_query, _execute_tools, _synthesize_response
- [ ] Multi-step execution con error handling
- [ ] Tool result caching per avoid redundant calls

**Week 10:**
- [ ] `ConversationContext`: message buffer, entity tracking
- [ ] Follow-up detection heuristics
- [ ] History compression con LLM summarization
- [ ] Context injection nel prompt synthesis

**Deliverables:**
- `src/llm/oracle_orchestrator.py`
- `src/llm/conversation_memory.py`
- Test: `tests/test_llm/test_orchestrator.py`

---

### **FASE 5: UI Integration (Week 11-12)**
**Obiettivo:** Update Streamlit UI per Oracle 2.0

- [ ] Refactor `pages/2_The_Oracle.py` per usare `OracleOrchestrator`
- [ ] UI per visualizzare `QueryPlan` (debug panel)
- [ ] Tool results expanders (show intermediate steps)
- [ ] Conversation history sidebar
- [ ] Export conversation (markdown/JSON)

**Deliverables:**
- Updated `pages/2_The_Oracle.py`
- New UI components: query plan visualization, tool trace

---

### **FASE 6: Advanced Features (Week 13-14)**
**Obiettivo:** Query decomposition + chain-of-thought

- [ ] `QueryDecomposer`: break complex queries into sub-queries
- [ ] Sub-query dependency graph (DAG)
- [ ] Parallel execution per independent sub-queries
- [ ] Result aggregation across sub-queries
- [ ] Chain-of-thought prompting per synthesis

**Deliverables:**
- `src/llm/query_decomposer.py`
- `src/llm/chain_of_thought.py`
- Test: complex multi-hop queries

---

### **FASE 7: Testing & Optimization (Week 15-16)**
**Obiettivo:** End-to-end testing + performance tuning

- [ ] Benchmark suite: 50 test queries across all intents
- [ ] Latency profiling: identify bottlenecks
- [ ] Caching strategy: tool results, embeddings, LLM responses
- [ ] Error recovery: graceful degradation se tool fails
- [ ] Cost tracking: Gemini API usage per query type

**Deliverables:**
- `tests/test_integration/test_oracle_e2e.py`
- Performance report + optimization recommendations
- Cost analysis spreadsheet

---

## 📊 METRICHE DI SUCCESSO

### **Functional Metrics**
- **Query Coverage:** 95%+ delle query user matchano almeno 1 intent
- **Tool Accuracy:** 90%+ tool executions successful (non-error)
- **Synthesis Quality:** 4/5 avg rating da human evaluation (50 query campionarie)
- **Follow-up Tracking:** 85%+ follow-up detected correttamente

### **Performance Metrics**
- **Latency P50:** < 3 seconds (SIMPLE queries)
- **Latency P95:** < 10 seconds (COMPLEX queries)
- **LLM Cost:** < $0.05 per query (avg)
- **DB Query Time:** < 500ms per tool execute

### **User Experience Metrics**
- **User Satisfaction:** 4+/5 feedback score
- **Query Reformulation Rate:** < 20% (users don't need to rephrase)
- **Session Depth:** Avg 4+ queries per session (conversational engagement)

---

## 🛠️ TECHNICAL STACK

### **Core Dependencies**
```python
# requirements.txt additions
pydantic>=2.0  # Schema validation
networkx>=3.0  # Graph tool (optional, se serve algoritmi complessi)
sqlparse>=0.4  # SQL query parsing/validation
```

### **File Struttura (Preview)**
```
src/llm/
├── oracle_engine.py              # LEGACY (mantenere per backward compatibility)
├── oracle_orchestrator.py        # NEW: main entry point
├── query_router.py               # NEW: intent classification + planning
├── query_decomposer.py           # NEW: complex query decomposition
├── conversation_memory.py        # NEW: conversational state
├── chain_of_thought.py           # NEW: CoT prompting
├── schemas.py                    # Extended: QueryPlan, ToolResult, etc.
├── query_analyzer.py             # EXISTING (keep)
├── report_generator.py           # EXISTING (keep)
├── tools/
│   ├── __init__.py
│   ├── base.py                   # BaseTool, ToolResult
│   ├── registry.py               # ToolRegistry
│   ├── rag_tool.py               # RAGTool (refactored from OracleEngine)
│   ├── sql_tool.py               # NEW
│   ├── aggregation_tool.py       # NEW
│   ├── graph_tool.py             # NEW
│   └── market_tool.py            # NEW

tests/test_llm/
├── test_tool_registry.py
├── test_query_router.py
├── test_tools/
│   ├── test_rag_tool.py
│   ├── test_sql_tool.py
│   ├── test_aggregation_tool.py
│   ├── test_graph_tool.py
│   └── test_market_tool.py
├── test_orchestrator.py
└── test_oracle_e2e.py
```

---

## 🚀 QUICK START (Post-Implementation)

### **Esempio 1: Factual Query con RAGTool**
```python
from src.llm.oracle_orchestrator import OracleOrchestrator

orchestrator = OracleOrchestrator(db=db, llm=llm)

response = orchestrator.process_query("Cosa è successo a Taiwan il 15 febbraio?")

# Query Plan:
# - intent: FACTUAL
# - complexity: SIMPLE
# - tools: ['rag_search']

print(response['answer'])
# Output: Detailed answer citando [Tool 1: rag_search]
```

---

### **Esempio 2: Analytical Query con SQLTool + AggregationTool**
```python
response = orchestrator.process_query(
    "Mostrami il trend degli articoli su cybersecurity negli ultimi 3 mesi"
)

# Query Plan:
# - intent: ANALYTICAL
# - complexity: MEDIUM
# - tools: ['sql_query', 'aggregation']
# - steps:
#   1. sql_query(query="SELECT COUNT(*), DATE(published_date) FROM articles WHERE category='CYBER' ...") 
#   2. aggregation(type='trend_over_time', data=sql_result)

print(response['answer'])
# Output: Trend analysis with chart data + insights
```

---

### **Esempio 3: Comparative Query con Multi-Tool**
```python
response = orchestrator.process_query(
    "Confronta la postura militare della Cina vs. 6 mesi fa"
)

# Query Plan:
# - intent: COMPARATIVE
# - complexity: COMPLEX
# - requires_decomposition: True
# - sub_queries: 
#   ["Postura militare Cina oggi", "Postura militare Cina 6 mesi fa"]
# - tools: ['rag_search', 'rag_search', 'graph_navigation']
# - steps:
#   1. rag_search(query='Cina military', filters={'end_date': today})
#   2. rag_search(query='Cina military', filters={'end_date': 6_months_ago})
#   3. LLM synthesis (compare results)

print(response['answer'])
# Output: Comparative analysis citing both time periods
```

---

## 🔐 SECURITY CONSIDERATIONS

### **SQLTool Safety**
- ✅ **Whitelist** di tabelle accessibili (no sensitive tables)
- ✅ **Blacklist** di keywords pericolose (INSERT, UPDATE, DROP, etc.)
- ✅ **Prepared statements** (no SQL injection)
- ✅ **Read-only mode** (solo SELECT, no mutations)
- ✅ **Query timeout** (max 10s execution)
- ✅ **Result limit** (max 1000 rows)

### **LLM Prompt Injection Defense**
- ✅ **Input sanitization** (strip command characters)
- ✅ **XML delimiter wrapping** per tool results (anti-confusion)
- ✅ **Role-based prompting** (clear USER vs. ASSISTANT vs. TOOL sections)
- ✅ **Output validation** (check for leaked system prompts)

---

## 📚 DOCUMENTATION PLAN

### **Developer Docs**
- [ ] `docs/oracle_2.0_architecture.md` - Architettura diagram + spiegazione
- [ ] `docs/tool_development_guide.md` - Come creare nuovi tools
- [ ] `docs/query_routing_logic.md` - Intent classification rules
- [ ] `docs/api_reference.md` - API docs per OracleOrchestrator

### **User Docs**
- [ ] `docs/user_guide_oracle.md` - Come usare Oracle 2.0 (query examples)
- [ ] `docs/advanced_queries.md` - Esempi di query complesse
- [ ] `docs/troubleshooting.md` - Common issues + solutions

---

## 🎓 TRAINING DATA (Few-Shot Examples per Query Router)

```python
# Esempi per intent classification (in query_router.py)
INTENT_EXAMPLES = {
    "FACTUAL": [
        "Cosa è successo a Taiwan negli ultimi 7 giorni?",
        "Chi ha attaccato l'infrastruttura energetica in Ucraina?",
        "Quali sanzioni ha imposto l'UE alla Russia ieri?"
    ],
    "ANALYTICAL": [
        "Trend degli attacchi ransomware negli ultimi 6 mesi",
        "Top 10 entità più menzionate nel database",
        "Distribuzione degli articoli per categoria e fonte"
    ],
    "NARRATIVE": [
        "Qual è lo stato della storyline Russia-Ucraina?",
        "Mostrami le storylines collegate alla crisi dei semiconduttori",
        "Come si è evoluta la narrativa sul conflitto Iran-Israele?"
    ],
    "MARKET": [
        "Mostrami i trade signals bullish per il settore defense",
        "Quali ticker hanno PE undervalued + high confidence?",
        "Come si correla il VIX con le crisi geopolitiche?"
    ],
    "COMPARATIVE": [
        "Confronta la postura militare della Cina oggi vs. 6 mesi fa",
        "Differenze tra report di dicembre e gennaio su Taiwan",
        "NATO vs. CSTO: confronto capacità militari"
    ]
}
```

---

## ✅ CHECKLIST PRE-DEPLOYMENT

- [ ] Tutti i test unitari passano (coverage >80%)
- [ ] Test integrazione E2E con 50 query diverse
- [ ] Performance benchmark < soglia latency
- [ ] Security audit: SQL injection, prompt injection
- [ ] Documentation completa (dev + user)
- [ ] Backward compatibility verificata (old OracleEngine still works)
- [ ] Monitoring setup: Sentry + logging dashboard
- [ ] Cost tracking attivo (Gemini API usage)
- [ ] User feedback form integrato in UI

---

## 🎯 KPI DASHBOARD (Post-Launch)

```sql
-- Esempi di query per monitoraggio
-- (da integrare in Streamlit dashboard)

-- 1. Query Intent Distribution
SELECT 
    metadata->>'query_plan'->>'intent' as intent,
    COUNT(*) as count
FROM oracle_query_log
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY intent;

-- 2. Avg Latency by Complexity
SELECT 
    metadata->>'query_plan'->>'complexity' as complexity,
    AVG((metadata->>'execution_time')::float) as avg_latency
FROM oracle_query_log
GROUP BY complexity;

-- 3. Top Tools Used
SELECT 
    jsonb_array_elements_text(metadata->'query_plan'->'tools') as tool,
    COUNT(*) as usage_count
FROM oracle_query_log
GROUP BY tool
ORDER BY usage_count DESC;

-- 4. User Satisfaction (feedback tracking)
SELECT 
    AVG((feedback->>'rating')::float) as avg_rating,
    COUNT(*) as feedback_count
FROM oracle_feedback
WHERE created_at >= NOW() - INTERVAL '30 days';
```

---

## 🚧 KNOWN LIMITATIONS & FUTURE WORK

### **Limitations (MVP)**
1. **Parallel tool execution:** Attualmente sequenziale, può essere parallelizzato per INDEPENDENT tools
2. **Query caching:** Nessun caching semantico (duplicate queries re-execute)
3. **Streaming responses:** LLM synthesis non è streamed (tutto in un colpo)
4. **Multi-modal:** Solo text, no images/charts/maps

### **Future Work (Post-MVP)**
1. **Plot generation tool:** Genera charts/graphs automaticamente (matplotlib/plotly)
2. **Map tool:** Visualizza eventi su mappa geografica (folium)
3. **Export tool:** Export risultati in PDF/Excel/CSV
4. **Scheduling tool:** "Notify me when X happens" (proactive alerts)
5. **Multi-language:** Support EN/IT/ES query naturali
6. **Voice input:** Whisper integration per query vocali

---

## 📞 SUPPORT & CONTRIBUTION

### **Questions?**
- Developer: @tommasofazzi
- Docs: `docs/oracle_2.0_architecture.md`
- Issues: GitHub Issues

### **Contributing**
1. Fork repo
2. Create feature branch: `feature/oracle-tool-xyz`
3. Follow `docs/tool_development_guide.md`
4. Submit PR with tests + docs

---

**END OF DOCUMENT**

*Last updated: 2026-03-03*
*Version: 1.0 (Oracle 2.0 Development Plan)*
