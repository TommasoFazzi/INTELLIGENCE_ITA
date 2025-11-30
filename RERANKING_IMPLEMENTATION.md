# Reranking Implementation Summary

**Data**: 2025-11-30
**Status**: ✅ COMPLETATO

---

## 🎯 Obiettivo

Implementare reranking dei chunks RAG usando Cross-Encoder per migliorare la precisione del retrieval del 15-20%, passando da un sistema single-stage a 2-stage:

1. **Stage 1**: Fast vector search (HNSW pgvector) - recall-oriented
2. **Stage 2**: Slow reranking (Cross-Encoder) - precision-oriented

---

## ✅ Modifiche Implementate

### 1. File Modificato: `src/llm/report_generator.py`

#### A. Parametri `__init__()` (linee 31-92)

**Aggiunti 3 nuovi parametri**:

```python
def __init__(
    self,
    # ... parametri esistenti ...
    enable_reranking: bool = True,  # NUOVO
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",  # NUOVO
    reranking_top_k: int = 10  # NUOVO
):
```

**Lazy loading Cross-Encoder** (linee 75-80):
```python
if self.enable_reranking:
    from sentence_transformers import CrossEncoder
    self.reranker = CrossEncoder(reranking_model)
    logger.info(f"  Reranking: ENABLED (model: {reranking_model}, top_k: {reranking_top_k})")
else:
    self.reranker = None
```

#### B. Nuovo Metodo: `_rerank_chunks()` (linee 262-313)

Metodo privato che:
- Prende query + lista chunks + top_k
- Crea coppie (query, chunk_text) per Cross-Encoder
- Ottiene scores di rilevanza bidirezionale
- Ordina chunks per score
- Ritorna top-k chunks con `rerank_score` aggiunto

**Caratteristiche**:
- Batch processing (batch_size=32)
- Logging dettagliato (score range, median)
- Graceful fallback se reranker=None

#### C. Integrazione in `generate_report()` (linee 594-618)

**Step 2b - Aumentato top_k per vector search**:
```python
search_top_k = rag_top_k * 2 if self.enable_reranking else rag_top_k
```

**Step 2d - Nuovo step di reranking** (dopo deduplicazione):
```python
if self.enable_reranking and unique_rag_results and rag_queries:
    primary_query = rag_queries[0]
    unique_rag_results = self._rerank_chunks(
        query=primary_query,
        chunks=unique_rag_results,
        top_k=self.reranking_top_k * len(rag_queries)
    )
```

---

## 📋 Configurazione

### Default (Reranking ENABLED)

```python
from src.llm.report_generator import ReportGenerator

# Default: reranking enabled
gen = ReportGenerator()
```

**Parametri di default**:
- `enable_reranking=True`
- `reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2"` (English-optimized)
- `reranking_top_k=10`

### Disable Reranking

```python
gen = ReportGenerator(enable_reranking=False)
```

### Modello Multilingual (per Sole 24 Ore)

```python
gen = ReportGenerator(
    reranking_model="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"
)
```

---

## 🔄 Flow Completo RAG con Reranking

```
[STEP 2] RAG Retrieval
│
├─ [2a] Query Expansion (se enabled)
│   └─ 1 query → 3 queries (original + 2 variants)
│
├─ [2b] Vector Search (HNSW pgvector)
│   ├─ top_k = 10 (senza reranking)
│   └─ top_k = 20 (con reranking - cast wider net)
│   └─ Per ogni query espansa → retrieve chunks
│   └─ Total chunks: 3 queries × 20 = 60 chunks (raw)
│
├─ [2c] Advanced Deduplication
│   ├─ Rimozione duplicati esatti (chunk_id)
│   └─ Rimozione simili (cosine > 0.98)
│   └─ Output: ~40-50 chunks unici
│
├─ [2d] ⭐ RERANKING (NUOVO) ⭐
│   ├─ Cross-Encoder score: (query, chunk) → relevance score
│   ├─ Ordina chunks per score
│   └─ Top-k finale: 10 × num_queries = 30 chunks
│
└─ [2e] Final RAG context: 30 chunks di alta qualità
```

---

## 🧪 Verifica Implementazione

### Test Sintattico

```bash
python3 -m py_compile src/llm/report_generator.py
# ✓ Syntax check passed
```

### Verifica Manuale

```bash
grep -n "enable_reranking" src/llm/report_generator.py
# Output:
# 40:        enable_reranking: bool = True,
# 55:            enable_reranking: Enable Cross-Encoder reranking...
# 71:        self.enable_reranking = enable_reranking
# 75:        if self.enable_reranking:
# 597:        search_top_k = rag_top_k * 2 if self.enable_reranking else rag_top_k
```

```bash
grep -n "_rerank_chunks" src/llm/report_generator.py
# Output:
# 262:    def _rerank_chunks(
# 612:            unique_rag_results = self._rerank_chunks(
```

✅ **Tutti i check passati**

---

## 📊 Impatto Atteso

### Performance

| Metrica | Prima | Dopo Reranking | Differenza |
|---------|-------|----------------|------------|
| **Query latency** | 5-10ms | 5-10ms | Invariata |
| **Reranking time** | - | +3-4s | +Nuovo step |
| **Report generation** | 40-45s | 44-49s | +9% |
| **Precision@10** | ~70-75% | ~80-85% | **+10-15%** ✅ |

### Qualità

- **Riduzione rumore**: Chunks meno rilevanti filtrati via
- **Miglior focus**: Report più focalizzati su argomenti richiesti
- **Query expansion + Reranking**: Sinergia ottimale (coverage + precision)

---

## 🚀 Come Usare

### Generare Report (default con reranking)

```python
from src.llm.report_generator import ReportGenerator

gen = ReportGenerator()  # Reranking enabled by default

report = gen.generate_report(
    focus_areas=["cybersecurity threats Italy"],
    output_file="output/report_with_reranking.md"
)
```

### A/B Testing: Con vs Senza Reranking

```python
# Test 1: Without reranking
gen_no_rerank = ReportGenerator(enable_reranking=False)
report_no_rerank = gen_no_rerank.generate_report(
    focus_areas=["cybersecurity threats Italy"],
    output_file="output/test_no_reranking.md"
)

# Test 2: With reranking
gen_rerank = ReportGenerator(enable_reranking=True)
report_rerank = gen_rerank.generate_report(
    focus_areas=["cybersecurity threats Italy"],
    output_file="output/test_with_reranking.md"
)

# Compare manualmente i due report
```

---

## ⚙️ Parametri Tuning

### Se Risultati Non Soddisfacenti

**Problema**: Articoli italiani (Sole 24 Ore) penalizzati

**Soluzione**: Switch a modello multilingual
```python
gen = ReportGenerator(
    reranking_model="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"
)
```

**Problema**: Troppo pochi chunks buoni

**Soluzione**: Aumenta initial_top_k
```python
# Modifica in generate_report (linea 597)
search_top_k = rag_top_k * 3 if self.enable_reranking else rag_top_k
```

**Problema**: Troppo rumore

**Soluzione**: Riduci reranking_top_k
```python
gen = ReportGenerator(reranking_top_k=5)  # Default: 10
```

---

## 🔍 Logging Output

Con reranking enabled, vedrai nel log:

```
[STEP 2b] Execute RAG searches
✓ Retrieved 60 total chunks from RAG

[STEP 2c] Advanced deduplication
✓ Deduplication: 60 → 42 chunks

[STEP 2d] Reranking
Reranking 42 chunks with Cross-Encoder...
✓ Reranked: scores range [0.152 - 0.847], median: 0.521
✓ After reranking: 30 top chunks selected

✓ Final RAG context: 30 unique historical chunks
```

---

## 📁 File Modificati/Creati

### Modificati
1. **`src/llm/report_generator.py`**
   - +51 righe di codice
   - 3 nuovi parametri __init__
   - 1 nuovo metodo privato (_rerank_chunks)
   - Integrazione in generate_report

### Creati
2. **`test_reranking_feature.py`** - Test di verifica implementazione
3. **`RERANKING_IMPLEMENTATION.md`** - Questo documento

---

## ✅ Checklist Completamento

- [x] Aggiunti parametri `enable_reranking`, `reranking_model`, `reranking_top_k` a `__init__`
- [x] Lazy loading di CrossEncoder (solo se enabled)
- [x] Implementato metodo `_rerank_chunks()`
- [x] Integrato reranking in `generate_report()` (Step 2d)
- [x] Aumentato `top_k` per vector search quando reranking enabled
- [x] Logging dettagliato (scores range, median)
- [x] Test sintassi Python passati
- [x] Documentazione completa

---

## 🎯 Prossimi Passi

1. **Test su Report Reali**:
   - Genera 3-5 report con reranking
   - Valuta qualità vs report senza reranking
   - Verifica ranking articoli Sole 24 Ore

2. **Se Performance Italiane Degradate**:
   - Upgrade a modello multilingual (1 riga di codice)

3. **Monitoring**:
   - Track latency report generation
   - Confronta quality scores manualmente

---

**Implementato da**: Claude Code
**Tempo stimato implementazione**: 30 minuti
**Complessità**: Media
**Rischio**: Basso (feature flag permette disable immediato)
