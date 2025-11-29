# Test Suite per INTELLIGENCE_ITA

Test automatizzati per verificare il corretto funzionamento del sistema.

## 📊 Statistiche Attuali

```
✅ 223 test totali
⚡ Tempo esecuzione: ~6.5 secondi
📦 Moduli testati: Ingestion (+ dedup), NLP, Storage (business logic + content hash), LLM (query expansion)
```

## 🗂️ Struttura Test

```
tests/
├── conftest.py                        # Fixtures condivise
├── test_setup.py                      # 9 test - Verifica setup
├── test_ingestion/
│   ├── test_feed_parser.py           # 18 test - RSS parsing
│   ├── test_content_extractor.py     # 20 test - Content extraction
│   └── test_pipeline_dedup.py        # 12 test - Hash deduplication (Phase 1)
├── test_nlp/
│   ├── test_text_cleaning.py         # 26 test - Text cleaning
│   ├── test_chunking.py              # 18 test - Semantic chunking
│   ├── test_embeddings.py            # 20 test - Embedding generation
│   ├── test_entities.py              # 19 test - Named Entity Recognition
│   └── test_nlp_processor.py         # 34 test - Full NLP pipeline
├── test_storage/
│   ├── test_database_logic.py        # 20 test - Database business logic
│   └── test_content_hash_dedup.py    # 10 test - Content hash dedup (Phase 2)
├── test_llm/
│   └── test_query_expansion.py       # 17 test - Query expansion & deduplication
└── test_e2e/                          # TODO
```

## 🚀 Come Eseguire i Test

### Tutti i test
```bash
pytest
```

### Test specifici
```bash
# Solo test ingestion
pytest tests/test_ingestion/

# Solo test NLP
pytest tests/test_nlp/

# Solo test Storage
pytest tests/test_storage/

# Solo test LLM
pytest tests/test_llm/

# Solo FeedParser
pytest tests/test_ingestion/test_feed_parser.py

# Solo text cleaning
pytest tests/test_nlp/test_text_cleaning.py

# Solo query expansion
pytest tests/test_llm/test_query_expansion.py

# Solo un test specifico
pytest tests/test_ingestion/test_feed_parser.py::test_parse_feed_success
```

### Con opzioni utili
```bash
# Verbose (mostra ogni test)
pytest -v

# Solo test veloci (esclude slow)
pytest -m "not slow"

# Solo unit test
pytest -m unit

# Con coverage
pytest --cov=src --cov-report=html
```

## 📝 Test Coverage

### ✅ Ingestion (50 test - COMPLETATO)

**FeedParser (18 test)**
- Inizializzazione e configurazione
- Parsing RSS feed validi e non validi
- Estrazione metadata (title, date, authors, tags)
- Filtro per categoria
- Gestione errori

**ContentExtractor (20 test)**
- Estrazione con Trafilatura
- Fallback a Newspaper3k
- Batch processing
- Gestione errori HTTP
- Timeout handling

**Pipeline Deduplication - Phase 1 (12 test)**
- deduplicate_by_quick_hash(): MD5 hash di link + title
- Rimozione duplicati esatti in-memory
- Preservazione prima occorrenza
- Gestione edge cases (campi mancanti, unicode)
- Integrazione in run() prima di content extraction

### ✅ NLP (117 test - COMPLETATO)

**Text Cleaning (26 test)**
- Normalizzazione whitespace (spazi, tab, newline)
- Rimozione markdown links e bracket
- Rimozione pattern comuni di rumore (ads, social media, etc.)
- Preservazione punteggiatura e unicode
- Gestione edge cases (empty, None, whitespace-only)

**Semantic Chunking (18 test)**
- Chunking basato su frasi complete con spaCy
- Gestione overlap tra chunk consecutivi
- Rispetto limiti di dimensione configurabili
- Metadata (word_count, sentence_count)
- Preservazione integrità del testo

**Embedding Generation (20 test)**
- Generazione embeddings singoli con SentenceTransformers
- Batch embedding per chunk multipli
- Dimensione corretta (384-dim)
- Conversione a liste per JSON serialization
- Gestione input vuoti/None con zero vectors

**Named Entity Recognition (19 test)**
- Estrazione entità con spaCy NER
- Organizzazione per tipo (PER, ORG, LOC, etc.)
- Metadata entità (text, label, posizioni)
- Supporto multilingue
- Gestione duplicati ed edge cases

**Full NLP Pipeline (34 test)**
- Preprocessing linguistico (tokenization, lemmatization, POS)
- process_article() - pipeline completa
- process_batch() - processing batch multipli articoli
- get_processing_stats() - statistiche processing
- Integrazione tutti i componenti
- Gestione errori e fallback

### ✅ Storage (30 test - Business Logic)

**Database Logic (20 test)**
- Inizializzazione con connection URL o env vars
- Validazione input (skip articoli senza NLP data)
- Batch save statistics tracking (saved/skipped/errors)
- Semantic search query building (con/senza categoria filter)
- Upsert logic per approval feedback
- Error handling (return empty on failure)
- Connection pool management

**Content Hash Deduplication - Phase 2 (10 test)**
- Computazione MD5 hash da clean_text
- Deduplicazione basata su content_hash (7-day window)
- Skip articoli con link diverso ma contenuto identico
- Gestione unicode e contenuti lunghi
- Link check PRIMA di content hash (efficienza)
- Integration con existing link dedup

**Note**: Questi test verificano la **logica di business** senza richiedere database reale.
Per test di integrazione completi (schema SQL, pgvector, queries reali),
eseguire test separati con PostgreSQL + pgvector configurato.

### ✅ LLM (17 test - Query Expansion)

**Query Expansion (17 test)**
- expand_rag_queries(): generazione varianti semantiche con LLM
- Filtering varianti invalide (troppo corte/lunghe)
- Gestione fallback quando LLM fallisce
- Rimozione numerazione dalle varianti
- deduplicate_chunks_advanced(): dedup con similarity
- Rimozione duplicati esatti (chunk_id)
- Rimozione duplicati semantici (cosine similarity > 0.98)
- Gestione chunks con/senza embeddings
- Integrazione in generate_report()
- Configurazione (enable/disable, variants count, similarity threshold)

### ⏳ End-to-End (TODO)
- Pipeline completa

## 🎯 Best Practices Utilizzate

### 1. AAA Pattern (Arrange-Act-Assert)
```python
def test_example():
    # Arrange: prepara dati
    parser = FeedParser()

    # Act: esegui funzione
    result = parser.parse_feed(url)

    # Assert: verifica risultato
    assert len(result) > 0
```

### 2. Fixtures per Setup Comune
```python
@pytest.fixture
def extractor():
    return ContentExtractor(timeout=10)

def test_with_fixture(extractor):
    result = extractor.extract_content(url)
    assert result is not None
```

### 3. Mocking per Dependency Isolation
```python
@patch('trafilatura.fetch_url')
def test_with_mock(mock_fetch, extractor):
    mock_fetch.return_value = "<html>...</html>"
    result = extractor.extract_with_trafilatura(url)
    assert result is not None
```

### 4. Markers per Categorizzazione
```python
@pytest.mark.unit
def test_fast():
    assert True

@pytest.mark.slow
def test_expensive():
    # Test lungo...
    pass
```

## 📚 Cosa Hai Imparato

1. **Setup pytest** - conftest.py, pytest.ini, fixtures
2. **Unit testing** - Test di singole funzioni isolate
3. **Mocking** - Mock di API esterne (RSS, HTTP, LLM)
4. **Parametrize** - Test multipli con dati diversi
5. **Error handling** - Test che verificano gestione errori
6. **Edge cases** - Test di casi limite

## 🔧 Configurazione

### pytest.ini
Configurazione globale per pytest (markers, opzioni, etc.)

### conftest.py
Fixtures condivise disponibili a tutti i test:
- `sample_article` - Articolo di esempio
- `sample_embedding` - Embedding 384-dim
- `sample_articles_batch` - Batch di articoli
- `test_db_url` - URL database di test

### requirements-dev.txt
Dipendenze per testing:
- pytest
- pytest-cov (coverage)
- pytest-timeout
- pytest-mock
- responses (HTTP mocking)

## 🎓 Tips & Tricks

### Debug di un test fallito
```bash
# Mostra dettagli completi
pytest test_file.py::test_name -vv

# Mostra print statements
pytest test_file.py::test_name -s

# Stoppa al primo errore
pytest test_file.py -x
```

### Coverage report
```bash
# Genera report HTML
pytest --cov=src --cov-report=html

# Apri report nel browser
open htmlcov/index.html
```

### Test in parallelo
```bash
# Installa plugin
pip install pytest-xdist

# Run con 4 workers
pytest -n 4
```

## 📈 Prossimi Passi

1. ✅ Test Ingestion (COMPLETATO - 50 test)
   - ✅ Feed Parser (18 test)
   - ✅ Content Extractor (20 test)
   - ✅ Pipeline Dedup Phase 1 (12 test)
2. ✅ Test NLP (COMPLETATO - 117 test)
3. ✅ Test Storage Business Logic (COMPLETATO - 30 test)
   - ✅ Database Logic (20 test)
   - ✅ Content Hash Dedup Phase 2 (10 test)
4. ✅ Test LLM Query Expansion (COMPLETATO - 17 test)
5. ⏳ Test Storage Integration (richiede PostgreSQL + pgvector)
6. ⏳ Test LLM Report Generation (article filtering, prompt formatting)
7. ⏳ Test HITL (dashboard, feedback)
8. ⏳ Test End-to-End (pipeline completa)
9. ⏳ CI/CD (GitHub Actions)

## 🎉 Deduplication Implementation

**Phase 1 - Hash Dedup (COMPLETATO)**
- In-memory deduplication basata su MD5(link + title[:100])
- Integrato in `IngestionPipeline.run()` dopo RSS parsing
- ~5-10% riduzione articoli duplicati prevista
- 12 test, zero risk (no DB changes)

**Phase 2 - Content Hash Dedup (COMPLETATO)**
- Database deduplication basata su MD5(clean_text)
- 7-day lookback window per performance
- Skip articoli con contenuto identico da fonti diverse
- ~10-15% riduzione duplicati contenuto prevista
- 10 test, SQL migration pronta

**Phase 3 - Similarity Dedup (POSTICIPATA)**
- Deduplicazione semantica con embeddings
- Pianificato per futuro

---

**Status**: 🟢 Test suite attiva e funzionante (223 test, ~6.5s)
**Last update**: 2025-11-29
