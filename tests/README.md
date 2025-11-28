# Test Suite per INTELLIGENCE_ITA

Test automatizzati per verificare il corretto funzionamento del sistema.

## 📊 Statistiche Attuali

```
✅ 47 test totali
⚡ Tempo esecuzione: ~0.3 secondi
📦 Moduli testati: Ingestion (FeedParser, ContentExtractor)
```

## 🗂️ Struttura Test

```
tests/
├── conftest.py                        # Fixtures condivise
├── test_setup.py                      # 9 test - Verifica setup
├── test_ingestion/
│   ├── test_feed_parser.py           # 18 test - RSS parsing
│   └── test_content_extractor.py     # 20 test - Content extraction
├── test_nlp/                          # TODO
├── test_storage/                      # TODO
├── test_llm/                          # TODO
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

# Solo FeedParser
pytest tests/test_ingestion/test_feed_parser.py

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

### ✅ Ingestion (38 test - COMPLETATO)

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

### ⏳ NLP (TODO)
- Text cleaning
- Semantic chunking
- Embedding generation
- Named Entity Recognition

### ⏳ Storage (TODO)
- Database operations
- Vector search
- Connection pooling

### ⏳ LLM (TODO)
- Article filtering
- Report generation
- RAG context

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

1. ✅ Test Ingestion (COMPLETATO)
2. ⏳ Test NLP (preprocessing, chunking, embeddings)
3. ⏳ Test Storage (database, vector search)
4. ⏳ Test LLM (filtering, report generation)
5. ⏳ Test HITL (dashboard, feedback)
6. ⏳ Test End-to-End (pipeline completa)
7. ⏳ CI/CD (GitHub Actions)

---

**Status**: 🟢 Test suite attiva e funzionante
**Last update**: 2025-11-28
