# INTELLIGENCE_ITA 🇮🇹

Sistema intelligente di analisi notizie con elaborazione NLP, generazione di insight e RAG (Retrieval-Augmented Generation) per report giornalieri contestualizzati.

## 📋 Descrizione

INTELLIGENCE_ITA è un sistema completo che:

1. **Raccoglie** notizie in tempo reale da feed RSS configurabili (geopolitica, economia, tecnologia, cybersecurity)
2. **Estrae** il testo completo degli articoli tramite web scraping intelligente
3. **Elabora** i contenuti con tecniche NLP avanzate per creare embedding e categorizzazioni
4. **Genera** report giornalieri con insight rilevanti e analisi cross-settoriale tramite LLM
5. **Impara** continuamente grazie al feedback umano (HITL) e al sistema RAG per contestualizzare le analisi future

## 🏗️ Architettura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  RSS Feed APIs  │────▶│  Data Ingestion  │────▶│  NLP Processing │
│  (33 sources)   │     │  + Full Text     │     │  + Embeddings   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Daily Report   │◀────│  LLM Generation  │◀────│  Vector DB RAG  │
│  (Human Review) │     │  (Gemini 2.5)    │     │  (pgvector)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         ▲
         │                       ▼                         │
         │              ┌──────────────────┐               │
         │              │  Trade Signals   │               │
         │              │  + Market Data   │               │
         │              │  (Yahoo Finance) │               │
         │              └──────────────────┘               │
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                          Feedback Loop
```

## 📂 Struttura del Progetto

```
INTELLIGENCE_ITA/
├── config/
│   ├── feeds.yaml              # Configurazione feed RSS (33 fonti)
│   ├── top_50_tickers.yaml     # Whitelist ticker per Trade Signals
│   └── entity_blocklist.yaml   # Blocklist entità rumorose
├── src/
│   ├── ingestion/              # Moduli di acquisizione dati (async con aiohttp)
│   │   ├── feed_parser.py      # Parser RSS/Atom multi-fonte (fetch parallelo)
│   │   ├── content_extractor.py # Estrazione testo completo (concorrente)
│   │   └── pipeline.py         # Pipeline orchestrata (singolo asyncio.run)
│   ├── nlp/                    # Elaborazione NLP
│   │   ├── processing.py       # Pulizia, NER, chunking
│   │   └── embeddings.py       # Generazione embedding vettoriali
│   ├── storage/                # Database e persistenza
│   │   └── database.py         # PostgreSQL + pgvector
│   ├── llm/                    # Generazione report
│   │   ├── report_generator.py # LLM + RAG + Trade Signals
│   │   └── schemas.py          # Pydantic schemas per validazione
│   ├── integrations/           # Integrazioni esterne (Sprint 3)
│   │   └── market_data.py      # Yahoo Finance API
│   ├── hitl/                   # Human-in-the-Loop
│   │   └── dashboard.py        # Streamlit dashboard
│   └── utils/
│       └── logger.py           # Sistema di logging
├── scripts/
│   ├── check_setup.py          # Verifica configurazione sistema
│   ├── process_nlp.py          # NLP processing pipeline
│   ├── load_to_database.py     # Caricamento DB
│   ├── generate_report.py      # Generazione report (+ Trade Signals)
│   ├── backfill_market_data.py # Backfill dati Yahoo Finance
│   └── run_dashboard.sh        # Avvio dashboard HITL
├── migrations/
│   ├── 004_add_market_intelligence_schema.sql
│   └── 005_add_trade_signals.sql  # Tabella trade_signals
├── data/                       # Dati temporanei (gitignored)
├── reports/                    # Report generati (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Setup Iniziale

### 1. Prerequisiti

- Python 3.9+
- PostgreSQL 14+ (con estensione pgvector per il RAG)
- API Key per LLM (Gemini, OpenAI, o Anthropic)

### 2. Installazione

```bash
# Clone della repository
git clone <repository-url>
cd INTELLIGENCE_ITA

# Creazione ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installazione dipendenze
pip install -r requirements.txt

# Download modello spaCy per italiano
python -m spacy download en_core_web_sm

# Configurazione variabili d'ambiente
cp .env.example .env
# Modifica .env con le tue credenziali
```

### 3. Configurazione Database (PostgreSQL + pgvector)

```sql
-- Crea database
CREATE DATABASE intelligence_ita;

-- Connettiti al database
\c intelligence_ita

-- Abilita estensione pgvector per il RAG
CREATE EXTENSION vector;

-- Verifica installazione
SELECT * FROM pg_extension WHERE extname = 'vector';
```

Aggiorna `.env` con i dettagli di connessione:
```
DATABASE_URL=postgresql://user:password@localhost:5432/intelligence_ita
```

### 4. Configurazione API Key LLM

Nel file `.env`, aggiungi la chiave API del provider scelto:

```bash
# Gemini (Google)
GEMINI_API_KEY=your_api_key_here

# Oppure OpenAI
# OPENAI_API_KEY=your_api_key_here

# Oppure Anthropic Claude
# ANTHROPIC_API_KEY=your_api_key_here
```

### 5. Verifica Setup Sistema

Prima di eseguire la pipeline, verifica che tutto sia configurato correttamente:

```bash
python scripts/check_setup.py
```

Questo script controlla automaticamente:

- ✅ **Python Version**: Verifica Python >= 3.9
- ✅ **Environment File**: Controlla `.env` e presenza di `DATABASE_URL` e `GEMINI_API_KEY`
- ✅ **Database Connection**: Testa connessione a PostgreSQL
- ✅ **pgvector Extension**: Verifica che l'estensione vector sia installata
- ✅ **Database Tables**: Controlla presenza di tabelle (`articles`, `chunks`, `reports`, `report_feedback`)
- ✅ **Database Content**: Verifica se ci sono articoli nel database
- ✅ **spaCy Models**: Controlla che `en_core_web_sm` sia installato
- ✅ **Sentence Transformers**: Verifica libreria per embeddings
- ✅ **Gemini API**: Valida configurazione API key
- ✅ **Data Directories**: Verifica presenza di `data/`, `reports/`, `logs/`

**Output esempio**:
```
============================================================
INTELLIGENCE_ITA - System Setup Check
============================================================

✓ Python Version - v3.12.0 ✓ OK
✓ .env File - DATABASE_URL e GEMINI_API_KEY configurati
✓ Database Connection - PostgreSQL connesso
✓ pgvector Extension - v0.5.1 installata
✓ Database Tables - 4 tabelle presenti
✓ Database Content - 134 articoli, 183 chunks
✓ spaCy Models - en_core_web_sm caricato
✓ Sentence Transformers - Library disponibile
✓ Gemini API Key - Configurata
✓ Data Directories - data/, reports/, logs/ presenti

============================================================
✓ Tutti i controlli superati (10/10)

Il sistema è pronto per l'uso!
```

Se qualche controllo fallisce, lo script fornisce istruzioni su come risolvere il problema.

## 🔧 Utilizzo

### Pipeline Completa (Esecuzione Giornaliera)

```bash
# 1. Ingestion: Raccolta articoli ultimi 24h
python -m src.ingestion.pipeline

# 2. NLP Processing: Embedding e chunking
python scripts/process_nlp.py

# 3. Database: Caricamento in PostgreSQL
python scripts/load_to_database.py

# 4. Report Generation: LLM + RAG
python scripts/generate_report.py

# 5. HITL Review: Dashboard interattiva
./scripts/run_dashboard.sh
```

### Fase 1: Data Ingestion

Raccolta notizie da 33 feed RSS con filtro temporale:

```bash
# Articoli ultimi 24h (default)
python -m src.ingestion.pipeline

# Articoli ultimi 3 giorni
python -c "
from src.ingestion.pipeline import IngestionPipeline
pipeline = IngestionPipeline()
articles = pipeline.run(extract_content=True, max_age_days=3)
"
```

**Filtraggio per categoria:**
```python
from src.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

# Solo intelligence/geopolitica
articles = pipeline.run(category='intelligence', max_age_days=1)

# Solo economia e tech
articles = pipeline.run(category='tech_economy', max_age_days=1)
```

**Deduplication Automatica** (attiva di default):

La pipeline include un sistema di deduplicazione a 2 fasi per ridurre articoli duplicati del 20-25%:

- **Phase 1 - Hash Dedup** (in-memory): Rimuove duplicati esatti basati su hash di link+titolo (5-10% articoli)
- **Phase 2 - Content Hash Dedup** (database): Rimuove contenuti simili basati su hash del testo pulito (10-15% articoli)

**Benefici**:
- ✅ Riduzione costi processing (NLP, embeddings, LLM)
- ✅ Riduzione storage database
- ✅ Miglioramento qualità report (meno ridondanza)

Vedi [DEDUPLICATION_IMPLEMENTATION.md](DEDUPLICATION_IMPLEMENTATION.md) per dettagli tecnici completi.

### Fase 2: NLP Processing

Elaborazione con embedding e chunking:

```bash
# Processing automatico (trova ultimo file di ingestion)
python scripts/process_nlp.py

# Processing di file specifico
python scripts/process_nlp.py --input data/articles_20251123_100000.json

# Processing con custom chunk size
python scripts/process_nlp.py --chunk-size 1000 --chunk-overlap 100
```

**Output:** `data/articles_nlp_YYYYMMDD_HHMMSS.json`
- Clean text, entities, embeddings, chunks

### Fase 3: Database Storage

Caricamento in PostgreSQL con pgvector:

```bash
# Init schema (prima volta)
python scripts/load_to_database.py --init-only

# Carica ultimo file NLP
python scripts/load_to_database.py

# Carica file specifico
python scripts/load_to_database.py --input data/articles_nlp_20251123.json

# Statistiche database
python -c "
from src.storage.database import DatabaseManager
db = DatabaseManager()
stats = db.get_statistics()
print(f'Articoli: {stats[\"total_articles\"]}')
print(f'Chunks: {stats[\"total_chunks\"]}')
"
```

### Fase 4: Report Generation

Generazione report intelligence con LLM + RAG:

```bash
# Report giornaliero (default)
python scripts/generate_report.py

# Report ultimi 3 giorni
python scripts/generate_report.py --days 3

# Con modello più potente
python scripts/generate_report.py --model gemini-2.5-flash

# Solo visualizza (non salvare file)
python scripts/generate_report.py --no-save
```

**Output:**
- `reports/intelligence_report_YYYYMMDD_HHMMSS.json` (strutturato)
- `reports/intelligence_report_YYYYMMDD_HHMMSS.md` (markdown)

#### Pipeline Macro-First (Sprint 3 - Raccomandato)

La nuova pipeline serializzata ottimizza costi API e qualità dei segnali:

```bash
# Pipeline completa con Trade Signals
python scripts/generate_report.py --macro-first

# Solo segnali report-level (più veloce, -90% costi)
python scripts/generate_report.py --macro-first --skip-article-signals

# Con parametri custom
python scripts/generate_report.py --macro-first --days 3 --top-articles 100
```

**Flusso Macro-First:**
1. **Genera Macro Report** → Analisi RAG completa
2. **Condensa Contesto** → ~500 token (vs 5000+ token originali)
3. **Estrai Report Signals** → Segnali high-conviction (sintesi multi-articolo)
4. **Filtra Articoli con Ticker** → Solo articoli con ticker whitelist
5. **Estrai Article Signals** → Segnali per-articolo con alignment check
6. **Salva in DB** → Tabella `trade_signals` normalizzata

**Benefici:**
- ✅ Riduzione costi API ~90% (contesto condensato)
- ✅ Segnali più accurati (macro alignment check)
- ✅ Schema strutturato Pydantic (validazione automatica)
- ✅ Persistenza normalizzata per analytics

#### Trade Signals Schema

Ogni segnale include:

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `ticker` | string | Simbolo ticker (es. LMT, TSM) |
| `signal` | enum | BULLISH, BEARISH, NEUTRAL, WATCHLIST |
| `timeframe` | enum | SHORT_TERM, MEDIUM_TERM, LONG_TERM |
| `rationale` | string | Motivazione specifica |
| `confidence` | float | 0.0-1.0 (report: 0.7+, article: variabile) |
| `alignment_score` | float | Allineamento con narrativa macro |

**RAG Reranking** (attivo di default):

Il sistema usa un approccio 2-stage per massimizzare qualità del retrieval:

- **Stage 1 - Vector Search** (recall-oriented): HNSW approximate nearest neighbor su pgvector (~50ms, top-20 chunks)
- **Stage 2 - Cross-Encoder Reranking** (precision-oriented): Modello `cross-encoder/ms-marco-MiniLM-L-6-v2` per reranking bidirezionale (~3-4s, top-10 chunks finali)

**Benefici**:
- ✅ Migliora precision del RAG del 15-20%
- ✅ Riduce rumore nei chunks selezionati
- ✅ Report più focalizzati sugli argomenti richiesti

Per disabilitare: `ReportGenerator(enable_reranking=False)`

Vedi [RERANKING_IMPLEMENTATION.md](RERANKING_IMPLEMENTATION.md) per dettagli tecnici completi.

### Fase 5: HITL Review

Dashboard interattiva per revisione:

```bash
# Avvia dashboard Streamlit
./scripts/run_dashboard.sh

# Oppure manualmente
source venv/bin/activate
streamlit run src/hitl/dashboard.py
```

**Apre browser su:** http://localhost:8501

**Workflow nella dashboard:**
1. Clicca "➕ Genera Nuovo Report" (oppure seleziona esistente)
2. Tab "Bozza LLM": leggi versione originale
3. Tab "Versione Finale": modifica il testo
4. Aggiungi rating (1-5 stelle) e commenti
5. Clicca "💾 Salva Bozza" o "✅ Approva"
6. Tab "Fonti & Feedback": vedi articoli usati e storico modifiche

### Output Esempio

Il sistema salva i dati in `data/articles_YYYYMMDD_HHMMSS.json`:

```json
[
  {
    "title": "Breaking: Major Cyber Attack on Infrastructure",
    "link": "https://...",
    "published": "2024-01-15T10:30:00",
    "source": "CyberScoop",
    "category": "intelligence",
    "subcategory": "cybersecurity",
    "summary": "RSS feed summary...",
    "full_content": {
      "text": "Full article text extracted...",
      "author": "John Doe",
      "extraction_method": "trafilatura"
    },
    "extraction_success": true
  }
]
```

## 📊 Fonti Configurate

Breaking News: 1 feed

The Moscow Times
Intelligence & Geopolitics: 12 feed

ASEAN Beat, Asia Times, BleepingComputer, China Power
CyberScoop, Defense One, Diálogo Américas, ECFR
Foreign Affairs POLITICO, Krebs on Security, Security (The Diplomat), SpaceNews
Middle East & North Africa: 3 feed

Al Jazeera English, Middle East Eye, The Jerusalem Post
Defense & Military: 3 feed

Breaking Defense, War on the Rocks, Janes Defence Weekly
Think Tanks: 3 feed

CSIS, Council on Foreign Relations, Chatham House
Americas: 1 feed

Americas Quarterly
Africa: 2 feed

African Arguments, ISS Africa
Tech & Economia: 8 feed

Euronews Business, ECB Press Releases, ECB Monetary Policy
Il Sole 24 ORE, OilPrice, Ars Technica Policy
Supply Chain Dive, Semiconductor Engineering
**Totale: 33 fonti attive**

## 🛠️ Stato Sviluppo

### ✅ Fase 1: Data Ingestion (COMPLETATA)
- [x] Parser RSS multi-fonte (33 feed attivi)
- [x] **Ingestion asincrona** con aiohttp (fetch parallelo di tutti i feed)
- [x] **Estrazione contenuto concorrente** con semaforo (max 10 paralleli)
- [x] Estrazione full-text con Trafilatura + Newspaper3k + Cloudscraper
- [x] Filtro per data (solo articoli ultimi 24h)
- [x] Export JSON con metadata completo
- [x] Deduplicazione automatica (hash + content-based)

### ✅ Fase 2: NLP Processing (COMPLETATA)
- [x] Pulizia e normalizzazione testo (spaCy)
- [x] Named Entity Recognition (PERSON, ORG, GPE, DATE)
- [x] Chunking semantico con overlap (500 words, 50 overlap)
- [x] Embedding generation (384-dim, paraphrase-multilingual-MiniLM-L12-v2)
- [x] Ticker mapping per geopolitical market movers

### ✅ Fase 3: Storage & RAG (COMPLETATA)
- [x] Schema PostgreSQL con pgvector
- [x] Connection pooling e batch inserts
- [x] Semantic search con HNSW index
- [x] Cross-encoder reranking (ms-marco-MiniLM)

### ✅ Fase 4: LLM Report Generation (COMPLETATA)
- [x] Integrazione Google Gemini API (2.5 Flash)
- [x] RAG context retrieval + query expansion
- [x] Prompt engineering strutturato
- [x] Export JSON + Markdown
- [x] Script CLI: `scripts/generate_report.py`

### ✅ Fase 5: Human-in-the-Loop (COMPLETATA)
- [x] Streamlit dashboard per revisione report
- [x] Editor interattivo con diff tracking
- [x] Sistema rating e feedback (1-5 stelle)
- [x] Database schema per report e feedback
- [x] Workflow: Draft → Reviewed → Approved

### ✅ Sprint 3: Trade Signals & Market Intelligence (COMPLETATA)
- [x] **Pipeline Macro-First**: Report → Condense → Signals
- [x] **Trade Signals Extraction**: BULLISH/BEARISH/NEUTRAL per ticker
- [x] **Ticker Whitelist**: 50+ ticker geopoliticamente rilevanti (`config/top_50_tickers.yaml`)
- [x] **Macro Alignment Check**: Segnali article-level con score di allineamento
- [x] **Pydantic Schemas**: Validazione strutturata (`src/llm/schemas.py`)
- [x] **Database Normalizzato**: Tabella `trade_signals` con FK a reports/articles
- [x] **Yahoo Finance Integration**: `src/integrations/market_data.py`
- [x] **Backfill Script**: `scripts/backfill_market_data.py`

### 🔄 Fase 6: Automazione (PROSSIMA)
- [ ] Scheduler per esecuzione giornaliera (cron/systemd)
- [ ] Email automation per distribuzione report
- [ ] Alert system per eventi critici
- [ ] Dashboard monitoring e analytics

## 📖 Note di Sviluppo

### Gestione degli URL RSS

Alcuni feed potrebbero richiedere aggiustamenti:
- **Euronews**: Verifica l'URL RSS per contenuti testuali (non solo video)
- **Il Sole 24 ORE**: L'URL potrebbe variare, controlla la sezione RSS del sito

### Estrazione Testo Completo

Il sistema usa due strategie:
1. **Trafilatura** (primario): Ottimizzato per articoli di news
2. **Newspaper3k** (fallback): Più generico ma meno accurato

Alcuni siti potrebbero richiedere autenticazione o bloccare lo scraping.

### Performance

- **Parsing RSS (async)**: ~30-60 secondi per tutti i 33 feed in parallelo (aiohttp)
- **Estrazione Full Text (concurrent)**: ~60-90 secondi per batch (10 paralleli con semaforo)
- **Batch completo**: ~2-3 minuti (vs ~20-30 minuti sequenziale)

## 🤝 Contributi

Questo è un progetto in sviluppo attivo. Per contribuire:

1. Crea un branch per la feature
2. Implementa i cambiamenti
3. Testa con `pytest tests/`
4. Apri una Pull Request

## 📝 Licenza

[Specifica la licenza del progetto]

## 🔗 Risorse

### Documentazione

- [Phase 4: LLM Report Generation](docs/PHASE4_REPORT_GENERATION.md)
- [Phase 5: HITL Dashboard](docs/PHASE5_HITL.md)

### Librerie Utilizzate

- [aiohttp](https://docs.aiohttp.org/) - HTTP client asincrono per fetch parallelo dei feed RSS
- [Trafilatura](https://trafilatura.readthedocs.io/) - Web scraping e text extraction
- [spaCy](https://spacy.io/) - NLP e Named Entity Recognition
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search per PostgreSQL
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [Streamlit](https://streamlit.io/) - Dashboard interattiva HITL
- [Google Gemini](https://ai.google.dev/) - LLM per report generation
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance API per dati mercato
- [Pydantic](https://docs.pydantic.dev/) - Validazione schema Trade Signals

### Tool e Tecnologie

- **Database**: PostgreSQL 14+ con pgvector extension
- **NLP Models**: en_core_web_sm (spaCy), paraphrase-multilingual-MiniLM-L12-v2
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Vector Dimension**: 384-dim embeddings
- **LLM**: Gemini 2.5 Flash (default, free tier)
- **Market Data**: Yahoo Finance (via yfinance)

---

**Status Progetto**: 🟢 Fasi 1-5 + Sprint 3 Completate | 🔄 Fase 6 in Pianificazione

**Ultima modifica**: 2026-02-15
