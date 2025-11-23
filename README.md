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
│  (25+ sources)  │     │  + Full Text     │     │  + Embeddings   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Daily Report   │◀────│  LLM Generation  │◀────│  Vector DB RAG  │
│  (Human Review) │     │  (Gemini/GPT)    │     │  (pgvector)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                                 ▲
         └─────────────── Feedback Loop ──────────────────┘
```

## 📂 Struttura del Progetto

```
INTELLIGENCE_ITA/
├── config/
│   └── feeds.yaml              # Configurazione feed RSS (25+ fonti)
├── src/
│   ├── ingestion/              # Moduli di acquisizione dati
│   │   ├── feed_parser.py      # Parser RSS/Atom multi-fonte
│   │   ├── content_extractor.py # Estrazione testo completo
│   │   └── pipeline.py         # Pipeline orchestrata
│   ├── nlp/                    # [TODO] Elaborazione NLP
│   │   ├── preprocessor.py     # Pulizia e normalizzazione
│   │   └── embeddings.py       # Generazione embedding vettoriali
│   ├── storage/                # [TODO] Database e persistenza
│   │   └── database.py         # PostgreSQL + pgvector
│   ├── llm/                    # [TODO] Generazione report
│   │   └── report_generator.py # Integrazione LLM (Gemini/GPT)
│   └── utils/
│       └── logger.py           # Sistema di logging
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
python -m spacy download it_core_news_lg

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

## 🔧 Utilizzo

### Fase 1: Ingestione Dati (Attualmente Implementata)

Il sistema può già raccogliere notizie da 25+ fonti configurate in `config/feeds.yaml`:

```bash
# Test parsing feed RSS (solo metadata)
python -m src.ingestion.pipeline

# Estrazione completa con full-text (più lento)
python -c "
from src.ingestion.pipeline import IngestionPipeline
pipeline = IngestionPipeline()
articles = pipeline.run(extract_content=True)
"
```

**Filtraggio per categoria:**
```python
from src.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

# Solo notizie di intelligence/geopolitica
articles = pipeline.run(category='intelligence')

# Solo economia e tech
articles = pipeline.run(category='tech_economy')
```

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

### Breaking News (2)
- The Moscow Times
- Times of India

### Intelligence & Geopolitics (13)
- ASEAN Beat, Asia Times
- BleepingComputer, CyberScoop
- Defense One, Krebs on Security
- NYT International, ECFR
- The Diplomat (Security, China Power)
- SpaceNews

### Tech & Economia (8)
- Euronews Business
- ECB Press Releases & Monetary Policy
- Handelsblatt, Il Sole 24 ORE
- OilPrice, Ars Technica Policy

**Totale: 23 fonti attive**

## 🛠️ Prossimi Sviluppi

### Fase 2: NLP Processing (Prossimo Step)
- [ ] Pulizia e normalizzazione testo (spaCy)
- [ ] Named Entity Recognition (NER) per categorizzazione
- [ ] Generazione embedding con Sentence Transformers
- [ ] Salvataggio in database vettoriale (pgvector)

### Fase 3: Storage & RAG
- [ ] Schema database PostgreSQL
- [ ] Implementazione ricerca semantica
- [ ] Sistema di categorizzazione automatica

### Fase 4: LLM Report Generation
- [ ] Prompt engineering per insight cross-settoriali
- [ ] Integrazione API LLM (Gemini/GPT)
- [ ] Template report giornaliero

### Fase 5: Human-in-the-Loop (HITL)
- [ ] Interfaccia web (Streamlit) per revisione
- [ ] Sistema di feedback e correzioni
- [ ] Loop di miglioramento continuo

### Fase 6: Automazione
- [ ] Scheduler per esecuzione giornaliera
- [ ] Sistema di notifiche
- [ ] Dashboard di monitoraggio

## 📖 Note di Sviluppo

### Gestione degli URL RSS

Alcuni feed potrebbero richiedere aggiustamenti:
- **Euronews**: Verifica l'URL RSS per contenuti testuali (non solo video)
- **Il Sole 24 ORE**: L'URL potrebbe variare, controlla la sezione RSS del sito
- **NYT**: Richiede subscription per alcuni contenuti

### Estrazione Testo Completo

Il sistema usa due strategie:
1. **Trafilatura** (primario): Ottimizzato per articoli di news
2. **Newspaper3k** (fallback): Più generico ma meno accurato

Alcuni siti potrebbero richiedere autenticazione o bloccare lo scraping.

### Performance

- **Parsing RSS**: ~1-2 secondi per feed
- **Estrazione Full Text**: ~2-5 secondi per articolo
- **Batch completo (23 fonti, ~500 articoli/giorno)**: ~20-30 minuti

## 🤝 Contributi

Questo è un progetto in sviluppo attivo. Per contribuire:

1. Crea un branch per la feature
2. Implementa i cambiamenti
3. Testa con `pytest tests/`
4. Apri una Pull Request

## 📝 Licenza

[Specifica la licenza del progetto]

## 🔗 Risorse

- [Trafilatura Documentation](https://trafilatura.readthedocs.io/)
- [spaCy Italian Models](https://spacy.io/models/it)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Sentence Transformers](https://www.sbert.net/)

---

**Status Progetto**: 🟡 Fase 1 Completata (Data Ingestion) - In sviluppo Fase 2 (NLP)

**Ultima modifica**: 2024-01-15
