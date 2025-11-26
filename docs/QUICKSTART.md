# Quickstart Guide - INTELLIGENCE_ITA

Guida rapida per iniziare ad usare il sistema di intelligence news analysis.

## Prerequisiti

1. **Python 3.9+** installato
2. **PostgreSQL 14+** in esecuzione
3. **Gemini API Key** (gratuito su https://makersuite.google.com/app/apikey)

## Setup Rapido (5 minuti)

```bash
# 1. Clone e setup ambiente
git clone <repository-url>
cd INTELLIGENCE_ITA
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download modello NLP inglese
python -m spacy download en_core_web_sm

# 3. Setup database
psql -U postgres -c "CREATE DATABASE intelligence_ita;"
psql -d intelligence_ita -c "CREATE EXTENSION vector;"

# 4. Configura environment
cp .env.example .env
# Modifica .env con le tue credenziali:
#   DATABASE_URL=postgresql://user:pass@localhost:5432/intelligence_ita
#   GEMINI_API_KEY=your_key_here

# 5. Inizializza schema database
python scripts/load_to_database.py --init-only
```

## Prima Esecuzione (10 minuti)

```bash
# 1. Raccogli articoli ultimi 24h (2-3 min)
python -m src.ingestion.pipeline

# 2. Processa con NLP (3-4 min)
python scripts/process_nlp.py

# 3. Carica nel database (1 min)
python scripts/load_to_database.py

# 4. Genera report intelligence (10-15 sec)
python scripts/generate_report.py

# 5. Apri dashboard per revisione
./scripts/run_dashboard.sh
```

Il browser si aprirà automaticamente su http://localhost:8501

## Workflow Giornaliero

### Metodo 1: Manuale con Dashboard

```bash
# Avvia dashboard
./scripts/run_dashboard.sh

# Nella UI:
# 1. Clicca "➕ Genera Nuovo Report"
# 2. Attendi 10-20 secondi
# 3. Revisiona la bozza generata dall'LLM
# 4. Modifica se necessario
# 5. Valuta qualità (1-5 stelle)
# 6. Clicca "✅ Approva"
```

### Metodo 2: CLI Step-by-Step

```bash
# 1. Ingestion (nuovi articoli)
python -m src.ingestion.pipeline

# 2. NLP Processing
python scripts/process_nlp.py

# 3. Load to Database
python scripts/load_to_database.py

# 4. Generate Report
python scripts/generate_report.py

# 5. Il report è salvato in reports/intelligence_report_YYYYMMDD_HHMMSS.md
cat reports/intelligence_report_*.md | tail -n 100
```

### Metodo 3: One-Liner (Futuro - Fase 6)

```bash
# Script unificato (da implementare in Fase 6)
./scripts/daily_pipeline.sh
```

## Struttura Output

```
INTELLIGENCE_ITA/
├── data/
│   ├── articles_20251125_090000.json       # Raw ingestion
│   └── articles_nlp_20251125_090500.json   # NLP processed
├── reports/
│   ├── intelligence_report_20251125_091000.json  # Structured
│   └── intelligence_report_20251125_091000.md    # Readable
└── logs/
    └── app_20251125.log                    # System logs
```

## Risoluzione Problemi Comuni

### "No articles found in last 24h"

Alcuni feed potrebbero non pubblicare tutti i giorni. Prova:
```bash
# Espandi a 3 giorni
python -m src.ingestion.pipeline
# Poi in Python:
from src.ingestion.pipeline import IngestionPipeline
pipeline = IngestionPipeline()
articles = pipeline.run(max_age_days=3)
```

### "GEMINI_API_KEY not found"

```bash
# Verifica .env
cat .env | grep GEMINI_API_KEY

# Se vuoto, aggiungi:
echo "GEMINI_API_KEY=your_key_here" >> .env

# Riavvia script
python scripts/generate_report.py
```

### "Database connection failed"

```bash
# Verifica PostgreSQL in esecuzione
pg_isready

# Testa connessione
psql -d intelligence_ita -c "SELECT 1;"

# Verifica .env
cat .env | grep DATABASE_URL
```

### "pgvector extension not found"

```bash
# Su macOS con Homebrew PostgreSQL
brew install pgvector

# Su Linux/Docker
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Poi abilita extension
psql -d intelligence_ita -c "CREATE EXTENSION vector;"
```

### Dashboard non si apre

```bash
# Verifica Streamlit installato
streamlit --version

# Se manca
pip install streamlit

# Porta occupata? Usa altra porta
streamlit run src/hitl/dashboard.py --server.port 8502
```

## FAQ

**Q: Quanto tempo richiede l'esecuzione giornaliera?**
A: ~5-10 minuti totali:
- Ingestion: 2-3 min
- NLP: 3-4 min
- Database: <1 min
- Report: 10-15 sec
- Review: 5-10 min (umano)

**Q: Quanti articoli raccoglie al giorno?**
A: Dipende dalle fonti, tipicamente 100-200 articoli nelle 24h.

**Q: Il sistema funziona solo in inglese?**
A: Attualmente sì (modello en_core_web_sm). Gli articoli RSS sono principalmente in inglese. L'embedding model è multilingua ma ottimizzato per inglese.

**Q: Posso usare un altro LLM oltre a Gemini?**
A: Sì, modifica `src/llm/report_generator.py` per usare OpenAI o Anthropic. Il codice è modulare.

**Q: Il feedback HITL migliora automaticamente i prompt?**
A: Non ancora. Il feedback è salvato nel DB ma l'integrazione automatica nei prompt è pianificata per una fase futura.

**Q: Posso eseguire il sistema su un server senza interfaccia grafica?**
A: Sì, la dashboard Streamlit può girare in modalità headless. Accedi via browser remoto.

**Q: Come aggiungo nuove fonti RSS?**
A: Modifica `config/feeds.yaml` aggiungendo nuove entries con URL e categoria.

## Performance Benchmark

Test su MacBook Air M1 con PostgreSQL locale:

| Fase | Articoli | Tempo | Note |
|------|----------|-------|------|
| Ingestion (24h) | 134 | 2m 15s | 23 feed, extract_content=True |
| NLP Processing | 134 | 3m 45s | Embeddings + chunking |
| Database Load | 134 + 183 chunks | 0.8s | Batch insert con pooling |
| Report Gen | 1 report | 12s | Gemini 1.5 Flash, 5 RAG queries |
| HITL Review | 1 report | 5-10m | Dipende da revisore |

**Total automation time**: ~6-7 minuti (senza HITL review)

## Costi Operativi

### Completamente Gratuito
- **Gemini 1.5 Flash**: 15 req/min, 1M tokens/day (free tier)
- **NLP Models**: Locali (spaCy, Sentence Transformers)
- **Database**: PostgreSQL locale
- **Streamlit**: Open source

### Costo con Scale-up
Se superi free tier Gemini:
- **Gemini 1.5 Flash**: $0.075 per 1M input tokens
- **Report giornaliero**: ~10K tokens = $0.00075 per report
- **Mensile**: ~$0.02/mese per 30 report

**Praticamente gratis anche a scala.**

## Support

Per problemi o domande:
1. Controlla [docs/PHASE5_HITL.md](docs/PHASE5_HITL.md) sezione Troubleshooting
2. Verifica logs in `logs/app_YYYYMMDD.log`
3. Apri issue su GitHub repository

## License

MIT License - see LICENSE file
