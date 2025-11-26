# Phase 5: Human-in-the-Loop (HITL) Dashboard

## Panoramica

La Fase 5 implementa un sistema **Human-in-the-Loop** per la revisione manuale dei report generati dall'LLM. Utilizza **Streamlit** per creare un'interfaccia web interattiva che permette di:

1. **Visualizzare** i report generati dall'LLM
2. **Modificare** il contenuto prima dell'approvazione
3. **Valutare** la qualità dei report (1-5 stelle)
4. **Salvare feedback** per migliorare i futuri prompt
5. **Approvare** le versioni finali

## Architettura

```
┌──────────────────────────────────────────────────────────────┐
│                   HITL Workflow                               │
│                                                               │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐  │
│  │   LLM       │─────▶│   Streamlit  │─────▶│  Database  │  │
│  │  Report     │      │   Dashboard  │      │  (reports, │  │
│  │ Generator   │      │              │      │  feedback) │  │
│  └─────────────┘      └──────────────┘      └────────────┘  │
│         │                     │                     │        │
│         │                     │                     │        │
│    1. Genera            2. Umano               3. Salva     │
│       bozza             revisiona              versione     │
│                         e corregge             finale +     │
│                                               feedback      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Feedback Loop: migliora prompt per futuri report    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Componenti

### 1. Dashboard Streamlit

**Location:** [src/hitl/dashboard.py](../src/hitl/dashboard.py)

**Caratteristiche:**

#### Interfaccia Utente
- **Layout wide** con sidebar per navigazione
- **Design responsive** con colonne e cards
- **Syntax highlighting** per markdown
- **Status badges** colorati (Draft, Revisionato, Approvato)

#### Funzionalità Principali

**A. Generazione Report**
- Pulsante "Genera Nuovo Report" nella sidebar
- Progress indicator durante generazione (10-20 secondi)
- Auto-salvataggio nel database con status 'draft'

**B. Selezione Report**
- Lista cronologica dei report esistenti
- Filtro per status (draft, reviewed, approved)
- Quick info: data, numero articoli, status icon

**C. Visualizzazione Report**
- **Tab 1: Bozza LLM** - Contenuto originale generato (read-only)
- **Tab 2: Versione Finale** - Editor per modifiche
- **Tab 3: Fonti & Feedback** - Articoli, RAG context, storico feedback

**D. Editor Report**
- Text area espandibile per editing
- Live word count e diff con originale
- Rating slider (1-5 stelle)
- Campo note/commenti per feedback
- Pulsanti "Salva Bozza" e "Approva"

**E. Statistiche Database**
- Totale articoli e chunks
- Report generati per status
- Articoli recenti (ultimi 7 giorni)

### 2. Database Schema

**Tabelle Aggiunte:**

#### `reports` Table
Salva i report generati dall'LLM:

```sql
CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    report_date DATE NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE,
    model_used TEXT,
    draft_content TEXT NOT NULL,        -- Bozza LLM
    final_content TEXT,                 -- Versione corretta
    status TEXT DEFAULT 'draft',        -- draft/reviewed/approved
    metadata JSONB,                     -- focus_areas, counts, etc.
    sources JSONB,                      -- Links articoli e chunks
    human_reviewed_at TIMESTAMP,
    human_reviewer TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

#### `report_feedback` Table
Salva il feedback umano per analisi e miglioramento:

```sql
CREATE TABLE report_feedback (
    id SERIAL PRIMARY KEY,
    report_id INTEGER REFERENCES reports(id),
    section_name TEXT,                  -- Sezione modificata
    feedback_type TEXT,                 -- correction/addition/removal/rating
    original_text TEXT,                 -- Testo LLM originale
    corrected_text TEXT,                -- Testo corretto
    comment TEXT,                       -- Note umane
    rating INTEGER CHECK (1-5),         -- Valutazione qualità
    created_at TIMESTAMP
);
```

### 3. Metodi Database Aggiunti

**In [src/storage/database.py](../src/storage/database.py):**

#### `save_report(report: Dict) -> int`
Salva report LLM nel database con status 'draft'.

```python
report_id = db.save_report(report)
# Returns: report ID
```

#### `get_report(report_id: int) -> Dict`
Recupera report completo per visualizzazione.

```python
report = db.get_report(report_id)
# Returns: {id, draft_content, final_content, status, metadata, sources, ...}
```

#### `get_all_reports(limit: int) -> List[Dict]`
Lista tutti i report ordinati per data (più recenti prima).

```python
reports = db.get_all_reports(limit=20)
# Returns: [{id, report_date, status, metadata, ...}, ...]
```

#### `update_report(report_id, final_content, status, reviewer) -> bool`
Aggiorna report con versione corretta e cambia status.

```python
success = db.update_report(
    report_id=123,
    final_content="Report corretto...",
    status='approved',
    reviewer='tommaso.fazzi@example.com'
)
```

#### `save_feedback(report_id, section_name, feedback_type, ...) -> int`
Salva feedback umano per una sezione del report.

```python
feedback_id = db.save_feedback(
    report_id=123,
    section_name="Executive Summary",
    feedback_type='correction',
    original_text="Original LLM text...",
    corrected_text="Corrected text...",
    comment="L'LLM ha omesso un dettaglio importante",
    rating=4
)
```

#### `get_report_feedback(report_id: int) -> List[Dict]`
Recupera tutto il feedback per un report.

```python
feedback = db.get_report_feedback(report_id=123)
# Returns: [{section_name, feedback_type, original_text, corrected_text, ...}, ...]
```

## Workflow Completo

### Step 1: Genera Report Iniziale

```bash
# Opzione A: Via dashboard (consigliato)
./scripts/run_dashboard.sh
# Poi clicca "Genera Nuovo Report" nella UI

# Opzione B: Via CLI
python scripts/generate_report.py
```

Il report viene salvato nel database con:
- `status = 'draft'`
- `draft_content` = testo LLM
- `final_content` = NULL

### Step 2: Revisione Umana

1. Apri la dashboard: `./scripts/run_dashboard.sh`
2. Seleziona il report dalla sidebar
3. Leggi la **Tab "Bozza LLM"**
4. Passa alla **Tab "Versione Finale"**
5. Modifica il testo nell'editor
6. Aggiungi valutazione (1-5 stelle)
7. Scrivi note/commenti (opzionale)

### Step 3: Salvataggio

**Opzione A: Salva come Revisionato**
- Pulsante "💾 Salva Bozza"
- Status → 'reviewed'
- Puoi continuare a modificare

**Opzione B: Approva Direttamente**
- Pulsante "✅ Approva"
- Status → 'approved'
- Versione finale bloccata

Il sistema salva:
- `final_content` = testo modificato
- `human_reviewed_at` = timestamp
- `human_reviewer` = nome revisore
- Record in `report_feedback` con rating e commenti

### Step 4: Utilizzo Feedback (Futuro)

Il feedback salvato può essere usato per:

1. **Analisi Qualità**
   ```sql
   -- Report che richiedono più correzioni
   SELECT r.id, COUNT(f.id) as feedback_count
   FROM reports r
   JOIN report_feedback f ON r.id = f.report_id
   WHERE f.feedback_type = 'correction'
   GROUP BY r.id
   ORDER BY feedback_count DESC;
   ```

2. **Miglioramento Prompt**
   - Identificare sezioni con più correzioni
   - Adattare prompt per quelle sezioni
   - Aggiungere esempi di "good output"

3. **Fine-tuning LLM** (avanzato)
   - Usare `draft_content` + `final_content` come training pairs
   - Dataset: {input: prompt+context, output: final_content}

## Configurazione

### Requisiti

```bash
# Già installati
pip install streamlit

# Aggiunti a requirements.txt
streamlit==1.51.0
```

### Environment Variables

Nessuna nuova variabile richiesta. Usa quelle esistenti:

```bash
# .env
DATABASE_URL=postgresql://user:pass@localhost:5432/intelligence_ita
GEMINI_API_KEY=your_key_here
```

### Configurazione Streamlit (opzionale)

Crea `.streamlit/config.toml` per customizzazione:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
headless = false
enableCORS = false
```

## Utilizzo

### Avvio Dashboard

```bash
# Metodo 1: Script dedicato (consigliato)
./scripts/run_dashboard.sh

# Metodo 2: Direttamente con Streamlit
source venv/bin/activate
streamlit run src/hitl/dashboard.py

# Metodo 3: Con custom port
streamlit run src/hitl/dashboard.py --server.port 8502
```

La dashboard si aprirà automaticamente nel browser all'indirizzo:
**http://localhost:8501**

### Workflow Tipico

**Scenario 1: Revisione Daily Report**

```
09:00 - Cron job genera report automatico
09:15 - Apri dashboard, seleziona report del giorno
09:20 - Leggi bozza LLM, noti alcune imprecisioni
09:25 - Correggi executive summary e una sezione
09:30 - Valuti 4/5, aggiungi nota "Buono ma mancava contesto X"
09:31 - Clicchi "Approva"
09:32 - Report finale pronto per distribuzione
```

**Scenario 2: Analisi Multi-Report**

```
# Genera report per ultimi 7 giorni
python scripts/generate_report.py --days 7

# Apri dashboard
./scripts/run_dashboard.sh

# Revisiona report storico per identificare pattern
# Esempio: L'LLM tende a omettere i link alle fonti originali
# → Aggiorna prompt per enfatizzare citazioni
```

### Esempi di Feedback Utili

**Tipo: Correction**
- Original: "China announced new AI regulations"
- Corrected: "China announced new AI regulations affecting companies with >10M users"
- Comment: "LLM omesso il dettaglio critico sul threshold utenti"

**Tipo: Addition**
- Section: "Geopolitical Events"
- Added: "Menzione importante: accordo India-USA su tech transfer"
- Comment: "LLM non ha incluso questo sviluppo rilevante"

**Tipo: Removal**
- Section: "Economic Trends"
- Removed: "Paragrafo su crypto markets"
- Comment: "Fuori scope per intelligence geopolitica"

**Tipo: Rating**
- Rating: 5/5
- Comment: "Eccellente sintesi, ben strutturato, fonti accurate"

## Database Query Utili

### Report Non Revisionati

```sql
SELECT id, report_date, generated_at,
       metadata->>'recent_articles_count' as articles
FROM reports
WHERE status = 'draft'
ORDER BY report_date DESC;
```

### Report con Più Feedback

```sql
SELECT r.id, r.report_date, COUNT(f.id) as feedback_count,
       AVG(f.rating) as avg_rating
FROM reports r
LEFT JOIN report_feedback f ON r.id = f.report_id
GROUP BY r.id, r.report_date
ORDER BY feedback_count DESC;
```

### Tipi di Feedback Più Comuni

```sql
SELECT feedback_type, COUNT(*) as count
FROM report_feedback
GROUP BY feedback_type
ORDER BY count DESC;
```

### Report Approvati Questa Settimana

```sql
SELECT id, report_date, human_reviewer, 
       final_content
FROM reports
WHERE status = 'approved'
  AND human_reviewed_at > NOW() - INTERVAL '7 days'
ORDER BY report_date DESC;
```

## Features Avanzate

### 1. Diff Visualizer

La dashboard mostra automaticamente le differenze tra bozza e versione finale:
- **Word count** prima e dopo
- **Highlighted changes** (in sviluppo)
- **Section-by-section comparison**

### 2. Feedback Analytics

Nella tab "Fonti & Feedback" puoi vedere:
- Storico completo delle modifiche
- Rating trend nel tempo
- Sezioni che richiedono più correzioni

### 3. Multi-User Support

Il sistema supporta più revisori:
- Campo `human_reviewer` salva chi ha approvato
- Timestamp preciso di ogni revisione
- Audit trail completo nel database

### 4. Export Capabilities

I report approvati possono essere esportati in:
- **Markdown** (già implementato)
- **PDF** (con libreria reportlab - future)
- **Email** (con smtplib - future)

## Estensioni Future

### Phase 5.1: Advanced Editor

- **Rich text editor** (Quill, TinyMCE)
- **Side-by-side diff view** (draft vs edited)
- **Section-level editing** con collapse/expand
- **Comment threads** per feedback collaborativo

### Phase 5.2: Feedback Learning

- **Automated prompt improvement** basato su feedback patterns
- **Few-shot examples** dalle versioni approvate
- **Quality prediction** (ML model per predire se report richiederà correzioni)

### Phase 5.3: Workflow Automation

- **Approval routing** (draft → reviewer → approver)
- **Email notifications** per nuovi report
- **Slack integration** per alert
- **Scheduled exports** dei report approvati

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"

```bash
# Assicurati di essere nel virtual environment
source venv/bin/activate

# Reinstalla streamlit
pip install streamlit

# Verifica installazione
streamlit --version
```

### "Database connection failed"

```bash
# Verifica PostgreSQL in esecuzione
psql -d intelligence_ita -c "SELECT 1;"

# Controlla DATABASE_URL in .env
cat .env | grep DATABASE_URL

# Testa connessione
python -c "from src.storage.database import DatabaseManager; db = DatabaseManager(); print('OK')"
```

### "GEMINI_API_KEY not found"

```bash
# Aggiungi al file .env
echo "GEMINI_API_KEY=your_key_here" >> .env

# Verifica
cat .env | grep GEMINI_API_KEY

# Riavvia dashboard
./scripts/run_dashboard.sh
```

### Dashboard non si apre nel browser

```bash
# Controlla se porta 8501 è occupata
lsof -i :8501

# Usa porta diversa
streamlit run src/hitl/dashboard.py --server.port 8502

# Apri manualmente
open http://localhost:8501
```

### "No recent articles found"

```bash
# Il database deve avere articoli recenti (< 24h)
# Esegui pipeline completa:

# 1. Ingestion
python -m src.ingestion.pipeline

# 2. NLP Processing
python scripts/process_nlp.py

# 3. Load to DB
python scripts/load_to_database.py

# 4. Verifica
psql -d intelligence_ita -c "SELECT COUNT(*) FROM articles WHERE published_date > NOW() - INTERVAL '1 day';"
```

## Shortcut Keys (Streamlit)

- **`R`** - Ricarica app
- **`C`** - Pulisci cache
- **`H`** - Mostra shortcuts
- **`Ctrl+Enter`** - Submit form (quando in text area)

## Best Practices

### Per Revisori

1. **Leggi prima la bozza completa** - Non iniziare a modificare subito
2. **Verifica le fonti** - Controlla che citazioni siano accurate
3. **Mantieni lo stile** - Tono professionale e analitico
4. **Sii specifico nei commenti** - "Aggiunto contesto mancante su X" è meglio di "Migliorato"
5. **Usa rating correttamente**:
   - 5/5 = Perfetto, nessuna modifica
   - 4/5 = Buono, piccole correzioni
   - 3/5 = Accettabile, modifiche moderate
   - 2/5 = Scarso, molte correzioni
   - 1/5 = Inutilizzabile, riscrittura completa

### Per Sviluppatori

1. **Monitora feedback patterns** - Quali sezioni richiedono più correzioni?
2. **Itera sui prompt** - Usa feedback per migliorare prompt LLM
3. **Track metrics** - Tempo di revisione, % modifiche, rating medio
4. **Versioning** - Considera git-like versioning per i report

## API Reference

### Funzioni Principali

```python
# Inizializza dashboard
initialize_session_state()

# Genera nuovo report
report_id = generate_new_report()

# Visualizza selettore report
display_report_selector()

# Visualizza report con editor
display_report_viewer(report)

# Visualizza statistiche
display_statistics()

# Get status badge HTML
badge_html = get_status_badge('approved')
```

## Performance

### Caricamento Dashboard
- **First load**: ~2-3 secondi (connessione DB + cache models)
- **Report selection**: ~100-200ms (query database)
- **Edit actions**: Istantaneo (session state)
- **Save/Approve**: ~50-100ms (INSERT/UPDATE)

### Generazione Report da Dashboard
- **Database queries**: ~100ms
- **Embedding generation**: ~200ms
- **LLM generation**: ~5-15 secondi
- **Save to DB**: ~50ms
- **Total**: ~6-16 secondi

### Scalabilità
- **Concurrent users**: 10+ (connection pooling)
- **Reports in DB**: Testato con 100+
- **Memory usage**: ~200MB (Streamlit + models)

## Security Considerations

### Autenticazione (Future)

Per production, aggiungi autenticazione:

```python
# Streamlit-authenticator
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'intelligence_dashboard',
    'auth_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show dashboard
    main()
elif authentication_status == False:
    st.error('Username/password errati')
```

### Access Control

Considera RBAC (Role-Based Access Control):
- **Analyst**: Può generare e visualizzare report
- **Reviewer**: Può modificare e salvare bozze
- **Approver**: Può approvare versioni finali
- **Admin**: Accesso completo + statistiche

### Data Privacy

- **Database**: Usa SSL per connessione PostgreSQL
- **API Keys**: Mai hardcodare, sempre da .env
- **Logs**: Non loggare contenuto sensibile dei report
- **Backups**: Cifra backup del database

## Testing

### Test Manuale

```bash
# 1. Avvia dashboard
./scripts/run_dashboard.sh

# 2. Genera report di test
# Clicca "Genera Nuovo Report" nella UI

# 3. Testa editing
# Modifica testo, salva, verifica database:
psql -d intelligence_ita -c "SELECT id, status, final_content IS NOT NULL FROM reports ORDER BY id DESC LIMIT 1;"

# 4. Testa feedback
# Aggiungi rating e commento, verifica:
psql -d intelligence_ita -c "SELECT * FROM report_feedback ORDER BY id DESC LIMIT 1;"

# 5. Testa approvazione
# Approva report, verifica status:
psql -d intelligence_ita -c "SELECT id, status, human_reviewed_at FROM reports WHERE status = 'approved' ORDER BY id DESC LIMIT 1;"
```

### Test Automatico (Future)

```python
# tests/test_hitl.py
def test_report_workflow():
    db = DatabaseManager()
    
    # Generate mock report
    report = {'report_text': 'Test report', 'metadata': {}, 'sources': {}}
    report_id = db.save_report(report)
    assert report_id is not None
    
    # Update with human edits
    success = db.update_report(
        report_id, 
        final_content="Edited test report",
        status='approved',
        reviewer='test@example.com'
    )
    assert success
    
    # Verify feedback
    feedback_id = db.save_feedback(
        report_id, 
        section_name="Test",
        feedback_type='correction',
        rating=5
    )
    assert feedback_id is not None
```

## Metrics da Monitorare

### Quality Metrics
- **Average rating** per report
- **% reports con modifiche** (final ≠ draft)
- **Tempo medio revisione** (human_reviewed_at - generated_at)
- **Feedback per category** (quale sezione richiede più correzioni)

### Efficiency Metrics
- **Reports generated per day**
- **Reports approved per day**
- **Backlog** (draft reports non revisionati)
- **Reviewer productivity** (report/hour)

### Improvement Metrics
- **Rating trend** (migliora nel tempo?)
- **Correction rate trend** (diminuisce?)
- **Time to approval trend** (più veloce?)

## Next Steps

Dopo Phase 5, considera:

1. **Phase 6: Automation**
   - Cron job per generazione giornaliera
   - Email automation per distribuzione report
   - Alert system per eventi critici

2. **Phase 7: Advanced Analytics**
   - Trend analysis dashboard
   - Entity tracking over time
   - Topic modeling su report storici

3. **Phase 8: Multi-Language**
   - Support per report in italiano
   - Translation layer per fonti multilingua

## License

MIT License - see [LICENSE](../LICENSE)
