# Config Context

## Purpose
YAML-based configuration files that define the system's data sources, financial instrument mappings, and NER quality filters. These configurations control what data enters the pipeline and how entities are processed.

## Architecture Role
Configuration layer that parameterizes the ingestion and NLP modules. Changes to these files affect data collection scope (feeds), financial signal extraction (tickers), and entity quality (blocklist) without code modifications.

## Key Files

- `feeds.yaml` - RSS feed source definitions (~33 feeds)
  - Categories: `breaking_news`, `intelligence`, `tech_economy`, `security`
  - Subcategories: `geopolitics`, `cybersecurity`, `defense`, `middle_east`, `americas`, `africa`, `think_tank`, `semiconductors`, etc.
  - Each feed: `name`, `url`, `category`, `subcategory`, `region`
  - Sources include: The Diplomat, Defense One, Al Jazeera, CSIS, CFR, ECB, OilPrice, Semiconductor Engineering

- `top_50_tickers.yaml` - Geopolitical market mover mappings
  - Sectors: `defense`, `semiconductors`, `energy`, `finance_macro`, `cyber_tech`
  - Each entry: `name`, `ticker`, `exchange`, `aliases`
  - Examples: LMT (Lockheed Martin), TSM (TSMC), XOM (ExxonMobil), JPM (JPMorgan)
  - Used by LLM to extract trade signals from geopolitical news

- `entity_blocklist.yaml` - NER false positive filters
  - Categories: `temporal` (days, months in IT/EN/ES/FR), `vague_references` ("The Government", "Officials"), `media_artifacts` (Reuters, BREAKING), `numbers_years`, `pronouns_articles`, `generic_words`, `nationality_adjectives`, `generic_roles`
  - `valid_acronyms` - Protected terms (UN, EU, NATO, G7, G20)
  - Used by entity cleaning scripts to filter garbage from knowledge graph

## Dependencies

- **Internal**: Used by `src/ingestion/`, `src/nlp/`, `src/llm/`, `scripts/`
- **External**: PyYAML for parsing

## Data Flow

- **Input**: Static configuration (edited manually)
- **Output**:
  - Feed URLs → `src/ingestion/feed_parser.py`
  - Ticker mappings → `src/llm/report_generator.py` for trade signals
  - Blocklist → `scripts/clean_entities.py`, `scripts/deep_clean_entities.py`
