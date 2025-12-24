"""
LLM Report Generator with RAG

Generates daily intelligence reports using:
- Recent articles from database (last 24h)
- Historical context from semantic search (RAG)
- Google Gemini LLM for report generation
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import google.generativeai as genai
from pydantic import ValidationError

from ..storage.database import DatabaseManager
from ..nlp.processing import NLPProcessor
from ..utils.logger import get_logger
from .schemas import IntelligenceReportMVP, IntelligenceReport

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates intelligence reports using LLM with RAG context.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        nlp_processor: Optional[NLPProcessor] = None,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        enable_query_expansion: bool = True,
        expansion_variants: int = 2,
        dedup_similarity: float = 0.98,
        enable_reranking: bool = True,
        reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranking_top_k: int = 15
    ):
        """
        Initialize report generator.

        Args:
            db_manager: Database manager instance (creates new if None)
            nlp_processor: NLP processor instance (creates new if None)
            gemini_api_key: Gemini API key (reads from env if None)
            model_name: Gemini model to use
            enable_query_expansion: Enable automatic query expansion for RAG
            expansion_variants: Number of query variants to generate per focus area
            dedup_similarity: Similarity threshold for chunk deduplication (0-1)
            enable_reranking: Enable Cross-Encoder reranking for better precision
            reranking_model: Cross-Encoder model to use for reranking
            reranking_top_k: Number of top chunks to keep after reranking
        """
        self.db = db_manager or DatabaseManager()
        self.nlp = nlp_processor or NLPProcessor(
            spacy_model="xx_ent_wiki_sm",
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Query expansion configuration
        self.enable_query_expansion = enable_query_expansion
        self.expansion_variants = expansion_variants
        self.dedup_similarity = dedup_similarity

        # Reranking configuration
        self.enable_reranking = enable_reranking
        self.reranking_top_k = reranking_top_k

        # Lazy load Cross-Encoder (only if enabled)
        if self.enable_reranking:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranking_model)
            logger.info(f"  Reranking: ENABLED (model: {reranking_model}, top_k: {reranking_top_k})")
        else:
            self.reranker = None

        # Configure Gemini
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or parameters")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # Load ticker whitelist for Trade Signal context
        self.ticker_whitelist = self._load_ticker_whitelist()
        if self.ticker_whitelist:
            total_tickers = sum(len(tickers) for tickers in self.ticker_whitelist.values())
            logger.info(f"  Ticker whitelist: {total_tickers} tickers loaded across {len(self.ticker_whitelist)} categories")

        logger.info(f"✓ Report generator initialized with {model_name}")
        if enable_query_expansion:
            logger.info(f"  Query expansion: ENABLED ({expansion_variants} variants, dedup threshold: {dedup_similarity})")

    def _load_ticker_whitelist(self) -> Dict[str, List[str]]:
        """
        Load top 50 ticker mappings from config/top_50_tickers.yaml

        Returns:
            Dict with structure: {
                'defense': ['LMT', 'RTX', 'NOC', ...],
                'semiconductors': ['TSM', 'NVDA', 'INTC', ...],
                ...
            }
        """
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / 'config' / 'top_50_tickers.yaml'

        if not config_path.exists():
            logger.warning(f"Ticker config not found at {config_path}, using empty whitelist")
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                ticker_config = yaml.safe_load(f)

            # Flatten to ticker-only list for prompt context
            tickers_by_category = {}
            all_tickers = []

            for category, companies in ticker_config.items():
                category_tickers = []
                for company in companies:
                    ticker = company['ticker']
                    category_tickers.append(ticker)
                    all_tickers.append(ticker)
                tickers_by_category[category] = category_tickers

            logger.debug(f"Loaded {len(all_tickers)} tickers across {len(tickers_by_category)} categories")
            return tickers_by_category

        except Exception as e:
            logger.error(f"Failed to load ticker whitelist: {e}")
            return {}

    def _format_ticker_whitelist(self) -> str:
        """
        Format ticker whitelist for prompt context

        Returns:
            Formatted string with tickers organized by category
        """
        if not self.ticker_whitelist:
            return "No ticker whitelist loaded"

        lines = []
        for category, tickers in self.ticker_whitelist.items():
            category_name = category.replace('_', ' ').title()
            lines.append(f"- {category_name}: {', '.join(tickers)}")

        return "\n".join(lines)

    def get_rag_context(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relevant historical context using RAG (semantic search).

        Args:
            query: Search query (e.g., "cybersecurity threats in Asia")
            top_k: Number of relevant chunks to retrieve
            category: Optional category filter

        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"Searching for RAG context: '{query}'")

        # Generate embedding for query
        query_embedding = self.nlp.embedding_model.encode(query).tolist()

        # Semantic search in database
        results = self.db.semantic_search(
            query_embedding=query_embedding,
            top_k=top_k,
            category=category
        )

        logger.info(f"✓ Found {len(results)} relevant chunks (similarity threshold applied)")
        return results

    def expand_rag_queries(self, queries: List[str]) -> List[str]:
        """
        Expand RAG queries using LLM to generate semantic variants.

        For each query, generates N variant sub-queries exploring different angles
        (economic, geopolitical, technological, etc.) to improve retrieval coverage.

        Args:
            queries: Original list of focus area queries

        Returns:
            Expanded list containing original queries + valid variants
        """
        if not self.enable_query_expansion:
            logger.info("Query expansion disabled - using original queries")
            return queries

        logger.info(f"Expanding {len(queries)} queries into {self.expansion_variants} variants each")
        expanded = []

        for query in queries:
            # Always include original query
            expanded.append(query)

            try:
                # Generate variant queries with Gemini Flash
                prompt = f"""Generate {self.expansion_variants} semantic variants of this intelligence query.

Original Query: "{query}"

Create {self.expansion_variants} alternative phrasings that explore different angles (economic impact, geopolitical implications, technological aspects, etc.) while maintaining the core intelligence focus.

Requirements:
- Each variant must be 5-15 words
- Must be related to: {query}
- Different perspective/angle from original
- Suitable for semantic search

Output ONLY the {self.expansion_variants} variant queries, one per line, without numbering or additional text."""

                response = self.model.generate_content(prompt)
                variants = response.text.strip().split('\n')

                # Filter and validate variants
                valid_variants = []
                for variant in variants:
                    variant = variant.strip()
                    # Remove numbering if present
                    if variant and variant[0].isdigit():
                        variant = variant.split('.', 1)[-1].strip()

                    # Validate length (5-15 words)
                    word_count = len(variant.split())
                    if 5 <= word_count <= 15 and variant.lower() != query.lower():
                        valid_variants.append(variant)

                # Limit to requested number of variants
                valid_variants = valid_variants[:self.expansion_variants]

                expanded.extend(valid_variants)
                logger.info(f"  '{query}' → +{len(valid_variants)} variants")

            except Exception as e:
                logger.warning(f"Query expansion failed for '{query}': {e} - using original only")
                continue

        logger.info(f"✓ Query expansion: {len(queries)} → {len(expanded)} total queries")
        return expanded

    def deduplicate_chunks_advanced(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Advanced deduplication using embedding similarity.

        Removes duplicate chunks based on:
        1. Exact chunk_id duplicates
        2. High embedding similarity (cosine > threshold)

        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and embeddings

        Returns:
            Deduplicated list of chunks
        """
        if not chunks:
            return []

        logger.info(f"Deduplicating {len(chunks)} chunks (threshold: {self.dedup_similarity})")

        # Step 1: Remove exact ID duplicates
        seen_ids = set()
        unique_by_id = []
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_by_id.append(chunk)

        if len(unique_by_id) < len(chunks):
            logger.info(f"  Removed {len(chunks) - len(unique_by_id)} exact ID duplicates")

        # Step 2: Similarity-based deduplication
        # Get embeddings from database for each chunk
        deduplicated = []
        for i, chunk in enumerate(unique_by_id):
            is_duplicate = False

            # Compare with already accepted chunks
            for accepted_chunk in deduplicated:
                # Calculate cosine similarity between embeddings
                # Note: chunks from DB should have embeddings available
                # If not available in chunk dict, we skip similarity check
                chunk_embedding = chunk.get('embedding')
                accepted_embedding = accepted_chunk.get('embedding')

                if chunk_embedding is not None and accepted_embedding is not None:
                    # Convert to numpy arrays for cosine similarity
                    vec1 = np.array(chunk_embedding)
                    vec2 = np.array(accepted_embedding)

                    # Cosine similarity
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                    if similarity > self.dedup_similarity:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(chunk)

        similarity_removed = len(unique_by_id) - len(deduplicated)
        if similarity_removed > 0:
            logger.info(f"  Removed {similarity_removed} similar chunks (cosine > {self.dedup_similarity})")

        logger.info(f"✓ Deduplication: {len(chunks)} → {len(deduplicated)} chunks")
        return deduplicated

    def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using Cross-Encoder for better precision.

        Uses bi-directional attention to score query-chunk pairs,
        providing more accurate relevance than cosine similarity alone.

        Args:
            query: Original search query
            chunks: List of chunks to rerank (from vector search)
            top_k: Number of top chunks to return

        Returns:
            Top-k reranked chunks with 'rerank_score' added
        """
        if not self.reranker or not chunks:
            return chunks[:top_k]

        logger.info(f"Reranking {len(chunks)} chunks with Cross-Encoder...")

        # Prepare pairs for Cross-Encoder: [(query, chunk_text), ...]
        pairs = []
        for chunk in chunks:
            # Chunks from database have 'content' field, not 'text'
            chunk_text = chunk.get('content', chunk.get('text', ''))
            if not chunk_text:  # Skip empty chunks
                chunk_text = ''
            pairs.append([query, chunk_text])

        # Get reranking scores (batch processing)
        scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)

        # Attach scores to chunks (handle NaN values)
        import math
        for i, chunk in enumerate(chunks):
            score = float(scores[i])
            # Replace NaN with 0.0 (lowest score)
            chunk['rerank_score'] = score if not math.isnan(score) else 0.0

        # Sort by rerank score (descending)
        reranked = sorted(chunks, key=lambda x: x.get('rerank_score', 0.0), reverse=True)

        # Log score distribution
        if reranked:
            top_score = reranked[0].get('rerank_score', 0.0)
            bottom_score = reranked[-1].get('rerank_score', 0.0)
            median_score = reranked[len(reranked)//2].get('rerank_score', 0.0)
            logger.info(
                f"✓ Reranked: scores range [{bottom_score:.3f} - {top_score:.3f}], "
                f"median: {median_score:.3f}"
            )

        return reranked[:top_k]

    def filter_relevant_articles(
        self,
        articles: List[Dict],
        focus_areas: List[str],
        top_n: int = 60,
        min_similarity: float = 0.30,
        min_fallback: int = 10
    ) -> List[Dict]:
        """
        Filter articles by relevance using cosine similarity with quality threshold.

        Implements a two-stage filtering approach:
        1. Quality Gate: Only articles with similarity >= min_similarity
        2. Quantity Limit: Take top N from those that passed quality gate
        3. Safety Net: If fewer than min_fallback articles pass, take top min_fallback regardless

        Args:
            articles: List of articles with embeddings
            focus_areas: List of focus area strings
            top_n: Maximum number of articles to return (default: 60)
            min_similarity: Minimum cosine similarity threshold (default: 0.30)
            min_fallback: Minimum articles to return even if below threshold (default: 10)

        Returns:
            Filtered list of relevant articles (between min_fallback and top_n)
        """
        import numpy as np

        if not articles:
            return []

        logger.info(f"Filtering {len(articles)} articles by relevance to focus areas...")
        logger.info(f"Parameters: top_n={top_n}, min_similarity={min_similarity}, min_fallback={min_fallback}")

        # Generate query embedding from focus areas
        query_text = " ".join(focus_areas)
        query_embedding = self.nlp.embedding_model.encode(query_text)

        # Calculate similarity for each article
        articles_with_similarity = []
        no_embedding_count = 0

        for article in articles:
            # Get article's full text embedding
            full_text_embedding = article.get('full_text_embedding')

            if full_text_embedding is None:
                no_embedding_count += 1
                logger.debug(f"Article '{article.get('title', 'Unknown')}' has no embedding, skipping")
                continue

            # Convert to numpy array if needed
            if isinstance(full_text_embedding, list):
                full_text_embedding = np.array(full_text_embedding)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, full_text_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(full_text_embedding)
            )

            articles_with_similarity.append({
                'article': article,
                'similarity': float(similarity)
            })

        if no_embedding_count > 0:
            logger.warning(f"{no_embedding_count} articles had no embeddings and were skipped")

        if not articles_with_similarity:
            logger.error("No articles with embeddings found for filtering")
            return []

        # Sort by similarity (descending)
        articles_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)

        # Log similarity distribution
        similarities = [x['similarity'] for x in articles_with_similarity]
        logger.info(f"Similarity distribution - Min: {min(similarities):.3f}, Max: {max(similarities):.3f}, "
                   f"Mean: {np.mean(similarities):.3f}, Median: {np.median(similarities):.3f}")

        # Stage 1: Filter by quality threshold
        above_threshold = [x for x in articles_with_similarity if x['similarity'] >= min_similarity]
        below_threshold_count = len(articles_with_similarity) - len(above_threshold)

        if below_threshold_count > 0:
            logger.info(f"Filtered out {below_threshold_count} articles below similarity threshold {min_similarity}")

        # Stage 2: Apply quantity limit (with fallback safety net)
        if len(above_threshold) >= min_fallback:
            # Normal path: take top N from articles above threshold
            selected_articles = above_threshold[:top_n]
            logger.info(f"✓ Selected {len(selected_articles)} articles from {len(above_threshold)} above threshold")

            # Warning if we're using fewer articles than expected
            if len(selected_articles) < 30:
                logger.warning(f"⚠ LOW RELEVANCE: Only {len(selected_articles)} articles met quality threshold. "
                             f"This suggests limited relevant news today.")
        else:
            # Fallback path: not enough articles above threshold, take top min_fallback regardless
            selected_articles = articles_with_similarity[:min_fallback]
            logger.warning(f"⚠ FALLBACK MODE ACTIVATED: Only {len(above_threshold)} articles above threshold {min_similarity}. "
                          f"Using emergency fallback: top {min_fallback} articles regardless of quality.")

        # Log final selection details
        if selected_articles:
            similarity_range = f"{selected_articles[-1]['similarity']:.3f} to {selected_articles[0]['similarity']:.3f}"
            avg_similarity = np.mean([x['similarity'] for x in selected_articles])
            logger.info(f"Final selection: {len(selected_articles)} articles "
                       f"(similarity range: {similarity_range}, avg: {avg_similarity:.3f})")

        return [item['article'] for item in selected_articles]

    def format_rag_context(self, rag_results: List[Dict]) -> str:
        """
        Format RAG search results into readable context for LLM.

        Args:
            rag_results: Results from semantic_search

        Returns:
            Formatted string with historical context
        """
        if not rag_results:
            return "No relevant historical context found."

        context_parts = []
        context_parts.append("=== RELEVANT HISTORICAL CONTEXT ===\n")

        for i, result in enumerate(rag_results, 1):
            pub_date = result.get('published_date', 'Unknown date')
            if pub_date and pub_date != 'Unknown date':
                pub_date = pub_date.strftime('%Y-%m-%d') if hasattr(pub_date, 'strftime') else str(pub_date)

            context_parts.append(
                f"\n[{i}] {result['title']}\n"
                f"Source: {result['source']} | Date: {pub_date} | "
                f"Category: {result['category']} | Similarity: {result['similarity']:.3f}\n"
                f"Relevant excerpt:\n{result['content']}\n"
                f"Link: {result['link']}"
            )

        return "\n".join(context_parts)

    def format_recent_articles(self, articles: List[Dict]) -> str:
        """
        Format recent articles for LLM prompt.

        Args:
            articles: List of recent articles from database

        Returns:
            Formatted string with recent news
        """
        if not articles:
            return "No recent articles found."

        formatted_parts = []
        formatted_parts.append("=== TODAY'S NEWS ARTICLES ===\n")

        for i, article in enumerate(articles, 1):
            pub_date = article.get('published_date', 'Unknown date')
            if pub_date and pub_date != 'Unknown date':
                pub_date = pub_date.strftime('%Y-%m-%d %H:%M') if hasattr(pub_date, 'strftime') else str(pub_date)

            entities = article.get('entities', {})
            entity_summary = []
            for entity_type in ['PERSON', 'ORG', 'GPE']:
                if entity_type in entities and entities[entity_type]:
                    top_entities = entities[entity_type][:3]  # Top 3 of each type
                    entity_summary.append(f"{entity_type}: {', '.join(top_entities)}")

            formatted_parts.append(
                f"\n[Article {i}]\n"
                f"Title: {article['title']}\n"
                f"Source: {article['source']} | Date: {pub_date} | Category: {article.get('category', 'N/A')}\n"
                f"Summary: {article.get('summary', 'No summary available')}\n"
            )

            if entity_summary:
                formatted_parts.append(f"Key entities: {' | '.join(entity_summary)}\n")

            # Include full text (truncated if too long)
            full_text = article.get('full_text', '')
            if full_text:
                if len(full_text) > 2000:
                    formatted_parts.append(f"Full text (excerpt): {full_text[:2000]}...\n")
                else:
                    formatted_parts.append(f"Full text: {full_text}\n")

            formatted_parts.append(f"Link: {article['link']}\n")

        return "\n".join(formatted_parts)

    def generate_structured_analysis(
        self,
        article_text: str,
        article_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON analysis for a single article (Sprint 2.1 MVP)

        Uses Gemini JSON mode with Pydantic validation for type-safe output.
        This is a NEW method separate from generate_report() to allow isolated testing.

        Args:
            article_text: Full text content of article
            article_metadata: Optional metadata (title, source, date, entities)

        Returns:
            Dictionary with:
            - success: bool (True if validation passed)
            - structured: dict (validated JSON output) if success=True
            - validation_errors: list of errors if success=False
            - raw_llm_output: str (original Gemini response for debugging)
        """
        logger.info("Generating structured analysis with JSON mode...")

        # Prepare metadata context (if provided)
        metadata_context = ""
        if article_metadata:
            metadata_parts = []
            if 'title' in article_metadata:
                metadata_parts.append(f"Title: {article_metadata['title']}")
            if 'source' in article_metadata:
                metadata_parts.append(f"Source: {article_metadata['source']}")
            if 'published_date' in article_metadata:
                metadata_parts.append(f"Date: {article_metadata['published_date']}")
            if 'entities' in article_metadata and article_metadata['entities']:
                # Format entities nicely
                entities = article_metadata['entities']
                if isinstance(entities, dict) and 'by_type' in entities:
                    entities_str = []
                    for etype, names in entities['by_type'].items():
                        if names:
                            entities_str.append(f"{etype}: {', '.join(names[:5])}")
                    metadata_parts.append(f"Key Entities: {' | '.join(entities_str)}")

            if metadata_parts:
                metadata_context = "\n".join(metadata_parts) + "\n\n"

        # Construct system instruction with JSON schema
        system_instruction = """You are a Senior Intelligence Analyst specializing in geopolitical risk assessment and investment implications.

Your task: Analyze the article and provide a structured intelligence assessment in JSON format.

OUTPUT REQUIREMENTS:
- Respond ONLY with valid JSON matching the schema below
- Use BLUF (Bottom Line Up Front) style for executive_summary
- Be concise but substantive (100-300 words for summary)
- Confidence score reflects certainty of your analysis (not article quality)

JSON SCHEMA (all fields REQUIRED):
{
  "title": "string (5-15 words, descriptive article title)",
  "category": "GEOPOLITICS | DEFENSE | ECONOMY | CYBER | ENERGY | OTHER",
  "executive_summary": "string (BLUF-style summary, 100-300 words)",
  "sentiment_label": "POSITIVE | NEUTRAL | NEGATIVE (investment/security outlook)",
  "confidence_score": float (0.0-1.0, your confidence in the assessment)
}

CATEGORY DEFINITIONS:
- GEOPOLITICS: Tensions, alliances, territorial disputes, diplomatic events
- DEFENSE: Military tech, weapons systems, defense spending, armed conflicts
- ECONOMY: Markets, trade, sanctions, economic policy, financial institutions
- CYBER: Cyberattacks, data breaches, espionage, critical infrastructure
- ENERGY: Oil, gas, renewables, OPEC, energy security, pipelines
- OTHER: Does not fit above categories clearly

SENTIMENT GUIDELINES:
- POSITIVE: Events likely to benefit markets, reduce risks, or improve stability
- NEGATIVE: Events increasing risks, market uncertainty, or instability
- NEUTRAL: Informational updates without clear directional impact

CONFIDENCE SCORE:
- 0.9-1.0: High confidence (verified facts, multiple sources)
- 0.7-0.8: Medium confidence (single source, or emerging story)
- 0.5-0.6: Low confidence (rumors, conflicting information, or highly speculative)

EXAMPLE OUTPUT:
{
  "title": "China Deploys Naval Forces Near Taiwan Strait",
  "category": "GEOPOLITICS",
  "executive_summary": "BLUF: China conducted large-scale naval exercises 100km from Taiwan coast on Dec 15, deploying 15 warships including 3 Type 055 destroyers. This represents the largest show of force since August 2024. Taiwan's defense ministry reports no direct incursions into territorial waters but increased surveillance flights. US 7th Fleet is monitoring. Investment implications: Heightened geopolitical risk premium likely for Taiwan-based semiconductor manufacturers (TSMC) and regional defense contractors. Short-term volatility expected in Asia-Pacific equity markets.",
  "sentiment_label": "NEGATIVE",
  "confidence_score": 0.85
}

Now analyze the article below and respond with JSON only:"""

        # User prompt with article content
        user_prompt = f"""Article to analyze:

{metadata_context}{article_text}

Respond with JSON analysis following the schema above:"""

        try:
            # Call Gemini with JSON mode
            response = self.model.generate_content(
                contents=[system_instruction, user_prompt],
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.3,  # Lower temperature for more consistent JSON
                }
            )

            raw_output = response.text
            logger.debug(f"Raw LLM output: {raw_output[:200]}...")

            # Validate with Pydantic
            try:
                validated_report = IntelligenceReportMVP.model_validate_json(raw_output)
                logger.info("✅ Pydantic validation PASSED")

                return {
                    'success': True,
                    'structured': validated_report.model_dump(),
                    'raw_llm_output': raw_output,
                    'validation_errors': []
                }

            except ValidationError as e:
                logger.warning(f"⚠️ Pydantic validation FAILED: {e}")
                # Fallback: Return raw parsed JSON with errors flagged
                try:
                    raw_json = json.loads(raw_output)
                except json.JSONDecodeError as json_err:
                    raw_json = {"error": "Invalid JSON", "raw": raw_output[:500]}

                return {
                    'success': False,
                    'validation_errors': [str(err) for err in e.errors()],
                    'raw_llm_output': raw_output,
                    'parsed_attempt': raw_json
                }

        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'validation_errors': [],
                'raw_llm_output': None
            }

    def generate_full_analysis(
        self,
        article_text: str,
        article_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate full schema analysis with Trade Signals (Sprint 2.2)

        Uses Gemini JSON mode with full Pydantic schema validation.
        Includes: Impact Score, Sentiment, Trade Signals, Key Entities, Markdown content.

        Args:
            article_text: Full text content of article
            article_metadata: Optional metadata (title, source, date, entities)

        Returns:
            Dictionary with:
            - success: bool (True if validation passed)
            - structured: dict (validated IntelligenceReport) if success=True
            - validation_errors: list of errors if success=False
            - raw_llm_output: str (original Gemini response for debugging)
        """
        logger.info("Generating FULL schema analysis with Trade Signals...")

        # Prepare metadata context (same as MVP)
        metadata_context = ""
        if article_metadata:
            metadata_parts = []
            if 'title' in article_metadata:
                metadata_parts.append(f"Title: {article_metadata['title']}")
            if 'source' in article_metadata:
                metadata_parts.append(f"Source: {article_metadata['source']}")
            if 'published_date' in article_metadata:
                metadata_parts.append(f"Date: {article_metadata['published_date']}")
            if 'entities' in article_metadata and article_metadata['entities']:
                entities = article_metadata['entities']
                if isinstance(entities, dict) and 'by_type' in entities:
                    entities_str = []
                    for etype, names in entities['by_type'].items():
                        if names:
                            entities_str.append(f"{etype}: {', '.join(names[:5])}")
                    metadata_parts.append(f"Key Entities: {' | '.join(entities_str)}")

            if metadata_parts:
                metadata_context = "\n".join(metadata_parts) + "\n\n"

        # Prepare ticker context from whitelist
        ticker_context = self._format_ticker_whitelist()

        # System instruction with full JSON schema
        system_instruction = f"""You are a Senior Investment Strategist specializing in Geopolitical Risk and Market Intelligence.

TASK: Analyze the article and provide a comprehensive intelligence assessment with ACTIONABLE TRADE SIGNALS.

OUTPUT REQUIREMENTS:
- Respond ONLY with valid JSON matching the schema below
- Use Markdown formatting: **bold** for key entities, [Article N] for citations
- Trade Signals: Only mention tickers if DIRECTLY relevant to article events
- Impact Score: Rate event severity (0=noise, 10=systemic crisis)
- Be concise but substantive (150-400 words for executive_summary)

JSON SCHEMA (all fields REQUIRED):
{{
  "title": "string (5-15 words, descriptive title)",
  "category": "GEOPOLITICS | DEFENSE | ECONOMY | CYBER | ENERGY",
  "impact": {{
    "score": integer (0-10, event severity),
    "reasoning": "string (why this score, 1-2 sentences)"
  }},
  "sentiment": {{
    "label": "POSITIVE | NEUTRAL | NEGATIVE",
    "score": float (-1.0 to +1.0, sentiment polarity)
  }},
  "key_entities": ["string", ...] (top 5-10 organizations, people, locations),
  "related_tickers": [
    {{
      "ticker": "string (e.g., 'LMT', 'TSM')",
      "signal": "BULLISH | BEARISH | NEUTRAL | WATCHLIST",
      "timeframe": "SHORT_TERM | MEDIUM_TERM | LONG_TERM",
      "rationale": "string (specific catalyst, 1-2 sentences)"
    }}
  ],
  "executive_summary": "string (BLUF-style summary with **markdown** formatting)",
  "analysis_content": "string (full markdown analysis with ## headings)",
  "confidence_score": float (0.0-1.0, your confidence in this analysis)
}}

CATEGORY DEFINITIONS:
- GEOPOLITICS: Tensions, alliances, territorial disputes, diplomatic events
- DEFENSE: Military tech, weapons systems, defense spending, armed conflicts
- ECONOMY: Markets, trade, sanctions, economic policy, financial institutions
- CYBER: Cyberattacks, data breaches, espionage, critical infrastructure
- ENERGY: Oil, gas, renewables, OPEC, energy security, pipelines

IMPACT SCORE (0-10):
- 0-2: Noise (routine diplomatic statement, minor local incident)
- 3-4: Noteworthy (significant development, limited geographic scope)
- 5-6: Important (regional crisis, major policy shift)
- 7-8: Critical (high escalation risk, global market impact)
- 9-10: Systemic (war, financial crisis, critical infrastructure failure)

SENTIMENT GUIDELINES:
- POSITIVE (+0.3 to +1.0): Events reducing risks, improving stability, bullish for markets
- NEUTRAL (-0.2 to +0.2): Informational, no clear directional impact
- NEGATIVE (-1.0 to -0.3): Events increasing risks, uncertainty, or instability

TRADE SIGNAL RULES:
1. **Only use tickers from the whitelist below** - DO NOT invent tickers
2. **Timeframe definitions**:
   - SHORT_TERM: <3 months (immediate tactical positioning)
   - MEDIUM_TERM: 3-12 months (quarterly earnings impact)
   - LONG_TERM: >1 year (structural shifts, multi-year trends)
3. **Signal types**:
   - BULLISH: Clear positive catalyst (contracts, earnings, favorable policy)
   - BEARISH: Clear negative catalyst (sanctions, loss of market, regulation)
   - NEUTRAL: No strong directional bias but worth monitoring
   - WATCHLIST: Potential future impact, awaiting catalyst
4. **Rationale must be SPECIFIC**: Include concrete numbers, contract values, dates, causal links

TICKER WHITELIST (ONLY use these):
{ticker_context}

If article does NOT mention any ticker-relevant events, return empty array for related_tickers.

MARKDOWN FORMATTING:
- Use **bold** for: Company names, key people, critical locations
- Use [Article N] format for source citations (if multiple articles)
- Use ## headings for analysis_content sections
- Example: "**Taiwan Semiconductor (TSM)** reported record Q4 earnings [Article 3]..."

CONFIDENCE SCORE:
- 0.9-1.0: High confidence (verified facts, multiple sources, clear causality)
- 0.7-0.8: Medium confidence (single source, emerging story, some uncertainty)
- 0.5-0.6: Low confidence (rumors, conflicting info, speculative analysis)

EXAMPLE OUTPUT:
{{
  "title": "China Naval Exercises Near Taiwan Escalate Tensions",
  "category": "GEOPOLITICS",
  "impact": {{
    "score": 7,
    "reasoning": "Largest PLA Navy deployment since 2024, high risk of miscalculation in contested waters"
  }},
  "sentiment": {{
    "label": "NEGATIVE",
    "score": -0.65
  }},
  "key_entities": ["China", "Taiwan", "US 7th Fleet", "TSMC", "Xi Jinping"],
  "related_tickers": [
    {{
      "ticker": "TSM",
      "signal": "BEARISH",
      "timeframe": "SHORT_TERM",
      "rationale": "Geopolitical risk premium spike; potential supply chain disruption fears impacting semiconductor sector valuations"
    }},
    {{
      "ticker": "LMT",
      "signal": "BULLISH",
      "timeframe": "MEDIUM_TERM",
      "rationale": "Increased demand for Aegis defense systems and F-35 fighter jets from Taiwan and regional allies (Japan, Australia) likely"
    }}
  ],
  "executive_summary": "BLUF: **China's People's Liberation Army Navy** deployed 15 warships including 3 Type 055 destroyers 100km from **Taiwan Strait** on Dec 15, marking the largest show of force since August 2024. **Taiwan's Ministry of Defense** confirmed no territorial water incursions but reported increased surveillance flights. **US 7th Fleet** is monitoring closely. Investment implications: Short-term volatility expected for **Taiwan Semiconductor (TSM)** and Asia-Pacific equities due to heightened geopolitical risk premium. Defense contractors like **Lockheed Martin (LMT)** and **Raytheon (RTX)** positioned to benefit from increased regional procurement.",
  "analysis_content": "## Military Deployment Details\\n\\nChina's naval exercise represents a significant escalation...\\n\\n## Market Impact Analysis\\n\\nTaiwan Semiconductor faces immediate valuation pressure...",
  "confidence_score": 0.85
}}

Now analyze the article below and respond with JSON only:"""

        # User prompt with article content
        user_prompt = f"""Article to analyze:

{metadata_context}{article_text}

Respond with JSON analysis following the full schema above:"""

        try:
            # Call Gemini with JSON mode
            # CRITICAL: Use temperature 0.2 (NOT 0.3) for analytical consistency
            response = self.model.generate_content(
                contents=[system_instruction, user_prompt],
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,  # Lower than MVP for trade signal precision
                    # NO max_output_tokens - let it generate full content
                }
                # Optional: Add timeout if 504 errors occur
                # request_options={"timeout": 600}
            )

            raw_output = response.text
            logger.debug(f"Raw LLM output: {raw_output[:200]}...")

            # Validate with Pydantic (IntelligenceReport full schema)
            try:
                validated_report = IntelligenceReport.model_validate_json(raw_output)
                logger.info("✅ Pydantic validation PASSED (Full Schema)")

                # Log extracted trade signals for visibility
                signals = validated_report.related_tickers
                if signals:
                    logger.info(f"  💰 Trade Signals: {len(signals)} extracted")
                    for sig in signals:
                        logger.info(f"     {sig.ticker}: {sig.signal} ({sig.timeframe})")
                else:
                    logger.info("  ℹ️  No trade signals (article not ticker-relevant)")

                return {
                    'success': True,
                    'structured': validated_report.model_dump(),
                    'raw_llm_output': raw_output,
                    'validation_errors': []
                }

            except ValidationError as e:
                logger.warning(f"⚠️ Pydantic validation FAILED: {e}")
                # Fallback: Return raw parsed JSON with errors flagged
                try:
                    raw_json = json.loads(raw_output)
                except json.JSONDecodeError as json_err:
                    raw_json = {"error": "Invalid JSON", "raw": raw_output[:500]}

                return {
                    'success': False,
                    'validation_errors': [str(err) for err in e.errors()],
                    'raw_llm_output': raw_output,
                    'parsed_attempt': raw_json
                }

        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'validation_errors': [],
                'raw_llm_output': None
            }

    def generate_report(
        self,
        focus_areas: Optional[List[str]] = None,
        days: int = 1,
        rag_queries: Optional[List[str]] = None,
        rag_top_k: int = 5,
        top_articles: int = 60,
        min_similarity: float = 0.30,
        min_fallback: int = 10
    ) -> Dict[str, Any]:
        """
        Generate intelligence report with RAG context.

        Args:
            focus_areas: List of topics to focus on (e.g., ["cybersecurity", "geopolitics"])
            days: Number of days to look back for recent articles
            rag_queries: Custom RAG search queries. If None, auto-generates from focus areas
            rag_top_k: Number of historical chunks per RAG query
            top_articles: Maximum number of top relevant articles to include (default: 60)
            min_similarity: Minimum cosine similarity threshold for relevance (default: 0.30)
            min_fallback: Minimum articles to return even if below threshold (default: 10)

        Returns:
            Dictionary with report content and metadata
        """
        logger.info("=" * 80)
        logger.info("GENERATING INTELLIGENCE REPORT")
        logger.info("=" * 80)

        # Default focus areas - aligned with feed coverage
        if focus_areas is None:
            focus_areas = [
                "cybersecurity threats, data breaches, and critical infrastructure vulnerabilities",
                "geopolitical tensions and power dynamics in Indo-Pacific region (China, Taiwan, ASEAN)",
                "Middle East conflicts, security developments, and regional stability (Israel, Iran, Arab states)",
                "defense technology, military procurement, and strategic weapons systems",
                "global supply chain disruptions, semiconductor industry, and critical materials",
                "energy markets, OPEC dynamics, and transition to renewables",
                "European Union policy, Russia-NATO relations, and transatlantic security",
                "space industry developments, satellite technology, and dual-use applications",
                "Africa security challenges, conflicts, and great power competition",
                "Latin America political developments and China's influence in the region",
                "economic policy shifts, central bank decisions, and financial market trends"
            ]

        # Step 1: Get recent articles
        logger.info(f"\n[STEP 1] Fetching articles from last {days} day(s)...")
        all_recent_articles = self.db.get_recent_articles(days=days)
        logger.info(f"✓ Retrieved {len(all_recent_articles)} recent articles")

        if not all_recent_articles:
            logger.warning("No recent articles found. Cannot generate report.")
            return {
                'success': False,
                'error': 'No recent articles available',
                'timestamp': datetime.now().isoformat()
            }

        # Step 1b: Filter articles by relevance to focus areas
        logger.info(f"\n[STEP 1b] Filtering articles by relevance...")
        recent_articles = self.filter_relevant_articles(
            articles=all_recent_articles,
            focus_areas=focus_areas,
            top_n=top_articles,
            min_similarity=min_similarity,
            min_fallback=min_fallback
        )

        if not recent_articles:
            logger.warning("No relevant articles found after filtering. Cannot generate report.")
            return {
                'success': False,
                'error': 'No relevant articles found',
                'timestamp': datetime.now().isoformat()
            }

        # Step 2: Get RAG context
        logger.info(f"\n[STEP 2] Retrieving historical context via RAG...")

        # Auto-generate RAG queries from focus areas if not provided
        if rag_queries is None:
            rag_queries = focus_areas

        # Step 2a: Expand queries (if enabled)
        expanded_queries = self.expand_rag_queries(rag_queries)

        # Step 2b: Execute RAG searches
        all_rag_results = []
        # Increase top_k if reranking is enabled (cast wider net)
        search_top_k = rag_top_k * 2 if self.enable_reranking else rag_top_k

        for query in expanded_queries:
            results = self.get_rag_context(query, top_k=search_top_k)
            all_rag_results.extend(results)

        logger.info(f"✓ Retrieved {len(all_rag_results)} total chunks from RAG")

        # Step 2c: Advanced deduplication (ID + similarity)
        unique_rag_results = self.deduplicate_chunks_advanced(all_rag_results)

        # Step 2d: Reranking (if enabled)
        if self.enable_reranking and unique_rag_results and rag_queries:
            # Rerank using the original (first) query
            primary_query = rag_queries[0]
            unique_rag_results = self._rerank_chunks(
                query=primary_query,
                chunks=unique_rag_results,
                top_k=self.reranking_top_k * len(rag_queries)  # Scale by number of queries
            )

        logger.info(f"✓ Final RAG context: {len(unique_rag_results)} unique historical chunks")

        # Step 3: Format context for LLM
        logger.info(f"\n[STEP 3] Preparing prompt for LLM...")
        recent_articles_text = self.format_recent_articles(recent_articles)
        rag_context_text = self.format_rag_context(unique_rag_results)

        # Step 4: Construct prompt
        prompt = f"""You are an intelligence analyst generating a daily intelligence briefing.

**YOUR TASK:**
Analyze today's news articles and provide a comprehensive intelligence report focused on strategic relevance and actionable investment implications. Prioritize events that represent breaking points in existing trends and competition between major powers, even in seemingly peripheral regions.

**FOCUS AREAS:**
{chr(10).join(f"- {area}" for area in focus_areas)}

**PRIORITIZATION FRAMEWORK:**
Before writing the report, score each event using this system and prioritize those with highest scores:

1. Immediate Impact (0-3 points): Does this event immediately affect national security, financial markets, or critical infrastructure? A cyberattack on a power grid scores 3, a generic diplomatic statement scores 0.

2. Escalation Potential (0-3 points): Can this event rapidly degenerate? A military incident in a contested zone scores 3, a peaceful local protest scores 0. Always ask: can this trigger a chain reaction?

3. Critical Actor Involvement (0-2 points): Are nuclear states, major economies, or actors controlling strategic resources involved? China, USA, Russia, EU score 2, peripheral countries without significant alliances score 0 (st martin island).

4. Break from Historical Pattern (0-2 points): Does this event represent a rupture with recent history? If it breaks a 5+ year pattern, it scores 2; if it confirms existing trends, it scores 0. Example: Russia and Ukraine negotiating after two years of refusal scores 2.

5. Long-Term Strategic Relevance (0-3 points + bonus): Does this event involve control of critical resources, trade routes, or positioning in great power competition even if it seems peripheral today? Think five to ten years ahead, not just six months.

**SPECIAL RULE - PERIPHERAL STRATEGIC EVENTS:**
Assign a bonus of 2 additional points to events involving great power competition in regions considered "peripheral" but strategically positioned: Myanmar, East Africa (Djibouti, Horn of Africa), Central Asia, Arctic, small Pacific island states. Even if immediate impact seems low, these events reveal long-term dynamics in global geopolitical repositioning. When you identify such events, explicitly explain why the geographic position or resources involved amplify importance beyond surface appearance.

Events scoring 8-10 points require priority analysis. Events scoring 4-7 go in standard report. Events below 4 can be briefly mentioned.

**REPORT STRUCTURE:**

1. Executive Summary (200-300 words)
Highlight the most critical developments with focus on strategic breaks and shifts in great power dynamics.

2. Key Developments by Category (150-200 words each):
   - Cybersecurity 
   - Technology
   - Geopolitical Events 
   - Economic events

For each development, always identify specific actors (individuals, organizations, governments, groups), explain their motivations and causal relationships. Avoid impersonal language: instead of "tensions are rising," say "Russia and NATO are escalating tensions because..." Provide relationship context: explain how actors relate to each other (allies, adversaries, dependencies).

3. Trend Analysis (250-300 words)
Connect current events with historical patterns from the context. Identify whether events confirm or break from existing trends.

4. Actionable Insights: Investment Implications

For each significant development, provide a three-level structured analysis that portfolio managers can use immediately:

**Level 1 - Direct Beneficiaries (immediate exposure):**
Identify companies with direct exposure seeing impact on balance sheets within 1-2 quarters. Specify:
- Company names with exact tickers
- Contract size or revenue impact
- Specific catalysts with concrete numbers
Example: "Long defense contractors Lockheed Martin (LMT), Raytheon (RTX), Northrop Grumman (NOC) based on $4.2B Pentagon contract for THAAD systems announced today [Article 12]. Delivery scheduled Q2-Q3 2025, with potential extensions if Taiwan increases orders (likely scenario given Chinese military exercises last week). Monitor: LMT earnings call January 15 for 2025 guidance."

**Level 2 - Supply Chain & Correlated Markets:**
Trace the complete causal chain. A geopolitical event rarely hits only one isolated sector. If China blocks rare earth exports, analyze not just alternative producers like Lynas (ASX:LYC) or MP Materials (MP), but also permanent magnet manufacturers, EV makers depending on those magnets, utilities that ordered wind turbines that won't arrive on schedule. Map: geopolitical event → input shortage → companies with alternative inventory → companies revising guidance → end markets facing delays.

**Level 3 - Macro Market Impacts:**
Consider impacts on currencies, government bonds, and commodities. If India raises fertilizer tariffs in retaliation against Canada, this affects not just fertilizer producers but also agricultural futures, US farm loans, and Canadian dollar weakening from reduced exports. If Myanmar becomes a proxy battlefield between US and China, this shifts capital flows toward safe-haven assets, strengthens Japanese yen, and increases credit default swap premiums on ASEAN countries perceived as next on the list.

For each insight, always include: specific tickers, concrete catalysts with exact figures, timing, causal connections with other events, and next catalysts to monitor.

**ADDITIONAL GUIDELINES:**
- Cite specific articles with [Article N] references
- Use professional, analytical tone
- Prioritize events that are strategic break points, not just high-volume news
- When information is unverified or conflicting, use confidence indicators (High/Medium/Low) and cite multiple sources
- Never use generic language like "the sector could benefit" - always specify which companies, why, with what catalyst, and in what timeframe

---
{recent_articles_text}

---

{rag_context_text}

---

**Now generate the intelligence report:**
"""

        # Step 5: Generate report with Gemini
        logger.info(f"\n[STEP 4] Generating report with Gemini...")
        try:
            response = self.model.generate_content(prompt)
            report_text = response.text
            logger.info(f"✓ Report generated successfully ({len(report_text)} characters)")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

        # Step 6: Compile results
        report = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'report_text': report_text,
            'metadata': {
                'focus_areas': focus_areas,
                'recent_articles_count': len(recent_articles),
                'historical_chunks_count': len(unique_rag_results),
                'days_covered': days,
                'model_used': self.model.model_name
            },
            'sources': {
                'recent_articles': [
                    {
                        'title': a['title'],
                        'link': a['link'],
                        'source': a['source'],
                        'published_date': a['published_date'].isoformat() if hasattr(a['published_date'], 'isoformat') else str(a['published_date'])
                    }
                    for a in recent_articles
                ],
                'historical_context': [
                    {
                        'title': r['title'],
                        'link': r['link'],
                        'similarity': r['similarity']
                    }
                    for r in unique_rag_results
                ]
            }
        }

        logger.info("\n✓ Report generation complete")
        return report

    def save_report(self, report: Dict[str, Any], output_dir: str = "reports") -> Path:
        """
        Save report to file.

        Args:
            report: Report dictionary from generate_report()
            output_dir: Directory to save reports

        Returns:
            Path to saved report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"intelligence_report_{timestamp}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Report saved to: {report_file}")

        # Also save markdown version for easy reading
        md_file = output_path / f"intelligence_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Intelligence Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(report['report_text'])
            f.write("\n\n---\n\n")
            f.write(f"**Generated by:** {report['metadata']['model_used']}\n")
            f.write(f"**Sources:** {report['metadata']['recent_articles_count']} recent articles, "
                   f"{report['metadata']['historical_chunks_count']} historical chunks\n")

        logger.info(f"✓ Markdown version saved to: {md_file}")

        return report_file

    def run_daily_report(
        self,
        focus_areas: Optional[List[str]] = None,
        save: bool = True,
        save_to_db: bool = True,
        output_dir: str = "reports",
        top_articles: int = 60,
        min_similarity: float = 0.30,
        min_fallback: int = 10
    ) -> Dict[str, Any]:
        """
        Run complete daily report generation pipeline.

        Args:
            focus_areas: Topics to focus on
            save: Whether to save report to file
            save_to_db: Whether to save report to database (for HITL review)
            output_dir: Directory for saved reports
            top_articles: Maximum number of top relevant articles to include (default: 60)
            min_similarity: Minimum cosine similarity threshold (default: 0.30)
            min_fallback: Minimum articles to return even if below threshold (default: 10)

        Returns:
            Report dictionary with added 'report_id' if saved to database
        """
        logger.info("Starting daily intelligence report generation...")

        # Generate report
        report = self.generate_report(
            focus_areas=focus_areas,
            days=1,  # Last 24 hours
            rag_top_k=5,  # Top 5 historical chunks per focus area
            top_articles=top_articles,
            min_similarity=min_similarity,
            min_fallback=min_fallback
        )

        if not report['success']:
            logger.error(f"Report generation failed: {report.get('error')}")
            return report

        # Save to database (for HITL review)
        if save_to_db:
            report_id = self.db.save_report(report)
            if report_id:
                report['report_id'] = report_id
                logger.info(f"✓ Report saved to database with ID: {report_id}")
            else:
                logger.warning("Failed to save report to database")

        # Save to file
        if save:
            self.save_report(report, output_dir=output_dir)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("REPORT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {report['timestamp']}")
        logger.info(f"Recent articles analyzed: {report['metadata']['recent_articles_count']}")
        logger.info(f"Historical context chunks: {report['metadata']['historical_chunks_count']}")
        logger.info(f"Report length: {len(report['report_text'])} characters")

        if 'report_id' in report:
            logger.info(f"Database ID: {report['report_id']}")
            logger.info(f"Review at: http://localhost:8501 (run ./scripts/run_dashboard.sh)")

        return report


if __name__ == "__main__":
    import sys
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate intelligence report with optional query expansion"
    )

    parser.add_argument(
        '--no-query-expansion',
        action='store_true',
        help='Disable automatic query expansion for RAG (default: enabled)'
    )

    parser.add_argument(
        '--expansion-variants',
        type=int,
        default=2,
        help='Number of query variants to generate per focus area (default: 2)'
    )

    parser.add_argument(
        '--dedup-similarity',
        type=float,
        default=0.98,
        help='Similarity threshold for chunk deduplication, range 0-1 (default: 0.98)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Directory to save reports (default: reports)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save report to file (default: saves to file)'
    )

    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Do not save report to database (default: saves to DB)'
    )

    args = parser.parse_args()

    # Initialize generator with query expansion settings
    generator = ReportGenerator(
        enable_query_expansion=not args.no_query_expansion,
        expansion_variants=args.expansion_variants,
        dedup_similarity=args.dedup_similarity
    )

    # Custom focus areas (optional)
    focus_areas = [
        "cybersecurity threats and data breaches",
        "artificial intelligence developments",
        "geopolitical tensions in Asia and Middle East",
        "economic policy changes in Europe"
    ]

    # Generate and save report
    report = generator.run_daily_report(
        focus_areas=focus_areas,
        save=not args.no_save,
        save_to_db=not args.no_db,
        output_dir=args.output_dir
    )

    if report['success']:
        print("\n" + "=" * 80)
        print("GENERATED REPORT")
        print("=" * 80)
        print(report['report_text'])
        print("\n" + "=" * 80)
        sys.exit(0)
    else:
        print(f"\nError: {report.get('error')}")
        sys.exit(1)
