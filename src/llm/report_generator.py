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

import google.generativeai as genai

from ..storage.database import DatabaseManager
from ..nlp.processing import NLPProcessor
from ..utils.logger import get_logger

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
        model_name: str = "gemini-2.5-flash"
    ):
        """
        Initialize report generator.

        Args:
            db_manager: Database manager instance (creates new if None)
            nlp_processor: NLP processor instance (creates new if None)
            gemini_api_key: Gemini API key (reads from env if None)
            model_name: Gemini model to use
        """
        self.db = db_manager or DatabaseManager()
        self.nlp = nlp_processor or NLPProcessor(
            spacy_model="xx_ent_wiki_sm",
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Configure Gemini
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or parameters")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        logger.info(f"✓ Report generator initialized with {model_name}")

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

        # Default focus areas
        if focus_areas is None:
            focus_areas = [
                "cybersecurity threats and vulnerabilities",
                "geopolitical developments in Asia and Europe",
                "technology and economic trends"
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

        all_rag_results = []
        for query in rag_queries:
            results = self.get_rag_context(query, top_k=rag_top_k)
            all_rag_results.extend(results)

        # Remove duplicates (same chunk_id)
        unique_rag_results = []
        seen_ids = set()
        for result in all_rag_results:
            if result['chunk_id'] not in seen_ids:
                unique_rag_results.append(result)
                seen_ids.add(result['chunk_id'])

        logger.info(f"✓ Retrieved {len(unique_rag_results)} unique historical chunks")

        # Step 3: Format context for LLM
        logger.info(f"\n[STEP 3] Preparing prompt for LLM...")
        recent_articles_text = self.format_recent_articles(recent_articles)
        rag_context_text = self.format_rag_context(unique_rag_results)

        # Step 4: Construct prompt
        prompt = f"""You are an intelligence analyst generating a daily intelligence briefing.

**YOUR TASK:**
Analyze today's news articles and provide a comprehensive intelligence report. Use the historical context to identify trends, connections, and significance of current events.

**FOCUS AREAS:**
{chr(10).join(f"- {area}" for area in focus_areas)}

**REPORT STRUCTURE:**
1. Executive Summary (2-3 paragraphs highlighting the most critical developments)
2. Key Developments by Category:
   - Cybersecurity & Technology
   - Geopolitical Events
   - Economic Trends
3. Trend Analysis (connections with historical context)
4. Actionable Insights (what decision-makers should know)

**GUIDELINES:**
- Be concise but comprehensive
- Prioritize actionable intelligence over general news
- Cite specific articles with [Article N] references
- Connect current events with historical patterns from the context
- Highlight emerging threats and opportunities
- Use professional, analytical tone

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
    # Example usage
    import sys

    # Initialize generator
    generator = ReportGenerator()

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
        save=True,
        output_dir="reports"
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
