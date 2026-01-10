"""
Story Manager - Narrative Engine Core

Groups articles into storylines (ongoing narratives) using:
1. Sequential matching (hybrid vector + entity for incremental processing)
2. Batch clustering (DBSCAN on embeddings for initial/periodic clustering)

Enables "delta reporting" - only report what's new in each story.
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
from psycopg2.extras import Json

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..storage.database import DatabaseManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StoryManager:
    """
    Manages storylines - narrative threads that group related articles over time.

    Key concepts:
    - Storyline: An ongoing narrative (e.g., "Taiwan Semiconductor Tensions")
    - Matching: Deciding if a new article belongs to existing storyline(s)
    - Vector Drift: Storyline's semantic signature evolves with new articles
    - Decay: Inactive storylines lose momentum and eventually archive
    """

    # Thresholds (tunable)
    SIMILARITY_THRESHOLD = 0.72          # Min cosine similarity to match
    ENTITY_BOOST = 0.08                  # Bonus per 50%+ entity overlap
    DRIFT_WEIGHT_OLD = 0.90              # Weight for existing embedding
    DRIFT_WEIGHT_NEW = 0.10              # Weight for new article embedding
    MOMENTUM_DECAY_FACTOR = 0.7          # Weekly decay multiplier
    DORMANT_THRESHOLD = 0.3              # Below this = DORMANT status
    ARCHIVE_DAYS = 30                    # Days in DORMANT before ARCHIVED

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize StoryManager.

        Args:
            db_manager: Database manager instance (uses singleton if None)
        """
        self.db = db_manager or DatabaseManager()
        logger.info("StoryManager initialized")

    def find_matching_storylines(
        self,
        article_embedding: List[float],
        article_entities: List[str],
        category: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find storylines that match a new article.

        Uses hybrid matching:
        1. Vector similarity on current_embedding
        2. Entity overlap boost on key_entities

        Args:
            article_embedding: Article's embedding vector (384 dims)
            article_entities: List of entity names from article (GPE, ORG, PERSON)
            category: Optional category filter
            top_k: Max storylines to return

        Returns:
            List of matching storylines with scores
        """
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                # Vector similarity search on active storylines
                query = """
                    SELECT
                        id,
                        title,
                        summary,
                        key_entities,
                        category,
                        article_count,
                        1 - (current_embedding <=> %s::vector) AS similarity
                    FROM storylines
                    WHERE status = 'ACTIVE'
                """
                params = [article_embedding]

                if category:
                    query += " AND category = %s"
                    params.append(category)

                query += """
                    ORDER BY current_embedding <=> %s::vector
                    LIMIT %s
                """
                params.extend([article_embedding, top_k * 2])  # Get more, filter later

                cur.execute(query, params)
                candidates = cur.fetchall()

        # Calculate hybrid scores
        matches = []
        article_entities_set = set(e.lower() for e in article_entities)

        for row in candidates:
            storyline_id, title, summary, key_entities, cat, article_count, similarity = row

            # Skip if below threshold
            if similarity < self.SIMILARITY_THRESHOLD - self.ENTITY_BOOST:
                continue

            # Calculate entity overlap
            storyline_entities = set(e.lower() for e in (key_entities or []))
            if storyline_entities and article_entities_set:
                overlap = len(article_entities_set & storyline_entities)
                overlap_ratio = overlap / max(len(storyline_entities), 1)
            else:
                overlap_ratio = 0

            # Hybrid score: vector similarity + entity boost
            entity_boost = self.ENTITY_BOOST if overlap_ratio >= 0.5 else 0
            final_score = similarity + entity_boost

            if final_score >= self.SIMILARITY_THRESHOLD:
                matches.append({
                    'storyline_id': storyline_id,
                    'title': title,
                    'summary': summary,
                    'category': cat,
                    'article_count': article_count,
                    'similarity': similarity,
                    'entity_overlap': overlap_ratio,
                    'final_score': final_score
                })

        # Sort by final score and limit
        matches.sort(key=lambda x: x['final_score'], reverse=True)
        return matches[:top_k]

    def create_storyline(
        self,
        title: str,
        article_embedding: List[float],
        key_entities: List[str],
        category: Optional[str] = None,
        summary: Optional[str] = None
    ) -> int:
        """
        Create a new storyline from a seed article.

        Args:
            title: Storyline title (generated or from article)
            article_embedding: Seed article's embedding
            key_entities: Key entities from seed article
            category: Storyline category
            summary: Initial summary (optional)

        Returns:
            New storyline ID
        """
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO storylines (
                        title, summary, original_embedding, current_embedding,
                        key_entities, category, status, start_date, last_update,
                        article_count, momentum_score
                    ) VALUES (
                        %s, %s, %s::vector, %s::vector,
                        %s, %s, 'ACTIVE', CURRENT_DATE, NOW(),
                        1, 1.0
                    )
                    RETURNING id
                """, (
                    title, summary,
                    article_embedding, article_embedding,  # Same for new storyline
                    Json(key_entities), category
                ))
                storyline_id = cur.fetchone()[0]
            conn.commit()

        logger.info(f"Created storyline #{storyline_id}: '{title}' ({category})")
        return storyline_id

    def add_article_to_storyline(
        self,
        article_id: int,
        storyline_id: int,
        article_embedding: List[float],
        article_entities: List[str],
        relevance_score: float = 1.0,
        is_origin: bool = False
    ) -> None:
        """
        Add an article to a storyline and update storyline's embedding.

        Performs:
        1. Create junction table entry
        2. Vector drift: current_embedding = 0.9*old + 0.1*new
        3. Merge key_entities
        4. Update metrics (article_count, momentum_score, last_update)

        Args:
            article_id: Article ID
            storyline_id: Storyline ID
            article_embedding: Article's embedding vector
            article_entities: Article's entities
            relevance_score: How central is this article (0-1)
            is_origin: True if this article started the storyline
        """
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Insert junction record
                cur.execute("""
                    INSERT INTO article_storylines (
                        article_id, storyline_id, relevance_score, is_origin
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (article_id, storyline_id) DO NOTHING
                """, (article_id, storyline_id, relevance_score, is_origin))

                # 2. Get current embedding and entities
                cur.execute("""
                    SELECT current_embedding, key_entities
                    FROM storylines WHERE id = %s
                """, (storyline_id,))
                row = cur.fetchone()

                if row:
                    current_emb = np.array(row[0])
                    current_entities = set(row[1] or [])

                    # Vector drift
                    new_emb = np.array(article_embedding)
                    drifted_emb = (
                        self.DRIFT_WEIGHT_OLD * current_emb +
                        self.DRIFT_WEIGHT_NEW * new_emb
                    )
                    # Normalize to unit vector
                    drifted_emb = drifted_emb / np.linalg.norm(drifted_emb)

                    # Merge entities (keep top 10 by frequency later, for now just add)
                    merged_entities = list(current_entities | set(article_entities))[:15]

                    # 3. Update storyline
                    cur.execute("""
                        UPDATE storylines SET
                            current_embedding = %s::vector,
                            key_entities = %s,
                            article_count = article_count + 1,
                            momentum_score = LEAST(1.0, momentum_score + 0.1),
                            last_update = NOW()
                        WHERE id = %s
                    """, (drifted_emb.tolist(), Json(merged_entities), storyline_id))

            conn.commit()

        logger.debug(f"Added article #{article_id} to storyline #{storyline_id}")

    def assign_article(
        self,
        article_id: int,
        article_embedding: List[float],
        article_entities: List[str],
        article_title: str,
        category: Optional[str] = None
    ) -> List[int]:
        """
        Main entrypoint: Assign an article to storyline(s).

        Flow:
        1. Find matching storylines
        2. If matches: add to existing storyline(s)
        3. If no match: create new storyline

        Args:
            article_id: Article ID in database
            article_embedding: Article's embedding vector
            article_entities: List of entity names
            article_title: Article title (used for new storyline)
            category: Article category

        Returns:
            List of storyline IDs the article was assigned to
        """
        # Filter entities to key types (GPE, ORG, PERSON)
        clean_entities = [e for e in article_entities if e and len(e) > 2]

        # Find matching storylines
        matches = self.find_matching_storylines(
            article_embedding=article_embedding,
            article_entities=clean_entities,
            category=category
        )

        assigned_ids = []

        if matches:
            # Add to top matching storyline(s)
            # For now, just add to the best match
            best_match = matches[0]
            self.add_article_to_storyline(
                article_id=article_id,
                storyline_id=best_match['storyline_id'],
                article_embedding=article_embedding,
                article_entities=clean_entities,
                relevance_score=best_match['final_score']
            )
            assigned_ids.append(best_match['storyline_id'])
            logger.info(
                f"Article #{article_id} matched storyline '{best_match['title']}' "
                f"(score={best_match['final_score']:.2f})"
            )
        else:
            # Create new storyline
            storyline_id = self.create_storyline(
                title=article_title[:100],  # Use article title as initial storyline title
                article_embedding=article_embedding,
                key_entities=clean_entities[:10],
                category=category
            )
            # Link article as origin
            self.add_article_to_storyline(
                article_id=article_id,
                storyline_id=storyline_id,
                article_embedding=article_embedding,
                article_entities=clean_entities,
                relevance_score=1.0,
                is_origin=True
            )
            assigned_ids.append(storyline_id)
            logger.info(f"Article #{article_id} started new storyline #{storyline_id}")

        return assigned_ids

    def apply_decay(self) -> Dict[str, int]:
        """
        Apply momentum decay to storylines without recent activity.

        Rules:
        - No articles in 7 days: momentum *= 0.7
        - Momentum < 0.3: status = 'DORMANT'
        - DORMANT for 30 days: status = 'ARCHIVED'

        Returns:
            Stats dict with counts of decayed/dormant/archived storylines
        """
        stats = {'decayed': 0, 'dormant': 0, 'archived': 0}

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                # Decay momentum for stale ACTIVE storylines
                cur.execute("""
                    UPDATE storylines
                    SET momentum_score = momentum_score * %s
                    WHERE status = 'ACTIVE'
                    AND last_update < NOW() - INTERVAL '7 days'
                    RETURNING id
                """, (self.MOMENTUM_DECAY_FACTOR,))
                stats['decayed'] = cur.rowcount

                # Move low-momentum to DORMANT
                cur.execute("""
                    UPDATE storylines
                    SET status = 'DORMANT'
                    WHERE status = 'ACTIVE'
                    AND momentum_score < %s
                    RETURNING id
                """, (self.DORMANT_THRESHOLD,))
                stats['dormant'] = cur.rowcount

                # Archive long-dormant storylines
                cur.execute("""
                    UPDATE storylines
                    SET status = 'ARCHIVED'
                    WHERE status = 'DORMANT'
                    AND last_update < NOW() - INTERVAL '%s days'
                    RETURNING id
                """, (self.ARCHIVE_DAYS,))
                stats['archived'] = cur.rowcount

            conn.commit()

        if any(stats.values()):
            logger.info(
                f"Decay applied: {stats['decayed']} decayed, "
                f"{stats['dormant']} → DORMANT, {stats['archived']} → ARCHIVED"
            )

        return stats

    def get_active_storylines(
        self,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get active storylines for dashboard/reporting.

        Args:
            category: Optional category filter
            limit: Max storylines to return

        Returns:
            List of storyline dicts
        """
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        id, title, summary, status, category,
                        article_count, momentum_score, start_date, last_update,
                        key_entities,
                        EXTRACT(DAY FROM NOW() - start_date)::INTEGER AS days_active,
                        EXTRACT(DAY FROM NOW() - last_update)::INTEGER AS days_since_update
                    FROM storylines
                    WHERE status = 'ACTIVE'
                """
                params = []

                if category:
                    query += " AND category = %s"
                    params.append(category)

                query += " ORDER BY momentum_score DESC, last_update DESC LIMIT %s"
                params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            {
                'id': r[0],
                'title': r[1],
                'summary': r[2],
                'status': r[3],
                'category': r[4],
                'article_count': r[5],
                'momentum_score': r[6],
                'start_date': r[7],
                'last_update': r[8],
                'key_entities': r[9],
                'days_active': r[10],
                'days_since_update': r[11]
            }
            for r in rows
        ]

    def get_storyline_articles(
        self,
        storyline_id: int,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get articles belonging to a storyline.

        Args:
            storyline_id: Storyline ID
            limit: Max articles to return

        Returns:
            List of article dicts with relevance scores
        """
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        a.id, a.title, a.source, a.published_date, a.category,
                        als.relevance_score, als.is_origin, als.added_at
                    FROM articles a
                    JOIN article_storylines als ON a.id = als.article_id
                    WHERE als.storyline_id = %s
                    ORDER BY a.published_date DESC
                    LIMIT %s
                """, (storyline_id, limit))
                rows = cur.fetchall()

        return [
            {
                'id': r[0],
                'title': r[1],
                'source': r[2],
                'published_date': r[3],
                'category': r[4],
                'relevance_score': r[5],
                'is_origin': r[6],
                'added_at': r[7]
            }
            for r in rows
        ]


# Singleton instance
_story_manager: Optional[StoryManager] = None


def get_story_manager() -> StoryManager:
    """Get or create StoryManager singleton."""
    global _story_manager
    if _story_manager is None:
        _story_manager = StoryManager()
    return _story_manager


# =============================================================================
# Batch Clustering with DBSCAN
# =============================================================================

class BatchClusterer:
    """
    Batch clustering of articles using DBSCAN on embeddings.

    Instead of sequential greedy matching, processes all articles at once:
    1. Load all embeddings from time window
    2. Run DBSCAN clustering (cosine metric)
    3. For each cluster: generate LLM title
    4. Create storylines in database

    Advantages:
    - Global optimization (not greedy)
    - Better cluster boundaries
    - More coherent storylines
    """

    # DBSCAN parameters (tunable)
    DEFAULT_EPS = 0.28          # Max cosine distance (1 - similarity)
    DEFAULT_MIN_SAMPLES = 2     # Minimum articles to form a cluster

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        gemini_api_key: Optional[str] = None,
        eps: float = DEFAULT_EPS,
        min_samples: int = DEFAULT_MIN_SAMPLES
    ):
        """
        Initialize BatchClusterer.

        Args:
            db_manager: Database manager instance
            gemini_api_key: Gemini API key for title generation (uses env if None)
            eps: DBSCAN epsilon (max distance = 1 - min_similarity)
            min_samples: Minimum articles per cluster
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for batch clustering. Run: pip install scikit-learn")

        self.db = db_manager or DatabaseManager()
        self.eps = eps
        self.min_samples = min_samples

        # Initialize Gemini for title generation
        self.gemini_available = False
        if GEMINI_AVAILABLE:
            api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.gemini_available = True
                logger.info("BatchClusterer initialized with Gemini for LLM titles")
            else:
                logger.warning("No GEMINI_API_KEY found. Will use fallback title generation.")
        else:
            logger.warning("google-generativeai not installed. Will use fallback title generation.")

        logger.info(f"BatchClusterer initialized (eps={eps}, min_samples={min_samples})")

    def cluster_articles(
        self,
        days: int = 30,
        exclude_assigned: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute batch clustering and create storylines.

        Args:
            days: Time window in days (0 = all articles)
            exclude_assigned: Skip articles already in storylines
            dry_run: If True, analyze only without creating storylines

        Returns:
            {
                'clusters_found': int,
                'storylines_created': int,
                'articles_assigned': int,
                'noise_articles': int,
                'clusters': [{'cluster_id': int, 'title': str, 'article_count': int, 'articles': [...]}]
            }
        """
        logger.info(f"=== Batch Storyline Clustering (days={days}, dry_run={dry_run}) ===")

        # 1. Load articles with embeddings
        articles = self.db.get_all_article_embeddings(
            days=days,
            exclude_assigned=exclude_assigned
        )

        if not articles:
            logger.warning("No articles found for clustering")
            return {
                'clusters_found': 0,
                'storylines_created': 0,
                'articles_assigned': 0,
                'noise_articles': 0,
                'clusters': []
            }

        logger.info(f"Loaded {len(articles)} articles with embeddings")

        # Filter articles with valid embeddings
        valid_articles = [a for a in articles if a['embedding'] is not None]
        if len(valid_articles) < len(articles):
            logger.warning(f"Skipped {len(articles) - len(valid_articles)} articles with NULL embeddings")

        if len(valid_articles) < self.min_samples:
            logger.warning(f"Not enough articles ({len(valid_articles)}) for clustering (min={self.min_samples})")
            return {
                'clusters_found': 0,
                'storylines_created': 0,
                'articles_assigned': 0,
                'noise_articles': len(valid_articles),
                'clusters': []
            }

        # 2. Build embedding matrix
        embeddings = np.array([a['embedding'] for a in valid_articles])
        logger.info(f"Embedding matrix shape: {embeddings.shape}")

        # 3. Run DBSCAN
        logger.info(f"Running DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='cosine'
        )
        labels = dbscan.fit_predict(embeddings)

        # 4. Analyze clusters
        cluster_map = defaultdict(list)
        noise_count = 0

        for idx, label in enumerate(labels):
            if label == -1:
                noise_count += 1
            else:
                cluster_map[label].append(valid_articles[idx])

        num_clusters = len(cluster_map)
        logger.info(f"Found {num_clusters} clusters + {noise_count} noise articles")

        # 5. Generate titles and create storylines
        clusters_info = []
        storylines_created = 0
        articles_assigned = 0

        for cluster_id, cluster_articles in sorted(cluster_map.items()):
            # Generate title
            title = self._generate_cluster_title(cluster_articles)
            logger.info(f"  Cluster {cluster_id} ({len(cluster_articles)} articles): \"{title}\"")

            cluster_info = {
                'cluster_id': cluster_id,
                'title': title,
                'article_count': len(cluster_articles),
                'articles': [
                    {
                        'id': a['id'],
                        'title': a['title'],
                        'source': a['source'],
                        'category': a['category']
                    }
                    for a in cluster_articles
                ]
            }
            clusters_info.append(cluster_info)

            if not dry_run:
                # Create storyline
                storyline_id = self._create_storyline_from_cluster(
                    title=title,
                    articles=cluster_articles
                )
                if storyline_id:
                    storylines_created += 1
                    articles_assigned += len(cluster_articles)

        result = {
            'clusters_found': num_clusters,
            'storylines_created': storylines_created if not dry_run else 0,
            'articles_assigned': articles_assigned if not dry_run else 0,
            'noise_articles': noise_count,
            'clusters': clusters_info
        }

        logger.info(f"=== Clustering Complete ===")
        logger.info(f"  Clusters: {num_clusters}")
        logger.info(f"  Storylines created: {storylines_created if not dry_run else 'N/A (dry run)'}")
        logger.info(f"  Articles assigned: {articles_assigned if not dry_run else 'N/A (dry run)'}")
        logger.info(f"  Noise (unassigned): {noise_count}")

        return result

    def _generate_cluster_title(self, articles: List[Dict]) -> str:
        """
        Generate a descriptive title for a cluster using LLM.

        Falls back to simple heuristic if LLM not available.

        Args:
            articles: List of articles in cluster

        Returns:
            Generated storyline title
        """
        if self.gemini_available:
            return self._generate_title_with_llm(articles)
        else:
            return self._generate_title_fallback(articles)

    def _generate_title_with_llm(self, articles: List[Dict]) -> str:
        """
        Generate title using Gemini LLM.

        Args:
            articles: List of articles in cluster

        Returns:
            LLM-generated title
        """
        # Prepare article titles for prompt
        article_titles = [a['title'] for a in articles[:10]]  # Limit to 10 for prompt size
        titles_text = "\n".join(f"- {t}" for t in article_titles)

        prompt = f"""Sei un analista geopolitico esperto. Dato questo gruppo di articoli correlati, genera un TITOLO BREVE (massimo 8 parole) che descriva la narrativa comune.

Il titolo deve essere:
- Specifico e informativo (non generico)
- In italiano
- Senza punteggiatura finale
- Descrittivo del tema principale

Articoli:
{titles_text}

Rispondi SOLO con il titolo, nient'altro."""

        try:
            response = self.model.generate_content(prompt)
            title = response.text.strip()

            # Clean up common issues
            title = title.strip('"\'')
            if len(title) > 100:
                title = title[:100]

            # Rate limiting - small delay between calls
            time.sleep(0.3)

            return title

        except Exception as e:
            logger.error(f"LLM title generation failed: {e}")
            return self._generate_title_fallback(articles)

    def _generate_title_fallback(self, articles: List[Dict]) -> str:
        """
        Generate title using simple heuristics.

        Uses common words from article titles and entities.

        Args:
            articles: List of articles in cluster

        Returns:
            Heuristic-generated title
        """
        # Collect all entities
        all_entities = []
        for article in articles:
            entities = article.get('entities', {})
            if isinstance(entities, dict):
                by_type = entities.get('by_type', {})
                # Priority: GPE (locations), ORG (organizations), PERSON
                for etype in ['GPE', 'ORG', 'PERSON']:
                    all_entities.extend(by_type.get(etype, []))

        # Count entity frequency
        entity_counts = defaultdict(int)
        for entity in all_entities:
            entity_counts[entity] += 1

        # Get top entities
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_entity_names = [e[0] for e in top_entities]

        if top_entity_names:
            # Use category if available
            categories = [a.get('category') for a in articles if a.get('category')]
            category = categories[0] if categories else None

            if category and len(top_entity_names) >= 2:
                return f"{top_entity_names[0]} e {top_entity_names[1]} - {category}"
            elif top_entity_names:
                return f"Sviluppi su {', '.join(top_entity_names[:2])}"

        # Fallback to first article title (truncated)
        first_title = articles[0]['title'] if articles else "Storyline"
        if len(first_title) > 60:
            first_title = first_title[:60] + "..."
        return first_title

    def _create_storyline_from_cluster(
        self,
        title: str,
        articles: List[Dict]
    ) -> Optional[int]:
        """
        Create a storyline from a cluster of articles.

        Args:
            title: Generated storyline title
            articles: List of articles in cluster

        Returns:
            Storyline ID if created, None otherwise
        """
        if not articles:
            return None

        # Compute centroid embedding
        embeddings = np.array([a['embedding'] for a in articles if a['embedding'] is not None])
        if len(embeddings) == 0:
            return None

        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize

        # Collect key entities
        all_entities = []
        for article in articles:
            entities = article.get('entities', {})
            if isinstance(entities, dict):
                by_type = entities.get('by_type', {})
                for etype in ['GPE', 'ORG', 'PERSON']:
                    all_entities.extend(by_type.get(etype, []))

        # Count and get top entities
        entity_counts = defaultdict(int)
        for entity in all_entities:
            entity_counts[entity] += 1
        top_entities = [e for e, _ in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:15]]

        # Determine category
        categories = [a.get('category') for a in articles if a.get('category')]
        category = max(set(categories), key=categories.count) if categories else None

        # Find earliest article date
        dates = [a.get('published_date') for a in articles if a.get('published_date')]
        start_date = min(dates) if dates else None

        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create storyline
                    cur.execute("""
                        INSERT INTO storylines (
                            title, summary, original_embedding, current_embedding,
                            key_entities, category, status, start_date, last_update,
                            article_count, momentum_score
                        ) VALUES (
                            %s, %s, %s::vector, %s::vector,
                            %s, %s, 'ACTIVE', %s, NOW(),
                            %s, 1.0
                        )
                        RETURNING id
                    """, (
                        title,
                        f"Storyline con {len(articles)} articoli correlati",
                        centroid.tolist(),
                        centroid.tolist(),
                        Json(top_entities),
                        category,
                        start_date,
                        len(articles)
                    ))
                    storyline_id = cur.fetchone()[0]

                    # Link all articles
                    for i, article in enumerate(articles):
                        is_origin = (i == 0)  # First article is origin
                        cur.execute("""
                            INSERT INTO article_storylines (
                                article_id, storyline_id, relevance_score, is_origin
                            ) VALUES (%s, %s, %s, %s)
                            ON CONFLICT (article_id, storyline_id) DO NOTHING
                        """, (article['id'], storyline_id, 1.0, is_origin))

                conn.commit()

            logger.debug(f"Created storyline #{storyline_id}: '{title}' with {len(articles)} articles")
            return storyline_id

        except Exception as e:
            logger.error(f"Error creating storyline: {e}")
            return None

    def reset_storylines(self) -> Dict[str, int]:
        """
        Remove all existing storylines for fresh clustering.

        Use with caution - this deletes all storyline data!

        Returns:
            {'storylines_deleted': int, 'links_deleted': int}
        """
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Count before delete
                    cur.execute("SELECT COUNT(*) FROM article_storylines")
                    links_count = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM storylines")
                    storylines_count = cur.fetchone()[0]

                    # Delete (cascade will handle article_storylines)
                    cur.execute("DELETE FROM storylines")

                conn.commit()

            logger.warning(f"Reset: Deleted {storylines_count} storylines and {links_count} article links")
            return {
                'storylines_deleted': storylines_count,
                'links_deleted': links_count
            }

        except Exception as e:
            logger.error(f"Error resetting storylines: {e}")
            return {'storylines_deleted': 0, 'links_deleted': 0}


# Singleton for BatchClusterer
_batch_clusterer: Optional[BatchClusterer] = None


def get_batch_clusterer(**kwargs) -> BatchClusterer:
    """Get or create BatchClusterer singleton."""
    global _batch_clusterer
    if _batch_clusterer is None:
        _batch_clusterer = BatchClusterer(**kwargs)
    return _batch_clusterer
