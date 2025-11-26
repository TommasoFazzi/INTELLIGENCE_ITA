"""
Database Storage Module

Handles connection to PostgreSQL with pgvector extension,
schema initialization, and storage of articles with vector embeddings.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import Json, execute_batch
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL database with pgvector extension for RAG system.
    """

    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize database manager with connection pooling.

        Args:
            connection_url: PostgreSQL connection URL. If None, reads from environment.
        """
        # Get connection URL from environment or parameter
        if connection_url is None:
            connection_url = os.getenv('DATABASE_URL')
            if not connection_url:
                # Fallback to individual env vars
                db_host = os.getenv("DB_HOST", "localhost")
                db_name = os.getenv("DB_NAME", "intelligence_ita")
                db_user = os.getenv("DB_USER", "postgres")
                db_pass = os.getenv("DB_PASS", "")
                db_port = os.getenv("DB_PORT", "5432")
                connection_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

        self.connection_url = connection_url

        # Create connection pool (min 1, max 10 connections)
        try:
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=connection_url
            )
            logger.info(f"✓ Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Automatically returns connection to pool after use.
        """
        conn = self.pool.getconn()
        try:
            # Register pgvector type for this connection
            register_vector(conn)
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self.pool.putconn(conn)

    def init_db(self):
        """
        Initialize database schema: enable pgvector extension and create tables.
        """
        logger.info("Initializing database schema...")

        schema_sql = """
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;

        -- Articles table
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            link TEXT UNIQUE NOT NULL,
            published_date TIMESTAMP WITH TIME ZONE,
            source TEXT,
            category TEXT,
            subcategory TEXT,
            summary TEXT,
            full_text TEXT,
            entities JSONB,              -- Extracted entities (PERSON, ORG, GPE, etc.)
            nlp_metadata JSONB,          -- NLP statistics (word count, tokens, etc.)
            full_text_embedding vector(384),  -- Full article embedding for similarity
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Chunks table for RAG
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(384),       -- Chunk embedding for semantic search
            word_count INTEGER,
            sentence_count INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Performance indexes
        CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_date DESC);
        CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
        CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
        CREATE INDEX IF NOT EXISTS idx_chunks_article_id ON chunks(article_id);

        -- HNSW index for fast approximate nearest neighbor search
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
            USING hnsw (embedding vector_cosine_ops);

        CREATE INDEX IF NOT EXISTS idx_articles_full_embedding ON articles
            USING hnsw (full_text_embedding vector_cosine_ops);

        -- Reports table (Phase 4: LLM-generated reports)
        CREATE TABLE IF NOT EXISTS reports (
            id SERIAL PRIMARY KEY,
            report_date DATE NOT NULL,
            generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            model_used TEXT,
            draft_content TEXT NOT NULL,          -- Original LLM-generated report
            final_content TEXT,                   -- Human-edited version (NULL if not reviewed)
            status TEXT DEFAULT 'draft',          -- draft, reviewed, approved
            metadata JSONB,                       -- focus_areas, article_count, etc.
            sources JSONB,                        -- Links to source articles and chunks
            human_reviewed_at TIMESTAMP WITH TIME ZONE,
            human_reviewer TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Report feedback table (Phase 5: Human-in-the-Loop)
        CREATE TABLE IF NOT EXISTS report_feedback (
            id SERIAL PRIMARY KEY,
            report_id INTEGER REFERENCES reports(id) ON DELETE CASCADE,
            section_name TEXT,                    -- e.g., "Executive Summary", "Cybersecurity"
            feedback_type TEXT NOT NULL,          -- 'correction', 'addition', 'removal', 'rating'
            original_text TEXT,                   -- What LLM originally wrote
            corrected_text TEXT,                  -- What human changed it to
            comment TEXT,                         -- Human explanation/notes
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),  -- 1-5 stars for quality
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Report indexes
        CREATE INDEX IF NOT EXISTS idx_reports_date ON reports(report_date DESC);
        CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
        CREATE INDEX IF NOT EXISTS idx_report_feedback_report_id ON report_feedback(report_id);

        -- Update timestamp trigger
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS update_articles_updated_at ON articles;
        CREATE TRIGGER update_articles_updated_at
            BEFORE UPDATE ON articles
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();

        DROP TRIGGER IF EXISTS update_reports_updated_at ON reports;
        CREATE TRIGGER update_reports_updated_at
            BEFORE UPDATE ON reports
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
            logger.info("✓ Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

    def save_article(self, article: Dict[str, Any]) -> Optional[int]:
        """
        Save a single processed article with its chunks and embeddings.

        Args:
            article: Article dictionary with nlp_data

        Returns:
            Article ID if saved successfully, None if skipped (duplicate or error)
        """
        # Only save articles with successful NLP processing
        if not article.get('nlp_processing', {}).get('success', False):
            logger.debug(f"Skipping article without NLP data: {article.get('title', 'Unknown')[:50]}...")
            return None

        nlp_data = article.get('nlp_data', {})

        # Parse published date
        pub_date = article.get('published')
        if isinstance(pub_date, str):
            try:
                pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            except:
                pub_date = None

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if article already exists
                    cur.execute("SELECT id FROM articles WHERE link = %s", (article.get('link'),))
                    existing = cur.fetchone()

                    if existing:
                        logger.debug(f"Article already exists: {article.get('title', '')[:50]}...")
                        return None

                    # Insert article
                    cur.execute("""
                        INSERT INTO articles
                        (title, link, published_date, source, category, subcategory, summary,
                         full_text, entities, nlp_metadata, full_text_embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        article.get('title'),
                        article.get('link'),
                        pub_date,
                        article.get('source'),
                        article.get('category'),
                        article.get('subcategory'),
                        article.get('summary'),
                        nlp_data.get('clean_text', ''),
                        Json(nlp_data.get('entities', {})),
                        Json({
                            'original_length': nlp_data.get('original_length', 0),
                            'clean_length': nlp_data.get('clean_length', 0),
                            'num_tokens': nlp_data.get('preprocessed', {}).get('num_tokens', 0),
                            'num_sentences': nlp_data.get('preprocessed', {}).get('num_sentences', 0),
                            'entity_count': nlp_data.get('entities', {}).get('entity_count', 0)
                        }),
                        nlp_data.get('full_text_embedding', [])
                    ))

                    article_id = cur.fetchone()[0]

                    # Insert chunks in batch
                    chunks = nlp_data.get('chunks', [])
                    if chunks:
                        chunk_data = [
                            (
                                article_id,
                                idx,
                                chunk['text'],
                                chunk['embedding'],
                                chunk.get('word_count', 0),
                                chunk.get('sentence_count', 0)
                            )
                            for idx, chunk in enumerate(chunks)
                        ]

                        execute_batch(cur, """
                            INSERT INTO chunks
                            (article_id, chunk_index, content, embedding, word_count, sentence_count)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, chunk_data, page_size=100)

                    logger.debug(f"✓ Saved article {article_id} with {len(chunks)} chunks")
                    return article_id

        except Exception as e:
            logger.error(f"Error saving article '{article.get('title', 'Unknown')[:50]}...': {e}")
            return None

    def batch_save(self, articles: List[Dict]) -> Dict[str, int]:
        """
        Save multiple articles in batch.

        Args:
            articles: List of article dictionaries

        Returns:
            Statistics dictionary with counts
        """
        stats = {
            "saved": 0,
            "skipped": 0,
            "errors": 0,
            "total_chunks": 0
        }

        logger.info(f"Saving {len(articles)} articles to database...")

        for i, article in enumerate(articles):
            article_id = self.save_article(article)

            if article_id is not None:
                stats["saved"] += 1
                chunks_count = len(article.get('nlp_data', {}).get('chunks', []))
                stats["total_chunks"] += chunks_count
            elif article.get('nlp_processing', {}).get('success', False):
                stats["skipped"] += 1  # Duplicate
            else:
                stats["errors"] += 1  # No NLP data

            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(articles)} articles processed")

        logger.info(f"✓ Batch save complete: {stats['saved']} saved, "
                   f"{stats['skipped']} skipped (duplicates), "
                   f"{stats['errors']} errors (no NLP data)")
        logger.info(f"  Total chunks inserted: {stats['total_chunks']}")

        return stats

    def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.

        Args:
            query_embedding: Query embedding vector (384 dimensions)
            top_k: Number of results to return
            category: Optional category filter

        Returns:
            List of matching chunks with article metadata
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT
                            c.id as chunk_id,
                            c.content,
                            c.chunk_index,
                            c.word_count,
                            a.id as article_id,
                            a.title,
                            a.link,
                            a.source,
                            a.published_date,
                            a.category,
                            1 - (c.embedding <=> %s::vector) as similarity
                        FROM chunks c
                        JOIN articles a ON c.article_id = a.id
                        WHERE 1=1
                    """

                    params = [query_embedding]

                    if category:
                        query += " AND a.category = %s"
                        params.append(category)

                    query += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
                    params.extend([query_embedding, top_k])

                    cur.execute(query, params)

                    results = []
                    for row in cur.fetchall():
                        results.append({
                            'chunk_id': row[0],
                            'content': row[1],
                            'chunk_index': row[2],
                            'word_count': row[3],
                            'article_id': row[4],
                            'title': row[5],
                            'link': row[6],
                            'source': row[7],
                            'published_date': row[8],
                            'category': row[9],
                            'similarity': float(row[10])
                        })

                    return results

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    stats = {}

                    # Total articles
                    cur.execute("SELECT COUNT(*) FROM articles")
                    stats['total_articles'] = cur.fetchone()[0]

                    # Total chunks
                    cur.execute("SELECT COUNT(*) FROM chunks")
                    stats['total_chunks'] = cur.fetchone()[0]

                    # Articles by category
                    cur.execute("""
                        SELECT category, COUNT(*)
                        FROM articles
                        GROUP BY category
                        ORDER BY COUNT(*) DESC
                    """)
                    stats['by_category'] = dict(cur.fetchall())

                    # Recent articles count (last 7 days)
                    cur.execute("""
                        SELECT COUNT(*)
                        FROM articles
                        WHERE published_date > NOW() - INTERVAL '7 days'
                    """)
                    stats['recent_articles'] = cur.fetchone()[0]

                    # Top sources
                    cur.execute("""
                        SELECT source, COUNT(*)
                        FROM articles
                        GROUP BY source
                        ORDER BY COUNT(*) DESC
                        LIMIT 10
                    """)
                    stats['top_sources'] = dict(cur.fetchall())

                    return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def get_recent_articles(self, days: int = 1, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent articles from database.

        Args:
            days: Number of days to look back
            category: Optional category filter

        Returns:
            List of article dictionaries
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT
                            id, title, link, published_date, source, category,
                            subcategory, summary, full_text, entities, nlp_metadata, full_text_embedding
                        FROM articles
                        WHERE published_date > NOW() - INTERVAL '%s days'
                    """

                    params = [days]

                    if category:
                        query += " AND category = %s"
                        params.append(category)

                    query += " ORDER BY published_date DESC"

                    cur.execute(query, params)

                    articles = []
                    for row in cur.fetchall():
                        articles.append({
                            'id': row[0],
                            'title': row[1],
                            'link': row[2],
                            'published_date': row[3],
                            'source': row[4],
                            'category': row[5],
                            'subcategory': row[6],
                            'summary': row[7],
                            'full_text': row[8],
                            'entities': row[9],
                            'nlp_metadata': row[10],
                            'full_text_embedding': row[11]
                        })

                    return articles

        except Exception as e:
            logger.error(f"Error getting recent articles: {e}")
            return []

    def save_report(self, report: Dict[str, Any]) -> Optional[int]:
        """
        Save LLM-generated report to database.

        Args:
            report: Report dictionary from ReportGenerator

        Returns:
            Report ID if saved successfully, None otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO reports
                        (report_date, model_used, draft_content, status, metadata, sources)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        datetime.now().date(),
                        report.get('metadata', {}).get('model_used', 'unknown'),
                        report.get('report_text', ''),
                        'draft',
                        Json(report.get('metadata', {})),
                        Json(report.get('sources', {}))
                    ))

                    report_id = cur.fetchone()[0]
                    logger.info(f"✓ Report saved to database with ID: {report_id}")
                    return report_id

        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return None

    def get_report(self, report_id: int) -> Optional[Dict[str, Any]]:
        """
        Get report by ID.

        Args:
            report_id: Report ID

        Returns:
            Report dictionary or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, report_date, generated_at, model_used,
                               draft_content, final_content, status, metadata, sources,
                               human_reviewed_at, human_reviewer
                        FROM reports
                        WHERE id = %s
                    """, (report_id,))

                    row = cur.fetchone()
                    if not row:
                        return None

                    return {
                        'id': row[0],
                        'report_date': row[1],
                        'generated_at': row[2],
                        'model_used': row[3],
                        'draft_content': row[4],
                        'final_content': row[5],
                        'status': row[6],
                        'metadata': row[7],
                        'sources': row[8],
                        'human_reviewed_at': row[9],
                        'human_reviewer': row[10]
                    }

        except Exception as e:
            logger.error(f"Error getting report: {e}")
            return None

    def get_all_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all reports ordered by date (most recent first).

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of report dictionaries
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, report_date, generated_at, model_used, status,
                               metadata, human_reviewed_at
                        FROM reports
                        ORDER BY report_date DESC, generated_at DESC
                        LIMIT %s
                    """, (limit,))

                    reports = []
                    for row in cur.fetchall():
                        reports.append({
                            'id': row[0],
                            'report_date': row[1],
                            'generated_at': row[2],
                            'model_used': row[3],
                            'status': row[4],
                            'metadata': row[5],
                            'human_reviewed_at': row[6]
                        })

                    return reports

        except Exception as e:
            logger.error(f"Error getting reports: {e}")
            return []

    def update_report(
        self,
        report_id: int,
        final_content: str,
        status: str = 'reviewed',
        reviewer: Optional[str] = None
    ) -> bool:
        """
        Update report with human-edited content.

        Args:
            report_id: Report ID
            final_content: Human-edited report text
            status: Report status (reviewed, approved)
            reviewer: Name/email of reviewer

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE reports
                        SET final_content = %s,
                            status = %s,
                            human_reviewed_at = CURRENT_TIMESTAMP,
                            human_reviewer = %s
                        WHERE id = %s
                    """, (final_content, status, reviewer, report_id))

                    logger.info(f"✓ Report {report_id} updated with status: {status}")
                    return True

        except Exception as e:
            logger.error(f"Error updating report: {e}")
            return False

    def save_feedback(
        self,
        report_id: int,
        section_name: Optional[str],
        feedback_type: str,
        original_text: Optional[str] = None,
        corrected_text: Optional[str] = None,
        comment: Optional[str] = None,
        rating: Optional[int] = None
    ) -> Optional[int]:
        """
        Save human feedback for a report section.

        Args:
            report_id: Report ID
            section_name: Section being reviewed (e.g., "Executive Summary")
            feedback_type: Type of feedback (correction, addition, removal, rating)
            original_text: Original LLM text
            corrected_text: Human-corrected text
            comment: Human notes/explanation
            rating: Quality rating 1-5

        Returns:
            Feedback ID if saved successfully, None otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO report_feedback
                        (report_id, section_name, feedback_type, original_text,
                         corrected_text, comment, rating)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        report_id, section_name, feedback_type,
                        original_text, corrected_text, comment, rating
                    ))

                    feedback_id = cur.fetchone()[0]
                    logger.debug(f"✓ Feedback saved with ID: {feedback_id}")
                    return feedback_id

        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return None

    def upsert_approval_feedback(
        self,
        report_id: int,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ) -> Optional[int]:
        """
        Insert or update approval feedback for a report.
        If feedback already exists for this report, update it.
        Otherwise, create new feedback.

        Args:
            report_id: Report ID
            rating: Quality rating 1-5
            comment: Human notes/explanation

        Returns:
            Feedback ID if saved successfully, None otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Use PostgreSQL's ON CONFLICT to upsert
                    # First, check if feedback exists
                    cur.execute("""
                        SELECT id FROM report_feedback
                        WHERE report_id = %s AND feedback_type = 'rating'
                        LIMIT 1
                    """, (report_id,))

                    existing = cur.fetchone()

                    if existing:
                        # Update existing feedback
                        cur.execute("""
                            UPDATE report_feedback
                            SET rating = %s,
                                comment = %s,
                                created_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                            RETURNING id
                        """, (rating, comment, existing[0]))
                        feedback_id = cur.fetchone()[0]
                        logger.debug(f"✓ Feedback updated with ID: {feedback_id}")
                    else:
                        # Insert new feedback
                        cur.execute("""
                            INSERT INTO report_feedback
                            (report_id, section_name, feedback_type, rating, comment)
                            VALUES (%s, NULL, 'rating', %s, %s)
                            RETURNING id
                        """, (report_id, rating, comment))
                        feedback_id = cur.fetchone()[0]
                        logger.debug(f"✓ Feedback created with ID: {feedback_id}")

                    return feedback_id

        except Exception as e:
            logger.error(f"Error upserting approval feedback: {e}")
            return None

    def get_report_feedback(self, report_id: int) -> List[Dict[str, Any]]:
        """
        Get all feedback for a report.

        Args:
            report_id: Report ID

        Returns:
            List of feedback dictionaries
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, section_name, feedback_type, original_text,
                               corrected_text, comment, rating, created_at
                        FROM report_feedback
                        WHERE report_id = %s
                        ORDER BY created_at ASC
                    """, (report_id,))

                    feedback = []
                    for row in cur.fetchall():
                        feedback.append({
                            'id': row[0],
                            'section_name': row[1],
                            'feedback_type': row[2],
                            'original_text': row[3],
                            'corrected_text': row[4],
                            'comment': row[5],
                            'rating': row[6],
                            'created_at': row[7]
                        })

                    return feedback

        except Exception as e:
            logger.error(f"Error getting feedback: {e}")
            return []

    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent feedback across all reports.

        Args:
            limit: Maximum number of feedback entries to return

        Returns:
            List of feedback dictionaries with report info
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            rf.id, rf.report_id, rf.rating, rf.comment, rf.created_at,
                            r.report_date, r.human_reviewer
                        FROM report_feedback rf
                        JOIN reports r ON rf.report_id = r.id
                        WHERE rf.feedback_type = 'rating'
                        ORDER BY rf.created_at DESC
                        LIMIT %s
                    """, (limit,))

                    feedback = []
                    for row in cur.fetchall():
                        feedback.append({
                            'id': row[0],
                            'report_id': row[1],
                            'rating': row[2],
                            'comment': row[3],
                            'created_at': row[4],
                            'report_date': row[5],
                            'reviewer': row[6]
                        })

                    return feedback

        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []

    def close(self):
        """Close all connections in the pool."""
        if hasattr(self, 'pool'):
            self.pool.closeall()
            logger.info("Database connection pool closed")


if __name__ == "__main__":
    # Example usage
    db = DatabaseManager()
    db.init_db()
    print("Database initialized successfully!")

    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    print(f"  Total articles: {stats.get('total_articles', 0)}")
    print(f"  Total chunks: {stats.get('total_chunks', 0)}")
    print(f"  Recent articles (7 days): {stats.get('recent_articles', 0)}")
