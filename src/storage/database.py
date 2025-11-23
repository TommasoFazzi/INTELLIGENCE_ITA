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
