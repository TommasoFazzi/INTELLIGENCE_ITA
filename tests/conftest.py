"""
Configurazione pytest e fixtures condivise per i test.

Questo file viene automaticamente caricato da pytest e rende disponibili
le fixtures a tutti i test senza doverle importare.
"""

import pytest
import os
from datetime import datetime


# ============================================================================
# CONFIGURAZIONE DATABASE TEST
# ============================================================================

TEST_DB_URL = os.getenv(
    'TEST_DATABASE_URL',
    'postgresql://localhost:5432/intelligence_ita_test'
)


@pytest.fixture(scope='session')
def test_db_url():
    """
    URL del database di test.

    Scope 'session': creato una volta per tutta la sessione di test.
    """
    return TEST_DB_URL


# ============================================================================
# FIXTURES PER DATI DI TEST
# ============================================================================

@pytest.fixture
def sample_article():
    """
    Articolo di esempio per test.

    Returns:
        dict: Articolo con tutti i campi necessari
    """
    return {
        'title': 'Test Article: Cybersecurity Threats',
        'link': 'https://example.com/test-article',
        'published': datetime(2025, 11, 28, 10, 0, 0),
        'source': 'Test Source',
        'category': 'intelligence',
        'subcategory': 'cybersecurity',
        'summary': 'This is a test article summary.',
        'full_text': 'This is the full text content of the test article. It contains multiple sentences for testing purposes.',
        'extraction_success': True
    }


@pytest.fixture
def sample_articles_batch():
    """
    Batch di articoli di esempio per test che richiedono multipli articoli.

    Returns:
        list: Lista di 3 articoli
    """
    return [
        {
            'title': 'Article 1: Geopolitics',
            'link': 'https://example.com/article-1',
            'published': datetime(2025, 11, 28, 9, 0, 0),
            'source': 'Source A',
            'category': 'intelligence',
            'subcategory': 'geopolitics',
            'full_text': 'Geopolitical content here.'
        },
        {
            'title': 'Article 2: Cybersecurity',
            'link': 'https://example.com/article-2',
            'published': datetime(2025, 11, 28, 10, 0, 0),
            'source': 'Source B',
            'category': 'intelligence',
            'subcategory': 'cybersecurity',
            'full_text': 'Cybersecurity threats are increasing.'
        },
        {
            'title': 'Article 3: Economy',
            'link': 'https://example.com/article-3',
            'published': datetime(2025, 11, 28, 11, 0, 0),
            'source': 'Source C',
            'category': 'tech_economy',
            'subcategory': 'economy',
            'full_text': 'Economic growth in Asia continues.'
        }
    ]


@pytest.fixture
def sample_embedding():
    """
    Embedding di esempio (384 dimensioni).

    Returns:
        list: Vector di 384 float
    """
    return [0.1] * 384


@pytest.fixture
def sample_chunk():
    """
    Chunk di testo di esempio.

    Returns:
        dict: Chunk con metadata
    """
    return {
        'text': 'This is a sample text chunk for testing purposes. It contains enough content to be meaningful.',
        'position': 0,
        'word_count': 15,
        'embedding': [0.1] * 384
    }


# ============================================================================
# MARKERS PER CATEGORIZZARE I TEST
# ============================================================================

def pytest_configure(config):
    """
    Registra i markers personalizzati per pytest.

    Questo permette di categorizzare i test e eseguirli selettivamente:
    - pytest -m unit          # Solo unit test
    - pytest -m "not slow"    # Esclude test lenti
    - pytest -m integration   # Solo integration test
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, multiple components)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (slowest, full pipeline)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (can be skipped with -m 'not slow')"
    )


# ============================================================================
# FIXTURES PER DATABASE (da implementare quando necessario)
# ============================================================================

# Nota: Le fixtures per il database saranno aggiunte quando inizieremo
# a testare i moduli che interagiscono con PostgreSQL.
# Per ora lasciamo questo file semplice e funzionale.
