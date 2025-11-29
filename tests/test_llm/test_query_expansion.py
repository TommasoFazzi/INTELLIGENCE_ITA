"""
Test per Query Expansion e Deduplicazione Avanzata.

Questi test verificano:
- expand_rag_queries(): generazione varianti semantiche
- deduplicate_chunks_advanced(): deduplicazione con similarity
- Integrazione in generate_report()
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.llm.report_generator import ReportGenerator


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_dependencies():
    """Mock database, NLP processor, e Gemini model."""
    with patch('src.llm.report_generator.DatabaseManager') as mock_db, \
         patch('src.llm.report_generator.NLPProcessor') as mock_nlp, \
         patch('src.llm.report_generator.genai') as mock_genai:

        # Mock database
        db_instance = MagicMock()
        mock_db.return_value = db_instance

        # Mock NLP processor with embedding model
        nlp_instance = MagicMock()
        nlp_instance.embedding_model = MagicMock()
        nlp_instance.embedding_model.encode.return_value = np.array([0.1] * 384)
        mock_nlp.return_value = nlp_instance

        # Mock Gemini
        mock_genai.configure = Mock()
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        yield {
            'db': db_instance,
            'nlp': nlp_instance,
            'model': mock_model
        }


@pytest.fixture
def generator_with_expansion(mock_dependencies):
    """ReportGenerator con query expansion abilitata."""
    generator = ReportGenerator(
        enable_query_expansion=True,
        expansion_variants=2,
        dedup_similarity=0.98
    )
    return generator


@pytest.fixture
def generator_without_expansion(mock_dependencies):
    """ReportGenerator con query expansion disabilitata."""
    generator = ReportGenerator(
        enable_query_expansion=False,
        expansion_variants=2,
        dedup_similarity=0.98
    )
    return generator


# ============================================================================
# TEST: EXPAND_RAG_QUERIES
# ============================================================================

@pytest.mark.unit
def test_expand_rag_queries_disabled(generator_without_expansion):
    """Test: quando expansion è disabilitata, ritorna queries originali."""
    original_queries = ["cybersecurity threats", "AI developments"]

    result = generator_without_expansion.expand_rag_queries(original_queries)

    assert result == original_queries
    assert len(result) == 2


@pytest.mark.unit
def test_expand_rag_queries_enabled(generator_with_expansion, mock_dependencies):
    """Test: quando expansion è abilitata, genera varianti."""
    original_queries = ["cybersecurity threats"]

    # Mock LLM response con 2 varianti (must be 5-15 words each)
    mock_response = MagicMock()
    mock_response.text = """data breach incidents vulnerabilities and security exploits in systems
critical infrastructure security risks and protection strategies"""

    mock_dependencies['model'].generate_content.return_value = mock_response

    result = generator_with_expansion.expand_rag_queries(original_queries)

    # Dovrebbe contenere: original + 2 varianti = 3 queries
    assert len(result) == 3
    assert result[0] == "cybersecurity threats"  # Original sempre primo
    assert "data breach" in result[1].lower()
    assert "infrastructure" in result[2].lower()


@pytest.mark.unit
def test_expand_rag_queries_filters_invalid_variants(generator_with_expansion, mock_dependencies):
    """Test: filtra varianti troppo corte o troppo lunghe."""
    original_queries = ["AI"]

    # Mock response con varianti di lunghezza varia
    mock_response = MagicMock()
    mock_response.text = """short
    this is a valid AI development query with good length
    this is way too long and exceeds the maximum word count limit that we set which should be around fifteen words maximum"""

    mock_dependencies['model'].generate_content.return_value = mock_response

    result = generator_with_expansion.expand_rag_queries(original_queries)

    # Dovrebbe contenere: original + 1 variante valida (la seconda)
    assert len(result) == 2
    assert result[0] == "AI"
    assert "valid AI development" in result[1]


@pytest.mark.unit
def test_expand_rag_queries_handles_llm_failure(generator_with_expansion, mock_dependencies):
    """Test: fallback graceful quando LLM fallisce."""
    original_queries = ["cybersecurity"]

    # Mock LLM exception
    mock_dependencies['model'].generate_content.side_effect = Exception("API Error")

    result = generator_with_expansion.expand_rag_queries(original_queries)

    # Dovrebbe ritornare solo la query originale
    assert result == original_queries


@pytest.mark.unit
def test_expand_rag_queries_removes_numbering(generator_with_expansion, mock_dependencies):
    """Test: rimuove numerazione dalle varianti."""
    original_queries = ["geopolitics"]

    # Mock response con numerazione
    mock_response = MagicMock()
    mock_response.text = """1. regional power dynamics in Asia Pacific
2. diplomatic tensions and international relations"""

    mock_dependencies['model'].generate_content.return_value = mock_response

    result = generator_with_expansion.expand_rag_queries(original_queries)

    # Varianti non dovrebbero contenere "1." o "2."
    for variant in result[1:]:  # Skip original
        assert not variant.startswith("1.")
        assert not variant.startswith("2.")


@pytest.mark.unit
def test_expand_rag_queries_multiple_originals(generator_with_expansion, mock_dependencies):
    """Test: espande correttamente multiple queries originali."""
    original_queries = ["cybersecurity", "AI developments"]

    # Mock LLM responses (2 varianti per query)
    mock_response_1 = MagicMock()
    mock_response_1.text = """data breach incidents and security vulnerabilities
critical infrastructure protection measures and strategies"""

    mock_response_2 = MagicMock()
    mock_response_2.text = """machine learning applications in enterprise software
generative AI models and large language models"""

    mock_dependencies['model'].generate_content.side_effect = [mock_response_1, mock_response_2]

    result = generator_with_expansion.expand_rag_queries(original_queries)

    # 2 original + (2 varianti × 2 queries) = 6 total
    assert len(result) == 6
    assert "cybersecurity" in result
    assert "AI developments" in result


# ============================================================================
# TEST: DEDUPLICATE_CHUNKS_ADVANCED
# ============================================================================

@pytest.mark.unit
def test_deduplicate_chunks_empty_list(generator_with_expansion):
    """Test: lista vuota ritorna lista vuota."""
    result = generator_with_expansion.deduplicate_chunks_advanced([])
    assert result == []


@pytest.mark.unit
def test_deduplicate_chunks_removes_exact_id_duplicates(generator_with_expansion):
    """Test: rimuove duplicati con stesso chunk_id."""
    # Create truly different embeddings using random vectors
    np.random.seed(42)
    embedding_a = np.random.rand(384).tolist()
    embedding_b = np.random.rand(384).tolist()

    chunks = [
        {'chunk_id': 1, 'content': 'Text A', 'embedding': embedding_a},
        {'chunk_id': 1, 'content': 'Text A', 'embedding': embedding_a},  # Duplicate ID
        {'chunk_id': 2, 'content': 'Text B', 'embedding': embedding_b}
    ]

    result = generator_with_expansion.deduplicate_chunks_advanced(chunks)

    assert len(result) == 2
    assert result[0]['chunk_id'] == 1
    assert result[1]['chunk_id'] == 2


@pytest.mark.unit
def test_deduplicate_chunks_removes_similar_embeddings(generator_with_expansion):
    """Test: rimuove chunks con embeddings molto simili (> 0.98)."""
    # Due embeddings quasi identici (similarity ~0.99)
    embedding_a = np.random.rand(384).tolist()
    embedding_b = (np.array(embedding_a) + 0.001).tolist()  # Quasi identico

    chunks = [
        {'chunk_id': 1, 'content': 'Text A', 'embedding': embedding_a},
        {'chunk_id': 2, 'content': 'Text A paraphrased', 'embedding': embedding_b}
    ]

    result = generator_with_expansion.deduplicate_chunks_advanced(chunks)

    # Dovrebbe tenere solo il primo (il secondo è troppo simile)
    assert len(result) == 1
    assert result[0]['chunk_id'] == 1


@pytest.mark.unit
def test_deduplicate_chunks_keeps_dissimilar_embeddings(generator_with_expansion):
    """Test: mantiene chunks con embeddings diversi."""
    # Create orthogonal vectors (similarity ~0)
    np.random.seed(123)
    embedding_a = np.random.rand(384)
    embedding_b = np.random.rand(384)

    # Make them more orthogonal
    embedding_a = (embedding_a - 0.5).tolist()
    embedding_b = (embedding_b - 0.5).tolist()

    chunks = [
        {'chunk_id': 1, 'content': 'Text A', 'embedding': embedding_a},
        {'chunk_id': 2, 'content': 'Text B', 'embedding': embedding_b}
    ]

    result = generator_with_expansion.deduplicate_chunks_advanced(chunks)

    # Dovrebbe tenerli entrambi (dissimilari)
    assert len(result) == 2


@pytest.mark.unit
def test_deduplicate_chunks_without_embeddings(generator_with_expansion):
    """Test: gestisce chunks senza embedding field."""
    chunks = [
        {'chunk_id': 1, 'content': 'Text A'},  # No embedding
        {'chunk_id': 2, 'content': 'Text B'}   # No embedding
    ]

    result = generator_with_expansion.deduplicate_chunks_advanced(chunks)

    # Senza embeddings, può solo dedup per ID
    assert len(result) == 2


@pytest.mark.unit
def test_deduplicate_chunks_mixed_embeddings(generator_with_expansion):
    """Test: gestisce mix di chunks con/senza embeddings."""
    # Create dissimilar random embeddings
    np.random.seed(456)
    embedding_a = np.random.rand(384).tolist()
    embedding_c = np.random.rand(384).tolist()

    chunks = [
        {'chunk_id': 1, 'content': 'Text A', 'embedding': embedding_a},
        {'chunk_id': 2, 'content': 'Text B'},  # No embedding
        {'chunk_id': 3, 'content': 'Text C', 'embedding': embedding_c}
    ]

    result = generator_with_expansion.deduplicate_chunks_advanced(chunks)

    # Tutti dovrebbero essere mantenuti (IDs unici, no similarity check per chunk 2)
    assert len(result) == 3


# ============================================================================
# TEST: INTEGRAZIONE GENERATE_REPORT
# ============================================================================

@pytest.mark.unit
def test_generate_report_uses_expansion(generator_with_expansion, mock_dependencies):
    """Test: generate_report usa expand_rag_queries quando abilitato."""
    # Mock recent articles
    mock_dependencies['db'].get_recent_articles.return_value = [
        {
            'title': 'Article 1',
            'link': 'https://ex.com/1',
            'source': 'Source',
            'published_date': '2025-11-29',
            'full_text': 'Content about cybersecurity threats'
        }
    ]

    # Mock filter_relevant_articles
    with patch.object(generator_with_expansion, 'filter_relevant_articles') as mock_filter:
        mock_filter.return_value = mock_dependencies['db'].get_recent_articles.return_value

        # Mock expand_rag_queries
        with patch.object(generator_with_expansion, 'expand_rag_queries') as mock_expand:
            mock_expand.return_value = ["query1", "query2", "query3"]  # Expanded

            # Mock get_rag_context
            with patch.object(generator_with_expansion, 'get_rag_context') as mock_rag:
                mock_rag.return_value = []

                # Mock deduplicate
                with patch.object(generator_with_expansion, 'deduplicate_chunks_advanced') as mock_dedup:
                    mock_dedup.return_value = []

                    # Mock LLM response
                    mock_response = MagicMock()
                    mock_response.text = "Test report"
                    mock_dependencies['model'].generate_content.return_value = mock_response

                    # Generate report
                    generator_with_expansion.generate_report(focus_areas=["cybersecurity"])

                    # Verifica che expand_rag_queries sia stato chiamato
                    mock_expand.assert_called_once()

                    # Verifica che get_rag_context sia chiamato con expanded queries
                    assert mock_rag.call_count == 3  # 3 expanded queries

                    # Verifica che deduplicate sia stato chiamato
                    mock_dedup.assert_called_once()


@pytest.mark.unit
def test_generate_report_skips_expansion_when_disabled(generator_without_expansion, mock_dependencies):
    """Test: generate_report non espande quando expansion è disabilitata."""
    # Mock recent articles
    mock_dependencies['db'].get_recent_articles.return_value = [
        {
            'title': 'Article 1',
            'link': 'https://ex.com/1',
            'source': 'Source',
            'published_date': '2025-11-29',
            'full_text': 'Content'
        }
    ]

    # Mock filter_relevant_articles
    with patch.object(generator_without_expansion, 'filter_relevant_articles') as mock_filter:
        mock_filter.return_value = mock_dependencies['db'].get_recent_articles.return_value

        # Mock get_rag_context
        with patch.object(generator_without_expansion, 'get_rag_context') as mock_rag:
            mock_rag.return_value = []

            # Mock deduplicate
            with patch.object(generator_without_expansion, 'deduplicate_chunks_advanced') as mock_dedup:
                mock_dedup.return_value = []

                # Mock LLM response
                mock_response = MagicMock()
                mock_response.text = "Test report"
                mock_dependencies['model'].generate_content.return_value = mock_response

                # Generate report
                generator_without_expansion.generate_report(focus_areas=["cybersecurity"])

                # Verifica che get_rag_context sia chiamato solo 1 volta (no expansion)
                assert mock_rag.call_count == 1


# ============================================================================
# TEST: CONFIGURATION
# ============================================================================

@pytest.mark.unit
def test_init_with_expansion_enabled():
    """Test: inizializzazione con expansion abilitata."""
    with patch('src.llm.report_generator.DatabaseManager'), \
         patch('src.llm.report_generator.NLPProcessor'), \
         patch('src.llm.report_generator.genai'):

        generator = ReportGenerator(
            enable_query_expansion=True,
            expansion_variants=3,
            dedup_similarity=0.95
        )

        assert generator.enable_query_expansion is True
        assert generator.expansion_variants == 3
        assert generator.dedup_similarity == 0.95


@pytest.mark.unit
def test_init_with_expansion_disabled():
    """Test: inizializzazione con expansion disabilitata."""
    with patch('src.llm.report_generator.DatabaseManager'), \
         patch('src.llm.report_generator.NLPProcessor'), \
         patch('src.llm.report_generator.genai'):

        generator = ReportGenerator(
            enable_query_expansion=False,
            expansion_variants=2,
            dedup_similarity=0.98
        )

        assert generator.enable_query_expansion is False


@pytest.mark.unit
def test_init_default_values():
    """Test: valori di default corretti."""
    with patch('src.llm.report_generator.DatabaseManager'), \
         patch('src.llm.report_generator.NLPProcessor'), \
         patch('src.llm.report_generator.genai'):

        generator = ReportGenerator()

        assert generator.enable_query_expansion is True  # Default: enabled
        assert generator.expansion_variants == 2  # Default: 2
        assert generator.dedup_similarity == 0.98  # Default: 0.98
