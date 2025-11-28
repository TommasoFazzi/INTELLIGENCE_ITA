"""
Test per NLPProcessor - Embedding Generation.

Questi test verificano:
- Generazione embeddings per testo singolo
- Batch embedding per chunk
- Dimensione corretta degli embeddings
- Gestione edge cases (testo vuoto, None)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.nlp.processing import NLPProcessor


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def nlp_processor():
    """
    Fixture: NLPProcessor con SentenceTransformer mockato.
    """
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        # Mock spaCy
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer', 'ner']
        mock_nlp.max_length = 2000000
        mock_spacy.return_value = mock_nlp

        # Mock SentenceTransformer con embedding dimension 384
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Mock encode per ritornare vettori numpy
        def mock_encode(text, convert_to_numpy=True, show_progress_bar=True):
            if isinstance(text, str):
                # Singolo embedding
                return np.random.rand(384).astype(np.float32)
            elif isinstance(text, list):
                # Batch embeddings
                return np.random.rand(len(text), 384).astype(np.float32)

        mock_model.encode = mock_encode
        mock_transformer.return_value = mock_model

        processor = NLPProcessor()
        return processor


# ============================================================================
# TEST: SINGLE TEXT EMBEDDING
# ============================================================================

@pytest.mark.unit
def test_generate_embedding_returns_numpy_array(nlp_processor):
    """Test: genera embedding come numpy array."""
    text = "This is a test sentence for embedding."

    embedding = nlp_processor.generate_embedding(text)

    assert isinstance(embedding, np.ndarray)


@pytest.mark.unit
def test_generate_embedding_correct_dimension(nlp_processor):
    """Test: embedding ha dimensione corretta."""
    text = "Test sentence."

    embedding = nlp_processor.generate_embedding(text)

    # Dovrebbe essere 384-dimensional (paraphrase-multilingual-MiniLM-L12-v2)
    assert len(embedding) == 384


@pytest.mark.unit
def test_generate_embedding_not_zero_vector(nlp_processor):
    """Test: embedding non è vettore zero (per testo valido)."""
    text = "This is real content with meaning."

    embedding = nlp_processor.generate_embedding(text)

    # Non dovrebbe essere tutto zeri
    assert not np.allclose(embedding, np.zeros_like(embedding))


@pytest.mark.unit
def test_generate_embedding_handles_empty_text(nlp_processor):
    """Test: gestisce testo vuoto ritornando zero vector."""
    embedding = nlp_processor.generate_embedding("")

    # Dovrebbe ritornare zero vector
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384
    assert np.allclose(embedding, np.zeros(384))


@pytest.mark.unit
def test_generate_embedding_handles_none(nlp_processor):
    """Test: gestisce None input."""
    embedding = nlp_processor.generate_embedding(None)

    # Dovrebbe ritornare zero vector
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384
    assert np.allclose(embedding, np.zeros(384))


@pytest.mark.unit
def test_generate_embedding_handles_non_string(nlp_processor):
    """Test: gestisce input non-string."""
    embedding = nlp_processor.generate_embedding(12345)

    # Dovrebbe ritornare zero vector per input non valido
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384


# ============================================================================
# TEST: CHUNK EMBEDDINGS (BATCH)
# ============================================================================

@pytest.mark.unit
def test_generate_chunk_embeddings_adds_embeddings(nlp_processor):
    """Test: aggiunge embeddings a ogni chunk."""
    chunks = [
        {'text': 'First chunk text.', 'word_count': 3, 'sentence_count': 1},
        {'text': 'Second chunk text.', 'word_count': 3, 'sentence_count': 1},
        {'text': 'Third chunk text.', 'word_count': 3, 'sentence_count': 1}
    ]

    enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)

    # Ogni chunk dovrebbe avere embedding
    assert all('embedding' in chunk for chunk in enriched_chunks)


@pytest.mark.unit
def test_generate_chunk_embeddings_correct_dimension(nlp_processor):
    """Test: embeddings hanno dimensione corretta."""
    chunks = [
        {'text': 'Chunk text.', 'word_count': 2, 'sentence_count': 1}
    ]

    enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)

    assert len(enriched_chunks[0]['embedding']) == 384
    assert enriched_chunks[0]['embedding_dim'] == 384


@pytest.mark.unit
def test_generate_chunk_embeddings_preserves_original_data(nlp_processor):
    """Test: mantiene i dati originali del chunk."""
    chunks = [
        {'text': 'Test text.', 'word_count': 2, 'sentence_count': 1}
    ]

    enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)

    # Dati originali dovrebbero essere preservati
    assert enriched_chunks[0]['text'] == 'Test text.'
    assert enriched_chunks[0]['word_count'] == 2
    assert enriched_chunks[0]['sentence_count'] == 1


@pytest.mark.unit
def test_generate_chunk_embeddings_batch_processing(nlp_processor):
    """Test: processa multipli chunk in batch."""
    chunks = [
        {'text': f'Chunk {i} text.', 'word_count': 3, 'sentence_count': 1}
        for i in range(10)
    ]

    enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)

    assert len(enriched_chunks) == 10
    assert all('embedding' in chunk for chunk in enriched_chunks)


@pytest.mark.unit
def test_generate_chunk_embeddings_empty_list(nlp_processor):
    """Test: gestisce lista vuota."""
    chunks = []

    enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)

    assert enriched_chunks == []


@pytest.mark.unit
def test_generate_chunk_embeddings_adds_metadata(nlp_processor):
    """Test: aggiunge metadata embedding (dimension)."""
    chunks = [
        {'text': 'Test.', 'word_count': 1, 'sentence_count': 1}
    ]

    enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)

    # Dovrebbe avere embedding_dim field
    assert 'embedding_dim' in enriched_chunks[0]
    assert enriched_chunks[0]['embedding_dim'] == 384


# ============================================================================
# TEST: EMBEDDING AS LIST (FOR JSON SERIALIZATION)
# ============================================================================

@pytest.mark.unit
def test_chunk_embeddings_are_lists(nlp_processor):
    """Test: embeddings sono liste (per JSON serialization)."""
    chunks = [
        {'text': 'Test text.', 'word_count': 2, 'sentence_count': 1}
    ]

    enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)

    # Embedding dovrebbe essere lista (non numpy array)
    assert isinstance(enriched_chunks[0]['embedding'], list)


@pytest.mark.unit
def test_chunk_embeddings_contain_floats(nlp_processor):
    """Test: embeddings contengono valori float."""
    chunks = [
        {'text': 'Test.', 'word_count': 1, 'sentence_count': 1}
    ]

    enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)

    embedding = enriched_chunks[0]['embedding']

    # Tutti i valori dovrebbero essere float
    assert all(isinstance(val, (float, np.floating)) for val in embedding)


# ============================================================================
# TEST: DIFFERENT TEXT LENGTHS
# ============================================================================

@pytest.mark.unit
def test_generate_embedding_short_text(nlp_processor):
    """Test: genera embedding per testo breve."""
    text = "Hi."

    embedding = nlp_processor.generate_embedding(text)

    assert len(embedding) == 384


@pytest.mark.unit
def test_generate_embedding_long_text(nlp_processor):
    """Test: genera embedding per testo lungo."""
    text = " ".join(["This is sentence number " + str(i) for i in range(100)])

    embedding = nlp_processor.generate_embedding(text)

    # Dimensione dovrebbe essere sempre la stessa
    assert len(embedding) == 384


@pytest.mark.unit
def test_generate_embedding_multilingual(nlp_processor):
    """Test: gestisce testo multilingue."""
    texts = [
        "This is English text.",
        "Questo è testo italiano.",
        "Ceci est du texte français."
    ]

    embeddings = [nlp_processor.generate_embedding(text) for text in texts]

    # Tutti dovrebbero avere stessa dimensione
    assert all(len(emb) == 384 for emb in embeddings)


# ============================================================================
# TEST: REPRODUCIBILITY
# ============================================================================

@pytest.mark.unit
def test_generate_embedding_consistent_for_same_text():
    """Test: stesso testo genera stesso embedding (con seed fisso)."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer']
        mock_nlp.max_length = 2000000
        mock_spacy.return_value = mock_nlp

        # Mock che ritorna embedding deterministico
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        def mock_encode(text, convert_to_numpy=True, show_progress_bar=True):
            # Embedding deterministico basato su hash del testo
            np.random.seed(hash(text if isinstance(text, str) else str(text)) % (2**32))
            if isinstance(text, str):
                return np.random.rand(384).astype(np.float32)
            elif isinstance(text, list):
                return np.array([np.random.rand(384).astype(np.float32) for _ in text])

        mock_model.encode = mock_encode
        mock_transformer.return_value = mock_model

        processor = NLPProcessor()

        text = "Consistent test text."
        embedding1 = processor.generate_embedding(text)
        embedding2 = processor.generate_embedding(text)

        # Con seed fisso, dovrebbero essere identici
        np.testing.assert_array_almost_equal(embedding1, embedding2)


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================

@pytest.mark.unit
def test_generate_chunk_embeddings_handles_malformed_chunks(nlp_processor):
    """Test: gestisce chunk malformati."""
    chunks = [
        {'text': 'Good chunk.', 'word_count': 2, 'sentence_count': 1},
        {'no_text_field': 'bad'},  # Malformed
    ]

    # Dovrebbe gestire l'errore gracefully (o lanciare KeyError)
    # Dipende dall'implementazione - testiamo che non crashe l'intero batch
    try:
        enriched_chunks = nlp_processor.generate_chunk_embeddings(chunks)
        # Se riesce, verifica che il chunk buono sia processato
        assert 'embedding' in enriched_chunks[0]
    except KeyError:
        # Se lancia KeyError è accettabile
        pass


@pytest.mark.unit
def test_generate_embedding_special_characters(nlp_processor):
    """Test: gestisce caratteri speciali."""
    text = "Text with special chars: @#$%^&*()!"

    embedding = nlp_processor.generate_embedding(text)

    assert len(embedding) == 384
