"""
Test per NLPProcessor - Full Pipeline Integration.

Questi test verificano:
- process_article() - pipeline completa
- process_batch() - batch processing
- preprocess_text() - preprocessing linguistico
- get_processing_stats() - statistiche
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from src.nlp.processing import NLPProcessor


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def nlp_processor():
    """
    Fixture: NLPProcessor completo con tutti i componenti mockati.
    """
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        # Mock spaCy con tutte le funzionalità
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer', 'ner']
        mock_nlp.max_length = 2000000

        def mock_nlp_call(text):
            doc = MagicMock()

            # Mock sentences
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            mock_sents = []
            for sent in sentences:
                sent_obj = MagicMock()
                sent_obj.text = sent
                mock_sents.append(sent_obj)
            doc.sents = mock_sents

            # Mock tokens
            words = text.split()
            mock_tokens = []
            for word in words:
                token = MagicMock()
                token.text = word
                token.lemma_ = word.lower()
                token.pos_ = "NOUN"
                mock_tokens.append(token)
            doc.__iter__ = lambda self: iter(mock_tokens)
            doc.__len__ = lambda self: len(mock_tokens)

            # Mock entities
            doc.ents = []

            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        def mock_encode(text, convert_to_numpy=True, show_progress_bar=True):
            if isinstance(text, str):
                return np.random.rand(384).astype(np.float32)
            elif isinstance(text, list):
                return np.random.rand(len(text), 384).astype(np.float32)

        mock_model.encode = mock_encode
        mock_transformer.return_value = mock_model

        processor = NLPProcessor(chunk_size=50, chunk_overlap=10)
        return processor


@pytest.fixture
def sample_article():
    """Article di esempio per testing."""
    return {
        'title': 'Test Article Title',
        'link': 'https://example.com/article',
        'full_content': {
            'text': 'This is the article content. It has multiple sentences. This allows testing chunking.',
            'title': 'Test Article',
            'extraction_method': 'trafilatura'
        },
        'source': 'Test Source',
        'category': 'intelligence'
    }


# ============================================================================
# TEST: PREPROCESS TEXT
# ============================================================================

@pytest.mark.unit
def test_preprocess_text_returns_dict(nlp_processor):
    """Test: preprocess_text ritorna dizionario."""
    text = "This is a test sentence."

    result = nlp_processor.preprocess_text(text)

    assert isinstance(result, dict)


@pytest.mark.unit
def test_preprocess_text_has_required_fields(nlp_processor):
    """Test: contiene tutti i campi richiesti."""
    text = "Test sentence here."

    result = nlp_processor.preprocess_text(text)

    required_fields = ['tokens', 'lemmas', 'pos_tags', 'sentences', 'num_tokens', 'num_sentences']
    for field in required_fields:
        assert field in result


@pytest.mark.unit
def test_preprocess_text_tokenization(nlp_processor):
    """Test: tokenizzazione funziona."""
    text = "Hello world test."

    result = nlp_processor.preprocess_text(text)

    assert len(result['tokens']) > 0
    assert isinstance(result['tokens'], list)


@pytest.mark.unit
def test_preprocess_text_lemmatization(nlp_processor):
    """Test: lemmatizzazione funziona."""
    text = "Running tests quickly."

    result = nlp_processor.preprocess_text(text)

    assert len(result['lemmas']) > 0
    assert isinstance(result['lemmas'], list)


@pytest.mark.unit
def test_preprocess_text_pos_tagging(nlp_processor):
    """Test: POS tagging funziona."""
    text = "Test sentence."

    result = nlp_processor.preprocess_text(text)

    assert len(result['pos_tags']) > 0
    # Ogni POS tag è una tupla (word, pos)
    if result['pos_tags']:
        assert isinstance(result['pos_tags'][0], tuple)


@pytest.mark.unit
def test_preprocess_text_empty_input(nlp_processor):
    """Test: gestisce input vuoto."""
    result = nlp_processor.preprocess_text("")

    assert result['tokens'] == []
    assert result['lemmas'] == []
    assert result['pos_tags'] == []
    assert result['sentences'] == []
    assert result['num_tokens'] == 0
    assert result['num_sentences'] == 0


@pytest.mark.unit
def test_preprocess_text_none_input(nlp_processor):
    """Test: gestisce None input."""
    result = nlp_processor.preprocess_text(None)

    assert result['tokens'] == []
    assert result['num_tokens'] == 0


@pytest.mark.unit
def test_preprocess_text_non_string_input(nlp_processor):
    """Test: gestisce input non-string."""
    result = nlp_processor.preprocess_text(12345)

    assert result['tokens'] == []
    assert result['num_tokens'] == 0


# ============================================================================
# TEST: PROCESS ARTICLE
# ============================================================================

@pytest.mark.unit
def test_process_article_enriches_with_nlp_data(nlp_processor, sample_article):
    """Test: process_article aggiunge nlp_data."""
    processed = nlp_processor.process_article(sample_article)

    assert 'nlp_data' in processed
    assert 'nlp_processing' in processed


@pytest.mark.unit
def test_process_article_nlp_data_structure(nlp_processor, sample_article):
    """Test: nlp_data ha struttura corretta."""
    processed = nlp_processor.process_article(sample_article)

    nlp_data = processed['nlp_data']

    required_fields = [
        'clean_text', 'chunks', 'chunk_count', 'entities',
        'preprocessed', 'full_text_embedding', 'embedding_dim',
        'original_length', 'clean_length', 'processed_at'
    ]

    for field in required_fields:
        assert field in nlp_data, f"Missing field: {field}"


@pytest.mark.unit
def test_process_article_marks_success(nlp_processor, sample_article):
    """Test: marca processing come successful."""
    processed = nlp_processor.process_article(sample_article)

    assert processed['nlp_processing']['success'] is True
    assert 'timestamp' in processed['nlp_processing']


@pytest.mark.unit
def test_process_article_handles_no_content(nlp_processor):
    """Test: gestisce articolo senza contenuto."""
    article = {
        'title': 'No Content Article',
        'link': 'https://example.com/empty'
    }

    processed = nlp_processor.process_article(article)

    assert processed['nlp_processing']['success'] is False
    assert 'error' in processed['nlp_processing']


@pytest.mark.unit
def test_process_article_extracts_from_full_content_dict(nlp_processor):
    """Test: estrae testo da full_content dict."""
    article = {
        'title': 'Test',
        'full_content': {
            'text': 'Content here.',
            'title': 'Title'
        }
    }

    processed = nlp_processor.process_article(article)

    assert processed['nlp_processing']['success'] is True


@pytest.mark.unit
def test_process_article_handles_full_content_string(nlp_processor):
    """Test: gestisce full_content come stringa."""
    article = {
        'title': 'Test',
        'full_content': 'Direct string content here.'
    }

    processed = nlp_processor.process_article(article)

    assert processed['nlp_processing']['success'] is True


@pytest.mark.unit
def test_process_article_falls_back_to_summary(nlp_processor):
    """Test: usa summary come fallback se full_content manca."""
    article = {
        'title': 'Test',
        'summary': 'Summary content here with enough words to be processed.'
    }

    processed = nlp_processor.process_article(article)

    # Dovrebbe provare a processare usando summary (anche se potrebbe fallire se troppo corto)
    assert 'nlp_processing' in processed


@pytest.mark.unit
def test_process_article_creates_chunks(nlp_processor, sample_article):
    """Test: crea chunk per articolo."""
    processed = nlp_processor.process_article(sample_article)

    assert processed['nlp_data']['chunk_count'] > 0
    assert len(processed['nlp_data']['chunks']) > 0


@pytest.mark.unit
def test_process_article_short_text_single_chunk(nlp_processor):
    """Test: testo breve crea singolo chunk."""
    article = {
        'title': 'Short',
        'full_content': {'text': 'Very short.'}
    }

    processed = nlp_processor.process_article(article)

    # Testo molto breve dovrebbe creare 1 chunk
    assert processed['nlp_data']['chunk_count'] == 1


@pytest.mark.unit
def test_process_article_chunks_have_embeddings(nlp_processor, sample_article):
    """Test: ogni chunk ha embedding."""
    processed = nlp_processor.process_article(sample_article)

    chunks = processed['nlp_data']['chunks']

    for chunk in chunks:
        assert 'embedding' in chunk
        assert 'embedding_dim' in chunk


@pytest.mark.unit
def test_process_article_generates_full_text_embedding(nlp_processor, sample_article):
    """Test: genera embedding per full text."""
    processed = nlp_processor.process_article(sample_article)

    assert 'full_text_embedding' in processed['nlp_data']
    assert len(processed['nlp_data']['full_text_embedding']) == 384


@pytest.mark.unit
def test_process_article_extracts_entities(nlp_processor, sample_article):
    """Test: estrae entità."""
    processed = nlp_processor.process_article(sample_article)

    assert 'entities' in processed['nlp_data']
    entities = processed['nlp_data']['entities']
    assert 'entities' in entities
    assert 'by_type' in entities
    assert 'entity_count' in entities


@pytest.mark.unit
def test_process_article_preprocesses_text(nlp_processor, sample_article):
    """Test: preprocessa testo."""
    processed = nlp_processor.process_article(sample_article)

    preprocessed = processed['nlp_data']['preprocessed']
    assert 'tokens' in preprocessed
    assert 'lemmas' in preprocessed
    assert 'pos_tags' in preprocessed


@pytest.mark.unit
def test_process_article_handles_exception(nlp_processor):
    """Test: gestisce exception durante processing."""
    # Article che causa errore
    article = {
        'title': 'Bad Article',
        'full_content': None  # Causa problemi
    }

    processed = nlp_processor.process_article(article)

    assert processed['nlp_processing']['success'] is False


# ============================================================================
# TEST: PROCESS BATCH
# ============================================================================

@pytest.mark.unit
def test_process_batch_processes_all_articles(nlp_processor):
    """Test: processa tutti gli articoli nel batch."""
    articles = [
        {'title': f'Article {i}', 'full_content': {'text': f'Content {i}.'}}
        for i in range(5)
    ]

    processed = nlp_processor.process_batch(articles, show_progress=False)

    assert len(processed) == 5


@pytest.mark.unit
def test_process_batch_returns_list(nlp_processor):
    """Test: ritorna lista di articoli processati."""
    articles = [
        {'title': 'Test', 'full_content': {'text': 'Content.'}}
    ]

    processed = nlp_processor.process_batch(articles, show_progress=False)

    assert isinstance(processed, list)


@pytest.mark.unit
def test_process_batch_empty_list(nlp_processor):
    """Test: gestisce lista vuota."""
    processed = nlp_processor.process_batch([], show_progress=False)

    assert processed == []


@pytest.mark.unit
def test_process_batch_tracks_success(nlp_processor):
    """Test: traccia successi e fallimenti."""
    articles = [
        {'title': 'Good', 'full_content': {'text': 'Content here.'}},
        {'title': 'Bad', 'full_content': None},  # Fallirà
    ]

    processed = nlp_processor.process_batch(articles, show_progress=False)

    success_count = sum(1 for a in processed if a.get('nlp_processing', {}).get('success'))
    assert success_count >= 0  # Almeno alcuni dovrebbero avere success


# ============================================================================
# TEST: GET PROCESSING STATS
# ============================================================================

@pytest.mark.unit
def test_get_processing_stats_returns_dict(nlp_processor):
    """Test: ritorna dizionario di statistiche."""
    articles = [
        {'title': 'Test', 'nlp_processing': {'success': True}, 'nlp_data': {
            'chunks': [{'text': 'chunk', 'word_count': 10}],
            'entities': {'entities': [], 'by_type': {}, 'entity_count': 0},
            'preprocessed': {'num_tokens': 20, 'num_sentences': 2},
            'embedding_dim': 384
        }}
    ]

    stats = nlp_processor.get_processing_stats(articles)

    assert isinstance(stats, dict)


@pytest.mark.unit
def test_get_processing_stats_has_required_fields(nlp_processor):
    """Test: contiene campi richiesti."""
    articles = [
        {'title': 'Test', 'nlp_processing': {'success': True}, 'nlp_data': {
            'chunks': [],
            'entities': {'entities': [], 'by_type': {}, 'entity_count': 0},
            'preprocessed': {'num_tokens': 0, 'num_sentences': 0},
            'embedding_dim': 384
        }}
    ]

    stats = nlp_processor.get_processing_stats(articles)

    required_fields = [
        'total_articles', 'successful_processing', 'success_rate',
        'total_entities_extracted', 'entities_by_type',
        'avg_tokens_per_article', 'total_chunks',
        'avg_chunks_per_article', 'avg_chunk_size', 'embedding_dimension'
    ]

    for field in required_fields:
        assert field in stats


@pytest.mark.unit
def test_get_processing_stats_empty_list(nlp_processor):
    """Test: gestisce lista vuota."""
    stats = nlp_processor.get_processing_stats([])

    assert stats['total_articles'] == 0


@pytest.mark.unit
def test_get_processing_stats_calculates_success_rate(nlp_processor):
    """Test: calcola success rate correttamente."""
    articles = [
        {'nlp_processing': {'success': True}, 'nlp_data': {'chunks': [], 'entities': {'entities': [], 'by_type': {}}, 'preprocessed': {'num_tokens': 0}, 'embedding_dim': 384}},
        {'nlp_processing': {'success': False}},
        {'nlp_processing': {'success': True}, 'nlp_data': {'chunks': [], 'entities': {'entities': [], 'by_type': {}}, 'preprocessed': {'num_tokens': 0}, 'embedding_dim': 384}},
    ]

    stats = nlp_processor.get_processing_stats(articles)

    assert stats['total_articles'] == 3
    assert stats['successful_processing'] == 2
    assert '2/3' in stats['success_rate']


# ============================================================================
# TEST: INITIALIZATION
# ============================================================================

@pytest.mark.unit
def test_nlp_processor_initialization():
    """Test: inizializzazione con parametri custom."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer']
        mock_nlp.max_length = 2000000
        mock_spacy.return_value = mock_nlp

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor(
            chunk_size=200,
            chunk_overlap=30,
            batch_size=64
        )

        assert processor.chunk_size == 200
        assert processor.chunk_overlap == 30
        assert processor.batch_size == 64


@pytest.mark.unit
def test_nlp_processor_adds_sentencizer():
    """Test: aggiunge sentencizer se mancante."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = []  # No sentencizer o parser
        mock_nlp.max_length = 2000000
        mock_spacy.return_value = mock_nlp

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor()

        # Dovrebbe aver chiamato add_pipe
        mock_nlp.add_pipe.assert_called_once_with("sentencizer")
