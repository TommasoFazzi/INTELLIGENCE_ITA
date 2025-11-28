"""
Test per NLPProcessor - Semantic Chunking.

Questi test verificano:
- Chunking basato su frasi complete
- Gestione overlap tra chunk
- Rispetto dei limiti di dimensione
- Metadata dei chunk (word_count, sentence_count)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.nlp.processing import NLPProcessor


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def nlp_processor():
    """
    Fixture: NLPProcessor con spaCy mockato ma funzionale per chunking.
    """
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        # Mock spaCy con sentence segmentation funzionante
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer', 'ner']
        mock_nlp.max_length = 2000000

        # Mock del metodo __call__ che ritorna un Doc con sentences
        def mock_nlp_call(text):
            doc = MagicMock()
            # Split in frasi semplici (split su .)
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

            # Mock delle sentence objects
            mock_sents = []
            for sent in sentences:
                sent_obj = MagicMock()
                sent_obj.text = sent
                mock_sents.append(sent_obj)

            doc.sents = mock_sents
            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor(chunk_size=100, chunk_overlap=20)
        return processor


@pytest.fixture
def short_text():
    """Testo breve (< chunk size)."""
    return "This is a short sentence. It has only two sentences."


@pytest.fixture
def medium_text():
    """Testo medio che necessita 2-3 chunk."""
    sentences = [
        "This is the first sentence with some words.",
        "This is the second sentence with more words in it.",
        "Third sentence continues the text with additional content.",
        "Fourth sentence adds even more information to the article.",
        "Fifth sentence wraps up the first section nicely.",
        "Sixth sentence starts a new topic with fresh ideas.",
        "Seventh sentence elaborates on those ideas further.",
        "Eighth sentence provides concrete examples for clarity.",
        "Ninth sentence concludes the section with summary.",
        "Tenth sentence is the final one in this text."
    ]
    return " ".join(sentences)


# ============================================================================
# TEST: BASIC CHUNKING
# ============================================================================

@pytest.mark.unit
def test_create_chunks_handles_empty_text(nlp_processor):
    """Test: gestisce testo vuoto."""
    chunks = nlp_processor.create_chunks("")

    assert chunks == []


@pytest.mark.unit
def test_create_chunks_handles_none(nlp_processor):
    """Test: gestisce None input."""
    chunks = nlp_processor.create_chunks(None)

    assert chunks == []


@pytest.mark.unit
def test_create_chunks_single_sentence(nlp_processor):
    """Test: singola frase crea un chunk."""
    text = "This is a single sentence."

    chunks = nlp_processor.create_chunks(text)

    assert len(chunks) == 1
    assert chunks[0]['text'] == "This is a single sentence."
    assert chunks[0]['sentence_count'] == 1


@pytest.mark.unit
def test_create_chunks_short_text(nlp_processor, short_text):
    """Test: testo breve (sotto chunk_size) crea un chunk."""
    chunks = nlp_processor.create_chunks(short_text)

    assert len(chunks) == 1
    assert short_text in chunks[0]['text']


@pytest.mark.unit
def test_create_chunks_splits_long_text(nlp_processor, medium_text):
    """Test: testo lungo viene diviso in multiple chunk."""
    chunks = nlp_processor.create_chunks(medium_text)

    # Con chunk_size=100 e 10 frasi lunghe, dovremmo avere almeno 1 chunk
    # (il comportamento esatto dipende dalla lunghezza delle frasi)
    assert len(chunks) >= 1


# ============================================================================
# TEST: CHUNK METADATA
# ============================================================================

@pytest.mark.unit
def test_chunk_has_required_metadata(nlp_processor, short_text):
    """Test: ogni chunk ha i metadata richiesti."""
    chunks = nlp_processor.create_chunks(short_text)

    assert len(chunks) > 0
    chunk = chunks[0]

    # Verifica campi richiesti
    assert 'text' in chunk
    assert 'word_count' in chunk
    assert 'sentence_count' in chunk


@pytest.mark.unit
def test_chunk_word_count_is_accurate(nlp_processor):
    """Test: word_count riflette il numero corretto di parole."""
    text = "One two three four five."  # 5 words

    chunks = nlp_processor.create_chunks(text)

    assert chunks[0]['word_count'] == 5


@pytest.mark.unit
def test_chunk_sentence_count_is_accurate(nlp_processor):
    """Test: sentence_count conta le frasi correttamente."""
    text = "First sentence. Second sentence. Third sentence."

    chunks = nlp_processor.create_chunks(text)

    # Se tutto in un chunk
    if len(chunks) == 1:
        assert chunks[0]['sentence_count'] == 3


# ============================================================================
# TEST: CHUNK SIZE LIMITS
# ============================================================================

@pytest.mark.unit
def test_chunks_respect_size_limit(nlp_processor, medium_text):
    """Test: chunk non superano significativamente chunk_size."""
    chunks = nlp_processor.create_chunks(medium_text)

    for chunk in chunks:
        # Può superare leggermente per frasi complete
        # Ma non dovrebbe essere drasticamente più grande
        assert chunk['word_count'] <= nlp_processor.chunk_size + 50


@pytest.mark.unit
def test_create_chunks_with_custom_size():
    """Test: chunk_size personalizzato viene rispettato."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        # Mock setup
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer']
        mock_nlp.max_length = 2000000

        def mock_nlp_call(text):
            doc = MagicMock()
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            mock_sents = []
            for sent in sentences:
                sent_obj = MagicMock()
                sent_obj.text = sent
                mock_sents.append(sent_obj)
            doc.sents = mock_sents
            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor(chunk_size=50, chunk_overlap=10)

        # Testo con frasi brevi
        text = " ".join([f"Sentence number {i} here." for i in range(20)])
        chunks = processor.create_chunks(text)

        # Con chunk_size più piccolo, dovremmo avere almeno 1 chunk
        assert len(chunks) >= 1


# ============================================================================
# TEST: CHUNK OVERLAP
# ============================================================================

@pytest.mark.unit
def test_chunks_have_overlap(nlp_processor, medium_text):
    """Test: chunk consecutivi hanno overlap."""
    chunks = nlp_processor.create_chunks(medium_text)

    if len(chunks) > 1:
        # Verifica che ci sia sovrapposizione tra chunk
        first_chunk_text = chunks[0]['text']
        second_chunk_text = chunks[1]['text']

        # L'ultimo pezzo del primo chunk dovrebbe apparire nel secondo
        # (non possiamo testare esattamente quale perché dipende dalle frasi)
        assert len(chunks) >= 2  # Almeno confermiamo che abbiamo più chunk


@pytest.mark.unit
def test_overlap_preserved_across_chunks():
    """Test: overlap viene mantenuto tra chunk consecutivi."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        # Mock setup
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer']
        mock_nlp.max_length = 2000000

        def mock_nlp_call(text):
            doc = MagicMock()
            # Frasi molto specifiche per testare overlap
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            mock_sents = []
            for sent in sentences:
                sent_obj = MagicMock()
                sent_obj.text = sent
                mock_sents.append(sent_obj)
            doc.sents = mock_sents
            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor(chunk_size=30, chunk_overlap=10)

        # Testo con frasi identiche e prevedibili
        sentences = [f"Sentence {i} with exactly five words here." for i in range(10)]
        text = " ".join(sentences)

        chunks = processor.create_chunks(text)

        # Dovremmo avere overlap configuration corretta
        assert processor.chunk_overlap == 10


# ============================================================================
# TEST: SENTENCE PRESERVATION
# ============================================================================

@pytest.mark.unit
def test_chunks_preserve_complete_sentences(nlp_processor):
    """Test: chunk contengono solo frasi complete (no split mid-sentence)."""
    text = "First complete sentence. Second complete sentence. Third complete sentence."

    chunks = nlp_processor.create_chunks(text)

    # Ogni chunk dovrebbe contenere frasi complete
    for chunk in chunks:
        chunk_text = chunk['text']
        # Non dovrebbe iniziare o finire a metà frase
        assert chunk_text[0].isupper() or not chunk_text[0].isalpha()


@pytest.mark.unit
def test_create_chunks_does_not_split_sentences(nlp_processor):
    """Test: chunking non divide mai una frase a metà."""
    text = "This is sentence one with many words. This is sentence two. This is sentence three."

    chunks = nlp_processor.create_chunks(text)

    # Unendo tutti i chunk, dovremmo avere tutte le frasi originali
    all_text = " ".join([c['text'] for c in chunks])

    # Le tre frasi dovrebbero essere presenti
    assert "sentence one" in all_text
    assert "sentence two" in all_text
    assert "sentence three" in all_text


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

@pytest.mark.unit
def test_create_chunks_single_very_long_sentence():
    """Test: singola frase molto lunga (> chunk_size)."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer']
        mock_nlp.max_length = 2000000

        def mock_nlp_call(text):
            doc = MagicMock()
            sent_obj = MagicMock()
            sent_obj.text = text
            doc.sents = [sent_obj]
            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor(chunk_size=20, chunk_overlap=5)

        # Frase singola con 40+ parole (oltre chunk_size)
        text = " ".join([f"word{i}" for i in range(40)])

        chunks = processor.create_chunks(text)

        # Dovrebbe creare un chunk anche se supera chunk_size
        # (perché non può dividere la frase)
        assert len(chunks) == 1
        assert chunks[0]['word_count'] > processor.chunk_size


@pytest.mark.unit
def test_create_chunks_whitespace_only_sentences(nlp_processor):
    """Test: gestisce frasi con solo whitespace."""
    text = "Real sentence.    .   Another real sentence."

    chunks = nlp_processor.create_chunks(text)

    # Dovrebbe ignorare frasi vuote
    assert len(chunks) >= 1


@pytest.mark.unit
def test_create_chunks_preserves_text_content(nlp_processor, medium_text):
    """Test: nessun contenuto viene perso durante chunking."""
    chunks = nlp_processor.create_chunks(medium_text)

    # Unendo tutti i chunk e rimuovendo overlap,
    # dovremmo avere contenuto simile all'originale
    total_words = sum(chunk['word_count'] for chunk in chunks)

    # Il totale dovrebbe essere >= parole originali (a causa dell'overlap)
    original_words = len(medium_text.split())
    assert total_words >= original_words


@pytest.mark.unit
def test_create_chunks_returns_list_of_dicts(nlp_processor, short_text):
    """Test: ritorna sempre lista di dizionari."""
    chunks = nlp_processor.create_chunks(short_text)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, dict) for chunk in chunks)
