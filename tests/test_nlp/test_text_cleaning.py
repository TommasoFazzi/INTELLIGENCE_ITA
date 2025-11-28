"""
Test per NLPProcessor - Text Cleaning.

Questi test verificano:
- Rimozione whitespace eccessivi
- Rimozione markdown e link
- Rimozione pattern comuni di rumore
- Gestione edge cases (testo vuoto, None, etc.)
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
    Fixture: NLPProcessor con modelli mockati per velocità.
    """
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        # Mock spaCy model
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer', 'ner']
        mock_nlp.max_length = 2000000
        mock_spacy.return_value = mock_nlp

        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor()
        return processor


# ============================================================================
# TEST: WHITESPACE NORMALIZATION
# ============================================================================

@pytest.mark.unit
def test_clean_text_removes_double_spaces(nlp_processor):
    """Test: rimuove spazi doppi."""
    text = "This  has   multiple    spaces."

    result = nlp_processor.clean_text(text)

    assert "  " not in result
    assert result == "This has multiple spaces."


@pytest.mark.unit
def test_clean_text_removes_tabs_and_newlines(nlp_processor):
    """Test: normalizza tab e newline."""
    text = "Line 1\n\nLine 2\t\tLine 3"

    result = nlp_processor.clean_text(text)

    assert "\n" not in result
    assert "\t" not in result
    assert result == "Line 1 Line 2 Line 3"


@pytest.mark.unit
def test_clean_text_removes_excessive_whitespace(nlp_processor):
    """Test: rimuove whitespace eccessivi di ogni tipo."""
    text = "Text   with\n\n\nmany\t\t\tspaces"

    result = nlp_processor.clean_text(text)

    assert result == "Text with many spaces"


# ============================================================================
# TEST: MARKDOWN AND LINKS REMOVAL
# ============================================================================

@pytest.mark.unit
def test_clean_text_removes_markdown_links(nlp_processor):
    """Test: rimuove link markdown [text]."""
    text = "This article has [a link] and [another one] embedded."

    result = nlp_processor.clean_text(text)

    assert "[" not in result
    assert "]" not in result
    # La pulizia rimuove i bracket ma può lasciare spazi extra (normalizzati dopo)
    assert "article has" in result and "embedded" in result


@pytest.mark.unit
def test_clean_text_removes_brackets(nlp_processor):
    """Test: rimuove tutte le parentesi quadre con contenuto."""
    text = "Text [with brackets] and [metadata:value] here."

    result = nlp_processor.clean_text(text)

    assert "brackets" not in result
    assert "metadata" not in result


# ============================================================================
# TEST: COMMON NOISE PATTERNS
# ============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("noise_pattern", [
    "Follow us on Twitter",
    "Click here to subscribe",
    "Share this article",
    "Photo:",
    "Source:",
    "Read more:",
    "Subscribe to our newsletter",
    "Sign up for",
    "Related articles:",
    "More from",
    "Advertisement",
    "Sponsored content"
])
def test_clean_text_removes_common_noise(nlp_processor, noise_pattern):
    """Test: rimuove pattern comuni di rumore."""
    text = f"Real content here. {noise_pattern} More content."

    result = nlp_processor.clean_text(text)

    # Pattern dovrebbe essere rimosso (case insensitive)
    assert noise_pattern.lower() not in result.lower()


@pytest.mark.unit
def test_clean_text_case_insensitive_removal(nlp_processor):
    """Test: rimozione pattern è case-insensitive."""
    text = "Content here. FOLLOW US ON TWITTER More content."

    result = nlp_processor.clean_text(text)

    assert "twitter" not in result.lower()


@pytest.mark.unit
def test_clean_text_removes_multiple_noise_patterns(nlp_processor):
    """Test: rimuove multipli pattern di rumore."""
    text = "Article text. Advertisement Photo: test.jpg Click here to subscribe Related articles: end"

    result = nlp_processor.clean_text(text)

    # Tutti i pattern dovrebbero essere rimossi
    assert "advertisement" not in result.lower()
    assert "photo:" not in result.lower()
    assert "click here" not in result.lower()
    assert "related articles" not in result.lower()


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

@pytest.mark.unit
def test_clean_text_handles_empty_string(nlp_processor):
    """Test: gestisce stringa vuota."""
    result = nlp_processor.clean_text("")

    assert result == ""


@pytest.mark.unit
def test_clean_text_handles_none(nlp_processor):
    """Test: gestisce None input."""
    result = nlp_processor.clean_text(None)

    assert result == ""


@pytest.mark.unit
def test_clean_text_handles_whitespace_only(nlp_processor):
    """Test: gestisce testo con solo whitespace."""
    result = nlp_processor.clean_text("   \n\n\t\t   ")

    assert result == ""


@pytest.mark.unit
def test_clean_text_preserves_punctuation(nlp_processor):
    """Test: mantiene punteggiatura normale."""
    text = "Hello, world! This is a test. How are you?"

    result = nlp_processor.clean_text(text)

    assert "," in result
    assert "!" in result
    assert "." in result
    assert "?" in result


@pytest.mark.unit
def test_clean_text_preserves_unicode(nlp_processor):
    """Test: mantiene caratteri unicode (per italiano e altre lingue)."""
    text = "L'articolo è interessante. Più informazioni."

    result = nlp_processor.clean_text(text)

    assert "è" in result
    assert "Più" in result
    assert "L'" in result


@pytest.mark.unit
def test_clean_text_strips_leading_trailing_whitespace(nlp_processor):
    """Test: rimuove whitespace all'inizio e alla fine."""
    text = "   Content here   "

    result = nlp_processor.clean_text(text)

    assert result == "Content here"
    assert not result.startswith(" ")
    assert not result.endswith(" ")


# ============================================================================
# TEST: REAL-WORLD EXAMPLES
# ============================================================================

@pytest.mark.unit
def test_clean_text_realistic_article(nlp_processor):
    """Test: pulizia di un articolo realistico con rumore."""
    text = """
    Breaking News   [Image: news.jpg]

    This is the main   article   content.

    It has multiple paragraphs.

    Photo: Author Name
    Follow us on Twitter
    Subscribe to our newsletter
    """

    result = nlp_processor.clean_text(text)

    # Dovrebbe mantenere il contenuto principale
    assert "This is the main article content" in result
    assert "It has multiple paragraphs" in result

    # Dovrebbe rimuovere il rumore
    assert "Photo:" not in result
    assert "Follow us" not in result
    assert "Subscribe" not in result
    assert "[Image" not in result


@pytest.mark.unit
def test_clean_text_preserves_sentence_structure(nlp_processor):
    """Test: mantiene struttura delle frasi."""
    text = "First sentence. Second sentence. Third sentence."

    result = nlp_processor.clean_text(text)

    assert result == text
    assert result.count(".") == 3


@pytest.mark.unit
def test_clean_text_handles_html_like_content(nlp_processor):
    """Test: gestisce contenuto simil-HTML dopo scraping."""
    text = "Content here   [div]   More content   [/div]   End"

    result = nlp_processor.clean_text(text)

    # Dovrebbe rimuovere le parti tra []
    assert "[div]" not in result
    assert "[/div]" not in result
    # Verifica che il contenuto sia presente (gli spazi sono normalizzati)
    assert "Content here" in result and "More content" in result and "End" in result
