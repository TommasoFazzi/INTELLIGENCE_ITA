"""
Test per ContentExtractor - Estrazione testo da URL.

Questi test verificano:
- Estrazione con Trafilatura
- Fallback a Newspaper3k
- Gestione errori HTTP
- Batch extraction
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.ingestion.content_extractor import ContentExtractor


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def extractor():
    """Fixture: ContentExtractor con configurazione base."""
    return ContentExtractor(timeout=10)


@pytest.fixture
def mock_html_content():
    """HTML di esempio per i test."""
    return """
    <html>
        <head><title>Test Article</title></head>
        <body>
            <article>
                <h1>Test Article Title</h1>
                <p>This is the main content of the article.</p>
                <p>It contains multiple paragraphs.</p>
            </article>
        </body>
    </html>
    """


@pytest.fixture
def mock_trafilatura_response():
    """Mock della risposta JSON di Trafilatura."""
    return """{
        "title": "Test Article from Trafilatura",
        "author": "Test Author",
        "date": "2025-11-28",
        "text": "This is the extracted text content from Trafilatura.",
        "description": "Article description",
        "sitename": "Example Site"
    }"""


# ============================================================================
# TEST: INIZIALIZZAZIONE
# ============================================================================

@pytest.mark.unit
def test_content_extractor_initialization():
    """Test: ContentExtractor si inizializza correttamente."""
    extractor = ContentExtractor()

    assert extractor is not None
    assert extractor.timeout == 10
    assert extractor.user_agent is not None
    assert extractor.session is not None


@pytest.mark.unit
def test_content_extractor_custom_timeout():
    """Test: timeout personalizzato viene applicato."""
    extractor = ContentExtractor(timeout=30)

    assert extractor.timeout == 30


@pytest.mark.unit
def test_content_extractor_custom_user_agent():
    """Test: user agent personalizzato viene applicato."""
    custom_ua = "CustomBot/1.0"
    extractor = ContentExtractor(user_agent=custom_ua)

    assert extractor.user_agent == custom_ua
    assert extractor.session.headers['User-Agent'] == custom_ua


# ============================================================================
# TEST: EXTRACT WITH TRAFILATURA
# ============================================================================

@pytest.mark.unit
@patch('trafilatura.fetch_url')
@patch('trafilatura.extract')
def test_extract_with_trafilatura_success(mock_extract, mock_fetch, extractor, mock_html_content, mock_trafilatura_response):
    """Test: estrazione con Trafilatura funziona."""
    mock_fetch.return_value = mock_html_content
    mock_extract.return_value = mock_trafilatura_response

    result = extractor.extract_with_trafilatura("https://example.com/article")

    assert result is not None
    assert result['title'] == "Test Article from Trafilatura"
    assert result['text'] == "This is the extracted text content from Trafilatura."
    assert result['author'] == "Test Author"
    assert result['extraction_method'] == 'trafilatura'


@pytest.mark.unit
@patch('trafilatura.extract')
def test_extract_with_trafilatura_with_preloaded_html(mock_extract, extractor, mock_html_content, mock_trafilatura_response):
    """Test: Trafilatura può usare HTML già caricato."""
    mock_extract.return_value = mock_trafilatura_response

    result = extractor.extract_with_trafilatura("https://example.com", html=mock_html_content)

    assert result is not None
    assert result['extraction_method'] == 'trafilatura'
    # fetch_url non dovrebbe essere chiamato
    mock_extract.assert_called_once()


@pytest.mark.unit
@patch('trafilatura.fetch_url')
@patch('trafilatura.extract')
def test_extract_with_trafilatura_returns_none_on_failure(mock_extract, mock_fetch, extractor):
    """Test: Trafilatura ritorna None se estrazione fallisce."""
    mock_fetch.return_value = "<html><body>Empty</body></html>"
    mock_extract.return_value = None  # Extraction failed

    result = extractor.extract_with_trafilatura("https://example.com")

    assert result is None


@pytest.mark.unit
@patch('trafilatura.fetch_url')
def test_extract_with_trafilatura_handles_network_error(mock_fetch, extractor):
    """Test: gestisce errori di rete."""
    mock_fetch.side_effect = Exception("Network error")

    result = extractor.extract_with_trafilatura("https://example.com")

    assert result is None  # Non deve crashare


# ============================================================================
# TEST: EXTRACT WITH NEWSPAPER
# ============================================================================

@pytest.mark.unit
@patch('src.ingestion.content_extractor.NewspaperArticle')
def test_extract_with_newspaper_success(mock_article_class, extractor):
    """Test: estrazione con Newspaper3k funziona."""
    # Mock dell'oggetto Article
    mock_article = Mock()
    mock_article.text = "This is the article text from Newspaper3k."
    mock_article.title = "Test Article from Newspaper"
    mock_article.authors = ["Author 1", "Author 2"]
    mock_article.publish_date = None
    mock_article.meta_description = "Description"
    mock_article.source_url = "https://example.com"
    mock_article.top_image = "https://example.com/image.jpg"

    # Mock dei metodi download e parse
    mock_article.download = Mock()
    mock_article.parse = Mock()

    mock_article_class.return_value = mock_article

    result = extractor.extract_with_newspaper("https://example.com/article")

    assert result is not None
    assert result['text'] == "This is the article text from Newspaper3k."
    assert result['title'] == "Test Article from Newspaper"
    assert result['author'] == "Author 1, Author 2"
    assert result['extraction_method'] == 'newspaper3k'


@pytest.mark.unit
@patch('src.ingestion.content_extractor.NewspaperArticle')
def test_extract_with_newspaper_handles_no_text(mock_article_class, extractor):
    """Test: Newspaper ritorna None se non c'è testo."""
    mock_article = Mock()
    mock_article.text = ""  # No text extracted
    mock_article.download = Mock()
    mock_article.parse = Mock()

    mock_article_class.return_value = mock_article

    result = extractor.extract_with_newspaper("https://example.com")

    assert result is None


@pytest.mark.unit
@patch('src.ingestion.content_extractor.NewspaperArticle')
def test_extract_with_newspaper_handles_error(mock_article_class, extractor):
    """Test: gestisce errori durante download/parsing."""
    mock_article = Mock()
    mock_article.download.side_effect = Exception("Download failed")

    mock_article_class.return_value = mock_article

    result = extractor.extract_with_newspaper("https://example.com")

    assert result is None  # Non deve crashare


# ============================================================================
# TEST: EXTRACT CONTENT (Fallback Logic)
# ============================================================================

@pytest.mark.unit
@patch('trafilatura.fetch_url')
@patch('trafilatura.extract')
def test_extract_content_uses_trafilatura_first(mock_extract, mock_fetch, extractor, mock_html_content, mock_trafilatura_response):
    """Test: extract_content prova prima Trafilatura."""
    mock_fetch.return_value = mock_html_content
    mock_extract.return_value = mock_trafilatura_response

    result = extractor.extract_content("https://example.com/article")

    assert result is not None
    assert result['extraction_method'] == 'trafilatura'


@pytest.mark.unit
@patch('trafilatura.fetch_url')
@patch('trafilatura.extract')
@patch('src.ingestion.content_extractor.NewspaperArticle')
def test_extract_content_falls_back_to_newspaper(mock_article_class, mock_extract, mock_fetch, extractor):
    """Test: se Trafilatura fallisce, usa Newspaper3k."""
    # Trafilatura fallisce
    mock_fetch.return_value = "<html></html>"
    mock_extract.return_value = None

    # Newspaper3k funziona
    mock_article = Mock()
    mock_article.text = "Text from Newspaper"
    mock_article.title = "Title"
    mock_article.authors = []
    mock_article.publish_date = None
    mock_article.meta_description = None
    mock_article.source_url = "https://example.com"
    mock_article.top_image = None
    mock_article.download = Mock()
    mock_article.parse = Mock()

    mock_article_class.return_value = mock_article

    result = extractor.extract_content("https://example.com/article")

    assert result is not None
    assert result['extraction_method'] == 'newspaper3k'
    assert result['text'] == "Text from Newspaper"


@pytest.mark.unit
@patch('trafilatura.fetch_url')
@patch('trafilatura.extract')
@patch('src.ingestion.content_extractor.NewspaperArticle')
def test_extract_content_returns_none_when_both_fail(mock_article_class, mock_extract, mock_fetch, extractor):
    """Test: ritorna None se entrambi i metodi falliscono."""
    # Trafilatura fallisce
    mock_fetch.return_value = None
    mock_extract.return_value = None

    # Newspaper fallisce
    mock_article = Mock()
    mock_article.download.side_effect = Exception("Failed")

    mock_article_class.return_value = mock_article

    result = extractor.extract_content("https://example.com/article")

    assert result is None


# ============================================================================
# TEST: EXTRACT BATCH
# ============================================================================

@pytest.mark.unit
@patch.object(ContentExtractor, 'extract_content')
def test_extract_batch_processes_all_articles(mock_extract, extractor):
    """Test: extract_batch processa tutti gli articoli."""
    # Mock extract_content per ritornare successo
    mock_extract.return_value = {'text': 'Extracted content', 'title': 'Test'}

    articles = [
        {'title': 'Article 1', 'link': 'https://example.com/1'},
        {'title': 'Article 2', 'link': 'https://example.com/2'},
        {'title': 'Article 3', 'link': 'https://example.com/3'},
    ]

    results = extractor.extract_batch(articles)

    assert len(results) == 3
    assert all('full_content' in article for article in results)
    assert all(article['extraction_success'] for article in results)


@pytest.mark.unit
@patch.object(ContentExtractor, 'extract_content')
def test_extract_batch_handles_missing_url(mock_extract, extractor):
    """Test: gestisce articoli senza URL."""
    articles = [
        {'title': 'No URL Article'},  # Manca 'link'
    ]

    results = extractor.extract_batch(articles)

    assert len(results) == 1
    # Non deve aver chiamato extract_content
    mock_extract.assert_not_called()


@pytest.mark.unit
@patch.object(ContentExtractor, 'extract_content')
def test_extract_batch_tracks_success_and_failures(mock_extract, extractor):
    """Test: traccia successi e fallimenti."""
    # Prima chiamata successo, seconda fallimento
    mock_extract.side_effect = [
        {'text': 'Success'},
        None,
        {'text': 'Success again'}
    ]

    articles = [
        {'title': 'Article 1', 'link': 'https://example.com/1'},
        {'title': 'Article 2', 'link': 'https://example.com/2'},
        {'title': 'Article 3', 'link': 'https://example.com/3'},
    ]

    results = extractor.extract_batch(articles)

    assert results[0]['extraction_success'] is True
    assert results[1]['extraction_success'] is False
    assert results[2]['extraction_success'] is True


@pytest.mark.unit
@patch.object(ContentExtractor, 'extract_content')
def test_extract_batch_handles_exceptions(mock_extract, extractor):
    """Test: continua anche se un'estrazione lancia exception."""
    mock_extract.side_effect = Exception("Unexpected error")

    articles = [
        {'title': 'Article 1', 'link': 'https://example.com/1'},
    ]

    results = extractor.extract_batch(articles)

    assert len(results) == 1
    assert results[0]['extraction_success'] is False
    assert 'extraction_error' in results[0]


@pytest.mark.unit
@patch.object(ContentExtractor, 'extract_content')
def test_extract_batch_adds_timestamp(mock_extract, extractor):
    """Test: aggiunge timestamp all'estrazione."""
    mock_extract.return_value = {'text': 'Content'}

    articles = [{'title': 'Test', 'link': 'https://example.com'}]

    results = extractor.extract_batch(articles)

    assert 'extraction_timestamp' in results[0]


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

@pytest.mark.unit
@patch('trafilatura.fetch_url')
@patch('trafilatura.extract')
def test_extract_handles_empty_text(mock_extract, mock_fetch, extractor):
    """Test: gestisce testo vuoto dopo estrazione."""
    mock_fetch.return_value = "<html></html>"
    mock_extract.return_value = '{"text": "", "title": "Empty"}'

    result = extractor.extract_with_trafilatura("https://example.com")

    # Trafilatura ritorna dict ma con text vuoto
    assert result is not None
    assert result['text'] == ""


@pytest.mark.unit
def test_extract_batch_empty_list(extractor):
    """Test: extract_batch con lista vuota."""
    results = extractor.extract_batch([])

    assert results == []
