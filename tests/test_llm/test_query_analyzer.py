"""
Tests for QueryAnalyzer module.

Tests the pre-search filter extraction from natural language queries.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.query_analyzer import QueryAnalyzer, merge_filters, get_query_analyzer
from src.llm.schemas import ExtractedFilters


class TestExtractedFiltersSchema:
    """Test the Pydantic schema for extracted filters."""

    def test_valid_filters_full(self):
        """Test schema with all fields populated."""
        filters = ExtractedFilters(
            start_date="2024-12-01",
            end_date="2024-12-15",
            categories=["CYBER", "DEFENSE"],
            gpe_filter=["Russia", "Ukraine"],
            sources=["Reuters"],
            semantic_query="cyber attacks infrastructure",
            extraction_confidence=0.9
        )

        assert filters.start_date == "2024-12-01"
        assert filters.end_date == "2024-12-15"
        assert "CYBER" in filters.categories
        assert "Russia" in filters.gpe_filter
        assert filters.extraction_confidence == 0.9

    def test_valid_filters_minimal(self):
        """Test schema with only required fields."""
        filters = ExtractedFilters(
            semantic_query="Taiwan tensions",
            extraction_confidence=0.5
        )

        assert filters.start_date is None
        assert filters.categories is None
        assert filters.semantic_query == "Taiwan tensions"

    def test_invalid_confidence_too_high(self):
        """Test that confidence > 1.0 raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ExtractedFilters(
                semantic_query="test",
                extraction_confidence=1.5  # Invalid
            )

    def test_invalid_category(self):
        """Test that invalid category raises validation error."""
        with pytest.raises(Exception):
            ExtractedFilters(
                categories=["INVALID_CATEGORY"],
                semantic_query="test",
                extraction_confidence=0.5
            )


class TestMergeFilters:
    """Test the filter merging logic."""

    def test_ui_takes_precedence(self):
        """UI filters should override extracted filters."""
        extracted = {
            'start_date': datetime(2024, 12, 1),
            'end_date': datetime(2024, 12, 15),
            'categories': ['CYBER'],
            'gpe_filter': ['Russia'],
            'sources': None,
            'semantic_query': 'cyber attacks'
        }

        merged = merge_filters(
            extracted,
            ui_start_date=datetime(2024, 11, 1),  # Override
            ui_end_date=None,  # Use extracted
            ui_categories=['DEFENSE'],  # Override
            ui_gpe_filter=None,  # Use extracted
            ui_sources=['Bloomberg']  # Override
        )

        assert merged['start_date'] == datetime(2024, 11, 1)  # UI
        assert merged['end_date'] == datetime(2024, 12, 15)  # Extracted
        assert merged['categories'] == ['DEFENSE']  # UI
        assert merged['gpe_filter'] == ['Russia']  # Extracted
        assert merged['sources'] == ['Bloomberg']  # UI

    def test_extracted_only(self):
        """When no UI filters, use extracted."""
        extracted = {
            'start_date': datetime(2024, 12, 1),
            'categories': ['ECONOMY'],
            'gpe_filter': ['China', 'Taiwan'],
            'semantic_query': 'semiconductor tensions'
        }

        merged = merge_filters(extracted)

        assert merged['start_date'] == datetime(2024, 12, 1)
        assert merged['categories'] == ['ECONOMY']
        assert 'Taiwan' in merged['gpe_filter']

    def test_empty_extracted(self):
        """When extracted is empty, UI filters are used."""
        merged = merge_filters(
            {},
            ui_start_date=datetime(2024, 12, 1),
            ui_categories=['GEOPOLITICS']
        )

        assert merged['start_date'] == datetime(2024, 12, 1)
        assert merged['categories'] == ['GEOPOLITICS']


class TestQueryAnalyzerMocked:
    """Test QueryAnalyzer with mocked Gemini responses."""

    @pytest.fixture
    def mock_gemini_response(self):
        """Create a mock Gemini response."""
        mock_response = Mock()
        mock_response.text = '''
        {
            "start_date": "2024-12-08",
            "end_date": "2024-12-15",
            "categories": ["CYBER"],
            "gpe_filter": ["Russia"],
            "sources": null,
            "semantic_query": "cyber attacks infrastructure",
            "extraction_confidence": 0.85
        }
        '''
        return mock_response

    @patch('src.llm.query_analyzer.genai')
    def test_analyze_success(self, mock_genai, mock_gemini_response):
        """Test successful query analysis."""
        # Setup mock
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Quali attacchi cyber dalla Russia negli ultimi 7 giorni?")

        assert result['success'] is True
        assert result['filters']['gpe_filter'] == ['Russia']
        assert result['filters']['categories'] == ['CYBER']
        assert result['filters']['extraction_confidence'] == 0.85
        # Dates should be converted to datetime
        assert isinstance(result['filters']['start_date'], datetime)

    @patch('src.llm.query_analyzer.genai')
    def test_analyze_fallback_on_error(self, mock_genai):
        """Test fallback when Gemini fails."""
        # Setup mock to raise exception
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model

        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Test query")

        assert result['success'] is False
        assert result['filters']['semantic_query'] == "Test query"  # Original query
        assert result['filters']['extraction_confidence'] == 0.0
        assert 'error' in result

    @patch('src.llm.query_analyzer.genai')
    def test_analyze_invalid_json_fallback(self, mock_genai):
        """Test fallback when Gemini returns invalid JSON."""
        mock_response = Mock()
        mock_response.text = "This is not valid JSON"

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Test query")

        assert result['success'] is False
        assert result['filters']['semantic_query'] == "Test query"


class TestQueryAnalyzerDateParsing:
    """Test date string to datetime conversion."""

    def test_post_process_valid_dates(self):
        """Test conversion of valid ISO date strings."""
        filters = {
            'start_date': '2024-12-01',
            'end_date': '2024-12-15',
            'semantic_query': 'test'
        }

        # Create analyzer with mock
        with patch('src.llm.query_analyzer.genai'):
            analyzer = QueryAnalyzer()
            processed = analyzer._post_process_dates(filters)

        assert isinstance(processed['start_date'], datetime)
        assert processed['start_date'].year == 2024
        assert processed['start_date'].month == 12
        assert processed['start_date'].day == 1

    def test_post_process_invalid_dates(self):
        """Test handling of invalid date strings."""
        filters = {
            'start_date': 'not-a-date',
            'end_date': None,
            'semantic_query': 'test'
        }

        with patch('src.llm.query_analyzer.genai'):
            analyzer = QueryAnalyzer()
            processed = analyzer._post_process_dates(filters)

        assert processed['start_date'] is None  # Invalid date becomes None
        assert processed['end_date'] is None


class TestSingletonFactory:
    """Test the singleton factory function."""

    @patch('src.llm.query_analyzer.genai')
    def test_singleton_returns_same_instance(self, mock_genai):
        """get_query_analyzer should return the same instance."""
        # Reset singleton
        import src.llm.query_analyzer as qa_module
        qa_module._analyzer_instance = None

        instance1 = get_query_analyzer()
        instance2 = get_query_analyzer()

        assert instance1 is instance2


# =============================================================================
# Integration-style tests (these require actual Gemini API)
# Skip by default, run with: pytest -m integration
# =============================================================================

@pytest.mark.integration
@pytest.mark.skip(reason="Requires Gemini API key - run manually")
class TestQueryAnalyzerIntegration:
    """Integration tests with real Gemini API."""

    def test_real_temporal_query(self):
        """Test with real API - temporal query."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Cosa e successo a Taiwan negli ultimi 7 giorni?")

        assert result['success'] is True
        assert 'Taiwan' in result['filters'].get('gpe_filter', [])
        # Date should be within last week
        if result['filters'].get('start_date'):
            assert result['filters']['start_date'] >= datetime.now() - timedelta(days=10)

    def test_real_category_inference(self):
        """Test with real API - category inference."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Analisi delle minacce cyber dalla Russia")

        assert result['success'] is True
        assert 'CYBER' in result['filters'].get('categories', [])
        assert 'Russia' in result['filters'].get('gpe_filter', [])
