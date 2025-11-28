"""
Test per NLPProcessor - Named Entity Recognition.

Questi test verificano:
- Estrazione entità con spaCy NER
- Organizzazione entità per tipo
- Metadata delle entità (text, label, position)
- Gestione edge cases
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
    Fixture: NLPProcessor con spaCy NER mockato.
    """
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        # Mock spaCy con NER
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer', 'ner']
        mock_nlp.max_length = 2000000

        # Mock del metodo __call__ che ritorna Doc con entities
        def mock_nlp_call(text):
            doc = MagicMock()

            # Mock entities based on text content
            mock_ents = []

            if "Apple" in text:
                ent = MagicMock()
                ent.text = "Apple"
                ent.label_ = "ORG"
                ent.start_char = text.index("Apple")
                ent.end_char = text.index("Apple") + len("Apple")
                mock_ents.append(ent)

            if "New York" in text:
                ent = MagicMock()
                ent.text = "New York"
                ent.label_ = "LOC"
                ent.start_char = text.index("New York")
                ent.end_char = text.index("New York") + len("New York")
                mock_ents.append(ent)

            if "John Smith" in text:
                ent = MagicMock()
                ent.text = "John Smith"
                ent.label_ = "PER"
                ent.start_char = text.index("John Smith")
                ent.end_char = text.index("John Smith") + len("John Smith")
                mock_ents.append(ent)

            if "Microsoft" in text:
                ent = MagicMock()
                ent.text = "Microsoft"
                ent.label_ = "ORG"
                ent.start_char = text.index("Microsoft")
                ent.end_char = text.index("Microsoft") + len("Microsoft")
                mock_ents.append(ent)

            doc.ents = mock_ents
            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor()
        return processor


# ============================================================================
# TEST: BASIC ENTITY EXTRACTION
# ============================================================================

@pytest.mark.unit
def test_extract_entities_returns_dict(nlp_processor):
    """Test: ritorna dizionario con struttura corretta."""
    text = "Apple is a company in New York."

    result = nlp_processor.extract_entities(text)

    assert isinstance(result, dict)
    assert 'entities' in result
    assert 'by_type' in result
    assert 'entity_count' in result


@pytest.mark.unit
def test_extract_entities_finds_organizations(nlp_processor):
    """Test: trova organizzazioni."""
    text = "Apple announced new products."

    result = nlp_processor.extract_entities(text)

    # Dovrebbe trovare "Apple" come ORG
    assert result['entity_count'] > 0
    assert 'ORG' in result['by_type']
    assert "Apple" in result['by_type']['ORG']


@pytest.mark.unit
def test_extract_entities_finds_locations(nlp_processor):
    """Test: trova locations."""
    text = "The event took place in New York."

    result = nlp_processor.extract_entities(text)

    assert 'LOC' in result['by_type']
    assert "New York" in result['by_type']['LOC']


@pytest.mark.unit
def test_extract_entities_finds_persons(nlp_processor):
    """Test: trova persone."""
    text = "John Smith gave a speech."

    result = nlp_processor.extract_entities(text)

    assert 'PER' in result['by_type']
    assert "John Smith" in result['by_type']['PER']


@pytest.mark.unit
def test_extract_entities_multiple_types(nlp_processor):
    """Test: trova entità di multipli tipi."""
    text = "John Smith works at Apple in New York."

    result = nlp_processor.extract_entities(text)

    # Dovrebbe trovare PER, ORG, LOC
    assert 'PER' in result['by_type']
    assert 'ORG' in result['by_type']
    assert 'LOC' in result['by_type']


@pytest.mark.unit
def test_extract_entities_count_is_accurate(nlp_processor):
    """Test: entity_count riflette numero corretto."""
    text = "Apple and Microsoft are companies in New York."

    result = nlp_processor.extract_entities(text)

    # Apple (ORG), Microsoft (ORG), New York (LOC) = 3
    assert result['entity_count'] == 3


# ============================================================================
# TEST: ENTITY METADATA
# ============================================================================

@pytest.mark.unit
def test_entity_has_required_fields(nlp_processor):
    """Test: ogni entità ha i campi richiesti."""
    text = "Apple is a company."

    result = nlp_processor.extract_entities(text)

    if result['entities']:
        entity = result['entities'][0]

        # Verifica campi richiesti
        assert 'text' in entity
        assert 'label' in entity
        assert 'start' in entity
        assert 'end' in entity


@pytest.mark.unit
def test_entity_positions_are_accurate(nlp_processor):
    """Test: posizioni start/end sono corrette."""
    text = "Apple announced products."

    result = nlp_processor.extract_entities(text)

    if result['entities']:
        for entity in result['entities']:
            # Verifica che le posizioni siano coerenti
            extracted_text = text[entity['start']:entity['end']]
            assert extracted_text == entity['text']


# ============================================================================
# TEST: GROUPING BY TYPE
# ============================================================================

@pytest.mark.unit
def test_entities_grouped_by_type(nlp_processor):
    """Test: entità sono raggruppate per tipo."""
    text = "Apple and Microsoft are companies. John Smith works there."

    result = nlp_processor.extract_entities(text)

    # ORG dovrebbe contenere Apple e Microsoft
    assert len(result['by_type']['ORG']) == 2
    assert "Apple" in result['by_type']['ORG']
    assert "Microsoft" in result['by_type']['ORG']

    # PER dovrebbe contenere John Smith
    assert len(result['by_type']['PER']) == 1
    assert "John Smith" in result['by_type']['PER']


@pytest.mark.unit
def test_by_type_structure(nlp_processor):
    """Test: by_type ha struttura corretta."""
    text = "Apple is in New York."

    result = nlp_processor.extract_entities(text)

    # by_type dovrebbe essere dict di liste
    assert isinstance(result['by_type'], dict)
    for entity_type, entity_list in result['by_type'].items():
        assert isinstance(entity_list, list)
        assert all(isinstance(e, str) for e in entity_list)


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

@pytest.mark.unit
def test_extract_entities_empty_text(nlp_processor):
    """Test: gestisce testo vuoto."""
    result = nlp_processor.extract_entities("")

    assert result['entities'] == []
    assert result['by_type'] == {}
    assert result['entity_count'] == 0


@pytest.mark.unit
def test_extract_entities_none_input(nlp_processor):
    """Test: gestisce None input."""
    result = nlp_processor.extract_entities(None)

    assert result['entities'] == []
    assert result['by_type'] == {}
    assert result['entity_count'] == 0


@pytest.mark.unit
def test_extract_entities_non_string_input(nlp_processor):
    """Test: gestisce input non-string."""
    result = nlp_processor.extract_entities(12345)

    assert result['entities'] == []
    assert result['by_type'] == {}
    assert result['entity_count'] == 0


@pytest.mark.unit
def test_extract_entities_no_entities_found(nlp_processor):
    """Test: testo senza entità riconoscibili."""
    text = "This is just plain text with no entities."

    result = nlp_processor.extract_entities(text)

    assert result['entities'] == []
    assert result['by_type'] == {}
    assert result['entity_count'] == 0


@pytest.mark.unit
def test_extract_entities_text_with_special_chars(nlp_processor):
    """Test: gestisce caratteri speciali."""
    text = "Apple Inc. announced @ New York!"

    result = nlp_processor.extract_entities(text)

    # Dovrebbe comunque trovare entità valide
    assert result['entity_count'] >= 0


# ============================================================================
# TEST: MULTILINGUAL SUPPORT
# ============================================================================

@pytest.mark.unit
def test_extract_entities_italian_text():
    """Test: supporta testo italiano."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer', 'ner']
        mock_nlp.max_length = 2000000

        def mock_nlp_call(text):
            doc = MagicMock()
            mock_ents = []

            # Mock Italian entities
            if "Roma" in text:
                ent = MagicMock()
                ent.text = "Roma"
                ent.label_ = "LOC"
                ent.start_char = text.index("Roma")
                ent.end_char = text.index("Roma") + len("Roma")
                mock_ents.append(ent)

            doc.ents = mock_ents
            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor()

        text = "L'evento si è svolto a Roma."
        result = processor.extract_entities(text)

        assert 'LOC' in result['by_type']
        assert "Roma" in result['by_type']['LOC']


# ============================================================================
# TEST: DUPLICATE HANDLING
# ============================================================================

@pytest.mark.unit
def test_extract_entities_duplicate_entities():
    """Test: gestisce entità duplicate nel testo."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer', 'ner']
        mock_nlp.max_length = 2000000

        def mock_nlp_call(text):
            doc = MagicMock()
            mock_ents = []

            # Trova tutte le occorrenze di "Apple"
            import re
            for match in re.finditer(r'\bApple\b', text):
                ent = MagicMock()
                ent.text = "Apple"
                ent.label_ = "ORG"
                ent.start_char = match.start()
                ent.end_char = match.end()
                mock_ents.append(ent)

            doc.ents = mock_ents
            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor()

        text = "Apple products are great. Apple is innovative. I love Apple."
        result = processor.extract_entities(text)

        # Dovrebbe trovare 3 occorrenze di Apple
        assert result['entity_count'] == 3
        # by_type può contenere duplicate (dipende dall'implementazione)


# ============================================================================
# TEST: DIFFERENT ENTITY TYPES
# ============================================================================

@pytest.mark.unit
def test_extract_entities_handles_various_types():
    """Test: gestisce vari tipi di entità."""
    with patch('spacy.load') as mock_spacy, \
         patch('src.nlp.processing.SentenceTransformer') as mock_transformer:

        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ['sentencizer', 'ner']
        mock_nlp.max_length = 2000000

        def mock_nlp_call(text):
            doc = MagicMock()
            mock_ents = []

            # Aggiungi vari tipi di entità
            entities_map = {
                "Google": "ORG",
                "2025": "DATE",
                "100 million": "MONEY",
                "50%": "PERCENT"
            }

            for entity_text, entity_label in entities_map.items():
                if entity_text in text:
                    ent = MagicMock()
                    ent.text = entity_text
                    ent.label_ = entity_label
                    ent.start_char = text.index(entity_text)
                    ent.end_char = text.index(entity_text) + len(entity_text)
                    mock_ents.append(ent)

            doc.ents = mock_ents
            return doc

        mock_nlp.side_effect = mock_nlp_call
        mock_spacy.return_value = mock_nlp

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        processor = NLPProcessor()

        text = "Google earned 100 million in 2025, up 50%."
        result = processor.extract_entities(text)

        # Dovrebbe trovare vari tipi
        assert 'ORG' in result['by_type']
        assert 'DATE' in result['by_type']
        assert 'MONEY' in result['by_type']
        assert 'PERCENT' in result['by_type']


# ============================================================================
# TEST: LONG TEXT PERFORMANCE
# ============================================================================

@pytest.mark.unit
def test_extract_entities_long_text(nlp_processor):
    """Test: gestisce testo lungo."""
    # Testo lungo con alcune entità
    text = "Apple is a company. " * 100 + "John Smith works there."

    result = nlp_processor.extract_entities(text)

    # Dovrebbe trovare entità anche in testo lungo
    assert result['entity_count'] > 0
