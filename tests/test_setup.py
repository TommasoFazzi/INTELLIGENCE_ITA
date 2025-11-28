"""
Test di verifica setup pytest.

Questo file contiene test base per verificare che:
1. Pytest è configurato correttamente
2. Le fixtures funzionano
3. I markers funzionano
"""

import pytest


def test_pytest_works():
    """Test più semplice possibile: verifica che pytest funzioni."""
    assert True


def test_basic_math():
    """Test banale per verificare che gli assert funzionino."""
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"


@pytest.mark.unit
def test_marker_unit():
    """Test con marker 'unit'."""
    assert True


def test_fixture_sample_article(sample_article):
    """
    Test che usa la fixture sample_article.

    Verifica che la fixture sia caricata correttamente da conftest.py
    """
    assert 'title' in sample_article
    assert 'link' in sample_article
    assert sample_article['category'] == 'intelligence'
    assert len(sample_article['title']) > 0


def test_fixture_sample_embedding(sample_embedding):
    """Test che usa la fixture sample_embedding."""
    assert len(sample_embedding) == 384
    assert all(isinstance(x, (int, float)) for x in sample_embedding)


def test_fixture_sample_articles_batch(sample_articles_batch):
    """Test che usa la fixture sample_articles_batch."""
    assert len(sample_articles_batch) == 3
    assert all('title' in article for article in sample_articles_batch)


@pytest.mark.parametrize("input_value,expected", [
    (1, 1),
    (2, 2),
    ("test", "test"),
])
def test_parametrize_works(input_value, expected):
    """Test per verificare che @pytest.mark.parametrize funzioni."""
    assert input_value == expected
