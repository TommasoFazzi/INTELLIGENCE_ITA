"""
Stopword Filtering for Intelligence Domain

Rimuove termini troppo generici dalle query per migliorare la precisione della ricerca semantica.
Esempio: "intelligence Taiwan" → "Taiwan" (solo il named entity rilevante)

Usage:
    from src.utils.stopwords import clean_query

    query = "latest intelligence report on Taiwan tensions"
    cleaned = clean_query(query)  # → "Taiwan tensions"
"""

from typing import List, Set
import spacy

# Intelligence domain stopwords (da rimuovere dalla query)
INTELLIGENCE_STOPWORDS = {
    # Core domain terms
    "intelligence", "report", "briefing", "analysis", "strategic",
    "tactical", "operational", "summary", "update", "alert",

    # Document types
    "daily", "weekly", "monthly", "quarterly", "article", "news",

    # Analysis terms
    "trends", "developments", "situation", "overview", "assessment",
    "implications", "outlook", "forecast", "projection",

    # Generic query terms
    "latest", "recent", "new", "current", "ongoing", "emerging"
}

# Preserve these even if they appear generic (critical entities)
PRESERVE_TERMS = {
    # Organizations
    "nato", "un", "eu", "opec", "brics",

    # Military/Defense
    "cyber", "defense", "military", "security",

    # Economic
    "economy", "sanctions", "trade", "energy",

    # Tech
    "ai", "semiconductor", "chip", "technology"
}


class QueryCleaner:
    """Clean and filter queries for better semantic search."""

    def __init__(self):
        """Initialize query cleaner with spaCy NER model."""
        # Load spaCy for NER (preserve named entities)
        try:
            self.nlp = spacy.load("xx_ent_wiki_sm")
        except OSError:
            print("Warning: spaCy model 'xx_ent_wiki_sm' not found. Named entity preservation disabled.")
            print("To enable: python -m spacy download xx_ent_wiki_sm")
            self.nlp = None

    def clean_query(self, query: str, preserve_entities: bool = True) -> str:
        """
        Remove intelligence domain stopwords from query.

        Args:
            query: User's search query
            preserve_entities: If True, preserve named entities (GPE, ORG, PERSON)

        Returns:
            Cleaned query with stopwords removed

        Examples:
            >>> cleaner = QueryCleaner()
            >>> cleaner.clean_query("latest intelligence report Taiwan")
            'Taiwan'
            >>> cleaner.clean_query("recent cyber threats China")
            'cyber threats China'
        """
        # Extract entities to preserve
        entities_to_preserve = set()
        if preserve_entities and self.nlp:
            try:
                doc = self.nlp(query)
                for ent in doc.ents:
                    # Preserve GPE (countries), ORG (companies), PERSON (names), LOC (locations)
                    if ent.label_ in {"GPE", "ORG", "PERSON", "LOC"}:
                        entities_to_preserve.add(ent.text.lower())
            except Exception as e:
                # Fallback if NER fails
                print(f"Warning: NER failed ({e}), proceeding without entity preservation")

        # Tokenize and filter
        words = query.lower().split()
        filtered = []

        for word in words:
            # Keep if:
            # 1. Not a stopword, OR
            # 2. In preserve list, OR
            # 3. Is a named entity
            if (word not in INTELLIGENCE_STOPWORDS or
                word in PRESERVE_TERMS or
                word in entities_to_preserve):
                filtered.append(word)

        cleaned = " ".join(filtered)

        # Fallback: if query becomes empty, return original
        return cleaned if cleaned.strip() else query

    def get_stopwords(self) -> Set[str]:
        """Return the stopword set."""
        return INTELLIGENCE_STOPWORDS.copy()


# Singleton instance
_cleaner = QueryCleaner()


def clean_query(query: str) -> str:
    """
    Clean query by removing intelligence domain stopwords.

    Args:
        query: User's search query

    Returns:
        Cleaned query with stopwords removed

    Examples:
        >>> clean_query("latest intelligence report on Taiwan tensions")
        'Taiwan tensions'
        >>> clean_query("cyber security threats Russia")
        'cyber security threats Russia'
    """
    return _cleaner.clean_query(query)


def get_stopwords() -> Set[str]:
    """
    Get the set of intelligence domain stopwords.

    Returns:
        Set of stopword strings
    """
    return _cleaner.get_stopwords()
