"""
AI-powered bullet point generator for articles.

Uses T5 (Gemini 2.5 Flash-Lite) via LLMFactory to generate 3 concise bullet
points summarizing the key facts of an article, similar to Bloomberg Terminal's
AI summaries.
"""

import json
import time
from typing import Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)

BULLET_GENERATION_PROMPT = (
    "Sei un analista di intelligence. Genera 3 bullet points concisi in italiano "
    "che sintetizzano i fatti chiave di questo articolo.\n\n"
    "REGOLE:\n"
    "- Ogni bullet deve essere 1-2 frasi massimo (15-30 parole)\n"
    "- Concentrati su fatti oggettivi, non opinioni\n"
    "- Ordina per importanza (più rilevante per primo)\n"
    "- Usa linguaggio da intelligence analyst (preciso, conciso, formale)\n"
    "- Se il testo è troppo breve o non contiene informazioni sufficienti, rispondi solo con il testo presente\n\n"
    "FORMATO RISPOSTA: JSON array con 3 stringhe, es.:\n"
    '["Fatto 1", "Fatto 2", "Fatto 3"]\n\n'
    "TITOLO: {title}\n"
    "TESTO (massimo 800 caratteri): {snippet}"
)

# Rate limit between LLM calls (seconds)
RATE_LIMIT_SECONDS = 0.1


def _parse_bullets(response: str) -> Optional[List[str]]:
    """
    Parse bullet points from LLM response — handles multiple formats:
    - JSON array: ["b1", "b2", "b3"]
    - JSON object: {"bullets": ["b1", ...]} or {"bullet_points": [...]}
    - Markdown-wrapped JSON: ```json\n[...]\n```
    - Plain text: lines starting with -, *, or numbers
    """
    text = response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()

    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            items = [str(b).strip() for b in parsed if b]
            return items[:3] if items else None
        if isinstance(parsed, dict):
            for key in ("bullets", "bullet_points", "items", "punti", "risultati"):
                val = parsed.get(key)
                if isinstance(val, list):
                    items = [str(b).strip() for b in val if b]
                    return items[:3] if items else None
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: parse plain-text bullet lines
    lines = [l.strip() for l in response.splitlines()]
    bullets = []
    for line in lines:
        if not line:
            continue
        # Strip leading -, *, •, or "1." / "1)"
        clean = line.lstrip("-*•").strip()
        if len(clean) > 2 and len(clean) < 10 and clean[0].isdigit():
            clean = clean.lstrip("0123456789.)").strip()
        if len(clean) > 10:
            bullets.append(clean)
    return bullets[:3] if bullets else None


class BulletGenerator:
    """Generates 3 bullet point summaries for articles using T5 (Gemini 2.5 Flash-Lite)."""

    def __init__(self):
        from ..llm.llm_factory import LLMFactory
        self._llm = LLMFactory.get("t5")
        logger.info("BulletGenerator: T5 (Gemini 2.5 Flash-Lite) initialized")

    def generate_bullets(self, article: Dict) -> Optional[List[str]]:
        """
        Generate 3 bullet point summaries for a single article.

        Args:
            article: Article dict with 'title' and either 'full_content' or 'summary'

        Returns:
            List of 3 bullet point strings, or None if generation failed
        """
        title = article.get('title', 'Unknown')

        # Extract text (prefer clean_text if available from NLP processing)
        if 'nlp_data' in article and 'clean_text' in article['nlp_data']:
            text = article['nlp_data']['clean_text']
        else:
            # Fallback to full_content or summary
            full_content = article.get('full_content', {})
            if isinstance(full_content, dict):
                text = full_content.get('text', article.get('summary', ''))
            else:
                text = article.get('summary', '')

        if not text or len(text.strip()) < 50:
            logger.warning(f"Article '{title[:50]}...' has insufficient text for bullet generation")
            return None

        # Truncate to reasonable length for LLM
        snippet = text[:800]

        try:
            prompt = BULLET_GENERATION_PROMPT.format(title=title, snippet=snippet)

            response = self._llm.generate(
                prompt,
                max_tokens=300,
                temperature=0.3,
                json_mode=True,
            )

            if not response:
                logger.warning(f"Empty response for article '{title[:50]}...'")
                return None

            bullets = _parse_bullets(response)
            if bullets:
                logger.debug(f"Generated {len(bullets)} bullet(s) for '{title[:50]}...'")
                return bullets

            logger.warning(f"Invalid JSON format from model for '{title[:50]}...': {response[:100]}")
            return None
        except Exception as e:
            logger.warning(f"Error generating bullets for '{title[:50]}...': {e}")
            return None

    def generate_batch(self, articles: List[Dict], skip_errors: bool = True, show_progress: bool = True) -> List[Dict]:
        """
        Generate bullet points for a batch of articles.

        Args:
            articles: List of article dictionaries
            skip_errors: If True, silently skip articles that fail generation
            show_progress: Whether to log progress

        Returns:
            List of articles enriched with 'bullet_points' in nlp_data
        """
        logger.info(f"Generating bullet points for {len(articles)} articles...")

        processed = []
        success_count = 0

        for i, article in enumerate(articles):
            try:
                bullets = self.generate_bullets(article)

                if bullets:
                    if 'nlp_data' not in article:
                        article['nlp_data'] = {}
                    article['nlp_data']['bullet_points'] = bullets
                    success_count += 1
                elif not skip_errors:
                    logger.warning(f"Failed to generate bullets for article {i+1}/{len(articles)}")

                processed.append(article)

                # Rate limiting
                if i < len(articles) - 1:
                    time.sleep(RATE_LIMIT_SECONDS)

            except Exception as e:
                if skip_errors:
                    logger.warning(f"Skipping article {i+1} due to error: {e}")
                    processed.append(article)
                else:
                    raise

            if show_progress and (i + 1) % 5 == 0:
                logger.info(f"Progress: {i + 1}/{len(articles)} articles processed ({success_count} with bullets)")

        logger.info(f"Bullet generation complete: {success_count}/{len(articles)} articles with bullets")
        return processed
