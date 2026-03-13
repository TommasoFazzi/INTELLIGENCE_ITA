"""
AI-powered bullet point generator for articles.

Uses Gemini 2.0 Flash to generate 3 concise bullet points summarizing
the key facts of an article, similar to Bloomberg Terminal's AI summaries.
"""

import os
import time
from typing import List, Dict, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Timeout for Gemini calls (seconds)
BULLET_GENERATION_TIMEOUT = 10

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


class BulletGenerator:
    """Generates 3 bullet point summaries for articles using Gemini Flash."""

    def __init__(self, gemini_api_key: str = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai is required. pip install google-generativeai")

        api_key = (gemini_api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')).strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set")

        genai.configure(api_key=api_key, transport='rest')
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("BulletGenerator: Gemini Flash initialized")

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

            response = self.model.generate_content(
                prompt,
                request_options={"timeout": BULLET_GENERATION_TIMEOUT}
            )

            if not response or not response.text:
                logger.warning(f"Empty response for article '{title[:50]}...'")
                return None

            # Parse JSON response
            import json
            response_text = response.text.strip()

            # Try to extract JSON from response (in case model wraps it in markdown or extra text)
            if '[' in response_text and ']' in response_text:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                json_str = response_text[start_idx:end_idx]
            else:
                json_str = response_text

            bullets = json.loads(json_str)

            # Validate format: should be list of 3 strings
            if isinstance(bullets, list) and len(bullets) >= 1:
                # Truncate to max 3 bullets
                bullets = bullets[:3]
                # Ensure all are strings
                bullets = [str(b).strip() for b in bullets if b]
                if bullets:
                    logger.debug(f"Generated {len(bullets)} bullet(s) for '{title[:50]}...'")
                    return bullets if len(bullets) >= 1 else None

            logger.warning(f"Invalid JSON format from model for '{title[:50]}...'")
            return None

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response for '{title[:50]}...': {e}")
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
                    # Add bullets to article's nlp_data
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
