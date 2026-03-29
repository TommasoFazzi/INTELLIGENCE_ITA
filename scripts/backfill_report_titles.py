"""
One-time script: generate LLM titles for existing reports that lack metadata.title.

Usage (on server):
    docker compose -p app exec backend python scripts/backfill_report_titles.py [--dry-run]

Requires: GEMINI_API_KEY env var (reads from .env if present).
"""
import argparse
import json
import logging
import os
import sys

import google.generativeai as genai

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_bluf(text: str) -> str:
    """
    Extract content for title generation: prefer H2/H3 headings (day-specific)
    over the boilerplate executive summary opening paragraph.
    """
    if not text:
        return ""
    # Pass 1: collect H2/H3 headings (most specific content)
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('## ') or stripped.startswith('### '):
            heading = stripped.lstrip('#').strip()
            lower = heading.lower()
            if any(skip in lower for skip in ('executive summary', 'strategic', 'investment', 'trade signal', 'macro', 'overview', 'key development', 'conclusion')):
                continue
            headings.append(heading)
            if len(headings) >= 5:
                break
    if headings:
        return ' | '.join(headings)[:400]
    # Pass 2: fallback to first real prose paragraph
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#') or stripped.startswith('---') or stripped.startswith('|') or stripped.startswith('*'):
            continue
        clean = stripped.replace('**', '').replace('*', '').strip()
        if len(clean) > 40:
            return clean[:300]
    return ""


def generate_title(model: genai.GenerativeModel, report_date: str, focus_areas: list, bluf: str) -> str:
    """Generate a headline title via Gemini 2.0 Flash."""
    themes = ", ".join(focus_areas[:3]) if focus_areas else "geopolitics"
    prompt = (
        "You are an intelligence editor. Generate a concise, descriptive headline "
        f"(maximum 80 characters) for a geopolitical intelligence briefing dated {report_date}.\n"
        f"Key themes: {themes}\n"
        f"Opening paragraph: {bluf[:250]}\n"
        "Rules: return ONLY the headline text. No quotes, no trailing punctuation, no prefixes."
    )
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=80),
            request_options={"timeout": 30},
        )
        raw = resp.text.strip().strip('"').strip("'")
        if raw.endswith('.'):
            raw = raw[:-1]
        return raw[:80]
    except Exception as e:
        logger.warning(f"  Title generation failed: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Backfill LLM titles for reports missing metadata.title")
    parser.add_argument("--dry-run", action="store_true", help="Preview titles without writing to DB")
    args = parser.parse_args()

    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.error("GEMINI_API_KEY not set")
        sys.exit(1)

    genai.configure(api_key=api_key, transport='rest')
    model = genai.GenerativeModel('gemini-2.0-flash')

    db = DatabaseManager()

    # Fetch reports missing a title
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, report_date, report_type,
                       metadata->>'title' as existing_title,
                       metadata->'focus_areas' as focus_areas,
                       LEFT(COALESCE(final_content, draft_content), 3000) as content_preview
                FROM reports
                WHERE metadata->>'title' IS NULL OR TRIM(metadata->>'title') = ''
                ORDER BY report_date DESC
            """)
            rows = cur.fetchall()

    logger.info(f"Found {len(rows)} reports without a title.")

    if not rows:
        logger.info("Nothing to do.")
        return

    for row in rows:
        report_id, report_date, report_type, existing_title, focus_areas_raw, content_preview = row

        # Parse focus_areas from JSONB (comes as list or None)
        if isinstance(focus_areas_raw, list):
            focus_areas = focus_areas_raw
        elif isinstance(focus_areas_raw, str):
            try:
                focus_areas = json.loads(focus_areas_raw)
            except Exception:
                focus_areas = []
        else:
            focus_areas = []

        bluf = extract_bluf(content_preview or "")
        date_str = str(report_date)

        logger.info(f"  [{report_id}] {date_str} ({report_type}) — generating title...")
        title = generate_title(model, date_str, focus_areas, bluf)

        if not title:
            logger.warning(f"  [{report_id}] Skipping — could not generate title")
            continue

        logger.info(f"  [{report_id}] → {title!r}")

        if not args.dry_run:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE reports
                        SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('title', %s::text),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (title, report_id),
                    )
                conn.commit()

    if args.dry_run:
        logger.info("Dry-run complete — no changes written.")
    else:
        logger.info(f"Done — {len(rows)} reports updated.")


if __name__ == "__main__":
    main()
