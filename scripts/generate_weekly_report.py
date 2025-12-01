#!/usr/bin/env python3
"""
Generate weekly meta-analysis intelligence report.

This script analyzes the evolution of trends across the week by reading
daily intelligence reports, rather than raw news articles.

Usage:
    python scripts/generate_weekly_report.py                # Generate for last 7 days
    python scripts/generate_weekly_report.py --days 14      # Generate for last 14 days
    python scripts/generate_weekly_report.py --no-save-db   # Don't save to database
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import DatabaseManager
from src.llm.report_generator import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_weekly_reports_from_db(db: DatabaseManager, days: int = 7) -> List[Dict]:
    """
    Retrieve daily reports from the last N days using priority logic.

    Priority: approved > reviewed > draft for each date.

    Args:
        db: Database manager instance
        days: Number of days to look back (default: 7)

    Returns:
        List of report dictionaries ordered by date
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Fetching daily reports from {start_date} to {end_date}...")

    # Use priority logic: None = approved > reviewed > draft
    reports = db.get_reports_by_date_range(
        start_date=start_date,
        end_date=end_date,
        status_filter=None,  # Priority logic
        report_type='daily'
    )

    if len(reports) < 3:
        logger.warning(
            f"Only {len(reports)} daily reports found. "
            f"Minimum 3 recommended for meaningful meta-analysis."
        )

    logger.info(f"✓ Retrieved {len(reports)} daily reports for meta-analysis")
    return reports


def aggregate_metadata(reports: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate metadata across weekly reports.

    Args:
        reports: List of daily report dictionaries

    Returns:
        Aggregated metadata dictionary
    """
    all_focus_areas = []
    total_articles = 0
    all_sources = []

    for report in reports:
        metadata = report.get('metadata', {})
        sources = report.get('sources', {})

        # Collect focus areas
        all_focus_areas.extend(metadata.get('focus_areas', []))

        # Count articles
        total_articles += metadata.get('recent_articles_count', 0)

        # Collect source articles
        all_sources.extend(sources.get('recent_articles', []))

    # Deduplicate sources by link
    unique_sources = {s.get('link'): s for s in all_sources if s.get('link')}.values()

    return {
        'focus_areas_frequency': all_focus_areas,
        'total_articles_analyzed': total_articles,
        'unique_sources_count': len(unique_sources),
        'unique_sources': list(unique_sources),
        'reports_count': len(reports),
        'date_range': {
            'start': reports[0]['report_date'].isoformat() if reports else None,
            'end': reports[-1]['report_date'].isoformat() if reports else None
        }
    }


def generate_weekly_prompt(reports: List[Dict], aggregated_meta: Dict) -> str:
    """
    Generate prompt for weekly meta-analysis.

    Args:
        reports: List of daily reports (ordered by date)
        aggregated_meta: Aggregated metadata

    Returns:
        Complete prompt string for LLM
    """

    system_prompt = """SEI UN SENIOR STRATEGIC ANALYST DI UN THINK TANK GLOBALE.

COMPITO: Scrivere il "WEEKLY INTELLIGENCE BRIEFING".
Non riassumere le notizie. Identifica l'EVOLUZIONE dei trend osservando i report giornalieri.

STRUTTURA REPORT:
1. 🚨 THE BIG PICTURE
   - Qual è stata la narrazione dominante della settimana?
   - Come si sono spostati gli equilibri globali?
   - Quali sono i cambiamenti di momentum (chi sta accelerando, chi sta rallentando)?

2. 📈 TREND WATCH
   - ESCALATION: Cosa è peggiorato rispetto a inizio settimana? Quali crisi si sono intensificate?
   - DE-ESCALATION: Cosa è migliorato? Quali tensioni si sono allentate?
   - SEGNALI DEBOLI → FORTI: Quali pattern emergenti stanno consolidando?
   - DISCONTINUITÀ: Eventi imprevisti che hanno cambiato il quadro?

3. 🏆 WINNERS & LOSERS
   - Chi (Nazioni, Aziende, Leader, Alleanze) esce rafforzato da questa settimana?
   - Chi è indebolito? Chi ha perso credibilità o influenza?
   - Quali strategie si sono rivelate efficaci? Quali hanno fallito?

4. 🔮 NEXT WEEK FORECAST
   - Basandoti sui trend identificati, cosa dobbiamo aspettarci?
   - Quali sono le decisioni chiave attese?
   - Dove concentrare l'attenzione?

STILE:
- Sintetico, Diretto, Analitico
- Usa bullet points numerati
- Evidenzia CAMBIAMENTI non fatti statici
- Focus su EVOLUZIONE TEMPORALE (come le storie si sono sviluppate)
- Cita specifici giorni quando rilevante (es. "Lunedì... poi Mercoledì...")
"""

    # Build context from daily reports
    context_parts = []
    for report in reports:
        # Format date in Italian
        date_obj = report['report_date']
        # Get Italian weekday name
        weekday_names = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
        weekday = weekday_names[date_obj.weekday()]
        date_str = f"{weekday} {date_obj.strftime('%d %B %Y')}"

        # Prefer final_content (human-reviewed) over draft_content
        content = report.get('final_content') or report.get('draft_content', '')
        status = report.get('status', 'draft')

        # Add status indicator
        status_indicator = "✓ Approved" if status == 'approved' else f"({status})"

        context_parts.append(
            f"=== REPORT {date_str.upper()} {status_indicator} ===\n{content}\n"
        )

    context = "\n".join(context_parts)

    # Metadata summary
    meta_summary = f"""
METADATA AGGREGATO SETTIMANA:
- Periodo: {aggregated_meta['date_range']['start']} → {aggregated_meta['date_range']['end']}
- Report analizzati: {aggregated_meta['reports_count']}
- Articoli totali considerati: {aggregated_meta['total_articles_analyzed']}
- Fonti uniche: {aggregated_meta['unique_sources_count']}
"""

    return system_prompt + "\n" + meta_summary + "\n\nREPORT GIORNALIERI:\n" + context


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate weekly meta-analysis intelligence report"
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to analyze (default: 7)'
    )
    parser.add_argument(
        '--no-save-db',
        action='store_true',
        help='Do not save weekly report to database'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for report file (default: reports)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash',
        help='Gemini model to use (default: gemini-2.5-flash)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("WEEKLY META-ANALYSIS REPORT GENERATION")
    logger.info("=" * 80)
    logger.info(f"Analyzing last {args.days} days")

    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        logger.error("GEMINI_API_KEY not found in environment")
        logger.error("Please add it to your .env file")
        return 1

    # Step 1: Connect to database
    logger.info("\n[STEP 1] Connecting to database...")
    try:
        db = DatabaseManager()
        logger.info("✓ Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 1

    # Step 2: Retrieve daily reports
    logger.info(f"\n[STEP 2] Retrieving daily reports (last {args.days} days)...")
    try:
        reports = get_weekly_reports_from_db(db, days=args.days)

        if not reports:
            logger.error("No daily reports found in database")
            logger.error("Generate some daily reports first with: python scripts/generate_report.py")
            return 1

        logger.info(f"✓ Found {len(reports)} daily reports")

        # Log which reports were found
        for report in reports:
            status_emoji = "✓" if report['status'] == 'approved' else "○"
            logger.info(f"  {status_emoji} {report['report_date']} ({report['status']})")

    except Exception as e:
        logger.error(f"Failed to retrieve reports: {e}")
        return 1

    # Step 3: Aggregate metadata
    logger.info("\n[STEP 3] Aggregating metadata...")
    try:
        aggregated_meta = aggregate_metadata(reports)
        logger.info(f"✓ Aggregated metadata from {aggregated_meta['reports_count']} reports")
        logger.info(f"  Total articles analyzed: {aggregated_meta['total_articles_analyzed']}")
        logger.info(f"  Unique sources: {aggregated_meta['unique_sources_count']}")
    except Exception as e:
        logger.error(f"Failed to aggregate metadata: {e}")
        return 1

    # Step 4: Generate weekly prompt
    logger.info("\n[STEP 4] Generating meta-analysis prompt...")
    try:
        prompt = generate_weekly_prompt(reports, aggregated_meta)
        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4  # Rough estimate
        logger.info(f"✓ Prompt generated ({prompt_length} chars, ~{estimated_tokens} tokens)")
    except Exception as e:
        logger.error(f"Failed to generate prompt: {e}")
        return 1

    # Step 5: Call Gemini for meta-analysis
    logger.info(f"\n[STEP 5] Generating weekly meta-analysis with {args.model}...")
    try:
        generator = ReportGenerator(model_name=args.model)
        
        response = generator.model.generate_content(
            prompt,
            generation_config={'temperature': 0.3}  # più deterministico per analisi
        )
        
        weekly_report_text = response.text
        logger.info(f"✓ Meta-analysis generated ({len(weekly_report_text)} characters)")
    except Exception as e:
        logger.error(f"Failed to generate meta-analysis: {e}")
        return 1

    # Step 6: Save to file
    logger.info(f"\n[STEP 6] Saving report to file...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"WEEKLY_REPORT_{timestamp}.md"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Weekly Intelligence Meta-Analysis\n")
            f.write(f"**Week of**: {aggregated_meta['date_range']['start']} to {aggregated_meta['date_range']['end']}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Reports analyzed**: {aggregated_meta['reports_count']}\n")
            f.write(f"**Model**: {args.model}\n\n")
            f.write("---\n\n")
            f.write(weekly_report_text)

        logger.info(f"✓ Report saved to: {output_file}")

    except Exception as e:
        logger.error(f"Failed to save report to file: {e}")
        return 1

    # Step 7: Save to database (optional)
    if not args.no_save_db:
        logger.info("\n[STEP 7] Saving weekly report to database...")
        try:
            # Prepare report dictionary for database
            weekly_report_dict = {
                'report_text': weekly_report_text,
                'report_type': 'weekly',  # Mark as weekly report
                'metadata': {
                    'model_used': args.model,
                    'days_analyzed': args.days,
                    'reports_count': aggregated_meta['reports_count'],
                    'total_articles_analyzed': aggregated_meta['total_articles_analyzed'],
                    'unique_sources_count': aggregated_meta['unique_sources_count'],
                    'date_range': aggregated_meta['date_range']
                },
                'sources': {
                    'daily_reports': [
                        {
                            'report_id': r['id'],
                            'report_date': r['report_date'].isoformat(),
                            'status': r['status']
                        } for r in reports
                    ],
                    'aggregated_sources': list(aggregated_meta['unique_sources'])[:100]  # Limit to 100
                }
            }

            report_id = db.save_report(weekly_report_dict)

            if report_id:
                logger.info(f"✓ Weekly report saved to database (ID: {report_id})")
                logger.info(f"✓ Review at: http://localhost:8501")
            else:
                logger.warning("Failed to save weekly report to database")

        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            # Don't fail entire script if DB save fails
    else:
        logger.info("\n[STEP 7] Skipping database save (--no-save-db flag)")

    # Step 8: Print report
    logger.info("\n" + "=" * 80)
    logger.info("WEEKLY META-ANALYSIS REPORT")
    logger.info("=" * 80)
    print(f"\n{weekly_report_text}\n")
    logger.info("=" * 80)

    # Final summary
    logger.info(f"\n✓ Weekly meta-analysis complete!")
    logger.info(f"  Period: {aggregated_meta['date_range']['start']} → {aggregated_meta['date_range']['end']}")
    logger.info(f"  Reports analyzed: {aggregated_meta['reports_count']}")
    logger.info(f"  Output: {output_file}")

    db.close()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
