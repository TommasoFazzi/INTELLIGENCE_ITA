#!/usr/bin/env python3
"""
Generate daily intelligence report using LLM with RAG.

Usage:
    python scripts/generate_report.py                    # Generate with default settings
    python scripts/generate_report.py --days 3           # Include last 3 days
    python scripts/generate_report.py --no-save          # Don't save to file
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.report_generator import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate intelligence report with RAG")
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days to look back for articles (default: 1)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save report to file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for reports (default: reports)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash',
        help='Gemini model to use (default: gemini-2.5-flash)'
    )
    parser.add_argument(
        '--top-articles',
        type=int,
        default=60,
        help='Maximum number of top relevant articles to include (default: 60)'
    )
    parser.add_argument(
        '--min-similarity',
        type=float,
        default=0.30,
        help='Minimum similarity threshold for article relevance (default: 0.30)'
    )
    parser.add_argument(
        '--min-articles',
        type=int,
        default=10,
        help='Minimum number of articles to use even if below threshold (default: 10)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("INTELLIGENCE REPORT GENERATION")
    logger.info("=" * 80)

    # Check for API key
    import os
    if not os.getenv('GEMINI_API_KEY'):
        logger.error("GEMINI_API_KEY not found in environment")
        logger.error("Please add it to your .env file or export it:")
        logger.error("  export GEMINI_API_KEY='your-api-key-here'")
        return 1

    # Initialize report generator
    try:
        logger.info(f"\n[STEP 1] Initializing report generator with {args.model}...")
        generator = ReportGenerator(model_name=args.model)
        logger.info("✓ Report generator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize report generator: {e}")
        return 1

    # Define focus areas
    focus_areas = [
    # Cybersecurity: Aggiungiamo l'intento malevolo e l'infrastruttura
    "cybersecurity threats, state-sponsored cyber attacks, ransomware campaigns, and critical infrastructure vulnerabilities",
    
    # Tech: Aggiungiamo la dimensione strategica (chip/supply chain)
    "breakthroughs in artificial intelligence, semiconductor supply chain shifts, and dual-use technology regulations",
    
    # Geopolitica (Generale): Rendiamola più attiva
    "escalation of military conflicts, diplomatic ruptures, and changing alliances in NATO, Russia, China, and Middle East",
    
    # NUOVO: Geografia dei Conflitti (Specifico per la tua richiesta)
    "territorial control changes, strategic military movements, maritime security in choke points, and border disputes",
    
    # Economia: Colleghiamola alla geopolitica
    "global economic impact of sanctions, energy market volatility, and trade protectionism policies"
]

    logger.info(f"\n[STEP 2] Focus areas:")
    for area in focus_areas:
        logger.info(f"  - {area}")

    # Generate report
    try:
        logger.info(f"\n[STEP 3] Generating report (analyzing last {args.days} day(s))...")
        logger.info(f"Filtering parameters: top_articles={args.top_articles}, "
                   f"min_similarity={args.min_similarity}, min_fallback={args.min_articles}")
        report = generator.generate_report(
            focus_areas=focus_areas,
            days=args.days,
            rag_top_k=5,
            top_articles=args.top_articles,
            min_similarity=args.min_similarity,
            min_fallback=args.min_articles
        )

        if not report['success']:
            logger.error(f"Report generation failed: {report.get('error')}")
            return 1

        logger.info("✓ Report generated successfully")

    except Exception as e:
        logger.error(f"Error during report generation: {e}", exc_info=True)
        return 1

    # Save report to files
    if not args.no_save:
        try:
            logger.info(f"\n[STEP 4] Saving report to {args.output_dir}/...")
            report_file = generator.save_report(report, output_dir=args.output_dir)
            logger.info(f"✓ Report saved to files successfully")
        except Exception as e:
            logger.error(f"Error saving report to files: {e}")
            return 1

    # Save report to database (for HITL dashboard)
    try:
        logger.info(f"\n[STEP 5] Saving report to database for HITL review...")
        report_id = generator.db.save_report(report)
        if report_id:
            logger.info(f"✓ Report saved to database with ID: {report_id}")
            logger.info(f"✓ You can now review it at: http://localhost:8501")
        else:
            logger.warning("Failed to save report to database")
    except Exception as e:
        logger.error(f"Error saving report to database: {e}")
        # Don't fail the entire script if DB save fails

    # Print report summary
    logger.info("\n" + "=" * 80)
    logger.info("REPORT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nGenerated: {report['timestamp']}")
    logger.info(f"Model: {report['metadata']['model_used']}")
    logger.info(f"Recent articles analyzed: {report['metadata']['recent_articles_count']}")
    logger.info(f"Historical context chunks: {report['metadata']['historical_chunks_count']}")
    logger.info(f"Report length: {len(report['report_text'])} characters")

    # Print report text
    print("\n" + "=" * 80)
    print("INTELLIGENCE REPORT")
    print("=" * 80)
    print(report['report_text'])
    print("\n" + "=" * 80)

    # Print sources
    print("\nSOURCES:")
    print(f"\nRecent Articles ({len(report['sources']['recent_articles'])}):")
    for i, article in enumerate(report['sources']['recent_articles'][:10], 1):
        print(f"  [{i}] {article['title']}")
        print(f"      {article['source']} - {article['published_date']}")
        print(f"      {article['link']}")

    if len(report['sources']['recent_articles']) > 10:
        print(f"  ... and {len(report['sources']['recent_articles']) - 10} more")

    print(f"\nHistorical Context ({len(report['sources']['historical_context'])}):")
    for i, ctx in enumerate(report['sources']['historical_context'][:5], 1):
        print(f"  [{i}] {ctx['title']} (similarity: {ctx['similarity']:.3f})")
        print(f"      {ctx['link']}")

    if len(report['sources']['historical_context']) > 5:
        print(f"  ... and {len(report['sources']['historical_context']) - 5} more")

    logger.info("\n✓ Report generation complete!")
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
