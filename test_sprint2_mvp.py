#!/usr/bin/env python3
"""
Sprint 2.1 MVP Test - Structured JSON Output Validation

Tests generate_structured_analysis() on real articles from database.
Success criteria: 95%+ validation success rate on 20 articles.

Usage:
    python3 test_sprint2_mvp.py --count 20
    python3 test_sprint2_mvp.py --count 50 --verbose
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.database import DatabaseManager
from src.llm.report_generator import ReportGenerator
from dotenv import load_dotenv

load_dotenv()


def test_structured_analysis(article_count: int = 20, verbose: bool = False):
    """
    Test structured analysis on real articles from database

    Args:
        article_count: Number of articles to test (default: 20)
        verbose: Print detailed output for each article
    """
    print("=" * 80)
    print("SPRINT 2.1 MVP TEST - Structured JSON Output Validation")
    print("=" * 80)
    print(f"Testing with {article_count} recent articles from database\n")

    # Initialize database and report generator
    db = DatabaseManager()
    generator = ReportGenerator(
        db_manager=db,
        model_name="gemini-2.5-flash"
    )

    # Fetch recent articles
    print(f"[1/3] Fetching {article_count} recent articles...")
    articles = db.get_recent_articles(days=7)[:article_count]

    if not articles:
        print("❌ ERROR: No articles found in database")
        return

    if len(articles) < article_count:
        print(f"⚠️  WARNING: Only found {len(articles)} articles (requested {article_count})")

    print(f"✅ Retrieved {len(articles)} articles\n")

    # Test each article
    print(f"[2/3] Running structured analysis on each article...")
    print("-" * 80)

    results = {
        'success': [],
        'validation_failed': [],
        'llm_error': []
    }

    for i, article in enumerate(articles, 1):
        article_title = article['title'][:60] + "..." if len(article['title']) > 60 else article['title']

        print(f"\n[{i}/{len(articles)}] {article_title}")
        print(f"Source: {article['source']} | Category: {article.get('category', 'N/A')}")

        # Prepare metadata
        metadata = {
            'title': article['title'],
            'source': article['source'],
            'published_date': article.get('published_date'),
            'entities': article.get('entities')
        }

        # Generate structured analysis
        try:
            result = generator.generate_structured_analysis(
                article_text=article['full_text'],
                article_metadata=metadata
            )

            if result['success']:
                structured = result['structured']
                results['success'].append({
                    'article_id': article['id'],
                    'title': article['title'],
                    'structured': structured
                })

                print(f"✅ SUCCESS - Validated")
                if verbose:
                    print(f"   Title: {structured['title']}")
                    print(f"   Category: {structured['category']}")
                    print(f"   Sentiment: {structured['sentiment_label']}")
                    print(f"   Confidence: {structured['confidence_score']:.2f}")
                    print(f"   Summary: {structured['executive_summary'][:100]}...")

            else:
                results['validation_failed'].append({
                    'article_id': article['id'],
                    'title': article['title'],
                    'errors': result.get('validation_errors', []),
                    'raw': result.get('raw_llm_output', '')[:200]
                })

                print(f"⚠️  VALIDATION FAILED")
                if verbose:
                    print(f"   Errors: {result['validation_errors']}")
                    print(f"   Raw output preview: {result.get('raw_llm_output', '')[:150]}...")

        except Exception as e:
            results['llm_error'].append({
                'article_id': article['id'],
                'title': article['title'],
                'error': str(e)
            })
            print(f"❌ LLM ERROR: {e}")

    # Calculate statistics
    print("\n" + "=" * 80)
    print("[3/3] TEST RESULTS SUMMARY")
    print("=" * 80)

    total = len(articles)
    success_count = len(results['success'])
    validation_fail_count = len(results['validation_failed'])
    llm_error_count = len(results['llm_error'])

    success_rate = (success_count / total) * 100 if total > 0 else 0

    print(f"\nTotal Articles Tested: {total}")
    print(f"✅ Success: {success_count} ({success_rate:.1f}%)")
    print(f"⚠️  Validation Failed: {validation_fail_count} ({validation_fail_count/total*100:.1f}%)")
    print(f"❌ LLM Errors: {llm_error_count} ({llm_error_count/total*100:.1f}%)")

    # Category distribution (from successful validations)
    if results['success']:
        categories = {}
        sentiments = {}
        confidence_scores = []

        for item in results['success']:
            cat = item['structured']['category']
            sent = item['structured']['sentiment_label']
            conf = item['structured']['confidence_score']

            categories[cat] = categories.get(cat, 0) + 1
            sentiments[sent] = sentiments.get(sent, 0) + 1
            confidence_scores.append(conf)

        print(f"\n📊 Category Distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat}: {count} ({count/success_count*100:.1f}%)")

        print(f"\n😊 Sentiment Distribution:")
        for sent, count in sorted(sentiments.items(), key=lambda x: x[1], reverse=True):
            print(f"   {sent}: {count} ({count/success_count*100:.1f}%)")

        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"\n🎯 Average Confidence Score: {avg_confidence:.3f}")
        print(f"   Range: {min(confidence_scores):.2f} - {max(confidence_scores):.2f}")

    # Show validation errors (if any)
    if results['validation_failed'] and verbose:
        print(f"\n⚠️  VALIDATION ERRORS DETAILS:")
        for i, fail in enumerate(results['validation_failed'][:5], 1):
            print(f"\n{i}. {fail['title'][:60]}...")
            print(f"   Errors: {fail['errors']}")

    # Final verdict
    print("\n" + "=" * 80)
    if success_rate >= 95:
        print("🎉 TEST PASSED - Success rate >= 95%")
        print("✅ Ready to proceed to Sprint 2.2 (full schema)")
    elif success_rate >= 80:
        print("⚠️  TEST MARGINAL - Success rate 80-95%")
        print("💡 Recommend: Review validation errors and refine prompt")
    else:
        print("❌ TEST FAILED - Success rate < 80%")
        print("💡 Recommend: Debug JSON mode issues before proceeding")

    print("=" * 80)

    # Cleanup
    db.close()

    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test Sprint 2.1 MVP structured analysis on real articles'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=20,
        help='Number of articles to test (default: 20)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output for each article'
    )

    args = parser.parse_args()

    success_rate = test_structured_analysis(
        article_count=args.count,
        verbose=args.verbose
    )

    # Exit code: 0 if success >= 95%, 1 otherwise
    sys.exit(0 if success_rate >= 95 else 1)
