#!/usr/bin/env python3
"""
Sprint 2.2 Full Schema Validation Test

Tests the new generate_full_analysis() method with:
- IntelligenceReport full schema validation
- Trade Signal extraction
- Impact Score assessment
- Markdown formatting detection
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from src.storage.database import DatabaseManager
from src.llm.report_generator import ReportGenerator


def test_full_schema():
    """Test full schema on 5 recent articles"""

    print("=" * 80)
    print("SPRINT 2.2 TEST - Full Schema & Trade Signals")
    print("=" * 80)

    # Initialize
    db = DatabaseManager()
    gen = ReportGenerator()

    # Get 5 recent articles (last 3 days to ensure enough content)
    articles = db.get_recent_articles(days=3)[:5]

    if len(articles) < 5:
        print(f"⚠️  WARNING: Only {len(articles)} articles found (need 5 for full test)")
        print("  Run ingestion pipeline to fetch more articles first")
        return 1

    print(f"\nTesting on {len(articles)} recent articles...")
    print("-" * 80)

    # Test statistics
    stats = {
        'success': 0,
        'failed': 0,
        'signals_found': 0,
        'impact_scores': [],
        'has_markdown': 0
    }

    # Test each article
    for i, art in enumerate(articles, 1):
        title = art['title'][:60] + "..." if len(art['title']) > 60 else art['title']
        print(f"\n[{i}/5] Analyzing: {title}")

        # Generate full analysis
        result = gen.generate_full_analysis(
            article_text=art['full_text'],
            article_metadata={
                'title': art['title'],
                'source': art['source'],
                'published_date': art.get('published_date')
            }
        )

        # Check validation success
        if result['success']:
            data = result['structured']

            # Extract metrics
            impact_score = data['impact']['score']
            sentiment = data['sentiment']['label']
            confidence = data['confidence_score']

            print(f"   ✅ SUCCESS")
            print(f"      Category: {data['category']}")
            print(f"      Impact: {impact_score}/10 - {data['impact']['reasoning'][:60]}...")
            print(f"      Sentiment: {sentiment} ({data['sentiment']['score']:.2f})")
            print(f"      Confidence: {confidence:.2f}")

            # Check Trade Signals
            signals = data['related_tickers']
            if signals:
                print(f"      💰 TRADE SIGNALS: {len(signals)}")
                for sig in signals:
                    print(f"         {sig['ticker']}: {sig['signal']} ({sig['timeframe']})")
                    rationale_preview = sig['rationale'][:70] + "..." if len(sig['rationale']) > 70 else sig['rationale']
                    print(f"         └─ {rationale_preview}")
                stats['signals_found'] += len(signals)
            else:
                print(f"      ℹ️  No trade signals (article not ticker-relevant)")

            # Check Markdown formatting
            if "**" in data['executive_summary'] or "**" in data['analysis_content']:
                print(f"      ✨ Markdown formatting detected")
                stats['has_markdown'] += 1

            # Check Key Entities
            entities = data['key_entities']
            if entities:
                entities_preview = ', '.join(entities[:5])
                print(f"      🏢 Entities: {entities_preview}")

            stats['success'] += 1
            stats['impact_scores'].append(impact_score)

        else:
            print(f"   ❌ FAILED")
            if 'error' in result:
                print(f"      Error: {result['error'][:100]}")
            if result.get('validation_errors'):
                print(f"      Validation errors:")
                for err in result['validation_errors'][:3]:  # Show first 3 errors
                    print(f"         - {err}")
            stats['failed'] += 1

    # Final Report
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    success_rate = (stats['success'] / 5) * 100
    print(f"Success Rate: {success_rate:.1f}% ({stats['success']}/5)")
    print(f"Trade Signals Extracted: {stats['signals_found']}")
    print(f"Articles with Markdown: {stats['has_markdown']}/5")

    if stats['impact_scores']:
        avg_impact = sum(stats['impact_scores']) / len(stats['impact_scores'])
        print(f"Average Impact Score: {avg_impact:.1f}/10")
        print(f"Impact Range: {min(stats['impact_scores'])}-{max(stats['impact_scores'])}")

    # Success criteria
    print("\n" + "-" * 80)
    print("SUCCESS CRITERIA:")

    criteria_met = []

    # Criterion 1: ≥80% success rate
    if success_rate >= 80:
        print("✅ Success Rate ≥80%")
        criteria_met.append(True)
    else:
        print(f"❌ Success Rate <80% (got {success_rate:.1f}%)")
        criteria_met.append(False)

    # Criterion 2: At least 1 trade signal extracted
    if stats['signals_found'] >= 1:
        print(f"✅ Trade Signals Extracted: {stats['signals_found']}")
        criteria_met.append(True)
    else:
        print(f"⚠️  No trade signals extracted (may be normal if no ticker-relevant news)")
        criteria_met.append(True)  # Not a hard failure

    # Criterion 3: Markdown formatting present
    if stats['has_markdown'] >= 3:
        print(f"✅ Markdown Formatting: {stats['has_markdown']}/5 articles")
        criteria_met.append(True)
    else:
        print(f"⚠️  Limited Markdown: {stats['has_markdown']}/5 articles")
        criteria_met.append(False)

    print("=" * 80)

    # Final verdict
    if all(criteria_met):
        print("\n🎉 TEST PASSED - Ready for production")
        print("\nNext steps:")
        print("1. Tag release: git tag v2.2.0-trade-signals")
        print("2. Update README with full schema usage")
        print("3. Monitor production for 48h")
        return 0
    else:
        print("\n❌ TEST FAILED - Needs debugging")
        print("\nDebug checklist:")
        print("1. Check Gemini API quota/rate limits")
        print("2. Review validation_errors in failed articles")
        print("3. Test with longer timeout if 504 errors")
        print("4. Verify ticker whitelist loaded correctly")
        return 1


if __name__ == "__main__":
    exit_code = test_full_schema()
    sys.exit(exit_code)
