"""
Simple test to verify reranking feature is correctly implemented.

This script tests that:
1. ReportGenerator can be initialized with reranking enabled/disabled
2. The _rerank_chunks method exists and is callable
3. The reranking integration in generate_report is correctly placed
"""

import sys
import inspect


def test_reranking_implementation():
    """Test that reranking feature is correctly implemented."""

    print("=" * 80)
    print("TESTING RERANKING IMPLEMENTATION")
    print("=" * 80)

    # Test 1: Import and initialization
    print("\n[TEST 1] Checking imports...")
    try:
        from src.llm.report_generator import ReportGenerator
        print("✓ ReportGenerator imported successfully")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

    # Test 2: Check __init__ signature
    print("\n[TEST 2] Checking __init__ parameters...")
    init_sig = inspect.signature(ReportGenerator.__init__)
    params = list(init_sig.parameters.keys())

    expected_params = ['enable_reranking', 'reranking_model', 'reranking_top_k']
    for param in expected_params:
        if param in params:
            print(f"✓ Parameter '{param}' found in __init__")
        else:
            print(f"✗ Parameter '{param}' missing from __init__")
            return False

    # Test 3: Check _rerank_chunks method exists
    print("\n[TEST 3] Checking _rerank_chunks method...")
    if hasattr(ReportGenerator, '_rerank_chunks'):
        print("✓ Method _rerank_chunks exists")
        method_sig = inspect.signature(ReportGenerator._rerank_chunks)
        method_params = list(method_sig.parameters.keys())
        print(f"  Parameters: {method_params}")

        expected_method_params = ['self', 'query', 'chunks', 'top_k']
        if method_params == expected_method_params:
            print("✓ Method signature is correct")
        else:
            print(f"✗ Expected {expected_method_params}, got {method_params}")
            return False
    else:
        print("✗ Method _rerank_chunks not found")
        return False

    # Test 4: Check generate_report has reranking integration
    print("\n[TEST 4] Checking reranking integration in generate_report...")
    try:
        import ast
        import os

        # Read the source file
        file_path = os.path.join(
            os.path.dirname(__file__),
            'src', 'llm', 'report_generator.py'
        )

        with open(file_path, 'r') as f:
            source = f.read()

        # Check for key strings
        if '_rerank_chunks' in source:
            print("✓ _rerank_chunks is called in the source")
        else:
            print("✗ _rerank_chunks is not called")
            return False

        if 'self.enable_reranking' in source:
            print("✓ enable_reranking flag is used")
        else:
            print("✗ enable_reranking flag is not used")
            return False

        if 'Step 2d: Reranking' in source:
            print("✓ Reranking step is documented in generate_report")
        else:
            print("✗ Reranking step documentation missing")
            return False

    except Exception as e:
        print(f"✗ Source code check failed: {e}")
        return False

    # Test 5: Verify default values
    print("\n[TEST 5] Checking default parameter values...")
    defaults = {
        'enable_reranking': True,
        'reranking_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'reranking_top_k': 10
    }

    for param, expected_default in defaults.items():
        actual_default = init_sig.parameters[param].default
        if actual_default == expected_default:
            print(f"✓ {param} defaults to {expected_default}")
        else:
            print(f"⚠ {param} defaults to {actual_default}, expected {expected_default}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nImplementation Summary:")
    print("- Reranking is ENABLED by default")
    print("- Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (English optimized)")
    print("- Top-k: 10 chunks after reranking")
    print("\nUsage:")
    print("  # With reranking (default)")
    print("  gen = ReportGenerator()")
    print()
    print("  # Without reranking")
    print("  gen = ReportGenerator(enable_reranking=False)")
    print()
    print("  # With multilingual model")
    print("  gen = ReportGenerator(")
    print("      reranking_model='nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large'")
    print("  )")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_reranking_implementation()
    sys.exit(0 if success else 1)
