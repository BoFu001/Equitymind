"""
research/2026-08-16_tag_normalization/test_phone_query.py

One-off stress test: "which phone-selling companies are good?" — a
deliberately colloquial query with no direct match in the 263-tag
vocabulary (the closest candidate, "consumer electronics", has only
one company: AAPL). Tests how each method degrades when the query
doesn't cleanly map to an existing tag.

Reuses compare_methods.py's functions without modifying that file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compare_methods import compare

if __name__ == "__main__":
    compare("which phone-selling companies are good")
