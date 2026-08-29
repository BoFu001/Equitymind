"""
research/2026-08-16_synonym_merge/candidate_synonyms.py

Manually identified candidate synonym pairs from a full read-through
of the 262-tag vocabulary (2026-08-16), beyond the ones already merged
(plurals in Research Log 04; ai/artificial intelligence and the
healthcare/health care gap found later the same day).

Each entry: (tag_to_be_merged_away, canonical_form)
Canonical form chosen per three rules established this session:
  1. singular over plural
  2. full form over abbreviation
  3. no-space compound over spaced form

These are candidates, not yet applied — this script only defines the
list and reports what merging them would look like, before any file
is actually modified.
"""

CANDIDATE_SYNONYMS = {
    "health care": "healthcare",
    "discount retailing": "discount retail",
    "property-casualty insurance": "property and casualty insurance",
    "data center services": "data center",
    "fintech": "financial technology",
}
