Examples
========

recommend_usage.py
    Generate a protocol recommendation with no database, using the dict
    substrate (DictPatientState + DictSimilarity). Self-contained and
    CI-friendly. Run: python examples/recommend_usage.py

The production entry point is `from ai_cdss import RecommendationService`
(DB-backed: fetch cohort -> score -> recommend -> persist). See
scripts/compute_ppf.py and scripts/simulate_phrase.py for DB-backed jobs.
