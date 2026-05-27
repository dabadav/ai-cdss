"""
Generate a protocol recommendation without a database.
======================================================

The production entry point is `RecommendationService` (reads a cohort
from the DB, scores it, recommends, persists). For a self-contained,
CI-friendly demo we drive the engine directly with the **dict
substrate** — `DictPatientState` + `DictSimilarity` — which replaces the
old `DataLoaderMock`. No DB, no DataFrame boilerplate.

    patient state (scores + prior week)  ─┐
                                          ├─►  Recommender.recommend  ─►  RecommendationResult
    protocol similarity                  ─┘

Run:  python examples/recommend_usage.py
"""
from ai_cdss.engine import DictPatientState, DictSimilarity, ProtocolRow
from ai_cdss.recommender import Recommender

# --- One patient's per-protocol state -------------------------------
# `days` non-empty == prescribed last week. Here 200/201/202/203 were
# prescribed; 204-209 are the unprescribed alternative pool. Scores
# vary so the MVT criterion swaps the low-scoring prescribed protocols
# for higher-scoring alternatives.
PATIENT_ID = 1

rows = [
    # prescribed last week (have DAYS) — mixed scores. usage_week > 0
    # means the week was actually played (else the engine treats it as a
    # skipped week and just repeats the prior schedule).
    ProtocolRow(PATIENT_ID, 200, score=0.30, days=[0, 1], usage=4, usage_week=2),
    ProtocolRow(PATIENT_ID, 201, score=0.40, days=[2, 3], usage=4, usage_week=2),
    ProtocolRow(PATIENT_ID, 202, score=0.95, days=[4],    usage=4, usage_week=2),
    ProtocolRow(PATIENT_ID, 203, score=0.90, days=[0],    usage=4, usage_week=2),
    # unprescribed alternative pool — never used, higher scores
    ProtocolRow(PATIENT_ID, 204, score=0.80, usage=0),
    ProtocolRow(PATIENT_ID, 205, score=0.75, usage=0),
    ProtocolRow(PATIENT_ID, 206, score=0.70, usage=0),
    ProtocolRow(PATIENT_ID, 207, score=0.65, usage=0),
    ProtocolRow(PATIENT_ID, 208, score=0.60, usage=0),
    ProtocolRow(PATIENT_ID, 209, score=0.55, usage=0),
]
state = DictPatientState.from_rows(PATIENT_ID, rows)

# --- Pairwise protocol similarity (asymmetric dict) -----------------
# Each removed protocol needs candidates to substitute toward.
similarity = DictSimilarity({
    (200, 204): 0.91, (200, 205): 0.40,
    (201, 205): 0.88, (201, 206): 0.42,
    (202, 207): 0.50, (203, 208): 0.50,
})

# --- Recommend ------------------------------------------------------
engine = Recommender(scoring=state, n=6, days=5, protocols_per_day=2)
result = engine.recommend(patient_id=PATIENT_ID, protocol_similarity=similarity)

# `result` is a RecommendationResult — every intermediate artifact is a
# public attribute (sklearn-style introspection).
print(f"branch:      {result.branch}")
print(f"mvt_mean:    {result.mvt_mean:.3f}  (cohort-wide threshold)")
print(f"swaps:       {result.n_swaps}")
for swap in result.swap_decisions:
    print(f"  removed {swap.removed_id} -> added {swap.protocol_id} "
          f"(tier={swap.tier}, sim={swap.similarity})")
print(f"top-up adds: {result.n_topup}")
print(f"final set:   {result.final_protocols}")
print("\nschedule (one row per protocol-day):")
print(result.recommendations.explode("DAYS").to_string(index=False))
