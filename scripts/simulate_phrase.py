"""
Temporal replay: re-score a cohort at successive weekly scoring dates.
======================================================================

The scoring pipeline windows each patient's sessions to
``[clinical_start, min(clinical_end, scoring_date)]``. So scoring the
SAME cohort at different ``scoring_date`` values replays how the scores
(and therefore the recommendations) evolve week over week — useful for
backtests and for eyeballing CDSS behavior over a trial.

Fetch the cohort once (one DB round-trip), then loop scoring dates; the
pipeline re-windows internally per call.

Run:  python scripts/simulate_phrase.py --study 1 --weeks 4
"""
import argparse
import logging

import pandas as pd

from ai_cdss.constants import CLINICAL_START, PATIENT_ID, SCORE
from ai_cdss.data import RGSCohortRepository
from ai_cdss.pipeline import DataPipeline
from ai_cdss.recommender import Recommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=int, nargs="+", default=[1])
    parser.add_argument("--weeks", type=int, default=4, help="Number of weekly snapshots.")
    parser.add_argument("--recommend", action="store_true",
                        help="Also run the engine at each snapshot, not just scoring.")
    args = parser.parse_args()

    repository = RGSCohortRepository()
    pipeline = DataPipeline()

    patient_ids = repository.fetch_and_validate_patients(study_ids=args.study)
    if not patient_ids:
        logging.warning("No patients for study %s.", args.study)
        return

    cohort = repository.find(patient_ids)            # one fetch
    if cohort.missing_ppf:
        raise RuntimeError(f"PPF missing for {cohort.missing_ppf}; compute PPF first.")

    # Weekly scoring dates anchored at the earliest clinical_start.
    start = pd.to_datetime(cohort.patient[CLINICAL_START]).min()
    scoring_dates = [start + pd.Timedelta(weeks=w) for w in range(1, args.weeks + 1)]

    summary = []
    for ts in scoring_dates:
        scores = pipeline.process(cohort, ts)        # re-windows internally
        summary.append({
            "scoring_date": ts.date().isoformat(),
            "rows": len(scores),
            "mean_score": round(float(scores[SCORE].mean()), 4) if len(scores) else None,
        })

        if args.recommend:
            engine = Recommender(scoring=scores)
            for pid in patient_ids:
                result = engine.recommend(pid, cohort.similarity)
                logging.info(
                    "ts=%s patient=%s branch=%s swaps=%d final=%s",
                    ts.date(), pid, result.branch, result.n_swaps, result.final_protocols,
                )

    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
