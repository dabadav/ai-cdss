"""
Compute + persist PPF and protocol similarity for a study cohort.
=================================================================

Offline job, run on patient enrollment or after a protocol-set change.
PPF (patient-protocol fit) and the protocol similarity matrix are
precomputed and written to ``~/.ai_cdss/output/`` so the recommender can
read them at recommend time (it never computes them on the hot path).

Outputs:
    ~/.ai_cdss/output/ppf.parquet           (PATIENT_ID, PROTOCOL_ID, PPF, CONTRIB)
    ~/.ai_cdss/output/protocol_similarity.csv (PROTOCOL_A, PROTOCOL_B, SIMILARITY)

Both steps are exposed on `RecommendationService` (which wraps
`precompute.py` + the repository's offline accessors). This script just
resolves the cohort and calls them.

Run:  python scripts/compute_ppf.py --study 2
"""
import argparse
import logging

from ai_cdss import RecommendationService

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=int, nargs="+", default=[2],
        help="Study ID(s) whose patients get PPF computed.",
    )
    args = parser.parse_args()

    service = RecommendationService()

    # Resolve the cohort (DB), then compute + persist both artifacts.
    patient_ids = service.repository.fetch_and_validate_patients(study_ids=args.study)
    if not patient_ids:
        logging.warning("No patients resolved for study %s — nothing to do.", args.study)
        return

    ppf_result = service.compute_patient_fit(patient_ids)
    print(ppf_result)

    similarity_result = service.compute_protocol_similarity()
    print(similarity_result)


if __name__ == "__main__":
    main()
