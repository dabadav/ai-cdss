# src/pipeline.py
import logging
import math
from typing import Dict, List, Optional

import pandas as pd
from ai_cdss.constants import (
    DAYS,
    PATIENT_ID,
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_ID,
    SCORE,
    SIMILARITY,
    USAGE,
    USAGE_WEEK,
)
from ai_cdss.models import ScoringSchema
from pandera.typing import DataFrame

logger = logging.getLogger(__name__)


class CDSS:
    """
    Clinical Decision Support System (CDSS) Class.

    This system provides personalized recommendations for patients based on scoring data.
    It allows protocol recommendations, scheduling, and prescription adjustments.

    Parameters
    ----------
    scoring : DataFrame
        A DataFrame containing patient protocol scores.
    n : int, optional
        Number of top protocols to recommend, by default 12.
    days : int, optional
        Number of days for scheduling, by default 7.
    protocols_per_day : int, optional
        Maximum number of protocols per day, by default 5.
    """

    def __init__(
        self,
        scoring: pd.DataFrame,
        n: int = 12,
        days: int = 7,
        protocols_per_day: int = 5,
    ):
        """
        Initialize the Clinical Decision Support System.
        """
        self.scoring = scoring
        self.n = n
        self.days = days
        self.protocols_per_day = protocols_per_day

    ###########################################################################
    # Recommendation method
    ###########################################################################

    def recommend(self, patient_id: int, protocol_similarity) -> pd.DataFrame:
        """
        Recommend prescriptions for a patient.
        """
        if not self._has_patient_data(patient_id):
            raise ValueError(f"Patient {patient_id} has no data.")

        prescriptions = self._get_prescriptions(patient_id)

        if prescriptions.empty:
            return self._generate_new_recommendations(patient_id)

        if self._is_week_skipped(prescriptions):
            return self._repeat_prescriptions(prescriptions)

        return self._update_existing_recommendations(
            patient_id, prescriptions, protocol_similarity
        )

    ###########################################################################
    # Patient Bootstrap
    ###########################################################################

    def _generate_new_recommendations(self, patient_id: int) -> pd.DataFrame:
        # Generate a new schedule of protocols for a patient with no prescriptions
        top_protocols = self._get_top_protocols(patient_id)
        schedule = self._schedule_protocols(top_protocols)

        # Build the recommendations DataFrame
        rows: list[dict] = []
        seen = {}
        for day, protocol_ids in schedule.items():
            for protocol_id in protocol_ids:
                if protocol_id not in seen:
                    row = self._get_scores(patient_id, protocol_id)
                    row["DAYS"] = [day]
                    row["PROTOCOL_ID"] = protocol_id
                    row["PATIENT_ID"] = patient_id
                    seen[protocol_id] = row
                else:
                    seen[protocol_id]["DAYS"].append(day)
        rows.extend(seen.values())
        recommendations = (
            pd.DataFrame(rows).sort_values(by="PROTOCOL_ID").reset_index(drop=True)
        )
        recommendations.attrs = self.scoring.attrs
        return recommendations

    def _get_top_protocols(self, patient_id: int) -> List[int]:
        """
        Select the top N protocols for a patient based on scores.
        """
        patient_data = self.scoring[self.scoring[PATIENT_ID] == patient_id]
        top_protocols = patient_data.nlargest(self.n, SCORE)[PROTOCOL_ID].tolist()
        return top_protocols

    def _schedule_protocols(self, protocols: List[int]) -> Dict[int, List[int]]:
        """
        Distribute protocols across days while ensuring constraints.
        """
        schedule: Dict[int, List[int]] = {
            day: [] for day in range(0, self.days)
        }  # Days are 1-indexed
        total_slots = self.days * self.protocols_per_day

        if protocols:
            # Repeat protocols as needed to fill the total slots
            repeated_protocols = (protocols * math.ceil(total_slots / len(protocols)))[
                :total_slots
            ]

            # Distribute protocols evenly across days
            for i, protocol in enumerate(repeated_protocols):
                day = i % self.days  # Distribute protocols in a round-robin fashion

                if protocol not in schedule[day]:
                    schedule[day].append(protocol)

        return schedule  # protocol: [day, ...]

    ###########################################################################
    # Prescription Updates (Substitution Logic)
    ###########################################################################

    def _update_existing_recommendations(
        self, patient_id: int, prescriptions: pd.DataFrame, protocol_similarity
    ) -> pd.DataFrame:
        """
        Update recommendations by swapping out underperforming protocols for better alternatives.
        """
        # Identify protocols to swap and those to exclude from substitution
        protocols_to_swap: list[int] = self._decide_prescription_swap(patient_id)
        protocols_excluded: list[int] = prescriptions[PROTOCOL_ID].tolist()

        # Start with prescriptions that are not being swapped
        updated_rows: list[dict] = prescriptions[
            ~prescriptions[PROTOCOL_ID].isin(protocols_to_swap)
        ].to_dict("records")

        # Swap out underperforming protocols
        for protocol_id in protocols_to_swap:
            substitute_row = self._swap_protocol(
                patient_id,
                protocol_id,
                prescriptions,
                protocol_similarity,
                protocols_excluded=protocols_excluded,
            )
            logger.debug(
                "Swapping %s for %s for patient %s",
                protocol_id,
                substitute_row[PROTOCOL_ID],
                patient_id,
            )
            updated_rows.append(substitute_row)
            protocols_excluded.append(substitute_row[PROTOCOL_ID])

        # Create the recommendations DataFrame
        recommendations = (
            pd.DataFrame(updated_rows)
            .sort_values(by=PROTOCOL_ID)
            .reset_index(drop=True)
        )
        recommendations.attrs = self.scoring.attrs
        return recommendations

    ###########################################################################
    # Marginal Value Theorem (Swapping Criteria)

    def _decide_prescription_swap(self, patient_id: int) -> List[int]:
        """
        Determine which prescriptions to swap based on their score.
        """
        prescriptions = self._get_prescriptions(patient_id)
        # Below protocols mean
        return prescriptions[
            prescriptions[SCORE].transform(lambda x: x < x.mean())
        ].PROTOCOL_ID.to_list()

    ###########################################################################
    # Substitution Logic

    def _swap_protocol(
        self,
        patient_id: int,
        protocol_id: int,
        prescriptions: pd.DataFrame,
        protocol_similarity,
        protocols_excluded: list[int],
    ) -> dict:
        """
        Find and return a substitute protocol row for a given protocol_id, or the same protocol if not found (all protocols are prescribed).
        """
        substitute = self._get_substitute(
            patient_id,
            protocol_id,
            protocol_similarity,
            protocols_excluded=protocols_excluded,
        )
        if substitute:
            substitute_row = self._get_scores(patient_id, substitute)
            substitute_row[DAYS] = prescriptions.loc[
                prescriptions[PROTOCOL_ID] == protocol_id, DAYS
            ].values[0]
            substitute_row[PROTOCOL_ID] = substitute
            substitute_row[PATIENT_ID] = patient_id
            return substitute_row

        # Else return same protocol
        return self._get_scores(patient_id, protocol_id)

    def _get_substitute(
        self,
        patient_id: int,
        protocol_id: int,
        protocol_similarity: pd.DataFrame,
        protocols_excluded: Optional[List[int]] = None,
    ) -> Optional[int]:
        """
        Find a suitable substitute for a given protocol.
        Returns the protocol ID of the substitute, or None if not found.
        """
        usage = self._get_patient_protocol_usage(patient_id)
        similarities = self._get_protocol_similarities(
            protocol_id, protocol_similarity, protocols_excluded
        )

        # Try to find unused protocols first
        unused_candidates = self._get_unused_candidates(usage)
        if unused_candidates:
            logger.info(
                "No usage for %s, selecting most similar from %s",
                protocol_id,
                unused_candidates,
            )
            return self._select_most_similar(unused_candidates, similarities)

        # Otherwise, pick from top 5 similar protocols the least used
        top_similar_protocols = self._get_top_similar_protocols(similarities, top_n=5)
        least_used_candidates = self._get_least_used_candidates(
            usage, top_similar_protocols
        )
        if least_used_candidates:
            logger.info(
                "No unused protocols for %s, selecting least used from %s",
                protocol_id,
                least_used_candidates,
            )
            return self._select_most_similar(least_used_candidates, similarities)

        # If no candidates found
        return None

    ###########################################################################
    # USAGE

    def _get_patient_protocol_usage(self, patient_id: int) -> pd.Series:
        """Return protocol usage for the given patient."""
        return self.scoring[self.scoring[PATIENT_ID] == patient_id].set_index(
            PROTOCOL_ID
        )[USAGE]

    def _get_unused_candidates(self, usage: pd.Series) -> List[int]:
        """Return protocol IDs with zero usage."""
        unused = usage[usage == 0].index.tolist()
        return unused

    def _get_least_used_candidates(
        self, usage: pd.Series, candidate_protocols: List[int]
    ) -> List[int]:
        """Return protocol IDs among candidates with the least usage."""
        candidate_usage = usage[usage.index.isin(candidate_protocols)]
        if candidate_usage.empty:
            return []
        min_usage = candidate_usage.min()
        return candidate_usage[candidate_usage == min_usage].index.tolist()

    ###########################################################################
    # SIMILARITY

    def _get_protocol_similarities(
        self,
        protocol_id: int,
        protocol_similarity: pd.DataFrame,
        protocol_excluded: Optional[List[int]],
    ) -> pd.DataFrame:
        """Return similarities for a protocol, excluding self and any excluded protocols."""
        similarities = protocol_similarity[
            protocol_similarity[PROTOCOL_A] == protocol_id
        ]
        similarities = similarities[
            similarities[PROTOCOL_A] != similarities[PROTOCOL_B]
        ]
        if protocol_excluded:
            similarities = similarities[
                ~similarities[PROTOCOL_B].isin(protocol_excluded)
            ]
        return similarities

    def _get_top_similar_protocols(
        self, similarities: pd.DataFrame, top_n: int = 5
    ) -> List[int]:
        """Return the protocol IDs of the top N most similar protocols."""
        return similarities.nlargest(top_n, SIMILARITY)[PROTOCOL_B].tolist()

    def _select_most_similar(
        self, candidates: List[int], similarities: pd.DataFrame
    ) -> Optional[int]:
        """Return the candidate protocol with the highest similarity."""
        candidate_similarities = similarities[similarities[PROTOCOL_B].isin(candidates)]
        if candidate_similarities.empty:
            return None
        max_sim = candidate_similarities[SIMILARITY].max()
        final_candidates = candidate_similarities[
            candidate_similarities[SIMILARITY] == max_sim
        ][PROTOCOL_B]
        return final_candidates.iloc[0] if not final_candidates.empty else None

    ###########################################################################
    # Validation Utilities
    ###########################################################################

    def _is_week_skipped(self, prescriptions: pd.DataFrame) -> bool:
        """
        Check if the prescriptions are skipped for the whole week.
        """
        # Apply a lambda function to each row: check if USAGE_WEEK >= number of DAYS for that prescription
        # If True for any row, .any() will return True (week is skipped for at least one prescription)
        return not prescriptions.apply(
            lambda x: x[USAGE_WEEK] >= len(x[DAYS]), axis=1
        ).any()

    def _has_patient_data(self, patient_id: int) -> bool:
        """Check if patient has scoring data."""
        patient_data = self.scoring[self.scoring[PATIENT_ID] == patient_id]
        return not patient_data.empty

    def _repeat_prescriptions(self, prescriptions) -> pd.DataFrame:
        """Repeat existing prescriptions when week was skipped."""
        logger.info(
            "Patient %s, skipped the whole week, cdss repeating prescriptions.",
            prescriptions[PATIENT_ID].iloc[0] if not prescriptions.empty else "unknown",
        )
        df = prescriptions.copy()
        df.attrs = getattr(self.scoring, "attrs", {})
        return df  # type: ignore

    ###########################################################################
    # General Utilities
    ###########################################################################

    def _get_scores(self, patient_id: int, protocol_id: int):
        """
        Retrieve scores for a given patient and protocol.
        """
        # Filter scoring DataFrame for the given patient and protocol
        return (
            self.scoring[
                (self.scoring[PATIENT_ID] == patient_id)
                & (self.scoring[PROTOCOL_ID] == protocol_id)
            ]
            .iloc[0]
            .to_dict()
        )

    def _get_prescriptions(self, patient_id: int):
        """
        Retrieve the current prescriptions for a patient.
        """
        patient_data = self.scoring[self.scoring[PATIENT_ID] == patient_id]
        prescriptions = patient_data[
            patient_data[DAYS].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ]
        return prescriptions
