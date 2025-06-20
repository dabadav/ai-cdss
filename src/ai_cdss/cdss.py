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
        scoring: DataFrame[ScoringSchema],
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

    def recommend(
        self, patient_id: int, protocol_similarity
    ) -> DataFrame[ScoringSchema]:
        """
        Recommend prescriptions for a patient.

        Parameters
        ----------
        patient_id : int
            The ID of the patient.
        protocol_similarity : DataFrame
            A DataFrame containing protocol similarity scores.

        Returns
        -------
        DataFrame
            A DataFrame mapping recommended protocol IDs to their scheduling details.
        """
        # Get scores for patient
        patient_data = self.scoring[self.scoring[PATIENT_ID] == patient_id]
        if patient_data.empty:
            return pd.DataFrame()

        # Get current prescriptions (which already include scores)
        prescriptions = self.get_prescriptions(patient_id)

        # Track protocol rows to output
        rows = []

        if not prescriptions.empty:

            # ALL_PRESCRIPTIONS_WEEK_USAGE = 0, Repeat prescriptions
            week_skipped = not prescriptions.apply(
                lambda x: True if x[USAGE_WEEK] >= len(x[DAYS]) else False, axis=1
            ).any()

            # Check this condition
            if week_skipped:
                logger.info(
                    f"Patient {patient_id}, skipped the whole week, cdss repeating prescriptions."
                )
                # Convert to DataFrame
                recommendations = prescriptions
                recommendations.attrs = self.scoring.attrs
                return recommendations

            # Identify which protocols need substitution
            protocols_to_swap = self.decide_prescription_swap(patient_id)
            protocols_excluded = prescriptions[PROTOCOL_ID].tolist()

            # Directly add non-swapped prescriptions
            rows.extend(
                prescriptions[
                    ~prescriptions[PROTOCOL_ID].isin(protocols_to_swap)
                ].to_dict("records")
            )

            # Swap selected protocols
            for protocol_id in protocols_to_swap:
                substitute = self.get_substitute(
                    patient_id,
                    protocol_id,
                    protocol_similarity,
                    protocol_excluded=protocols_excluded,
                )
                if substitute:
                    protocols_excluded.append(substitute)
                    substitute_row = self.get_scores(patient_id, substitute)
                    substitute_row["DAYS"] = prescriptions.loc[
                        prescriptions["PROTOCOL_ID"] == protocol_id, "DAYS"
                    ].values[0]
                    substitute_row["PROTOCOL_ID"] = substitute
                    substitute_row["PATIENT_ID"] = patient_id
                    rows.append(substitute_row)

        else:
            # No prescriptions → Generate new schedule
            top_protocols = self.get_top_protocols(patient_id)
            schedule = self.schedule_protocols(
                top_protocols
            )  # {day: [protocol_id, ...]}

            seen = {}  # protocol_id: row
            for day, protocol_ids in schedule.items():
                for protocol_id in protocol_ids:
                    if protocol_id not in seen:
                        row = self.get_scores(patient_id, protocol_id)
                        row["DAYS"] = [day]
                        row["PROTOCOL_ID"] = protocol_id
                        row["PATIENT_ID"] = patient_id
                        seen[protocol_id] = row
                    else:
                        seen[protocol_id]["DAYS"].append(day)

            rows.extend(seen.values())

        # Convert to DataFrame
        recommendations = (
            pd.DataFrame(rows).sort_values(by=PROTOCOL_ID).reset_index(drop=True)
        )
        recommendations.attrs = self.scoring.attrs
        return recommendations

    def schedule_protocols(self, protocols: List[int]):
        """
        Distribute protocols across days while ensuring constraints.

        Parameters
        ----------
        protocols : list of int
            List of protocol IDs to distribute.

        Returns
        -------
        dict
            A dictionary mapping days to scheduled protocols.
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

        return schedule

    def decide_prescription_swap(self, patient_id: int) -> List[int]:
        """
        Determine which prescriptions to swap based on their score.

        Parameters
        ----------
        patient_id : int
            The ID of the patient.

        Returns
        -------
        list of int
            List of protocol IDs to be swapped.
        """
        prescriptions = self.get_prescriptions(patient_id)
        # Below protocols mean
        return prescriptions[
            prescriptions[SCORE].transform(lambda x: x < x.mean())
        ].PROTOCOL_ID.to_list()

    def get_substitute(
        self,
        patient_id: int,
        protocol_id: int,
        protocol_similarity,
        protocol_excluded: Optional[List[int]] = None,
    ):
        """
        Find a suitable substitute for a given protocol.

        Behavior:
        - Choose 0 usage protocols starting from highest similarity.
        once they are all used,
        - Pick from top 5 similar protocols the least used?

        Parameters
        ----------
        patient_id : int
            The ID of the patient.
        protocol_id : int
            The protocol to be substituted.
        protocol_similarity : DataFrame
            A DataFrame containing protocol similarity scores.
        protocol_excluded : list of int, optional
            List of protocols to exclude from consideration, by default None.

        Returns
        -------
        int
            The ID of the substitute protocol, or None if no suitable substitute is found.
        """

        # Get protocol usage for the given patient and protocol
        usage = self.scoring[self.scoring[PATIENT_ID] == patient_id].set_index(
            PROTOCOL_ID
        )[USAGE]
        # Get protocol similarities
        similarities = protocol_similarity[
            (protocol_similarity[PROTOCOL_A] == protocol_id)
        ]

        # Drop rows where PROTOCOL_B is the same as PROTOCOL_A (self-similarity)
        similarities = similarities[
            similarities[PROTOCOL_A] != similarities[PROTOCOL_B]
        ]

        # Exclude protocols in the `protocol_excluded` list from similarities
        if protocol_excluded:
            similarities = similarities[
                ~similarities[PROTOCOL_B].isin(protocol_excluded)
            ]

        # Find the minimum usage value
        min_usage = usage.min()

        # Get candidates with the lowest usage
        candidates = usage[usage == min_usage].index

        # Among these candidates, select the one with the highest similarity
        candidate_similarities = similarities[similarities[PROTOCOL_B].isin(candidates)]

        # Find the maximum similarity among candidates
        if not candidate_similarities.empty:
            max_sim = candidate_similarities[SIMILARITY].max()

            final_candidates = candidate_similarities[
                candidate_similarities[SIMILARITY] == max_sim
            ][PROTOCOL_B]

            # Return the first candidate (or handle ties)
            return final_candidates.iloc[0] if not final_candidates.empty else None

        else:
            raise ValueError(f"No candidates for protocol {protocol_id}?")

    def get_top_protocols(self, patient_id: int) -> List[int]:
        """
        Select the top N protocols for a patient based on scores.

        Parameters
        ----------
        patient_id : int
            The ID of the patient.

        Returns
        -------
        list of int
            A list of top protocol IDs.
        """
        patient_data = self.scoring[self.scoring[PATIENT_ID] == patient_id]
        top_protocols = patient_data.nlargest(self.n, SCORE)[PROTOCOL_ID].tolist()
        return top_protocols

    def get_prescriptions(self, patient_id: int):
        """
        Retrieve the current prescriptions for a patient.

        Parameters
        ----------
        patient_id : int
            The ID of the patient.

        Returns
        -------
        DataFrame
            A DataFrame containing prescription details.
        """
        patient_data = self.scoring[self.scoring[PATIENT_ID] == patient_id]
        prescriptions = patient_data[
            patient_data[DAYS].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ]
        return prescriptions

    def get_scores(self, patient_id: int, protocol_id: int):
        """
        Retrieve scores for a given patient and protocol.

        Parameters
        ----------
        patient_id : int
            The ID of the patient.
        protocol_id : int
            The ID of the protocol.

        Returns
        -------
        dict
            A dictionary containing score details for the specified patient and protocol.
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
