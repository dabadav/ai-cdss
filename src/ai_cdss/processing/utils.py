import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from ai_cdss.constants import BY_ID, BY_PP, BY_PPS, DEFAULT_LOG_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Utils
# ---------------------------------------------------------------


def get_nth(df, col, groupby_col, session_index_col, n):
    """
    Return the nth session for each group, sorted by session index.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to select.
        groupby_col (str or list): Column(s) to group by.
        session_index_col (str): Column to sort within each group.
        n (int): Index of the session to select (can be negative).

    Returns:
        pd.DataFrame: DataFrame with the nth session for each group.
    """
    # Sort so that session index is in order within each protocol
    df_sorted = df.sort_values(by=[session_index_col])
    # Group by protocol and get the nth session (n can be negative)
    nth_sessions = df_sorted.groupby(groupby_col).nth(n)
    # Drop NaNs and select needed column
    return nth_sessions[BY_PP + [session_index_col, col]].dropna()


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on,
    how: Literal["left", "right", "outer", "inner"] = "left",
    export_dir: Path = DEFAULT_LOG_DIR,
    left_name: str = "left",
    right_name: str = "right",
) -> pd.DataFrame:
    """
    Perform a safe merge and independently report unmatched rows from left and right DataFrames.

    Parameters
    ----------
    left : pd.DataFrame
        Left DataFrame.
    right : pd.DataFrame
        Right DataFrame.
    on : str or list
        Column(s) to join on.
    how : str, optional
        Type of merge to be performed. Default is "left".
    export_dir : str, optional
        Directory to export discrepancy reports and logs.
    left_name : str, optional
        Friendly name for the left DataFrame, for logging.
    right_name : str, optional
        Friendly name for the right DataFrame, for logging.
    drop_inconsistencies : bool, optional
        If True, drop inconsistent rows (left-only). Default is False.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    # Prepare export directory
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Outer merge for discrepancy check
    discrepancy_check = left.merge(right, on=on, how="outer", indicator=True)

    left_only = discrepancy_check[discrepancy_check["_merge"] == "left_only"]
    right_only = discrepancy_check[discrepancy_check["_merge"] == "right_only"]

    expected_left_cols = BY_ID + ["SESSION_DURATION", "GAME_SCORE", "DM_VALUE"]
    available_left_cols = [
        col for col in expected_left_cols if col in left_only.columns
    ]

    expected_right_cols = BY_PPS + ["SESSION_DURATION", "GAME_SCORE", "DM_VALUE"]
    available_right_cols = [
        col for col in expected_right_cols if col in right_only.columns
    ]

    # Export and log discrepancies if found
    if not left_only.empty:
        export_file = os.path.join(export_dir, f"{left_name}_only_{timestamp}.csv")
        try:
            left_only[available_left_cols].to_csv(export_file, index=False)
        except KeyError:
            left_only.to_csv(export_file, index=False)

        logger.warning(
            "%d rows found only in '%s' DataFrame (see export: %s)",
            len(left_only),
            left_name,
            export_file,
        )

    if not right_only.empty:
        export_file = os.path.join(export_dir, f"{right_name}_only_{timestamp}.csv")
        try:
            right_only[available_right_cols].to_csv(export_file, index=False)
        except KeyError:
            right_only.to_csv(export_file, index=False)
        logger.warning(
            "%d rows from '%s' DataFrame did not match '%s' DataFrame (see export: %s)",
            len(right_only),
            right_name,
            left_name,
            export_file,
        )

    # Step 3: Actual requested merge
    merged = left.merge(right, on=on, how=how)

    return merged
