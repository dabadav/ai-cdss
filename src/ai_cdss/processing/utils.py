import logging
import os

import pandas as pd
from ai_cdss.constants import *

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Utils
# ---------------------------------------------------------------


def get_nth(df, col, groupby_col, session_index_col, n):
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
    how: str = "left",
    export_dir: str = "~/.ai_cdss/logs/",
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
    timestamp = pd.Timestamp.now()

    # Outer merge for discrepancy check
    discrepancy_check = left.merge(right, on=on, how="outer", indicator=True)

    left_only = discrepancy_check[discrepancy_check["_merge"] == "left_only"]
    right_only = discrepancy_check[discrepancy_check["_merge"] == "right_only"]

    # Export and log discrepancies if found
    if not left_only.empty:
        export_file = os.path.join(export_dir, f"{left_name}_only_{timestamp}.csv")
        try:
            left_only[
                BY_ID + ["SESSION_DURATION", "GAME_SCORE", "DM_VALUE", "PE_VALUE"]
            ].to_csv(export_file, index=False)
        except KeyError as e:
            left_only.to_csv(export_file, index=False)

        logger.warning(
            f"{len(left_only)} rows found only in '{left_name}' DataFrame "
            f"(see export: {export_file})"
        )

    if not right_only.empty:
        export_file = os.path.join(export_dir, f"{right_name}_only_{timestamp}.csv")
        try:
            right_only[
                BY_PPS + ["SESSION_DURATION", "GAME_SCORE", "DM_VALUE", "PE_VALUE"]
            ].to_csv(export_file, index=False)
        except KeyError as e:
            right_only.to_csv(export_file, index=False)
        logger.warning(
            f"{len(right_only)} rows from '{right_name}' DataFrame did not match '{left_name}' DataFrame "
            f"(see export: {export_file})"
        )

    # Step 3: Actual requested merge
    merged = left.merge(right, on=on, how=how)

    return merged
