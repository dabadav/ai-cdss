# %%
import logging
import os  # For cleanup
from pathlib import Path

import pandas as pd

# Configure a simple logger for debug output
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Define a temporary path for testing
TEST_DIR = Path("./test_parquet_upsert_data")
PPF_PARQUET_FILEPATH = TEST_DIR / "test_ppf_data.parquet"


def setup_test_environment():
    """Ensures a clean test directory and file before each test run."""
    if TEST_DIR.exists():
        import shutil

        shutil.rmtree(TEST_DIR)  # Remove previous test directory and its contents
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Setting up test environment in: {TEST_DIR.absolute()} ---")


def create_test_df(patient_id, protocol_id, ppf_score, contrib_list, other_col="value"):
    """Helper to create a single-row DataFrame for testing."""
    return pd.DataFrame(
        {
            "PATIENT_ID": [patient_id],
            "PROTOCOL_ID": [protocol_id],
            "PPF_SCORE": [ppf_score],
            "CONTRIB": [
                contrib_list
            ],  # Keep this as a list to test complex type handling
            "OTHER_COL": [other_col],
        }
    )


def run_upsert_logic(ppf_contrib_df):
    """
    Simulates the core upsert logic that would be in your FastAPI endpoint.
    Assumes PPF_PARQUET_FILEPATH is globally defined.
    """
    logger.debug("\n--- Running upsert with new data:\n%s", ppf_contrib_df.to_string())

    try:
        # Step 1: Read existing data if the file exists
        if PPF_PARQUET_FILEPATH.exists():
            existing_ppf = pd.read_parquet(PPF_PARQUET_FILEPATH)
            # Ensure 'PATIENT_ID' and 'PROTOCOL_ID' are regular columns, not index,
            # and have consistent data types for merging/filtering.
            # This is robust against previous saves where index might have been used.
            # Use reset_index(drop=True) to avoid making the old index a new column.
            existing_ppf = existing_ppf.reset_index(drop=True)

            # Ensure key columns are consistent types for merging
            # This is crucial for merge to work as expected
            existing_ppf["PATIENT_ID"] = existing_ppf["PATIENT_ID"].astype(int)
            ppf_contrib_df["PATIENT_ID"] = ppf_contrib_df["PATIENT_ID"].astype(int)
            existing_ppf["PROTOCOL_ID"] = existing_ppf["PROTOCOL_ID"].astype(
                str
            )  # Assuming Protocol IDs can be mixed (int/str)
            ppf_contrib_df["PROTOCOL_ID"] = ppf_contrib_df["PROTOCOL_ID"].astype(
                str
            )  # Ensure consistent string type

            logger.debug("Existing PPF before merge:\n%s", existing_ppf.to_string())

            # Step 2: Identify keys in the new data (ppf_contrib)
            new_or_updated_keys = ppf_contrib_df[["PATIENT_ID", "PROTOCOL_ID"]]

            # Step 3: Filter out rows from existing_ppf that are being updated by ppf_contrib
            # Use an anti-join to keep only old rows that DON'T have a match in new_or_updated_keys
            merged = existing_ppf.merge(
                new_or_updated_keys,
                on=["PATIENT_ID", "PROTOCOL_ID"],
                how="left",
                indicator=True,
            )
            filtered_existing_ppf = existing_ppf[merged["_merge"] == "left_only"]

            logger.debug(
                "Filtered existing data (rows not updated):\n%s",
                filtered_existing_ppf.to_string(),
            )

            # Step 4: Concatenate the filtered existing data with the new/updated data
            updated_ppf = pd.concat(
                [filtered_existing_ppf, ppf_contrib_df], ignore_index=True
            )

            logger.debug("Combined df after upsert:\n%s", updated_ppf.to_string())

            # Step 5: Overwrite the file with combined data
            # Always save with index=False unless your index is meaningful data itself
            updated_ppf.to_parquet(PPF_PARQUET_FILEPATH, index=False)
            print(
                f"Appended/Updated (by rewriting) to existing Parquet file: {PPF_PARQUET_FILEPATH}"
            )

        else:
            # If file doesn't exist, create it normally for the first time
            logger.debug(
                "Creating new Parquet file: %s with initial data:\n%s",
                PPF_PARQUET_FILEPATH,
                ppf_contrib_df.to_string(),
            )
            ppf_contrib_df.to_parquet(PPF_PARQUET_FILEPATH, index=False)
            print(f"Created new Parquet file: {PPF_PARQUET_FILEPATH}")

    except Exception as e:
        logger.error("Error during Parquet file upsert: %s", e, exc_info=True)
        raise  # Re-raise for test failure


def read_current_parquet_file():
    """Reads and returns the current content of the Parquet file."""
    if PPF_PARQUET_FILEPATH.exists():
        df = pd.read_parquet(PPF_PARQUET_FILEPATH)
        # Ensure consistent types after reading for comparison
        df["PATIENT_ID"] = df["PATIENT_ID"].astype(int)
        df["PROTOCOL_ID"] = df["PROTOCOL_ID"].astype(str)
        # Sort for consistent comparison in tests
        return df.sort_values(by=["PATIENT_ID", "PROTOCOL_ID"]).reset_index(drop=True)
    return pd.DataFrame()  # Return empty DataFrame if file doesn't exist


# --- Test Cases ---

# Test 1: Initial Write (file does not exist)
setup_test_environment()
initial_data = create_test_df(1, "P1", 0.8, [0.1, 0.2], "first_val")
run_upsert_logic(initial_data)
expected_df_t1 = initial_data.sort_values(by=["PATIENT_ID", "PROTOCOL_ID"]).reset_index(
    drop=True
)
assert_df_equal = pd.testing.assert_frame_equal(
    read_current_parquet_file(), expected_df_t1
)
print(
    f"Test 1 (Initial Write) passed. Content:\n{read_current_parquet_file().to_string()}"
)


# Test 2: Pure Insert (add new unique rows)
new_inserts = pd.concat(
    [
        create_test_df(1, "P2", 0.7, [0.3, 0.4], "new_val_1"),
        create_test_df(2, "P1", 0.9, [0.5, 0.6], "new_val_2"),
    ],
    ignore_index=True,
)
run_upsert_logic(new_inserts)
expected_df_t2 = (
    pd.concat([initial_data, new_inserts], ignore_index=True)
    .sort_values(by=["PATIENT_ID", "PROTOCOL_ID"])
    .reset_index(drop=True)
)
pd.testing.assert_frame_equal(read_current_parquet_file(), expected_df_t2)
print(
    f"Test 2 (Pure Insert) passed. Content:\n{read_current_parquet_file().to_string()}"
)


# Test 3: Pure Update (modify existing rows)
update_data = pd.concat(
    [
        create_test_df(
            1, "P1", 0.95, [0.9, 0.9], "UPDATED_VAL"
        ),  # Update existing (1, P1)
    ],
    ignore_index=True,
)
run_upsert_logic(update_data)
# Expected: (1, P1) is updated, (1, P2) and (2, P1) remain as they were after Test 2
expected_df_t3 = (
    pd.concat(
        [
            create_test_df(1, "P2", 0.7, [0.3, 0.4], "new_val_1"),  # From initial + T2
            create_test_df(2, "P1", 0.9, [0.5, 0.6], "new_val_2"),  # From initial + T2
            create_test_df(1, "P1", 0.95, [0.9, 0.9], "UPDATED_VAL"),  # Updated (1, P1)
        ],
        ignore_index=True,
    )
    .sort_values(by=["PATIENT_ID", "PROTOCOL_ID"])
    .reset_index(drop=True)
)
pd.testing.assert_frame_equal(read_current_parquet_file(), expected_df_t3)
print(
    f"Test 3 (Pure Update) passed. Content:\n{read_current_parquet_file().to_string()}"
)


# Test 4: Mixed - Insert new and Update existing (combination)
mixed_data = pd.concat(
    [
        create_test_df(2, "P1", 0.99, [1.0, 1.0], "UPDATED_VAL_2"),  # Update (2, P1)
        create_test_df(3, "P1", 0.88, [0.7, 0.7], "NEW_PATIENT_3"),  # Insert (3, P1)
    ],
    ignore_index=True,
)
run_upsert_logic(mixed_data)
# Expected: (1,P1) updated, (1,P2) unchanged, (2,P1) updated, (3,P1) inserted
expected_df_t4 = (
    pd.concat(
        [
            create_test_df(1, "P2", 0.7, [0.3, 0.4], "new_val_1"),  # unchanged
            create_test_df(1, "P1", 0.95, [0.9, 0.9], "UPDATED_VAL"),  # from T3
            create_test_df(2, "P1", 0.99, [1.0, 1.0], "UPDATED_VAL_2"),  # updated
            create_test_df(3, "P1", 0.88, [0.7, 0.7], "NEW_PATIENT_3"),  # inserted
        ],
        ignore_index=True,
    )
    .sort_values(by=["PATIENT_ID", "PROTOCOL_ID"])
    .reset_index(drop=True)
)
pd.testing.assert_frame_equal(read_current_parquet_file(), expected_df_t4)
print(
    f"Test 4 (Mixed Insert/Update) passed. Content:\n{read_current_parquet_file().to_string()}"
)


# Test 5: Empty ppf_contrib_df (should not modify the file)
print("\n--- Test 5: Empty ppf_contrib_df (should not modify file) ---")
current_content_before_empty_write = read_current_parquet_file()
run_upsert_logic(
    pd.DataFrame(columns=current_content_before_empty_write.columns)
)  # Empty DF with correct columns
pd.testing.assert_frame_equal(
    read_current_parquet_file(), current_content_before_empty_write
)
print(
    f"Test 5 (Empty ppf_contrib_df) passed. Content:\n{read_current_parquet_file().to_string()}"
)


# Clean up test files
# import shutil
# shutil.rmtree(TEST_DIR)
# print(f"\nCleaned up test directory: {TEST_DIR}")
# %%
