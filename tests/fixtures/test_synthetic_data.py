# import pytest

# def test_custom_synthetic_data(synthetic_data_factory):
#     session_df, timeseries_df, ppf_df = synthetic_data_factory(
#         num_patients=2,
#         num_sessions=2,
#         timepoints=20,
#         null_cols_session=[],
#         null_cols_timeseries=[]
#     )

#     # Ensure session consistency
#     session_ids_1 = set(session_df["SESSION_ID"])
#     session_ids_2 = set(timeseries_df["SESSION_ID"])
#     assert session_ids_2.issubset(session_ids_1)

#     # Print
#     print(timeseries_df.head())
#     print(session_df.head())
#     print(ppf_df.head())
