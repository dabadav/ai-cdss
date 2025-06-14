from ai_cdss.data_processor import DataProcessor, safe_merge
from ai_cdss.data_loader import DataLoaderMock, DataLoader
from ai_cdss.cdss import CDSS
from ai_cdss.constants import SESSION_DATE, RECENT_ADHERENCE, DELTA_DM, PPF, PATIENT_ID, PROTOCOL_ID, SCORE, DAYS, USAGE, BY_PP, BY_PPS, USAGE_WEEK
from rgs_interface.data.schemas import RecsysMetricsRow
from IPython.display import display
import pandas as pd
import pandas.testing as pdt
from functools import reduce

# def test_feature_computation_usage():
#     """Test the exclusion of sessions that are outside the study range.
    
#     Patient 12 — Study 405:
#     - Prescribed 3 protocols (233, 220, 231)
#     - Only 2 prescriptions had sessions:
#         - Protocol 220: 2 sessions
#         - Protocol 231: 1 session
#         - Protocol 233: 0 sessions (unused)
#     """
#     patient_id = 12
#     loader = DataLoader()
#     session = loader.load_session_data([patient_id])
#     print(f"\nSession data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(session)

#     processor = DataProcessor()
#     usage = processor.build_usage(session)
#     print(f"\nUsage... \n")
#     with pd.option_context('display.max_columns', None):
#         display(usage)

#     # Define expected result
#     expected = pd.DataFrame({
#         "PATIENT_ID": [12, 12, 12],
#         "PROTOCOL_ID": [220, 231, 233],
#         "USAGE": [3, 1, 0]
#     }).astype({"USAGE": "Int64"})

#     # Assert equality
#     pdt.assert_frame_equal(
#         usage.sort_values(by=["PATIENT_ID", "PROTOCOL_ID"]).reset_index(drop=True),
#         expected.sort_values(by=["PATIENT_ID", "PROTOCOL_ID"]).reset_index(drop=True),
#         check_dtype=False
#     )

#     # assert False

# def test_feature_computation_adherence_internal_ewma_nan():
#     """Test behavior of ewma function when nan values

#     Expected behavior:
#     - When day skipped ADHERENCE of that sessions set to nan
#     - EWMA function does not use or is influence by nan
#     """
#     patient_id = 13
#     loader = DataLoader()
#     session = loader.load_session_data([patient_id])
#     print(f"\nSession data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(session)

#     processor = DataProcessor()
#     adherence = processor.build_recent_adherence(session)
#     print(f"\n Adherence... \n")
#     with pd.option_context('display.max_columns', None, 'display.width', 1000):
#         display_adherence = adherence.sort_values(by=BY_PP)
#         display(display_adherence)

#     # Assert EWMA skips NaNs — check it's not NaN where it shouldn't be
#     ewma_vals = adherence['ADHERENCE_RECENT']
#     non_skipped = adherence[adherence['STATUS'] == 'CLOSED']
#     assert not ewma_vals[non_skipped.index].isna().any(), "EWMA should compute values for non-skipped days"

#     # Assert EWMA correctly aggregates — crude check based on order
#     # For protocol 205:
#     df_205 = adherence[adherence['PROTOCOL_ID'] == 205].sort_values(by='SESSION_INDEX')
#     assert df_205['ADHERENCE_RECENT'].iloc[-1] > 0, "EWMA for protocol 205 should be > 0"

# def test_feature_computation_adherence():
#     """Test the exclusion of sessions that are outside the study range
#     Patient 12 — Study 405:
#     - Prescribed 3 protocols (233, 220, 231)
#     - Only 2 prescriptions had sessions:
#         - Protocol 220: 3 sessions
#         - Protocol 231: 1 session
#         - Protocol 233: 0 sessions (unused)
#     """
#     patient_id = 12
#     loader = DataLoader()
#     session = loader.load_session_data([patient_id])
#     print(f"\nSession data... \n")
#     # with pd.option_context('display.max_columns', None):
#     #     display(session)
   
#     processor = DataProcessor()
#     adherence = processor.build_recent_adherence(session)
#     print(f"\n Adherence... \n")
#     with pd.option_context('display.max_columns', None):
#         display_adherence = adherence.sort_values(by=BY_PP)
#         # display_adherence = display_adherence[display_adherence.PROTOCOL_ID == 220]
#         display(display_adherence)

# def test_computation_adherence_week_day_dropout():
#     """
#     Here it is tested adherence behavior when a patient is skips a whole day (protocol agnostic).

#     Scenario:

#     insert PRESCRIPTIONS FOR PATIENT
#     insert SESSIONS EXCEPT FOR ONE DAY 
    
#     INSERT INTO `prescription_plus` (`PRESCRIPTION_ID`, `PATIENT_ID`, `PROTOCOL_ID`, `STARTING_DATE`, `ENDING_DATE`, `WEEKDAY`, `SESSION_DURATION`, `AR_MODE`) VALUES (NULL, '13', '205', '2025-06-16 00:00:01', '2025-06-30 00:00:01', 'THURSDAY', '300', 'NONE'); 
#     INSERT INTO `session_plus` (`SESSION_ID`, `PRESCRIPTION_ID`, `STARTING_DATE`, `ENDING_DATE`, `STATUS`, `PLATFORM`, `DEVICE`, `SESSION_LOG_PARSED`) VALUES (NULL, '8327', '2025-06-19 12:56:49', '2025-06-19 12:59:49', 'CLOSED', 'OTHER', NULL, '0');
    
#     Expected Behavior:
#     - Do not penalize adherence
#     - Adherence for whole day sessions is considered NaN
#     """
#     patient_id = 13
#     loader = DataLoader()
#     session = loader.load_session_data([patient_id])
#     print(f"\nSession data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(session)
   
#     processor = DataProcessor()
#     adherence = processor.build_recent_adherence(session)
#     print(f"\n Adherence... \n")
#     with pd.option_context('display.max_columns', None, 'display.width', 1000):
#         display_adherence = adherence.sort_values(by=BY_PP)
#         # display_adherence = display_adherence[display_adherence.PROTOCOL_ID == 220]
#         display(display_adherence)

#     assert False

# def test_feature_computation_usage_week():
#     from datetime import datetime, timedelta
#     patient_id = 12
#     loader = DataLoader()
#     session = loader.load_session_data([patient_id])

#     # Compute this week's Monday 00:00 and Sunday 23:59:59.999999
#     now = datetime.now()
#     week_start = now - timedelta(days=now.weekday())  # Monday
#     week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
#     week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
#     df = session[(session[SESSION_DATE] >= week_start) & (session[SESSION_DATE] < week_end)]

#     print(f"\n Sessions... \n")
#     with pd.option_context('display.max_columns', None):
#         # display_adherence = display_adherence[display_adherence.PROTOCOL_ID == 220]
#         display(session)

#     print(f"\n Filtered sessions... \n")
#     with pd.option_context('display.max_columns', None):
#         # display_adherence = display_adherence[display_adherence.PROTOCOL_ID == 220]
#         display(df)

# def test_feature_computation_days():

#     patient_id = 13
#     loader = DataLoader()
#     session = loader.load_session_data([patient_id])
#     print(f"\nSession data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(session)

#     processor = DataProcessor()
#     # scoring_date = processor._get_scoring_date()
#     scoring_date = pd.Timestamp("2025-06-23")
#     days = processor.build_prescription_days(session, scoring_date=scoring_date)
#     print(f"\n Active prescriptions... \n")
#     with pd.option_context('display.max_columns', None):
#         display_days = days.sort_values(by=BY_PP)
#         # display_adherence = display_adherence[display_adherence.PROTOCOL_ID == 220]
#         display(display_days)

    # assert False

# def test_feature_computation_dm():
#     """Test dm slope / delta generation
#     """
#     patient_id = 12
#     loader = DataLoader()
#     timeseries = loader.load_timeseries_data([patient_id])
#     print(f"\n Timeseries data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(timeseries)

#     processor = DataProcessor()
#     dm = processor.build_delta_dm(timeseries)
#     print(f"\n Delta DM... \n")
#     with pd.option_context('display.max_columns', None):
#         display(dm)

#     # assert False

# def test_feature_aggregation():
    
#     patient_id = 12
#     loader = DataLoader()
#     session = loader.load_session_data([patient_id])
#     timeseries = loader.load_timeseries_data([patient_id])
#     ppf = loader.load_ppf_data([patient_id])
#     print(f"\nSession data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(session)

#     processor = DataProcessor()

#     # Compute Session Features
#     dm_df = processor.build_delta_dm(timeseries)                # DELTA_DM
#     adherence_df = processor.build_recent_adherence(session)    # ADHERENCE_RECENT
#     usage_df = processor.build_usage(session)                   # USAGE
#     days_df = processor.build_prescription_days(session)        # DAYS
    
#     print(f"\nDM data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(dm_df)

#     print(f"\nAdherence data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(adherence_df)

#     print(f"\nUsage data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(usage_df)

#     print(f"\nActive prescriptions data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(days_df)

#     # Combine Session Features
#     feat_pp_df = reduce(lambda l, r: pd.merge(l, r, on=BY_PP, how='left'), [ppf, usage_df, days_df])
#     feat_pps_df = reduce(lambda l, r: pd.merge(l, r, on=BY_PPS, how='left'), [adherence_df, dm_df])
    
#     print(f"\nFeatures protocol-level data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(feat_pp_df)

#     print(f"\nFeatures session-level data... \n")
#     with pd.option_context('display.max_columns', None):
#         display(feat_pps_df)

#     # assert False

# def test_week_skip_patient():
#     """
#     Test Scenario where patient skips a whole week 
    
#     Expected behavior:
#     -> PRESCRIPTIONS REPEATED
#     """

#     # Display options
#     pd.set_option('display.max_columns', None)

#     patient_id = 12
#     loader = DataLoader()
#     session = loader.load_session_data([patient_id])
#     timeseries = loader.load_timeseries_data([patient_id])
#     ppf = loader.load_ppf_data([patient_id])
#     protocol_similarity = loader.load_protocol_similarity()

#     processor = DataProcessor()
#     scoring_df = processor.process_data(session_data=session, timeseries_data=timeseries, ppf_data=ppf, init_data=None)

#     print(f"\nScoring df data... \n")
#     with pd.option_context('display.width', 1000):
#         display(scoring_df)

#     #### Test Week Skip Logic
#     cdss = CDSS(scoring=scoring_df)

#     prescriptions = cdss.get_prescriptions(patient_id=patient_id)
#     print(f"\nPrescriptions data... \n")
#     with pd.option_context('display.width', 1000):
#         display(prescriptions)

#     week_skipped = not prescriptions.apply(lambda x: True if x[USAGE_WEEK] >= len(x[DAYS]) else False, axis=1).any()
#     print(f"Patient week dropout: {week_skipped}, REPEAT_PRESCRIPTIONS")

#     assert week_skipped == True

#     recommendations = cdss.recommend(patient_id=patient_id, protocol_similarity=protocol_similarity)
    
#     print(f"\nDisplay of Recommendations df... \n")
#     with pd.option_context('display.width', 1000):
#         display(recommendations)

#     pdt.assert_frame_equal(recommendations, prescriptions)

# def test_generate_expected_sessions():

#     import random

#     def generate_expected_sessions(start_date, end_date, target_weekday):
#         """
#         Generate all expected session dates between start_date and end_date for the given target weekday.
#         If the prescription end date is in the future, use today as the end limit (assuming future sessions are not yet done).

#         **NOT ROBUST TO INVALID TARGET_WEEKDAY**
#         """
#         expected_dates = []

#         # If the prescription is still ongoing, cap the end_date at today
#         if end_date is None:
#             return expected_dates  # no valid end date
        
#         # Find the first occurrence of the target weekday on or after start_date
#         if start_date.weekday() != target_weekday:
#             days_until_target = (target_weekday - start_date.weekday()) % 7
#             start_date = start_date + pd.Timedelta(days=days_until_target)
        
#         # Generate dates every 7 days (weekly) from the adjusted start_date up to end_date
#         current_date = start_date

#         while current_date <= end_date:
#             expected_dates.append(current_date)
#             current_date += pd.Timedelta(days=7)
        
#         return expected_dates

#     def mock_generate_expected_sessions(start, end, weekday):
#         """
#         Generate session dates between start and end for the given weekday index.
#         Weekday: 0=Monday, 1=Tuesday, ..., 6=Sunday
#         """
#         weekday_map = {
#             0: 'W-MON',
#             1: 'W-TUE',
#             2: 'W-WED',
#             3: 'W-THU',
#             4: 'W-FRI',
#             5: 'W-SAT',
#             6: 'W-SUN',
#         }

#         freq = weekday_map.get(weekday)
#         if freq is None:
#             return []

#         return list(pd.date_range(start=start, end=end, freq=freq))
    
#     # Edge Case: start date is the weekday
#     start = pd.Timestamp("2025-06-16")  # Monday
#     end = start + pd.Timedelta(days=7)
#     weekday = 0  # Monday
#     expected = mock_generate_expected_sessions(start, end, weekday)
#     actual = generate_expected_sessions(start, end, weekday)
#     assert list(actual) == list(expected), "Edge Case 1 failed"
    
#     # Edge Case 2: end date is the weekday
#     start = pd.Timestamp("2025-06-10")  # Tuesday
#     end = pd.Timestamp("2025-06-16")    # Monday (should catch next Monday)
#     weekday = 0  # Monday
#     expected = mock_generate_expected_sessions(start, end, weekday)
#     actual = generate_expected_sessions(start, end, weekday)
#     assert list(actual) == list(expected), "Edge Case 2 failed"

#     # Edge Case 3: start == end and it's the weekday
#     date = pd.Timestamp("2025-06-16")  # Monday
#     weekday = 0
#     expected = mock_generate_expected_sessions(date, date, weekday)
#     actual = generate_expected_sessions(date, date, weekday)
#     assert list(actual) == list(expected), "Edge Case 3 failed"

#     # Edge Case 4: start == end and it's NOT the weekday
#     date = pd.Timestamp("2025-06-16")  # Monday
#     weekday = 1  # Tuesday
#     expected = mock_generate_expected_sessions(date, date, weekday)
#     actual = generate_expected_sessions(date, date, weekday)
#     assert list(actual) == list(expected), "Edge Case 4 failed"

#     # Edge Case 5: invalid weekday
#     start = pd.Timestamp("2025-06-10")
#     end = pd.Timestamp("2025-06-20")
#     weekday = 10
#     expected = mock_generate_expected_sessions(start, end, weekday)
#     actual = generate_expected_sessions(start, end, weekday)
#     # assert list(actual) == list(expected), "Edge Case 5 failed"

#     # Edge Case 6: end < start
#     start = pd.Timestamp("2025-06-20")
#     end = pd.Timestamp("2025-06-10")
#     weekday = 2
#     expected = mock_generate_expected_sessions(start, end, weekday)
#     actual = generate_expected_sessions(start, end, weekday)
#     assert list(actual) == list(expected), "Edge Case 6 failed"

#     # Test with multiple randomized inputs
#     for _ in range(10):
#         weekday = random.randint(0, 6)
#         start = pd.Timestamp("2025-06-01") + pd.Timedelta(days=random.randint(0, 12))
#         end = start + pd.Timedelta(days=random.randint(7, 41))  # 1–3 weeks

#         expected = mock_generate_expected_sessions(start, end, weekday)
#         actual = generate_expected_sessions(start, end, weekday)

#         print(expected)
#         print(actual)

#         assert list(actual) == list(expected), f"Mismatch on weekday={weekday}, start={start}, end={end}"

# def test_include_missing_sessions_simple_case():
#     from ai_cdss.data_processor import include_missing_sessions

#     # Create fake session dataframe (as above)
#     df = pd.DataFrame([
#         {
#             "PATIENT_ID": 13,
#             "PRESCRIPTION_ID": 1001,
#             "PROTOCOL_ID": 200,
#             "SESSION_ID": None,
#             "PRESCRIPTION_STARTING_DATE": pd.Timestamp("2025-05-16"),
#             "PRESCRIPTION_ENDING_DATE": pd.Timestamp("2025-05-23"),
#             "WEEKDAY_INDEX": 0,  # Monday
#             "SESSION_DATE": pd.NaT,
#             "STATUS": None,
#             "SESSION_DURATION": None,
#             "REAL_SESSION_DURATION": None,
#             "PRESCRIBED_SESSION_DURATION": 300,
#             "ADHERENCE": None
#         }
#     ])

#     result = include_missing_sessions(df)
#     print(f'\ninclude missing ...\n')
#     with pd.option_context('display.max_columns', None, 'display.width', 1000):
#         display(result)

#     missing = result[result["STATUS"] == "NOT_PERFORMED"]
#     assert len(missing) == 1
#     assert missing.iloc[0]["SESSION_DATE"].weekday() == 0  # Monday
