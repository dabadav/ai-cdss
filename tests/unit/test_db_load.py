from ai_cdss.data_loader import DataLoaderMock, DataLoader
from ai_cdss.data_processor import include_missing_sessions
from ai_cdss.constants import SESSION_DATE, RECENT_ADHERENCE, DELTA_DM, PPF, PATIENT_ID, PROTOCOL_ID, SCORE, DAYS, USAGE, BY_PP, BY_PPS, USAGE_WEEK
import pandas as pd
import pandas.testing as pdt
from IPython.display import display

# def test_session_load():

#     loader = DataLoader()

#     session = loader.load_session_data([13])
#     with pd.option_context('display.max_columns', None, 'display.width', 1000):
#         print("Session data \n")
#         display(session)
#         # Includes prescribed but not performed sessions
#         print("\n All Session data \n")
#         df = include_missing_sessions(session)
#         display(df)

#     assert False