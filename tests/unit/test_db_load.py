# from ai_cdss.loaders import DataLoaderMock, DataLoader
# from ai_cdss.processing.features import include_missing_sessions
# from ai_cdss.constants import SESSION_DATE, RECENT_ADHERENCE, DELTA_DM, PPF, PATIENT_ID, PROTOCOL_ID, SCORE, DAYS, USAGE, BY_PP, BY_PPS, USAGE_WEEK, DM_VALUE
# import pandas as pd
# import pandas.testing as pdt
# from IPython.display import display

# def test_session_load():

#     loader = DataLoader()
#     ts = loader.load_session_data([12])

#     # print(loader.interface.engine.engine)
#     # session = loader.load_session_data([13])
#     with pd.option_context('display.max_columns', None, 'display.width', 1000):
#         print("Session data \n")
#         display(ts)

#     # grouped = ts.groupby(BY_PPS).agg({DM_VALUE: "mean"}).reset_index()
#     # with pd.option_context('display.max_columns', None, 'display.width', 1000):
#     #     print("Session mean data \n")
#     #     display(grouped)

#     # final = include_missing_sessions(grouped)
#     # with pd.option_context('display.max_columns', None, 'display.width', 1000):
#     #     print("Session mean data \n")
#     #     display(final)



#         # Includes prescribed but not performed sessions
#     #     print("\n All Session data \n")
#     #     df = include_missing_sessions(session)
#     #     display(df)

#     assert False



    # assert False


    # Monday 8-12??