# %%
import cProfile
import io
import logging
import pstats

import pandas as pd
from ai_cdss.interface import CDSSInterface
from ai_cdss.loaders import DataLoader
from ai_cdss.processing import DataProcessor
from IPython.display import display

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

loader = DataLoader(rgs_mode="plus")
timestamp = pd.Timestamp("2025-06-26 10:28:06")
processor = DataProcessor()
pr = cProfile.Profile()
pr.enable()

# The code you want to profile:
cdss_client = CDSSInterface(loader, processor)
print(cdss_client.compute_protocol_similarity())
print(
    cdss_client.recommend_for_study(
        study_id=[2],
        days=7,
        protocols_per_day=5,
        n=12,
        scoring_date=timestamp,
    )
)

pr.disable()
s = io.StringIO()
sortby = "cumulative"
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(30)  # Show top 30 lines
print(s.getvalue())

# %%
