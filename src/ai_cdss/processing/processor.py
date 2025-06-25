# ai_cdss/processing/processor.py
import logging
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional

import numpy as np
import pandas as pd
from ai_cdss.constants import *
from ai_cdss.models import DataUnitName, DataUnitSet, ScoringSchema, SessionSchema
from ai_cdss.processing.features import (
    apply_savgol_filter_groupwise,
    get_rolling_theilsen_slope,
    include_missing_sessions,
)
from ai_cdss.processing.pipeline import DataPipeline
from ai_cdss.processing.utils import get_nth, safe_merge
from pandas import Timestamp
from pandera.typing import DataFrame

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################
# ------ Data Processing ------ #
#################################

# ---------------------------------------------------------------------
# Data Processor Class


class DataProcessor:
    """
    Backward-compatible facade for the new DataPipeline.
    Provides the same interface as the old DataProcessor.
    """

    def __init__(self, *args, **kwargs):
        # Accept legacy args/kwargs for compatibility, but only use what you need
        self.pipeline = DataPipeline()

    def process_data(self, data, scoring_date):
        """
        Backward-compatible method for processing data.
        Delegates to DataPipeline.process.
        """
        return self.pipeline.process(data, scoring_date)
