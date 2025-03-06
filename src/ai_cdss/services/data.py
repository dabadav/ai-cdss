from datetime import date, timedelta
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from ai_cdss.models import SessionBatchSchema

class DataProcessor:
    """
    Data repository class

    This class will load and process SessionBatch data, Timeseries data, Patient data, Protocol data.
    """

    def __init__(self, patient_list, mapping_dict):
        self.patient_list = patient_list
        self.session_batch = None
        self.patient_data = None
        self.protocol_data = None
        self.mapping_dict = mapping_dict

    def get_session(self) -> DataFrame[SessionBatchSchema]:
        """
        Load session batch data from db
        """
        from recsys_interface.data.interface import fetch_rgs_data

        # Fetch data from two sources/modes and concatenate
        data_app = fetch_rgs_data(self.patient_list, rgs_mode="app", output_file="../../data/app_data.csv")
        data_plus = fetch_rgs_data(self.patient_list, rgs_mode="plus", output_file="../../data/plus_data.csv")
        session_batch = pd.concat([data_app, data_plus], ignore_index=True)
        
        return session_batch
        
    def get_protocol(self, protocol_path="../../data/protocol_attributes.csv"):
        """
        Load protocol attributes matrix
        """
        protocol_attributes = pd.read_csv(protocol_path, index_col=0)
        self.protocol_data = protocol_attributes

        return protocol_attributes

    def get_patient(self, clinical_path="../../data/clinical_scores.csv"):
        """
        Load Patient data
        """
        patient_data = pd.read_csv(clinical_path, index_col=0)
        self.patient_data = patient_data

        return patient_data
    
    def get_timeseries(self):
        """
        Fetch time-series records of Difficulty Modulators (DMs) and Performance Estimators (PEs) for given patients.
        Combines data from two modes (app and plus).
        """
        from recsys_interface.data.interface import fetch_timeseries_data
        dms_app = fetch_timeseries_data(patients_ids=self.patient_list, rgs_mode="app", output_file="../../data/app_timeseries.csv")
        dms_plus = fetch_timeseries_data(patients_ids=self.patient_list, rgs_mode="plus", output_file="../../data/plus_timeseries.csv")
        timeseries = pd.concat([dms_app, dms_plus], ignore_index=True)
        return timeseries

    def process_session_batch(self, session_batch: pd.DataFrame) -> DataFrame[SessionBatchSchema]:
        """
        Clean and standardize the session DataFrame:
        - Enforce expected data types for each column.
        - Drop sessions with missing critical values.
        - Convert date columns to datetime.date.
        - Map weekday names to numeric codes (0=Monday, ..., 6=Sunday).
        """
        # Define expected data types for key columns
        expected_dtypes = {
            "PATIENT_ID": "Int64",
            "HOSPITAL_ID": "Int64",
            "PARETIC_SIDE": "string",
            "UPPER_EXTREMITY_TO_TRAIN": "string",
            "HAND_RAISING_CAPACITY": "string",
            "COGNITIVE_FUNCTION_LEVEL": "string",
            "HAS_HEMINEGLIGENCE": "Int64",
            "GENDER": "string",
            "SKIN_COLOR": "string",
            "AGE": "Int64",
            "VIDEOGAME_EXP": "Int64",
            "COMPUTER_EXP": "Int64",
            "COMMENTS": "string",
            "PTN_HEIGHT_CM": "Int64",
            "ARM_SIZE_CM": "Int64",
            "PRESCRIPTION_ID": "Int64",
            "SESSION_ID": "Int64",
            "PROTOCOL_ID": "Int64",
            "PRESCRIPTION_STARTING_DATE": "datetime64[ns]",
            "PRESCRIPTION_ENDING_DATE": "datetime64[ns]",
            "SESSION_DATE": "datetime64[ns]",
            "STARTING_HOUR": "Int64",
            "STARTING_TIME_CATEGORY": "string",
            "STATUS": "string",
            "PROTOCOL_TYPE": "string",
            "AR_MODE": "string",
            "WEEKDAY": "string",
            "REAL_SESSION_DURATION": "Int64",
            "PRESCRIBED_SESSION_DURATION": "Int64",
            "SESSION_DURATION": "Int64",
            "ADHERENCE": "float64",
            "TOTAL_SUCCESS": "Int64",
            "TOTAL_ERRORS": "Int64",
            "SCORE": "float64"
        }
        # Enforce data types (will raise if incompatible types encountered)
        sessions_df = session_batch.astype(expected_dtypes, errors="raise")
        # Drop any sessions without a Prescription ID or Session Duration (critical for analysis)
        sessions_df.dropna(subset=["PRESCRIPTION_ID", "SESSION_DURATION"], inplace=True)
        # Convert date columns to datetime (then extract date part)
        sessions_df["SESSION_DATE"] = pd.to_datetime(sessions_df["SESSION_DATE"], errors="coerce").dt.date
        sessions_df["PRESCRIPTION_STARTING_DATE"] = pd.to_datetime(sessions_df["PRESCRIPTION_STARTING_DATE"], errors="coerce").dt.date
        sessions_df["PRESCRIPTION_ENDING_DATE"] = pd.to_datetime(sessions_df["PRESCRIPTION_ENDING_DATE"], errors="coerce").dt.date
        # Map weekday names to integer (Mon=0,...Sun=6) for easier handling
        weekday_map = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
                    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}
        sessions_df["WEEKDAY"] = sessions_df["WEEKDAY"].map(weekday_map)
        # session_batch = SessionBatchSchema.validate(sessions_df)
        self.session_batch = sessions_df

        return session_batch

    def process_timeseries_data(self, timeseries_df):
        """
        Process raw time-series data:
        - Ensure numeric types for parameter and performance values.
        - Aggregate entries with the same timestamp (seconds from start) within each session.
        - Compute exponential moving averages (EWMA) of parameter and performance values to smooth fluctuations.
        """
        # Cast parameter and performance values to float
        timeseries_df["PARAMETER_VALUE"] = timeseries_df["PARAMETER_VALUE"].astype(float)
        timeseries_df["PERFORMANCE_VALUE"] = timeseries_df["PERFORMANCE_VALUE"].astype(float)
        # Aggregate by session and time point (unique timestamp within a session), averaging multiple entries if any
        aggregated = timeseries_df.groupby(
            ["PATIENT_ID", "SESSION_ID", "PROTOCOL_ID", "GAME_MODE", "SECONDS_FROM_START"]
        ).agg({
            "PARAMETER_KEY": lambda x: list(set(x)),       # unique parameters at this time
            "PARAMETER_VALUE": "mean",                     # average parameter value
            "PERFORMANCE_KEY": "first",                    # assume same performance key per time, take first
            "PERFORMANCE_VALUE": "mean"                    # average performance value (usually only one)
        }).reset_index()
        # Compute Exponential Weighted Moving Average (EWMA) for values within each session time-series
        # This smooths out the instantaneous fluctuations in DMs and PEs.
        aggregated.sort_values(by=["PATIENT_ID", "SESSION_ID", "SECONDS_FROM_START"], inplace=True)
        aggregated["PARAMETER_VALUE_EWMA"] = aggregated.groupby(["PATIENT_ID", "SESSION_ID"])["PARAMETER_VALUE"]\
                                                    .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
        aggregated["PERFORMANCE_VALUE_EWMA"] = aggregated.groupby(["PATIENT_ID", "SESSION_ID"])["PERFORMANCE_VALUE"]\
                                                    .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
        return aggregated

    def expand_session_batch(self, session_batch: pd.DataFrame):
        """
        Augment the session DataFrame with rows for expected sessions that were *not* performed.
        For each prescription (identified by PRESCRIPTION_ID), generate sessions on the scheduled weekday 
        that do not exist in the recorded sessions, marking them as NOT_PERFORMED with zero adherence.
        """
        session_cols = [
            "SESSION_ID", "STARTING_HOUR", "STARTING_TIME_CATEGORY", "REAL_SESSION_DURATION",
            "SESSION_DURATION", "TOTAL_SUCCESS", "TOTAL_ERRORS", "SCORE"
        ]
        missing_sessions = []
        # Group sessions by prescription and find expected dates not present
        for _, group in session_batch.groupby("PRESCRIPTION_ID"):
            # Use the first row of each group to get prescription schedule info
            first_row = group.iloc[0]
            start = first_row.PRESCRIPTION_STARTING_DATE
            end = first_row.PRESCRIPTION_ENDING_DATE
            weekday = first_row.WEEKDAY
            if pd.isna(start) or pd.isna(end) or pd.isna(weekday):
                continue  # skip if any critical info is missing
            expected_dates = self.generate_expected_sessions(start, end, int(weekday))
            performed_dates = set(group["SESSION_DATE"].dropna().unique())
            # Any expected date not in performed_dates is a missed session
            for miss_date in expected_dates:
                if miss_date not in performed_dates:
                    new_row = first_row.copy()
                    new_row["SESSION_DATE"] = miss_date
                    new_row["STATUS"] = "NOT_PERFORMED"
                    new_row["ADHERENCE"] = 0.0
                    # Set all session outcome-related columns to NaN (since session didn't occur)
                    for col in session_cols:
                        new_row[col] = np.nan
                    missing_sessions.append(new_row)
        if missing_sessions:
            df_missing = pd.DataFrame(missing_sessions)
            sessions_df = pd.concat([sessions_df, df_missing], ignore_index=True)
            # Sort by prescription and date for chronological order
            sessions_df.sort_values(by=["PRESCRIPTION_ID", "PROTOCOL_ID", "SESSION_DATE"], inplace=True)
        return sessions_df

    def map_latent_to_clinical(self, protocol_attributes, agg_func=np.mean):
        """We need to collapse the protocol feature space into the clinical feature space.
        """
        df_clinical = pd.DataFrame(index=protocol_attributes.index)

        # Collapse using agg_func the protocol latent attributes    
        for clinical_scale, features in self.mapping_dict.items():
            df_clinical[clinical_scale] = protocol_attributes[features].apply(agg_func, axis=1)

        df_clinical.index = protocol_attributes["PROTOCOL_ID"]
        self.protocol_data = df_clinical

        return df_clinical
    
    @staticmethod
    def generate_expected_sessions(start_date, end_date, target_weekday):
        """
        Generate all expected session dates between start_date and end_date for the given target weekday.
        If the prescription end date is in the future, use today as the end limit (assuming future sessions are not yet done).
        """
        expected_dates = []
        today = date.today()
        # If the prescription is still ongoing, cap the end_date at today
        if end_date is None:
            return expected_dates  # no valid end date
        if end_date > today:
            end_date = today
        # Find the first occurrence of the target weekday on or after start_date
        if start_date.weekday() != target_weekday:
            days_until_target = (target_weekday - start_date.weekday()) % 7
            start_date = start_date + timedelta(days=days_until_target)
        # Generate dates every 7 days (weekly) from the adjusted start_date up to end_date
        current_date = start_date
        while current_date <= end_date:
            expected_dates.append(current_date)
            current_date += timedelta(days=7)
        return expected_dates
