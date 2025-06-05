# %%
import pandas as pd
from typing import List
import pandera as pa

# Assuming `ScoringSchema` is already defined as in your code
from ai_cdss.models import ScoringSchema
from rgs_interface.data.schemas import RecsysMetricsRow, PrescriptionStagingRow

import uuid

# Create a DataFrame with valid (alias) columns
fake_data = pd.DataFrame([
    {
        "PATIENT_ID": 2,
        "PROTOCOL_ID": 201,
        "ADHERENCE_RECENT": 0.87,
        "DELTA_DM": -0.05,
        "PPF": 0.92,
        "CONTRIB": [0.2, 0.3, 0.1, 0.25, 0.05],
        "SCORE": 1.23,
        "USAGE": 5,
        "DAYS": [0, 2]
    },
    {
        "PATIENT_ID": 3,
        "PROTOCOL_ID": 201,
        "ADHERENCE_RECENT": 0.95,
        "DELTA_DM": -0.02,
        "PPF": 0.89,
        "CONTRIB": [0.1, 0.2, 0.3, 0.15, 0.05],
        "SCORE": 1.45,
        "USAGE": 10,
        "DAYS": [1, 3, 5]
    }
])
from rgs_interface.data.schemas import PrescriptionStagingRow, WeekdayEnum, PrescriptionStatusEnum, RecsysMetricKeyEnum

# Instantiate and validate against schema
validated_df = ScoringSchema.validate(fake_data)
prescription_df = (
    validated_df
    .explode("DAYS")
    .rename(columns={"DAYS": "WEEKDAY"})
)
metrics_df = pd.melt(
    validated_df,
    id_vars=["PATIENT_ID", "PROTOCOL_ID"],
    value_vars=["DELTA_DM", "ADHERENCE_RECENT", "PPF"],
    var_name="KEY",
    value_name="VALUE"
)

# %%
from datetime import timedelta, date

def row_to_prescription(
    row: pd.Series,
    recommendation_id: uuid.UUID,
    start: date = date.today(),
    duration: int = 30,
    status: PrescriptionStatusEnum = PrescriptionStatusEnum.PENDING
) -> PrescriptionStagingRow:
    
    return PrescriptionStagingRow(
        patient_id=row["PATIENT_ID"],
        protocol_id=row["PROTOCOL_ID"],
        starting_date=start,
        ending_date=start + timedelta(days=7),
        weekday=list(WeekdayEnum)[row['WEEKDAY']],
        session_duration=duration,
        recommendation_id=recommendation_id, # ?? How ?
        weeks_since_start=1, # ?? How ?
        status=status,
    )

def row_to_metrics(
    row: pd.Series,
    recommendation_id: uuid.UUID,
    metric_date: date = date.today()
) -> RecsysMetricsRow:
    
    return RecsysMetricsRow(
        patient_id=row["PATIENT_ID"],
        protocol_id=row["PROTOCOL_ID"],
        recommendation_id=recommendation_id,
        metric_date=metric_date,
        metric_key=RecsysMetricKeyEnum[row["KEY"]],
        metric_value=row['VALUE']
    )
# %%
from rgs_interface.data.interface import DatabaseInterface

interface = DatabaseInterface()

unique_id = uuid.uuid4()
print(f"Unique ID for this recommendation: {type(unique_id)}")

for idx, row in prescription_df.iterrows():
    prescription = PrescriptionStagingRow.from_row(
        row=row,
        recommendation_id=unique_id,
    )
    interface.add_prescription_staging_entry(prescription)
    print(prescription.to_params_dict(), id)

for idx, row in metrics_df.iterrows():
    metrics = RecsysMetricsRow.from_row(
        row=row,
        recommendation_id=unique_id
    )
    interface.add_recsys_metric_entry(metrics)
    print(metrics.to_params_dict())
    # break
# %%
