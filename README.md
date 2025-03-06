
# Clinical Decision Support System

### Input

- SessionBatch: pd.DataFrame = (sessions, session_info)

### Output:

- WeeklyPrescriptionBatch

    - prescriptions: Dict[str, Dict[str, Any]] = (patient, protocol, prescription)

        prescription

        - day: List
        - score: int
        - factors: List
        - contributions: List
        - explanation: str

