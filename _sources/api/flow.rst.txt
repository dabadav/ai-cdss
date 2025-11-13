
Loader

- Session data (Patient Data + Session Data) [Adherence]
- Timeseries data (Session Data by Time) [DM, PE]

- Patient embeddings (Clinical subscales)
- Protocol embeddings (Protocol Attributes)
- Patient-Protocol Similarity (Clinical x Protocol)

Processing

- Session processing () [EWMA_ADHERENCE]
- Timeseries processing () [DELTA_DM, PE]
- Patient-Protocol calculation () [PPF]

Returns
- Patient Protocol Scores () [PATIENT, ADHERENCE, DELTA_DM, PPF, CONTRIB, ]


Business Logic

- 

Public Interface (FastAPI)