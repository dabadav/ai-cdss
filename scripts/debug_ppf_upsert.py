"""
Manual test of the PPF Parquet upsert logic.
=============================================

`precompute.persist_ppf` upserts by (PATIENT_ID, PROTOCOL_ID): the first
call creates the file, later calls replace rows for matching keys and
append the rest. This script exercises that real function (it used to
re-implement the upsert inline — now it tests the thing that ships).

Run:  python scripts/debug_ppf_upsert.py
"""
import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from ai_cdss.constants import CONTRIB, PATIENT_ID, PPF, PROTOCOL_ID
from ai_cdss.precompute import persist_ppf

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def row(patient_id: int, protocol_id: int, ppf: float, contrib: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        PATIENT_ID: [patient_id], PROTOCOL_ID: [protocol_id],
        PPF: [ppf], CONTRIB: [contrib],
    })


def read_back(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values([PATIENT_ID, PROTOCOL_ID]).reset_index(drop=True)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ppf_upsert_"))
    path = tmp / "ppf.parquet"
    try:
        # 1) Initial write — file created.
        persist_ppf(row(1, 200, 0.80, [0.1, 0.2]), path=path)
        df = read_back(path)
        assert len(df) == 1 and float(df.iloc[0][PPF]) == 0.80
        print("Test 1 (initial write) passed:\n", df.to_string(index=False))

        # 2) Pure insert — new keys appended, existing untouched.
        persist_ppf(row(1, 201, 0.70, [0.3, 0.4]), path=path)
        persist_ppf(row(2, 200, 0.90, [0.5, 0.6]), path=path)
        df = read_back(path)
        assert len(df) == 3
        print("Test 2 (pure insert) passed — 3 rows.")

        # 3) Update existing key — value replaced, row count unchanged.
        persist_ppf(row(1, 200, 0.55, [0.9, 0.9]), path=path)
        df = read_back(path)
        updated = df[(df[PATIENT_ID] == 1) & (df[PROTOCOL_ID] == 200)].iloc[0]
        assert len(df) == 3, f"expected 3 rows after update, got {len(df)}"
        assert float(updated[PPF]) == 0.55, "PPF for (1,200) was not upserted"
        print("Test 3 (update existing) passed — (1,200) PPF now 0.55, still 3 rows.")

        print("\nAll upsert tests passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
