
###################
### DEBUG UTILS ###
###################

# This debug class handles saving intermeditate steps of recommendation pipeline
# in the debug mode. It saves dfs to disk at the default debug location.

from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import pandas as pd

from ai_cdss.constants import DEFAULT_DEBUG_DIR

class DebugReport:
    """
    Debug class helper that:
    - manages a base debug dir
    - creates <base>/<run_id>[/<subdir>] on demand
    - dumps DataFrames to disk
    - returns a small JSON-friendly preview
    """

    def __init__(self, debug_path: str = DEFAULT_DEBUG_DIR):
        self.base_dir = Path(debug_path)
    
    # ---------- dirs ----------
    def ensure_dir(self, run_id: uuid.UUID, subdir: Optional[str] = None) -> Path:
        root = self.base_dir / str(run_id)
        if subdir:
            root = root / subdir
        root.mkdir(parents=True, exist_ok=True)
        return root

    # ---------- io ----------
    def dump_df(self, df: pd.DataFrame, run_id: uuid.UUID, filename: str,
                *, subdir: Optional[str] = None, format: str = "csv") -> str:
        out_dir = self.ensure_dir(run_id, subdir=subdir)
        out = out_dir / filename
        if format == "csv":
            df.to_csv(out, index=False)
        elif format == "parquet":
            out = out.with_suffix(".parquet")
            df.to_parquet(out, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return out.as_posix()

    def preview_df(self, df: pd.DataFrame, *, rows: int = 1,
                   round_ndigits: int = 6) -> Dict[str, Any]:
        head = df.head(rows).copy()
        num_cols = head.select_dtypes(include=["number"]).columns
        if len(num_cols):
            head[num_cols] = head[num_cols].round(round_ndigits)
        return {
            "rows": int(len(df)),
            "cols": list(df.columns),
            "sample": head.to_dict(orient="records"),
        }

    # ---------- bundles ----------
    def make_artifacts(self,
        *,
        run_id: uuid.UUID,
        scores: pd.DataFrame,
        recs: pd.DataFrame,
        presc: pd.DataFrame,
        metrics: pd.DataFrame,
        subdir: Optional[str] = None,
        format: str = "csv",
        preview: bool = False,
    ) -> Dict[str, Any]:
        files = {
            "scores":          self.dump_df(scores,  run_id, "scores.csv",          subdir=subdir, format=format),
            "recommendations": self.dump_df(recs,    run_id, "recommendations.csv", subdir=subdir, format=format),
            "prescriptions":   self.dump_df(presc,   run_id, "prescriptions.csv",   subdir=subdir, format=format),
            "metrics":         self.dump_df(metrics, run_id, "metrics.csv",         subdir=subdir, format=format),
        }

        if not preview:
            return {"files": files}
        
        else:
            previews = {
                "scores":          self.preview_df(scores),
                "recommendations": self.preview_df(recs),
                "prescriptions":   self.preview_df(presc),
                "metrics":         self.preview_df(metrics),
            }
            return {"files": files, "previews": previews}
