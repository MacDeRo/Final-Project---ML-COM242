"""Input validation and prediction helpers shared by the Streamlit app and tests."""
from __future__ import annotations

import json

import joblib
import pandas as pd

import config as C

REQUIRED = ["aadt", "prev_crashes", "lane_km", "year"]
HARD_LIMITS = {  # values outside these are rejected as impossible
    "aadt": (1, 2_000_000),
    "prev_crashes": (0, 100_000),
    "lane_km": (0.1, 20_000),
    "year": (2000, 2100),
}


def load_artifacts():
    model = joblib.load(C.MODEL_PATH)
    meta = json.loads(C.METADATA_PATH.read_text(encoding="utf-8"))
    return model, meta


def validate_frame(df: pd.DataFrame, ranges: dict | None = None):
    """Return (clean_df, problems). problems maps row index -> list of messages.

    Rows listed with an 'ERROR' message are not predicted; 'WARNING' rows are
    predicted but flagged because they lie outside the training range.
    """
    problems: dict[int, list[str]] = {}
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}. "
                         f"Expected: {', '.join(REQUIRED)}.")
    work = df.copy()
    for c in REQUIRED:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if "pandemic" not in work.columns:
        work["pandemic"] = work["year"].isin(C.PANDEMIC_YEARS).astype(float)
    work["pandemic"] = pd.to_numeric(work["pandemic"], errors="coerce")

    for i, row in work.iterrows():
        msgs = []
        for c in REQUIRED:
            v = row[c]
            if pd.isna(v):
                msgs.append(f"ERROR: '{c}' is missing or not a number")
                continue
            lo, hi = HARD_LIMITS[c]
            if not lo <= v <= hi:
                msgs.append(f"ERROR: '{c}' = {v:g} is outside the allowed range {lo:,g} to {hi:,g}")
            elif c != "year" and ranges and not ranges[c][0] <= v <= ranges[c][1]:
                msgs.append(f"WARNING: '{c}' = {v:g} is outside the training range "
                            f"{ranges[c][0]:g} to {ranges[c][1]:g}; the estimate is less reliable")
        if not pd.isna(row["year"]) and row["year"] != int(row["year"]):
            msgs.append("ERROR: 'year' must be a whole number")
        if row["pandemic"] not in (0, 1):
            msgs.append("ERROR: 'pandemic' must be 0 or 1")
        if msgs:
            problems[i] = msgs
    return work, problems


def predict_frame(model, work: pd.DataFrame, problems: dict) -> pd.DataFrame:
    ok = [i for i in work.index if not any(m.startswith("ERROR") for m in problems.get(i, []))]
    result = work.copy()
    result["predicted_crashes"] = float("nan")
    if ok:
        result.loc[ok, "predicted_crashes"] = model.predict(work.loc[ok, C.FEATURES])
    result["notes"] = [" | ".join(problems.get(i, [])) for i in work.index]
    return result
