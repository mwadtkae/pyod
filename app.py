import os
import joblib
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Body

app = FastAPI(title="PyOD Timesheet Scoring Service (RAW)", version="3.2")

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = BASE_DIR

# ---- HARDCODE LATEST MODEL FILE HERE ----
PINNED_MODEL_NAME = "hbos_hours_PASS_20260209_160558.joblib"
PINNED_MODEL_PATH = os.path.join(MODEL_DIR, PINNED_MODEL_NAME)

MODEL = None

TARGET_HOURS = 8.0  # required hours per day


# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
def startup():
    global MODEL

    if not os.path.exists(PINNED_MODEL_PATH):
        raise RuntimeError(f"Pinned model not found: {PINNED_MODEL_PATH}")

    MODEL = joblib.load(PINNED_MODEL_PATH)
    print("Loaded model:", PINNED_MODEL_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": PINNED_MODEL_NAME,
        "flags": {
            0: "PASS (exactly 8 hours)",
            1: "Below 8 hours",
            2: "Above 8 hours"
        }
    }

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def extract_entries(payload: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("rows") or payload.get("data")
        if isinstance(entries, list):
            return entries
    raise HTTPException(status_code=400, detail="Expected a JSON array or {entries:[...]}")

def _get_hours_value(e: Dict[str, Any], i: int) -> float:
    # IMPORTANT: do NOT use `or` chains; 0 is valid but falsy.
    if "hours" in e:
        h = e["hours"]
    elif "Time in hours" in e:
        h = e["Time in hours"]
    elif "Time" in e:
        h = e["Time"]
    elif "time_in_hours" in e:
        h = e["time_in_hours"]
    else:
        raise HTTPException(status_code=400, detail=f"Entry {i} missing 'hours'")

    try:
        return float(h)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Entry {i} invalid hours: {h}")

def _get_date_value(e: Dict[str, Any], i: int) -> str:
    d = e.get("date") or e.get("Timesheet Date")
    if not d:
        raise HTTPException(status_code=400, detail=f"Entry {i} missing 'date'")
    return str(d)

def normalise_by_date(entries: List[Dict[str, Any]]):
    """
    Collate entries by date:
      - totals_by_date: sum(hours) per date
      - row_dates: date per original row (to map back)
      - row_hours: raw hours per original row (for transparency)
    Returns:
      X_date (n_dates, 1), date_keys, row_dates, row_hours, totals_by_date
    """
    totals_by_date: Dict[str, float] = {}
    row_dates: List[str] = []
    row_hours: List[float] = []

    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise HTTPException(status_code=400, detail=f"Entry {i} must be an object")

        d = _get_date_value(e, i)
        h = _get_hours_value(e, i)

        totals_by_date[d] = totals_by_date.get(d, 0.0) + h
        row_dates.append(d)
        row_hours.append(h)

    date_keys = list(totals_by_date.keys())
    X_date = np.array([totals_by_date[d] for d in date_keys], dtype=float).reshape(-1, 1)

    return X_date, date_keys, row_dates, row_hours, totals_by_date

def flag_to_reason(flag: int) -> str:
    if flag == 0:
        return "OK (exactly 8 hours)"
    if flag == 1:
        return "Below required hours (< 8)"
    if flag == 2:
        return "Above required hours (> 8)"
    return "Unknown"

def score_and_flag_date_totals(model, X_date: np.ndarray, date_keys: List[str]):
    """
    Uses PyOD scores + threshold_ to decide anomaly,
    then maps anomaly direction to:
      0 = OK
      1 = Below 8
      2 = Above 8
    """
    scores = model.decision_function(X_date)
    flags: List[int] = []

    for total_hours, score in zip(X_date.flatten(), scores):
        if score <= model.threshold_:
            flags.append(0)
        else:
            flags.append(1 if total_hours < TARGET_HOURS else 2)

    score_by_date = dict(zip(date_keys, scores))
    flag_by_date = dict(zip(date_keys, flags))
    return score_by_date, flag_by_date


# -------------------------------------------------
# Scoring endpoint
# -------------------------------------------------
@app.post("/score_entries")
def score_entries(payload: Union[List[Dict[str, Any]], Dict[str, Any]] = Body(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    entries = extract_entries(payload)

    X_date, date_keys, row_dates, row_hours, totals_by_date = normalise_by_date(entries)
    score_by_date, flag_by_date = score_and_flag_date_totals(MODEL, X_date, date_keys)

    flags: List[int] = []
    reasons: List[str] = []

    for d in row_dates:
        flag = int(flag_by_date[d])
        flags.append(flag)

        if flag == 0:
            reasons.append("PASS")
        elif flag == 1:
            reasons.append("Below 8 hours")
        elif flag == 2:
            reasons.append("Above 8 hours")
        else:
            reasons.append("Unknown")

    return {
        "flags": flags,
        "reasons": reasons
    }
