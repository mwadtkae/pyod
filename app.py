import os
import json
import joblib
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Body

app = FastAPI(title="PyOD Timesheet Scoring Service (RAW)", version="3.1")

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = BASE_DIR

# ---- PIN YOUR *NEW* MODEL FILE HERE ----
PINNED_MODEL_NAME = "hours_hbos_model.joblib"

PINNED_MODEL_PATH = os.path.join(MODEL_DIR, PINNED_MODEL_NAME)

MODEL = None


# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
def startup():
    global MODEL

    if not os.path.exists(PINNED_MODEL_PATH):
        raise RuntimeError(f"Pinned model not found: {PINNED_MODEL_PATH}")

    MODEL = joblib.load(PINNED_MODEL_PATH)


    print("Loaded RAW model:", PINNED_MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

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

def normalise_by_date(entries: List[Dict[str, Any]]):
    """
    Collate entries by date:
    - Sum hours per date
    - Return:
        X_date      -> ndarray for model
        date_keys   -> list of dates (aligned to X)
        row_dates   -> date per original row (for remapping)
    """
    totals_by_date: Dict[str, float] = {}
    row_dates: List[str] = []

    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise HTTPException(status_code=400, detail=f"Entry {i} must be an object")

        date = e.get("date") or e.get("Timesheet Date")
        if not date:
            raise HTTPException(status_code=400, detail=f"Entry {i} missing 'date'")

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
            h = float(h)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Entry {i} invalid hours: {h}")


        totals_by_date[date] = totals_by_date.get(date, 0.0) + h
        row_dates.append(date)

    date_keys = list(totals_by_date.keys())
    X_date = np.array([totals_by_date[d] for d in date_keys], dtype=float).reshape(-1, 1)

    return X_date, date_keys, row_dates

# -------------------------------------------------
# Scoring endpoint
# -------------------------------------------------
@app.post("/score_entries")
def score_entries(payload: Union[List[Dict[str, Any]], Dict[str, Any]] = Body(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    entries = extract_entries(payload)

    # --- DATE-LEVEL NORMALISATION ---
    X_date, date_keys, row_dates = normalise_by_date(entries)

    # --- PyOD scoring (date totals) ---
    date_scores = MODEL.decision_function(X_date)
    date_flags = MODEL.predict(X_date).astype(int)

    # --- Map date results back to rows ---
    score_by_date = dict(zip(date_keys, date_scores))
    flag_by_date = dict(zip(date_keys, date_flags))

    row_scores = [float(score_by_date[d]) for d in row_dates]
    row_flags = [int(flag_by_date[d]) for d in row_dates]

    return {
        "flagged": row_flags,
        "scores": row_scores
    }
