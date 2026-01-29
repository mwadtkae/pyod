import os
import json
import joblib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body

from test import format_for_output

app = FastAPI(title="PyOD Timesheet Scoring Service (RAW)", version="3.0")

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR)

# ---- PIN YOUR MODEL FILE HERE ----
PINNED_MODEL_NAME = "hbos_raw_20260129_173937.joblib"
PINNED_META_NAME  = "hbos_raw_20260129_173937.meta.json"

PINNED_MODEL_PATH = os.path.join(MODEL_DIR, PINNED_MODEL_NAME)
PINNED_META_PATH  = os.path.join(MODEL_DIR, PINNED_META_NAME)

MODEL = None
META: Dict[str, Any] = {}

@app.on_event("startup")
def startup():
    global MODEL, META

    if not os.path.exists(PINNED_MODEL_PATH):
        raise RuntimeError(f"Pinned model not found: {PINNED_MODEL_PATH}")

    MODEL = joblib.load(PINNED_MODEL_PATH)

    META = {}
    if os.path.exists(PINNED_META_PATH):
        with open(PINNED_META_PATH, "r", encoding="utf-8") as f:
            META = json.load(f)

    print("Loaded RAW model:", PINNED_MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

def extract_entries(payload: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Accept either raw list OR {entries:[...]}
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("rows") or payload.get("data")
        if isinstance(entries, list):
            return entries
    raise HTTPException(status_code=400, detail="Expected a JSON array or {entries:[...]}")

def normalise(entries: List[Dict[str, Any]]) -> np.ndarray:
    """
    RAW scoring: each entry -> one feature vector [hours]
    Date is accepted but not used (you asked RAW entries).
    """
    hours = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise HTTPException(status_code=400, detail=f"Entry {i} must be an object")

        # Accept aliases
        h = (
            e.get("hours")
            or e.get("Time in hours")
            or e.get("Time")
            or e.get("time_in_hours")
        )
        if h is None:
            raise HTTPException(status_code=400, detail=f"Entry {i} missing 'hours'")

        try:
            hours.append(float(h))
        except Exception:
            raise HTTPException(status_code=400, detail=f"Entry {i} invalid hours: {h}")

    X = np.array(hours, dtype=float).reshape(-1, 1)
    return X

@app.post("/score_entries")
def score_entries(payload: Union[List[Dict[str, Any]], Dict[str, Any]] = Body(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    entries = extract_entries(payload)
    X = normalise(entries)

    # PyOD outputs
    scores = MODEL.decision_function(X)   # anomaly scores (float)
    flags = MODEL.predict(X).astype(int)  # 1 = outlier, 0 = normal

    return {
        "flagged": flags.tolist(),
        "scores": scores.tolist()
    }
