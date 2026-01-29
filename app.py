from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from pyod.models.iforest import IForest

app = FastAPI()

model = None
feature_count = None

@app.get("/")
def health():
    return {"status": "PyOD anomaly service ready"}

# -------- TRAIN (CSV or Excel) --------

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    global model, feature_count

    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file.file)
    else:
        return {"error": "Upload CSV or Excel file only"}

    # keep numeric columns only
    df = df.select_dtypes(include=[np.number])

    if df.empty:
        return {"error": "No numeric columns found"}

    X = df.values
    feature_count = X.shape[1]

    model = IForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(X)

    return {
        "status": "trained",
        "rows": X.shape[0],
        "features": feature_count
    }

# -------- DETECT ANOMALIES --------

@app.post("/detect")
def detect(data: list):
    if model is None:
        return {"error": "Train model first"}

    X = np.array(data)

    if X.shape[1] != feature_count:
        return {
            "error": f"Expected {feature_count} features, got {X.shape[1]}"
        }

    preds = model.predict(X)
    scores = model.decision_function(X)

    return {
        "anomaly": preds.tolist(),
        "score": scores.tolist()
    }
