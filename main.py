from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from pyod.models.iforest import IForest

app = FastAPI()

model = None   # global live model

@app.get("/")
def health():
    return {"status": "Upload Excel to train PyOD model"}

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    global model

    df = pd.read_excel(file.file)

    # Keep only numeric columns
    X = df.select_dtypes(include=[np.number]).values

    model = IForest(contamination=0.05)
    model.fit(X)

    return {
        "message": "Model trained successfully",
        "rows_used": len(X),
        "features": X.shape[1]
    }

@app.post("/detect")
def detect(data: list):
    if model is None:
        return {"error": "Train model first using Excel"}

    X = np.array(data)
    preds = model.predict(X)
    scores = model.decision_function(X)

    return {
        "anomaly": preds.tolist(),
        "score": scores.tolist()
    }
