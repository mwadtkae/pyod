from fastapi import FastAPI
import numpy as np
from pyod.models.iforest import IForest

app = FastAPI()

# Train model on startup (live)
X_train = np.random.rand(1000, 5)

model = IForest(
    n_estimators=100,
    contamination=0.05
)
model.fit(X_train)

@app.get("/")
def health():
    return {"status": "Live PyOD model running"}

@app.post("/detect")
def detect(data: list):
    X = np.array(data)
    preds = model.predict(X)   # 1 = anomaly
    scores = model.decision_function(X)

    return {
        "anomaly": preds.tolist(),
        "score": scores.tolist()
    }
