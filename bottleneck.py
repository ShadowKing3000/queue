from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('bottleneck_model.pkl')


class QueueInput(BaseModel):
    queue_length: int
    arrival_rate: float
    departure_rate: float
    time_of_day: int


@app.post("/predict")
def predict_bottleneck(input: QueueInput):
    # Convert input to numpy array (no pandas)
    X = np.array([
        [input.queue_length, input.arrival_rate,
         input.departure_rate, input.time_of_day]
    ])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "bottleneck": bool(prediction),
        "confidence": float(probability)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)