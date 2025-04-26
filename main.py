from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class WaitTimeInput(BaseModel):
    current_queue_length: int
    staff_count: int
    historical_throughput: float
    is_holiday: bool
    weather_condition: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Wait Time Prediction API!"}

model_bundle = joblib.load("wait_time_predictor.pkl")
models = model_bundle['models']
preprocessor = model_bundle['preprocessor']

@app.post("/predict")
def predict_wait_time(input_data: WaitTimeInput):
    print("Received input data:", input_data.dict())

    try:
        input_df = pd.DataFrame([input_data.dict()])
        processed = preprocessor.transform(input_df)

        prediction = {
            "predicted_wait_time_minutes": round(models['q0.5'].predict(processed)[0], 1),
            "confidence_interval": {
                "lower_bound": round(models['q0.1'].predict(processed)[0], 1),
                "upper_bound": round(models['q0.9'].predict(processed)[0], 1)
            }
        }
        print("Prediction result:", prediction)
        return prediction

    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)