from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn
from datetime import datetime

app = FastAPI()


class DayRequest(BaseModel):
    day_name: str
    hours_range: list = list(range(9, 18))  # Default 9AM-5PM


@app.get("/")
def read_root():
    return {"message": "Welcome to the Time Slot Recommendation API!"}


try:
    model = joblib.load("time_slot_model.pkl")
    feature_names = model.feature_names_in_
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")


@app.post("/recommend")
def recommend_time_slots(request: DayRequest):
    print("Received request for day:", request.day_name)

    try:
        default_hours = list(range(9, 18))

        day_data = pd.DataFrame({'hour_of_day': default_hours})
        current_month = datetime.now().month
        day_data['month'] = current_month
        day_data['is_weekend'] = 1 if request.day_name in ['Saturday', 'Sunday'] else 0

        for day_col in [col for col in feature_names if col.startswith('day_of_week_')]:
            day_data[day_col] = 1 if day_col == f'day_of_week_{request.day_name}' else 0

        day_data['predicted_wait'] = model.predict(day_data[feature_names])

        recommendations = day_data.sort_values('predicted_wait').head(3)

        results = []
        for _, row in recommendations.iterrows():
            results.append({
                "hour": row['hour_of_day'],
                "time": f"{row['hour_of_day']}:00",
                "predicted_wait_minutes": round(row['predicted_wait'], 1)
            })

        return {
            "best_slot": results[0],
            "alternative_slots": results[1:]
        }

    except Exception as e:
        print(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)