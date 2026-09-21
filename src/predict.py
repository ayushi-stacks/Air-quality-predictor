import joblib
import pandas as pd

MODEL_PATH = "models/xgboost_pm25_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_pm25(values):
    columns = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "rolling_3",
        "rolling_7",
        "rolling_14",
        "day_of_week",
        "day_of_year",
        "week_of_year"
    ]

    data = pd.DataFrame([values], columns=columns)

    return model.predict(data)[0]