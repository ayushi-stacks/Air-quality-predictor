import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Delhi PM2.5 Predictor",
    page_icon="🌫️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_pm25_model.pkl"
DATA_PATH = BASE_DIR / "data" / "delhi_pm25_daily.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    # Convert OpenAQ timezone-aware timestamps
    # into timezone-naive Indian local timestamps
    data["date"] = pd.to_datetime(
        data["date"],
        utc=True
    )

    data["date"] = (
        data["date"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    data = data.sort_values("date")
    data = data.reset_index(drop=True)

    return data


model = load_model()
df = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("Delhi PM2.5 Predictor")

st.caption(
    "Predict tomorrow's PM2.5 concentration using "
    "historical air-quality patterns and an XGBoost model."
)

st.divider()


# ============================================================
# DATA INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Available Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Start Date",
        df["date"].min().strftime("%d %b %Y")
    )

with col3:
    st.metric(
        "End Date",
        df["date"].max().strftime("%d %b %Y")
    )


st.divider()


# ============================================================
# DATE SELECTION
# ============================================================

st.subheader("Prediction Settings")

available_dates = df["date"].dt.date.tolist()

prediction_date = st.date_input(
    "Select the latest available date",
    value=available_dates[-1],
    min_value=available_dates[13],
    max_value=available_dates[-1]
)


# ============================================================
# HISTORICAL DATA
# ============================================================

prediction_timestamp = pd.Timestamp(
    prediction_date
)

historical = df[
    df["date"] <= prediction_timestamp
].copy()


# ============================================================
# VALIDATION
# ============================================================

if len(historical) < 14:

    st.error(
        "At least 14 days of historical data are required "
        "to generate the prediction."
    )

    st.stop()


# ============================================================
# GET RECENT PM2.5 VALUES
# ============================================================

lag_1 = historical["pm25"].iloc[-1]

lag_2 = historical["pm25"].iloc[-2]

lag_3 = historical["pm25"].iloc[-3]

lag_7 = historical["pm25"].iloc[-7]


# ============================================================
# ROLLING FEATURES
# ============================================================

rolling_3 = (
    historical["pm25"]
    .iloc[-3:]
    .mean()
)

rolling_7 = (
    historical["pm25"]
    .iloc[-7:]
    .mean()
)

rolling_14 = (
    historical["pm25"]
    .iloc[-14:]
    .mean()
)


# ============================================================
# NEXT DAY FEATURES
# ============================================================

next_day = (
    prediction_timestamp
    + pd.Timedelta(days=1)
)

day_of_week = next_day.dayofweek

day_of_year = next_day.dayofyear

week_of_year = next_day.isocalendar().week


# ============================================================
# CREATE MODEL INPUT
# ============================================================

features = pd.DataFrame(
    [[
        lag_1,
        lag_2,
        lag_3,
        lag_7,
        rolling_3,
        rolling_7,
        rolling_14,
        day_of_week,
        day_of_year,
        int(week_of_year)
    ]],
    columns=[
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
)


# ============================================================
# SHOW CURRENT DATA
# ============================================================

st.subheader("Current Air Quality")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Today's PM2.5",
        f"{lag_1:.2f} µg/m³"
    )

with col2:
    st.metric(
        "3-Day Average",
        f"{rolling_3:.2f} µg/m³"
    )

with col3:
    st.metric(
        "7-Day Average",
        f"{rolling_7:.2f} µg/m³"
    )

with col4:
    st.metric(
        "14-Day Average",
        f"{rolling_14:.2f} µg/m³"
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Tomorrow's PM2.5",
    type="primary",
    use_container_width=True
):

    prediction = model.predict(features)[0]

    # Prevent impossible negative predictions
    prediction = max(0, prediction)


    # --------------------------------------------------------
    # CHANGE
    # --------------------------------------------------------

    change = prediction - lag_1

    if lag_1 != 0:
        percentage_change = (
            change / lag_1
        ) * 100
    else:
        percentage_change = 0


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if prediction < 30:
        category = "Lower concentration"

    elif prediction < 60:
        category = "Moderate concentration"

    elif prediction < 90:
        category = "High concentration"

    else:
        category = "Very high concentration"


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        f"Prediction for {next_day.strftime('%d %B %Y')}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted PM2.5",
            f"{prediction:.2f} µg/m³"
        )

    with col2:

        st.metric(
            "Expected Change",
            f"{change:+.2f} µg/m³",
            delta=f"{percentage_change:+.1f}%"
        )

    with col3:

        st.metric(
            "Prediction Category",
            category
        )


    # ========================================================
    # RECENT TREND
    # ========================================================

    st.subheader(
        "Recent PM2.5 Trend"
    )

    chart_data = (
        historical
        .tail(30)
        .set_index("date")
    )

    st.line_chart(
        chart_data["pm25"],
        height=350
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.divider()

    st.subheader("Model Information")

    st.write(
        """
        The prediction is generated using an XGBoost regression
        model trained on historical New Delhi PM2.5 measurements.
        """
    )

    st.write(
        "**Features used:**"
    )

    st.write(
        """
        - Previous-day PM2.5
        - PM2.5 from previous days
        - 3-day rolling average
        - 7-day rolling average
        - 14-day rolling average
        - Day of week
        - Day of year
        - Week of year
        """
    )