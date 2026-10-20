"""Streamlit demo.  Run:  streamlit run app.py   (after training a model)"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from depression_ml.data import FEATURES

MODEL_PATH = Path("models/model.joblib")

st.set_page_config(page_title="Depression risk — ML demo", page_icon="🧠")
st.title("Depression risk — ML demo")
st.warning(
    "**This is a student machine-learning project, not a medical tool.** "
    "It cannot diagnose anything. If you are struggling, talk to a doctor or someone you trust.  \n"
    "🇫🇷 **3114** — numéro national de prévention du suicide (24/7, free).  \n"
    "🚨 Emergency: **15** or **112**."
)

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m depression_ml.train <csv>` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("survey"):
    age = st.slider("Age", 16, 40, 21)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    academic_pressure = st.slider("Academic pressure (1 = low, 5 = high)", 1, 5, 3)
    study_satisfaction = st.slider("Study satisfaction (1 = low, 5 = high)", 1, 5, 3)
    sleep = st.select_slider("Sleep per night", ["<5h", "5-6h", "6-7h", "7-8h", ">8h"], value="7-8h")
    study_hours = st.slider("Study hours per day", 0, 12, 6)
    financial_stress = st.slider("Financial stress (1 = low, 5 = high)", 1, 5, 3)
    diet = st.radio("Dietary habits", ["healthy", "moderate", "unhealthy"], horizontal=True)
    suicidal = st.radio("Have you ever had suicidal thoughts?", ["No", "Yes"], horizontal=True)
    family = st.radio("Family history of mental illness?", ["No", "Yes"], horizontal=True)
    submitted = st.form_submit_button("Estimate")

if submitted:
    sleep_hours = {"<5h": 4.5, "5-6h": 5.5, "6-7h": 6.5, "7-8h": 7.5, ">8h": 8.5}[sleep]
    row = pd.DataFrame([{
        "age": age,
        "academic_pressure": academic_pressure,
        "study_satisfaction": study_satisfaction,
        "sleep_hours": sleep_hours,
        "study_hours": study_hours,
        "financial_stress": financial_stress,
        "is_female": int(gender == "Female"),
        "suicidal_thoughts": int(suicidal == "Yes"),
        "family_history": int(family == "Yes"),
        "dietary_habits": diet,
    }])[FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    st.metric("Model score", f"{proba:.0%}")
    st.caption(
        "Probability the model assigns to the 'depression' label, based on patterns in one survey "
        "dataset. It is not a clinical assessment."
    )
    if suicidal == "Yes":
        st.info("You mentioned suicidal thoughts. You don't have to deal with this alone — call **3114** (free, 24/7).")
