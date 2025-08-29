# app.py
import streamlit as st
import pandas as pd
import json, pickle, os

BASE_DIR = os.path.dirname(__file__)

# Load model + artifacts
with open(os.path.join(BASE_DIR, "log_reg_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "feature_order.json"), "r") as f:
    FEATURE_ORDER = json.load(f)

try:
    with open(os.path.join(BASE_DIR, "best_threshold.json"), "r") as f:
        BEST_THR = float(json.load(f)["threshold"])
except Exception:
    BEST_THR = 0.5

st.set_page_config(page_title="HTN Risk (Logistic Regression)", layout="centered")
st.title("🩺 Hypertension Risk Prediction (Logistic Regression)")

# Sidebar inputs (survey-style)
st.sidebar.header("Inputs")

age = st.sidebar.slider("Age", 18, 100, 40)
smoking = 1 if st.sidebar.radio("Do you currently smoke?", ["No", "Yes"]) == "Yes" else 0

alcohol_freq = st.sidebar.selectbox(
    "Alcohol frequency",
    [
        "<1 per month (6)",
        "1-3 days per month (5)",
        "1-2 days per week (4)",
        "3-4 days per week (3)",
        "5-6 days per week (2)",
        "Daily (1)"
    ]
)
alcohol_code = int(alcohol_freq.split("(")[-1].strip(")"))
heavy_alcohol = 1 if alcohol_code in [1, 2] else 0

sex = st.sidebar.radio("Sex", ["Male", "Female"])
waist = st.sidebar.number_input("Waist (cm)", 40.0, 200.0, 85.0)
hip   = st.sidebar.number_input("Hip (cm)",   40.0, 200.0, 95.0)
whr   = waist / hip if hip > 0 else 0
whr_risk = 1 if ((sex == "Male" and whr > 0.90) or (sex == "Female" and whr > 0.85)) else 0

vig_work = st.sidebar.number_input("Vigorous work (min/week)", 0, 2000, 0)
mod_work = st.sidebar.number_input("Moderate work (min/week)", 0, 2000, 0)
vig_rec  = st.sidebar.number_input("Vigorous recreation (min/week)", 0, 2000, 0)
mod_rec  = st.sidebar.number_input("Moderate recreation (min/week)", 0, 2000, 0)
walk_cycle = st.sidebar.number_input("Walking/cycling (min/week)", 0, 2000, 0)
MET = vig_work*8 + mod_work*4 + vig_rec*8 + mod_rec*4 + walk_cycle*3.5

fbs = st.sidebar.number_input("Fasting Blood Sugar (mg/dl)", 50, 300, 90)
tc  = st.sidebar.number_input("Total Cholesterol (mg/dl)", 100, 400, 180)

# Build feature row in correct order
row = {
    "age": age,
    "smoking": smoking,
    "WHR_risk": whr_risk,
    "HEAVY_ALCOHOL_USE": heavy_alcohol,
    "PHYSICAL_ACTIVITY": MET,
    "FASTING_BLOOD_SUGAR": fbs,
    "TOTAL_CHOLESTEROL": tc
}
X_new = pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

st.subheader("Prediction")
prob = model.predict_proba(X_new)[0, 1]
pred = "⚠️ High HTN Risk" if prob >= BEST_THR else "✅ Normal Risk"
st.write(f"**{pred}** — probability: `{prob:.2f}`  (threshold: `{BEST_THR:.2f}`)")

st.markdown("### Input Summary")
st.dataframe(pd.DataFrame(row, index=["value"]).T)
st.caption(f"Computed WHR = {whr:.2f} → {'High' if whr_risk else 'Normal'} risk per WHO cutoffs.")
