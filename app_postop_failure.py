import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Postop Respiratory Failure (Minimal)", layout="centered")

st.title("Postoperative Respiratory Failure — Minimal App")

# ---- Load model path from sidebar ----
model_path = st.sidebar.text_input("Model file (.pkl)", value="random_forest_model.pkl")
threshold = st.sidebar.slider("Positive threshold", 0.05, 0.95, 0.50, 0.01)

@st.cache_resource(show_spinner=False)
def load_model(p):
    return joblib.load(p)

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---- EXACT 15 features (must match training order) ----
FEATURES = [
    "post_APACHEII_score",
    "post_SOFA_score",
    "post_albumin_g_l",
    "post_CRP_mg_l",
    "post_WBC_10e9_l",
    "post_HR_bpm",
    "post_glucose_mmol_l",
    "post_lactate_mmol_l",
    "post_PCT_ng_ml",
    "post_BUN_mmol_l",
    "post_SBP_mmhg",
    "post_PLT_10e9_l",
    "post_INR",
    "post_APTT_sec",
    "post_creatinine_umol_l",
]

st.subheader("Inputs")
vals = {}
for f in FEATURES:
    # simple numeric inputs with reasonable defaults
    vals[f] = st.number_input(f, value=0.0, format="%.4f")

if st.button("Predict"):
    X = pd.DataFrame([vals], columns=FEATURES)
    # predict_proba -> fallback to decision_function -> predict
    try:
        prob = float(model.predict_proba(X)[:, 1][0])
    except Exception:
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(X)[0])
            prob = 1 / (1 + np.exp(-score))
        else:
            prob = float(model.predict(X)[0])

    pred = int(prob >= threshold)
    st.success(f"Probability (positive class): {prob:.3f}")
    st.info(f"Predicted label: {'Positive' if pred==1 else 'Negative'}")
