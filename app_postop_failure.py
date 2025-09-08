import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Postop Respiratory Failure", layout="wide")
st.title("Postoperative Respiratory Failure — Predict")

# 固定参数
CSV_PATH   = "train_top.csv"
TARGET_COL = "group"
LABEL_MAP  = {"normal": 0, "rfpe": 1}
THRESHOLD  = 0.5   # 固定阈值

# ---------------- Load & Train ----------------
@st.cache_resource(show_spinner=True)
def load_and_train():
    df = pd.read_csv(CSV_PATH, index_col=0)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map(LABEL_MAP).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe.fit(X, y)

    defaults = X.median(numeric_only=True).to_dict()
    return pipe, list(X.columns), defaults

try:
    model, feature_names, default_vals = load_and_train()
    st.success("Model trained on train_top.csv ✔️")
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

# ---------------- Inputs & Predict ----------------
st.subheader("Input patient variables")

vals = {}
cols = st.columns(5)  # 5 列

for i, f in enumerate(feature_names):
    col = cols[i % 5]
    with col:
        dv = float(default_vals.get(f, 0.0))
        vals[f] = st.number_input(f, value=float(dv), format="%.6f")
    # 每 5 个变量换一行
    if (i + 1) % 5 == 0 and (i + 1) < len(feature_names):
        cols = st.columns(5)

if st.button("Predict"):
    X_new = pd.DataFrame([vals], columns=feature_names)
    proba = float(model.predict_proba(X_new)[:, 1][0])
    pred  = int(proba >= THRESHOLD)
    st.success(f"Probability of RFPE: {proba:.3f}")
    st.info(f"Predicted label: {'Positive (RFPE)' if pred==1 else 'Negative (Normal)'}")
