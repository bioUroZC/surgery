import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Postop Respiratory Failure — Minimal", layout="centered")
st.title("Postoperative Respiratory Failure — Minimal Train & Predict")

st.caption(
    "Reads train_top.csv from the repo, trains a simple model on the fly, "
    "then predicts from user inputs. No test set, no uploads."
)

# ---------------- Sidebar ----------------
csv_path   = st.sidebar.text_input("Train CSV path", value="train_top.csv")
target_col = st.sidebar.text_input("Target column", value="group")
pos_label  = st.sidebar.text_input("Positive label (→1)", value="rfpe")
neg_label  = st.sidebar.text_input("Negative label (→0)", value="normal")
threshold  = st.sidebar.slider("Positive threshold", 0.05, 0.95, 0.50, 0.01)

# ---------------- Load & Train ----------------
@st.cache_resource(show_spinner=True)
def load_and_train(train_csv: str, target: str,
                   pos: str, neg: str):
    df = pd.read_csv(train_csv, index_col=0)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {train_csv}. "
                         f"Available columns: {list(df.columns)}")

    X = df.drop(columns=[target])
    y_raw = df[target]

    # Map labels to 0/1 (if already numeric, keep as is)
    if y_raw.dtype.kind in "iuf":  # numeric
        y = (y_raw > 0).astype(int)
    else:
        y = y_raw.map({neg: 0, pos: 1}).astype(int)

    # Simple, fast, robust model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe.fit(X, y)

    # Use medians as sensible defaults
    defaults = X.median(numeric_only=True).to_dict()

    return pipe, list(X.columns), defaults

try:
    model, feature_names, default_vals = load_and_train(csv_path, target_col, pos_label, neg_label)
    st.success("Model trained on cloud ✔️")
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

# ---------------- Inputs & Predict ----------------
st.subheader("Input features")
vals = {}
for f in feature_names:
    # default to training median if numeric; fall back to 0.0
    dv = float(default_vals.get(f, 0.0))
    vals[f] = st.number_input(f, value=float(dv), format="%.6f")

if st.button("Predict"):
    try:
        X_new = pd.DataFrame([vals], columns=feature_names)
        proba = float(model.predict_proba(X_new)[:, 1][0])
        pred  = int(proba >= threshold)
        st.success(f"Probability (positive='{pos_label}'): {proba:.3f}")
        st.info(f"Predicted label: {'Positive' if pred==1 else 'Negative'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
