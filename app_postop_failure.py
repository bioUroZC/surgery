import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page Config ----------------
st.set_page_config(page_title="Perioperative RF & Mortality — Prediction", layout="wide")

# ---------------- Styles (lightweight CSS) ----------------
STYLES = """
<style>
div.block-container {padding-top: 1.2rem;}
.hero {display:flex; align-items:center; gap:14px; margin-bottom:0.6rem;}
.hero h1 {margin:0; font-size:1.6rem;}
.badge {padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.85rem;}
.badge-low {background:#ECFDF5; color:#065F46;}
.badge-mid {background:#FFFBEB; color:#92400E;}
.badge-high{background:#FEF2F2; color:#991B1B;}
.card {background:#fff; border-radius:16px; padding:16px 18px; box-shadow:0 6px 18px rgba(0,0,0,0.06);}
.subtle {color:#6b7280; font-size:.92rem;}
hr{border:none; height:1px; background:#eef2f7; margin:.8rem 0;}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ---------------- Fixed config ----------------
DATASETS = {
    "Occurrence (RFPE)": {
        "csv": "train_occurence.csv",
        "target": "group",
        "label_map": {"normal": 0, "rfpe": 1},
        "positive_name": "Respiratory Failure (RFPE)"
    },
    "28-day mortality": {
        "csv": "train_28days.csv",
        "target": "group",
        "label_map": None,
        "positive_name": "28-day mortality"
    },
    "90-day mortality": {
        "csv": "train_90days.csv",
        "target": "group",
        "label_map": None,
        "positive_name": "90-day mortality"
    },
}

# ---------------- Train (cached per task) ----------------
@st.cache_resource(show_spinner=False)
def train_for_task(task_key: str):
    cfg = DATASETS[task_key]
    df = pd.read_csv(cfg["csv"], index_col=0)

    if cfg["target"] not in df.columns:
        raise ValueError(f"Target column '{cfg['target']}' not found in {cfg['csv']}.")

    X = df.drop(columns=[cfg["target"]])
    y_raw = df[cfg["target"]]

    if y_raw.dtype.kind in "iuf":
        y = (y_raw > 0).astype(int)
    else:
        if cfg["label_map"]:
            y = y_raw.map(cfg["label_map"]).astype(int)
        else:
            vals = y_raw.dropna().unique().tolist()
            if len(vals) != 2:
                raise ValueError(f"Target in {cfg['csv']} must be binary; got: {vals}")
            mapping = {vals[0]: 0, vals[1]: 1}
            y = y_raw.map(mapping).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            random_state=42, n_jobs=-1
        ))
    ])
    pipe.fit(X, y)

    defaults = X.median(numeric_only=True).to_dict()
    return pipe, list(X.columns), defaults

# ---------------- Header ----------------
st.markdown(
    "<div class='hero'><h1>🫁 Perioperative Respiratory Failure & Mortality — Prediction</h1>"
    "<span class='subtle'>Single-page, cloud-trained Random Forest</span></div>", unsafe_allow_html=True
)

# ---------------- Task selector ----------------
task = st.segmented_control("Select task", list(DATASETS.keys()), selection_mode="single")
cfg = DATASETS[task]
pos_name = cfg["positive_name"]

try:
    model, feature_names, default_vals = train_for_task(task)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

# ---------------- Threshold presets ----------------
st.markdown("#### Prediction")
c1, c2, c3, c4 = st.columns([1,1,3,3])
with c1:
    preset = st.radio("Preset", ["High sensitivity", "Balanced", "High specificity"], index=1)
with c2:
    if preset == "High sensitivity":
        threshold = 0.30
    elif preset == "High specificity":
        threshold = 0.70
    else:
        threshold = 0.50
    threshold = st.slider("Threshold", 0.05, 0.95, float(threshold), 0.01, label_visibility="collapsed")
with c3:
    if st.button("Reset to median values", use_container_width=True):
        for f in feature_names:
            st.session_state[f"_in_{f}"] = float(default_vals.get(f, 0.0))
with c4:
    predict_clicked = st.button("🚀 Predict", type="primary", use_container_width=True)

# ---------------- Inputs (5 columns per row, in a card) ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
cols = st.columns(5)
vals = {}
for i, f in enumerate(feature_names):
    col = cols[i % 5]
    with col:
        dv = float(default_vals.get(f, 0.0))
        vals[f] = st.number_input(f, value=float(dv), format="%.6f", key=f"_in_{f}")
    if (i + 1) % 5 == 0 and (i + 1) < len(feature_names):
        cols = st.columns(5)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Predict & Result card ----------------
if predict_clicked:
    try:
        X_new = pd.DataFrame([vals], columns=feature_names)
        proba = float(model.predict_proba(X_new)[:, 1][0])
        pred  = int(proba >= threshold)

        # risk badge
        if proba < 0.20:
            bcls, btxt = "badge-low", "Low risk"
        elif proba < 0.50:
            bcls, btxt = "badge-mid", "Low–Moderate risk"
        elif proba < 0.80:
            bcls, btxt = "badge-mid", "Moderate–High risk"
        else:
            bcls, btxt = "badge-high", "High risk"

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1,1,2])
        with r1:
            st.metric("Predicted label", "Positive" if pred==1 else "Negative")
        with r2:
            st.metric("Probability", f"{proba:.3f}")
        with r3:
            st.markdown(f"<span class='badge {bcls}'>{btxt}</span>", unsafe_allow_html=True)

        # simple progress bar
        st.progress(min(max(proba, 0.0), 1.0), text=f"Probability of {pos_name}: {proba:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- Footer ----------------
st.markdown("<hr><div class='subtle'>For research/education purposes only.</div>", unsafe_allow_html=True)
