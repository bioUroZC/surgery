import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page Config ----------------
st.set_page_config(page_title="Postoperative Respiratory Failure and Pulmonary Embolism & Mortality — Prediction", layout="wide")

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
        "csv": "train_occurence.csv",   # your filename
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
DEFAULT_TASK = list(DATASETS.keys())[0]

# ---------------- Train (cached per task) ----------------
@st.cache_resource(show_spinner=False)
def train_for_task(task_key: str):
    cfg = DATASETS[task_key]
    df = pd.read_csv(cfg["csv"], index_col=0)

    if cfg["target"] not in df.columns:
        raise ValueError(f"Target column '{cfg['target']}' not found in {cfg['csv']}.")

    X = df.drop(columns=[cfg["target"]])
    y_raw = df[cfg["target"]]

    # unify to 0/1
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
    "<div class='hero'><h1>🫁 Postoperative Respiratory Failure and Pulmonary Embolism & Mortality — Prediction</h1></div>", 
    unsafe_allow_html=True
)


# ---------------- Task selector (radio for compatibility) ----------------
task = st.radio("Select task", list(DATASETS.keys()), index=0, horizontal=True)
# 保险起见：如果返回了意外值，回退到默认
if task not in DATASETS:
    task = DEFAULT_TASK
cfg = DATASETS[task]
pos_name = cfg["positive_name"]

# Train silently (cached)
try:
    model, feature_names, default_vals = train_for_task(task)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()


# ---------------- Threshold presets ----------------
st.markdown("#### Prediction")
c1, c2, c3 = st.columns([1,3,2])
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
    predict_clicked = st.button("🚀 Predict", type="primary", use_container_width=True)

# ---------------- Inputs (5 columns per row) ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
cols = st.columns(5)
vals = {}

for i, f in enumerate(feature_names):
    col = cols[i % 5]

    # 规则：score/age/bpm/sec/mode/PLT/ventilation 等用整数；其余保留 1 位小数
    is_int_like = any(
        kw in f.lower() for kw in
        ["score", "age", "bpm", "sec", "mode", "plt", "count"]
    )

    with col:
        dv = float(default_vals.get(f, 0.0))
        if is_int_like:
            # 整数：不要传 format；value 用 int；step=1
            vals[f] = st.number_input(
                f, value=int(round(dv)), step=1, key=f"_in_{f}"
            )
        else:
            # 小数：一位小数；value 用 float；step=0.1
            vals[f] = st.number_input(
                f, value=round(dv, 1), step=0.1, format="%.1f", key=f"_in_{f}"
            )

    if (i + 1) % 5 == 0 and (i + 1) < len(feature_names):
        cols = st.columns(5)

st.markdown("</div>", unsafe_allow_html=True)



# ---------------- Predict & Result ----------------
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
        st.progress(min(max(proba, 0.0), 1.0), text=f"Probability of {pos_name}: {proba:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
