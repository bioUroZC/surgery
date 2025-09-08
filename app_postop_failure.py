import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Perioperative RF & Mortality — Prediction", layout="wide")
st.title("Perioperative Respiratory Failure & Mortality — Prediction")

# ---------------- Fixed config ----------------
DATASETS = {
    "Occurrence (RFPE)": {
        "csv": "train_occurence.csv",        # 列含 group: normal / rfpe
        "target": "group",
        "label_map": {"normal": 0, "rfpe": 1},
        "positive_name": "RFPE"
    },
    "28-day mortality": {
        "csv": "train_28days.csv",           # 列含 group: 通常为 0/1
        "target": "group",
        "label_map": None,                    # 若为数字则直接用；若为字符串则自动 0/1
        "positive_name": "28-day death"
    },
    "90-day mortality": {
        "csv": "train_90days.csv",           # 列含 group: 通常为 0/1
        "target": "group",
        "label_map": None,
        "positive_name": "90-day death"
    },
}
THRESHOLD = 0.5   # 简化：固定 0.5

# ---------------- Helpers ----------------
@st.cache_resource(show_spinner=False)
def train_for_task(task_key: str):
    """Load CSV, build & fit a simple RF pipeline. Return (model, feature_names, defaults)."""
    cfg = DATASETS[task_key]
    df = pd.read_csv(cfg["csv"], index_col=0)

    if cfg["target"] not in df.columns:
        raise ValueError(f"Target column '{cfg['target']}' not found in {cfg['csv']}.")

    X = df.drop(columns=[cfg["target"]])
    y_raw = df[cfg["target"]]

    # 统一标签到 0/1
    if y_raw.dtype.kind in "iuf":
        y = (y_raw > 0).astype(int)
    else:
        # 如果提供了显式映射（如 normal/rfpe），优先使用；否则尝试自动二值化
        if cfg["label_map"]:
            y = y_raw.map(cfg["label_map"]).astype(int)
        else:
            # 将最常见的两个取值映射为 0/1（字典序或出现频次决定）
            vals = y_raw.dropna().unique().tolist()
            if len(vals) != 2:
                raise ValueError(f"Target in {cfg['csv']} must be binary; got: {vals}")
            mapping = {vals[0]: 0, vals[1]: 1}
            y = y_raw.map(mapping).astype(int)

    # 训练一个轻量 RF（树模型对尺度不敏感，但保留 StandardScaler 不影响）
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ))
    ])
    pipe.fit(X, y)

    # 用训练集的中位数作为默认值
    defaults = X.median(numeric_only=True).to_dict()

    return pipe, list(X.columns), defaults

# ---------------- UI: Task switch ----------------
task = st.radio(
    "Select task",
    list(DATASETS.keys()),
    horizontal=True
)

# 训练所选任务的模型
try:
    model, feature_names, default_vals = train_for_task(task)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

# ---------------- Inputs (5 columns per row) ----------------
st.subheader("Input variables")

vals = {}
cols = st.columns(5)
for i, f in enumerate(feature_names):
    col = cols[i % 5]
    with col:
        dv = float(default_vals.get(f, 0.0))
        vals[f] = st.number_input(f, value=dv, format="%.6f")
    if (i + 1) % 5 == 0 and (i + 1) < len(feature_names):
        cols = st.columns(5)

# ---------------- Predict ----------------
if st.button("Predict"):
    try:
        X_new = pd.DataFrame([vals], columns=feature_names)
        proba = float(model.predict_proba(X_new)[:, 1][0])
        pred  = int(proba >= THRESHOLD)
        pos_name = DATASETS[task]["positive_name"]

        st.markdown(f"**Probability of {pos_name}:** `{proba:.3f}`")
        st.markdown(f"**Predicted label:** {'🟥 Positive' if pred==1 else '🟩 Negative'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
