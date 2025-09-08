import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

"""
app_postop_failure.py — Single-model Streamlit app (English UI)
Purpose: Predict postoperative respiratory failure (binary) using exactly the 15 features
         defined in schema.json. CSV upload is NOT required; users input values one by one.
Requirements: streamlit, joblib, pandas, numpy, pillow (optional for images)
Run: streamlit run app_postop_failure.py
"""

st.set_page_config(page_title="Postoperative Respiratory Failure Risk", page_icon="🫁", layout="wide")

# ---------------- UI Style ----------------
STYLES = """
<style>
div.block-container {padding-top: 1.25rem;}
.kpi-card {background: #fff; padding: 18px 20px; border-radius: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.06);} 
.kpi-badge {padding: 6px 10px; border-radius: 999px; font-weight: 600;}
.badge-low {background: #ECFDF5; color: #065F46;}
.badge-mid {background: #FFFBEB; color: #92400E;}
.badge-high {background: #FEF2F2; color: #991B1B;}
.small {font-size: 12px; color: #6b7280;}
hr {border: none; height: 1px; background: #f1f5f9; margin: 1rem 0;}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ---------------- Helpers ----------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_schema(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # basic validation
        if not isinstance(data, dict) or "features" not in data or not isinstance(data["features"], dict):
            raise ValueError("schema.json must contain a 'features' object.")
        if len(data["features"]) == 0:
            raise ValueError("'features' is empty.")
        return data
    except Exception as e:
        st.error(f"Failed to read schema.json: {e}")
        return None

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings")
model_path = st.sidebar.text_input("Model file (.pkl)", value="random_forest_model.pkl")
schema_path = st.sidebar.text_input("Schema file (schema.json)", value="schema.json")
thresh = st.sidebar.slider("Positive threshold", 0.05, 0.95, 0.50, 0.01)
show_prob = st.sidebar.toggle("Show predicted probability", value=True)

schema = load_schema(schema_path)
if schema is None:
    st.stop()

# Always use the feature list from schema.json (order preserved)
FEATURE_NAMES = list(schema["features"].keys())

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.markdown("## 🫁 Postoperative Respiratory Failure — Risk Prediction")
st.caption("For research/decision support only. Not a substitute for clinical diagnosis.")

# ---------------- Form ----------------
st.markdown("#### Input features")
cols = st.columns(3)


def render_input(feat: str, idx: int):
    meta = schema["features"].get(feat, {})
    label = meta.get("label", feat)
    ftype = meta.get("type", "float")
    minv = meta.get("min", None)
    maxv = meta.get("max", None)
    default = meta.get("default", 0.0 if ftype != "int" else 0)
    help_text = meta.get("help", None)
    col = cols[idx % 3]
    with col:
        if ftype == "int":
            return st.number_input(label, value=int(default), min_value=minv, max_value=maxv, step=1, help=help_text, key=f"in_{feat}")
        elif ftype == "cat":
            choices = meta.get("choices", [])
            if not choices:
                choices = ["0", "1"]
            sel = st.selectbox(label, choices=choices, help=help_text, key=f"in_{feat}")
            mapping = meta.get("mapping", None)
            if mapping and sel in mapping:
                return mapping[sel]
            try:
                return float(sel)
            except Exception:
                return sel
        else:
            step = meta.get("step", 0.1)
            return st.number_input(label, value=float(default), min_value=minv, max_value=maxv, step=step, help=help_text, key=f"in_{feat}")

values = {}
for i, f in enumerate(FEATURE_NAMES):
    values[f] = render_input(f, i)

st.markdown("
")
c1, c2 = st.columns([1, 2])
with c1:
    run = st.button("🚀 Predict", type="primary")
with c2:
    st.caption("Adjust the threshold on the left to trade off sensitivity/specificity.")

# ---------------- Predict ----------------
if run:
    X = pd.DataFrame([values], columns=FEATURE_NAMES)
    # Try predict_proba; fallback to decision_function; else predict
    try:
        prob_pos = float(model.predict_proba(X)[:, 1][0])
    except Exception:
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(X)[0])
            prob_pos = 1 / (1 + np.exp(-score))
        else:
            prob_pos = float(model.predict(X)[0])
    pred = int(prob_pos >= thresh)

    # Risk badge
    if prob_pos < 0.2:
        badge_cls, badge_txt = "badge-low", "Low risk"
    elif prob_pos < 0.5:
        badge_cls, badge_txt = "badge-mid", "Low–moderate risk"
    elif prob_pos < 0.8:
        badge_cls, badge_txt = "badge-mid", "Moderate–high risk"
    else:
        badge_cls, badge_txt = "badge-high", "High risk"

    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    k1, k2, k3 = st.columns([1, 1, 2])
    with k1:
        st.metric("Predicted label", "Positive" if pred == 1 else "Negative")
    with k2:
        st.metric("Probability" if show_prob else "Threshold", f"{prob_pos:.3f}" if show_prob else f"{thresh:.2f}")
    with k3:
        st.markdown(f'<span class="kpi-badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.download_button(
        "💾 Download this result (CSV)",
        data=pd.DataFrame([{**values, "prob": prob_pos, "pred": pred}]).to_csv(index=False),
        file_name="single_prediction.csv",
        mime="text/csv",
    )

# ---------------- Optional: metrics / explainability ----------------
with st.expander("📈 Optional: show test metrics (metrics.csv)"):
    if os.path.exists("metrics.csv"):
        try:
            dfm = pd.read_csv("metrics.csv")
            st.dataframe(dfm)
        except Exception as e:
            st.warning(f"Failed to read metrics.csv: {e}")
    else:
        st.caption("metrics.csv not found.")

with st.expander("🧩 Optional: SHAP/global importance images"):
    from PIL import Image
    imgs = ["shap_summary_beeswarm.png", "shap_summary_bar.png"]
    found = False
    for p in imgs:
        if os.path.exists(p):
            st.image(Image.open(p), caption=os.path.basename(p), use_column_width=True)
            found = True
    if not found:
        st.caption("If you only have PDFs, consider saving PNGs in your training script: plt.savefig('xxx.png', dpi=200)")

st.markdown("""
<hr/>
<div class='small'>For research/education purposes only. Not for clinical decision-making without physician oversight.</div>
""", unsafe_allow_html=True)
