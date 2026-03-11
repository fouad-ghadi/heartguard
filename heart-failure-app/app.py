import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #080e1a !important;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px !important; }

[data-testid="stSidebar"] {
    background: #0d1526 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.78rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099cc) !important;
    color: #080e1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,212,170,0.35) !important;
}
.streamlit-expanderHeader {
    background: #0d1526 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}
.streamlit-expanderContent {
    background: #0d1526 !important;
    border: 1px solid #1e2d4a !important;
    border-top: none !important;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
</style>
""", unsafe_allow_html=True)


# ── Train model (same logic as your FastAPI app) ────────────────────────────────
@st.cache_resource
def train_model():
    """Replicates your original FastAPI training logic."""
    DATA_PATH = os.path.join(os.path.dirname(__file__), "model", "heart5.xls")

    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        # Demo mode: generate synthetic data with same structure
        st.warning("⚠️ Dataset not found at `model/heart5.xls` — running in demo mode with synthetic data.", icon="⚠️")
        np.random.seed(42)
        n = 303
        data = pd.DataFrame({
            "cp":       np.random.randint(0, 4, n),
            "trestbps": np.random.randint(94, 200, n),
            "chol":     np.random.randint(126, 564, n),
            "fbs":      np.random.randint(0, 2, n),
            "restecg":  np.random.randint(0, 3, n),
            "thalach":  np.random.randint(71, 202, n),
            "slope":    np.random.randint(0, 3, n),
            "target":   np.random.randint(0, 2, n),
        })

    X = data.drop("target", axis=1)[["cp","trestbps","chol","fbs","restecg","thalach","slope"]]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    return model, scaler, round(acc * 100, 1)


model, scaler, accuracy = train_model()


# ── Predict function (same as your original) ────────────────────────────────────
def predict_heart_disease(cp, trestbps, chol, fbs, restecg, thalach, slope):
    user_data = np.array([[cp, trestbps, chol, fbs, restecg, thalach, slope]])
    user_data = scaler.transform(user_data)
    prediction    = model.predict(user_data)
    probability   = model.predict_proba(user_data)[0]
    return int(prediction[0]), probability


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:2rem;'>
    <div>
        <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;
                    color:#f1f5f9;letter-spacing:-0.02em;'>
            Heart Disease Predictor
        </div>
        <div style='font-size:0.85rem;color:#475569;margin-top:4px;'>
            Random Forest Classifier · 7 Clinical Features
        </div>
    </div>
    <div style='text-align:right;'>
        <div style='font-family:DM Mono,monospace;font-size:1.4rem;font-weight:700;color:#00d4aa;'>{accuracy}%</div>
        <div style='font-size:0.72rem;color:#475569;letter-spacing:0.05em;text-transform:uppercase;'>Model Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar inputs (matching your RequestModel fields) ──────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    background:linear-gradient(135deg,#00d4aa,#0099cc);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            🫀 Patient Input
        </div>
        <div style='font-size:0.72rem;color:#475569;letter-spacing:0.08em;
                    text-transform:uppercase;margin-top:2px;'>
            Fill all fields below
        </div>
    </div>
    <hr style='border-color:#1e2d4a;margin:0.8rem 0 1.2rem;'>
    """, unsafe_allow_html=True)

    # chest_pain (cp)
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Typical Angina",
            1: "1 — Atypical Angina",
            2: "2 — Non-Anginal Pain",
            3: "3 — Asymptomatic"
        }[x]
    )

    # resting_blood_pressure (trestbps)
    trestbps = st.number_input(
        "Resting Blood Pressure (trestbps) mmHg",
        min_value=80, max_value=220, value=120, step=1
    )

    # cholesterol (chol)
    chol = st.number_input(
        "Cholesterol (chol) mg/dL",
        min_value=100, max_value=600, value=240, step=1
    )

    # sugar (fbs) — fasting blood sugar > 120 mg/dl
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL (fbs)",
        options=[0, 1],
        format_func=lambda x: "1 — Yes" if x else "0 — No"
    )

    # electrocardiographic_result (restecg)
    restecg = st.selectbox(
        "Resting ECG Result (restecg)",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Normal",
            1: "1 — ST-T Wave Abnormality",
            2: "2 — Left Ventricular Hypertrophy"
        }[x]
    )

    # max_heart_rate (thalach)
    thalach = st.number_input(
        "Max Heart Rate Achieved (thalach)",
        min_value=60, max_value=220, value=150, step=1
    )

    # slope
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Upsloping",
            1: "1 — Flat",
            2: "2 — Downsloping"
        }[x]
    )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Run Prediction", use_container_width=True, type="primary")

    st.markdown("""
    <div style='margin-top:1.5rem;padding:0.8rem;background:#0a1220;border-radius:8px;
                border:1px solid #1e2d4a;font-size:0.7rem;color:#475569;line-height:1.6;'>
        ⚕️ For decision support only. Not a substitute for clinical judgement.
    </div>
    """, unsafe_allow_html=True)


# ── Stat bar ────────────────────────────────────────────────────────────────────
def stat_card(col, icon, label, value, color="#94a3b8", sub=None):
    col.markdown(f"""
    <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:14px;
                padding:1.1rem 1.3rem;'>
        <div style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#475569;margin-bottom:0.3rem;'>{icon} {label}</div>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;color:{color};'>{value}</div>
        {f'<div style="font-size:0.7rem;color:#334155;margin-top:2px;">{sub}</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
bp_color  = "#ef4444" if trestbps > 140 else "#f59e0b" if trestbps > 120 else "#00d4aa"
ch_color  = "#ef4444" if chol > 240 else "#f59e0b" if chol > 200 else "#00d4aa"
hr_color  = "#ef4444" if thalach > 180 else "#00d4aa" if thalach > 100 else "#f59e0b"
cp_color  = "#ef4444" if cp == 3 else "#f59e0b" if cp == 2 else "#00d4aa"

stat_card(c1, "🩸", "Blood Pressure", f"{trestbps} mmHg", bp_color, ">140 = high")
stat_card(c2, "🧪", "Cholesterol",    f"{chol} mg/dL",   ch_color, ">240 = high")
stat_card(c3, "💓", "Max Heart Rate", f"{thalach} bpm",  hr_color)
stat_card(c4, "⚡", "Chest Pain",
          {0:"Typical",1:"Atypical",2:"Non-Anginal",3:"Asymptomatic"}[cp], cp_color)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)


# ── Result ──────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:600;
                letter-spacing:0.12em;text-transform:uppercase;color:#475569;
                margin-bottom:1rem;'>Prediction Result</div>
    """, unsafe_allow_html=True)

    if predict_btn or st.session_state.get("hd_predicted"):

        if predict_btn:
            pred, proba = predict_heart_disease(cp, trestbps, chol, fbs, restecg, thalach, slope)
            st.session_state["hd_pred"]      = pred
            st.session_state["hd_proba"]     = proba
            st.session_state["hd_predicted"] = True

        pred  = st.session_state["hd_pred"]
        proba = st.session_state["hd_proba"]
        prob_pct    = proba[1] * 100
        is_positive = pred == 1
        label       = "HEART DISEASE DETECTED" if is_positive else "NO HEART DISEASE"
        ring_color  = "#ef4444" if is_positive else "#00d4aa"
        bar_gradient= "linear-gradient(90deg,#ef4444,#dc2626)" if is_positive else "linear-gradient(90deg,#00d4aa,#0099cc)"
        glow        = "rgba(239,68,68,0.4)" if is_positive else "rgba(0,212,170,0.4)"

        st.markdown(f"""
        <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:16px;
                    padding:2rem;text-align:center;margin-bottom:1rem;'>

            <svg width="160" height="160" viewBox="0 0 160 160" style="margin-bottom:1rem;">
                <circle cx="80" cy="80" r="65" fill="none" stroke="#1e2d4a" stroke-width="12"/>
                <circle cx="80" cy="80" r="65" fill="none" stroke="{ring_color}" stroke-width="12"
                    stroke-dasharray="{408.4 * proba[1]:.1f} 408.4"
                    stroke-linecap="round"
                    transform="rotate(-90 80 80)"
                    style="filter:drop-shadow(0 0 6px {ring_color});"/>
                <text x="80" y="72" text-anchor="middle"
                    style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;fill:{ring_color};">
                    {prob_pct:.0f}%
                </text>
                <text x="80" y="92" text-anchor="middle"
                    style="font-family:DM Sans,sans-serif;font-size:10px;fill:#475569;letter-spacing:1px;">
                    PROBABILITY
                </text>
            </svg>

            <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;
                        color:{ring_color};letter-spacing:0.04em;margin-bottom:0.3rem;'>
                {label}
            </div>
            <div style='font-size:0.78rem;color:#475569;margin-bottom:1.2rem;'>
                Confidence: {max(proba)*100:.1f}%
            </div>

            <div style='background:#131f35;border-radius:999px;height:8px;overflow:hidden;'>
                <div style='width:{prob_pct:.1f}%;height:100%;border-radius:999px;
                            background:{bar_gradient};
                            box-shadow:0 0 10px {glow};'></div>
            </div>
            <div style='display:flex;justify-content:space-between;
                        font-size:0.65rem;color:#334155;margin-top:4px;'>
                <span>0% — Healthy</span><span>100% — Disease</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown(f"""
        <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:14px;
                    padding:1rem 1.2rem;'>
            <div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                        color:#475569;margin-bottom:0.8rem;'>Probability Breakdown</div>
            <div style='display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:0.5rem;'>
                <span style='font-size:0.82rem;color:#00d4aa;'>🟢 No Disease</span>
                <span style='font-family:DM Mono,monospace;font-size:0.9rem;color:#00d4aa;font-weight:600;'>
                    {proba[0]*100:.1f}%
                </span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-size:0.82rem;color:#ef4444;'>🔴 Heart Disease</span>
                <span style='font-family:DM Mono,monospace;font-size:0.9rem;color:#ef4444;font-weight:600;'>
                    {proba[1]*100:.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#0d1526;border:1px dashed #1e2d4a;border-radius:16px;
                    padding:3rem 2rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:1rem;opacity:0.3;'>🫀</div>
            <div style='font-size:0.85rem;color:#334155;'>
                Fill in patient data in the sidebar<br>
                and click <strong style="color:#475569;">Run Prediction</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Right: Feature importance ────────────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:600;
                letter-spacing:0.12em;text-transform:uppercase;color:#475569;
                margin-bottom:1rem;'>Feature Importance</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:16px;padding:1.4rem;'>", unsafe_allow_html=True)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    feature_names = ["Chest Pain", "Rest. BP", "Cholesterol",
                     "Fasting Sugar", "Rest. ECG", "Max HR", "Slope"]
    importances   = model.feature_importances_
    indices       = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0d1526")
    ax.set_facecolor("#0d1526")

    colors = ["#00d4aa" if i == indices[-1] else
              "#0d6e8a" if i in indices[-3:] else
              "#1e3a5f" for i in range(len(importances))]

    ax.barh([feature_names[i] for i in indices],
            importances[indices],
            color=[colors[i] for i in indices],
            edgecolor="none", height=0.55)

    ax.grid(axis="x", color="#1e2d4a", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("Importance Score", fontsize=8.5, color="#475569", labelpad=8)
    ax.set_title("Random Forest Feature Importance",
                 fontsize=9.5, fontweight="600", color="#94a3b8", pad=12, fontfamily="Syne")
    ax.tick_params(axis="y", labelsize=8.5, colors="#94a3b8", length=0)
    ax.tick_params(axis="x", labelsize=7.5, colors="#475569", length=0)
    for spine in ax.spines.values(): spine.set_visible(False)

    # value labels
    for i, (imp, idx) in enumerate(zip(importances[indices], indices)):
        ax.text(imp + 0.002, i, f"{imp:.3f}", va="center",
                fontsize=7.5, color="#64748b")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Model info expander ──────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
with st.expander("ℹ️  Model Details & How to Use"):
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"""
        **Model:** Random Forest Classifier  
        **Test Accuracy:** `{accuracy}%`  
        **Training Split:** 80% train / 20% test  
        **Scaler:** StandardScaler  
        **Features used:** 7 clinical parameters
        """)
    with d2:
        st.markdown("""
        **Input fields match your original API:**
        - `cp` → Chest Pain Type
        - `trestbps` → Resting Blood Pressure
        - `chol` → Cholesterol
        - `fbs` → Fasting Blood Sugar
        - `restecg` → ECG Result
        - `thalach` → Max Heart Rate
        - `slope` → ST Segment Slope
        """)
