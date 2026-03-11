import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

/* ── Medical background image with dark overlay ── */
.stApp {
    background:
        linear-gradient(135deg,
            rgba(8,14,26,0.93) 0%,
            rgba(8,14,26,0.78) 50%,
            rgba(8,14,26,0.93) 100%),
        url("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=1600&q=80")
        center center / cover no-repeat fixed !important;
}

/* Red glow left */
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0;
    width: 45%; height: 100%;
    background: radial-gradient(ellipse at 0% 50%,
        rgba(180,20,20,0.22) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}

/* Blue glow right */
.stApp::after {
    content: '';
    position: fixed; top: 0; right: 0;
    width: 50%; height: 100%;
    background: radial-gradient(ellipse at 100% 50%,
        rgba(0,140,210,0.20) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}

.block-container { position: relative; z-index: 1; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1300px !important; }

/* Glass card effect */
[data-testid="stExpander"],
div[style*="background:#0d1526"] {
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

/* Expander — make it look like a panel toggle */
.streamlit-expanderHeader {
    background: rgba(13,21,38,0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid #00d4aa !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: #00d4aa !important;
    padding: 0.8rem 1.2rem !important;
}
.streamlit-expanderContent {
    background: #0d1526 !important;
    border: 1px solid #1e2d4a !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.2rem !important;
}
/* Inputs */
input[type=number], .stSelectbox > div > div {
    background: #131f35 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099cc) !important;
    color: #080e1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,212,170,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Train model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "model", "heart5.xls")
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        np.random.seed(42); n = 303
        data = pd.DataFrame({
            "cp": np.random.randint(0,4,n), "trestbps": np.random.randint(94,200,n),
            "chol": np.random.randint(126,564,n), "fbs": np.random.randint(0,2,n),
            "restecg": np.random.randint(0,3,n), "thalach": np.random.randint(71,202,n),
            "slope": np.random.randint(0,3,n), "target": np.random.randint(0,2,n),
        })
    X = data.drop("target", axis=1)[["cp","trestbps","chol","fbs","restecg","thalach","slope"]]
    y = data["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    clf = RandomForestClassifier(random_state=42)
    clf.fit(sc.fit_transform(Xtr), ytr)
    acc = accuracy_score(yte, clf.predict(sc.transform(Xte)))
    return clf, sc, round(acc * 100, 1)

model, scaler, accuracy = train_model()


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem;'>
    <div style='display:flex;align-items:center;gap:1rem;'>
        <div style='width:54px;height:54px;background:linear-gradient(135deg,#c0392b,#e74c3c);
                    border-radius:14px;display:flex;align-items:center;justify-content:center;
                    box-shadow:0 0 20px rgba(192,57,43,0.5);flex-shrink:0;'>
            <svg width="30" height="28" viewBox="0 0 30 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M15 26S2 17.5 2 8.5C2 5 4.5 2 8 2c2.5 0 4.8 1.4 6.2 3.5L15 7l0.8-1.5C17.2 3.4 19.5 2 22 2c3.5 0 6 3 6 6.5C28 17.5 15 26 15 26Z"
                    fill="white" stroke="rgba(255,255,255,0.3)" stroke-width="0.5"/>
                <path d="M8 10 L12 14 L15 11 L18 16 L22 12"
                    stroke="rgba(192,57,43,0.7)" stroke-width="1.8"
                    stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            </svg>
        </div>
        <div>
            <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                        color:#f1f5f9;letter-spacing:-0.02em;'>Heart Disease Predictor</div>
            <div style='font-size:0.85rem;color:#475569;margin-top:2px;'>
                Random Forest Classifier · 7 Clinical Features</div>
        </div>
    </div>
    <div style='text-align:right;'>
        <div style='font-family:DM Mono,monospace;font-size:1.6rem;font-weight:700;color:#00d4aa;'>{accuracy}%</div>
        <div style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;'>Model Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── INPUT PANEL (expander = click to open/close, always works) ──────────────────
with st.expander("☰  Patient Input Panel — click to open / close", expanded=False):
    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        trestbps = st.number_input("🩸 Resting Blood Pressure (mmHg)", 80, 220, 120, 1)
    with r1c2:
        chol = st.number_input("🧪 Cholesterol (mg/dL)", 100, 600, 240, 1)
    with r1c3:
        thalach = st.number_input("💓 Max Heart Rate Achieved", 60, 220, 150, 1)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        cp = st.selectbox("⚡ Chest Pain Type", [0,1,2,3],
            format_func=lambda x: {0:"0 — Typical",1:"1 — Atypical",2:"2 — Non-Anginal",3:"3 — Asymptomatic"}[x])
    with r2c2:
        fbs = st.selectbox("🍬 Fasting Blood Sugar >120", [0,1],
            format_func=lambda x: "1 — Yes" if x else "0 — No")
    with r2c3:
        restecg = st.selectbox("📈 Resting ECG", [0,1,2],
            format_func=lambda x: {0:"0 — Normal",1:"1 — ST-T Abnormality",2:"2 — LV Hypertrophy"}[x])
    with r2c4:
        slope = st.selectbox("📉 ST Segment Slope", [0,1,2],
            format_func=lambda x: {0:"0 — Upsloping",1:"1 — Flat",2:"2 — Downsloping"}[x])

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    col_btn = st.columns([2, 1, 2])
    with col_btn[1]:
        predict_btn = st.button("⚡ Run Prediction", use_container_width=True)

# ── Save to session state ────────────────────────────────────────────────────────
if predict_btn:
    X = scaler.transform([[cp, trestbps, chol, fbs, restecg, thalach, slope]])
    st.session_state["pred"]      = int(model.predict(X)[0])
    st.session_state["proba"]     = model.predict_proba(X)[0]
    st.session_state["inputs"]    = dict(cp=cp,trestbps=trestbps,chol=chol,
                                         fbs=fbs,restecg=restecg,thalach=thalach,slope=slope)
    st.session_state["predicted"] = True

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# ── Stat bar (updates live as inputs change) ────────────────────────────────────
def stat_card(col, icon, label, value, color, sub=None):
    col.markdown(f"""
    <div style='background:rgba(13,21,38,0.82);border:1px solid #1e2d4a;border-radius:14px;
                padding:1.1rem 1.3rem;backdrop-filter:blur(12px);'>
        <div style='font-size:0.67rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#475569;margin-bottom:0.3rem;'>{icon} {label}</div>
        <div style='font-family:Syne,sans-serif;font-size:1.35rem;font-weight:700;color:{color};'>{value}</div>
        {f'<div style="font-size:0.68rem;color:#334155;margin-top:2px;">{sub}</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

s1,s2,s3,s4 = st.columns(4)
stat_card(s1,"🩸","Blood Pressure",f"{trestbps} mmHg",
          "#ef4444" if trestbps>140 else "#f59e0b" if trestbps>120 else "#00d4aa",">140 = high")
stat_card(s2,"🧪","Cholesterol",f"{chol} mg/dL",
          "#ef4444" if chol>240 else "#f59e0b" if chol>200 else "#00d4aa",">240 = high")
stat_card(s3,"💓","Max Heart Rate",f"{thalach} bpm",
          "#ef4444" if thalach>180 else "#00d4aa" if thalach>100 else "#f59e0b")
stat_card(s4,"⚡","Chest Pain",{0:"Typical",1:"Atypical",2:"Non-Anginal",3:"Asymptomatic"}[cp],
          "#ef4444" if cp==3 else "#f59e0b" if cp==2 else "#00d4aa")

st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)


# ── Result + Chart ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1.5], gap="large")

with col_l:
    st.markdown("""<div style='font-family:Syne,sans-serif;font-size:0.72rem;font-weight:600;
        letter-spacing:0.12em;text-transform:uppercase;color:#475569;margin-bottom:1rem;'>
        Prediction Result</div>""", unsafe_allow_html=True)

    if st.session_state.get("predicted"):
        pred  = st.session_state["pred"]
        proba = st.session_state["proba"]
        pp    = proba[1] * 100
        pos   = pred == 1
        rc    = "#ef4444" if pos else "#00d4aa"
        bg    = "linear-gradient(90deg,#ef4444,#dc2626)" if pos else "linear-gradient(90deg,#00d4aa,#0099cc)"
        glow  = "rgba(239,68,68,0.4)" if pos else "rgba(0,212,170,0.4)"
        lbl   = "HEART DISEASE DETECTED" if pos else "NO HEART DISEASE"

        st.markdown(f"""
        <div style='background:rgba(13,21,38,0.82);border:1px solid #1e2d4a;border-radius:16px;
                    padding:2rem;text-align:center;margin-bottom:1rem;backdrop-filter:blur(12px);'>
            <svg width="150" height="150" viewBox="0 0 160 160" style="margin-bottom:0.8rem;">
                <circle cx="80" cy="80" r="65" fill="none" stroke="#1e2d4a" stroke-width="12"/>
                <circle cx="80" cy="80" r="65" fill="none" stroke="{rc}" stroke-width="12"
                    stroke-dasharray="{408.4*proba[1]:.1f} 408.4" stroke-linecap="round"
                    transform="rotate(-90 80 80)" style="filter:drop-shadow(0 0 6px {rc});"/>
                <text x="80" y="72" text-anchor="middle"
                    style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;fill:{rc};">
                    {pp:.0f}%</text>
                <text x="80" y="92" text-anchor="middle"
                    style="font-size:10px;fill:#475569;letter-spacing:1px;">PROBABILITY</text>
            </svg>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                        color:{rc};letter-spacing:0.04em;margin-bottom:0.3rem;'>{lbl}</div>
            <div style='font-size:0.76rem;color:#475569;margin-bottom:1.2rem;'>
                Confidence: {max(proba)*100:.1f}%</div>
            <div style='background:#131f35;border-radius:999px;height:8px;overflow:hidden;'>
                <div style='width:{pp:.1f}%;height:100%;border-radius:999px;
                            background:{bg};box-shadow:0 0 10px {glow};'></div>
            </div>
            <div style='display:flex;justify-content:space-between;
                        font-size:0.63rem;color:#334155;margin-top:4px;'>
                <span>0% — Healthy</span><span>100% — Disease</span></div>
        </div>
        <div style='background:rgba(13,21,38,0.82);border:1px solid #1e2d4a;border-radius:14px;
                    padding:1rem 1.2rem;backdrop-filter:blur(12px);'>
            <div style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
                        color:#475569;margin-bottom:0.8rem;'>Probability Breakdown</div>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;'>
                <span style='font-size:0.82rem;color:#00d4aa;'>🟢 No Disease</span>
                <span style='font-family:DM Mono,monospace;color:#00d4aa;font-weight:600;'>
                    {proba[0]*100:.1f}%</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-size:0.82rem;color:#ef4444;'>🔴 Heart Disease</span>
                <span style='font-family:DM Mono,monospace;color:#ef4444;font-weight:600;'>
                    {proba[1]*100:.1f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(13,21,38,0.75);border:1px dashed #1e2d4a;border-radius:16px;
                    padding:3.5rem 2rem;text-align:center;backdrop-filter:blur(12px);'>
            <svg width="52" height="48" viewBox="0 0 30 28" fill="none" style="opacity:0.15;margin-bottom:1rem;">
                <path d="M15 26S2 17.5 2 8.5C2 5 4.5 2 8 2c2.5 0 4.8 1.4 6.2 3.5L15 7l0.8-1.5C17.2 3.4 19.5 2 22 2c3.5 0 6 3 6 6.5C28 17.5 15 26 15 26Z" fill="#00d4aa"/>
            </svg>
            <div style='font-size:0.85rem;color:#334155;'>
                Open the panel above ↑<br>fill in patient data<br>
                then click <strong style="color:#475569;">⚡ Run Prediction</strong>
            </div>
        </div>""", unsafe_allow_html=True)

with col_r:
    st.markdown("""<div style='font-family:Syne,sans-serif;font-size:0.72rem;font-weight:600;
        letter-spacing:0.12em;text-transform:uppercase;color:#475569;margin-bottom:1rem;'>
        Feature Importance</div>""", unsafe_allow_html=True)

    st.markdown("<div style='background:rgba(13,21,38,0.82);border:1px solid #1e2d4a;border-radius:16px;padding:1.4rem;backdrop-filter:blur(12px);'>", unsafe_allow_html=True)

    fn  = ["Chest Pain","Rest. BP","Cholesterol","Fasting Sugar","Rest. ECG","Max HR","Slope"]
    imp = model.feature_importances_
    idx = np.argsort(imp)
    clrs = ["#00d4aa" if i==idx[-1] else "#0d6e8a" if i in idx[-3:] else "#1e3a5f"
            for i in range(len(imp))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0d1526"); fig.patch.set_alpha(0.0)
    ax.set_facecolor("#0d1526"); ax.patch.set_alpha(0.0)
    ax.barh([fn[i] for i in idx], imp[idx], color=[clrs[i] for i in idx],
            edgecolor="none", height=0.55)
    for i, v in enumerate(imp[idx]):
        ax.text(v+0.003, i, f"{v:.3f}", va="center", fontsize=7.5, color="#64748b")
    ax.grid(axis="x", color="#1e2d4a", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Importance Score", fontsize=8, color="#475569", labelpad=8)
    ax.set_title("Random Forest Feature Importance", fontsize=9.5,
                 fontweight="600", color="#94a3b8", pad=10, fontfamily="Syne")
    ax.tick_params(axis="y", labelsize=8.5, colors="#94a3b8", length=0)
    ax.tick_params(axis="x", labelsize=7.5, colors="#475569", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Made by footer ───────────────────────────────────────────────────────────────
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:1.2rem 2rem;
            background:rgba(13,21,38,0.75);border:1px solid #1e2d4a;
            border-radius:14px;backdrop-filter:blur(12px);'>
    <div style='font-size:0.68rem;letter-spacing:0.15em;text-transform:uppercase;
                color:#475569;margin-bottom:0.5rem;'>Made by</div>
    <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                background:linear-gradient(135deg,#00d4aa,#0099cc);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        Fouad Ghadi &nbsp;·&nbsp; Yassine Ait Bella &nbsp;·&nbsp; Rabi Ilyas &nbsp;·&nbsp; Yahiaoui Ziyad &nbsp;·&nbsp; Chakir Mohamed
    </div>
    <div style='font-size:0.72rem;color:#334155;margin-top:0.4rem;'>
        Centrale Casablanca · Coding Week · March 2026
    </div>
</div>
""", unsafe_allow_html=True)
