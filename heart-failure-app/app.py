import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS + Hamburger HTML ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #080e1a !important;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1300px !important; }

.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099cc) !important;
    color: #080e1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,212,170,0.35) !important; }
.streamlit-expanderHeader { background: #0d1526 !important; border: 1px solid #1e2d4a !important; border-radius: 10px !important; color: #94a3b8 !important; }
.streamlit-expanderContent { background: #0d1526 !important; border: 1px solid #1e2d4a !important; border-top: none !important; }

/* ── Hamburger button ── */
#hg-btn {
    position: fixed; top: 14px; left: 14px; z-index: 9999;
    width: 44px; height: 44px;
    background: #0d1526; border: 1px solid #1e2d4a; border-radius: 10px;
    cursor: pointer; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 5px;
    transition: background 0.2s, box-shadow 0.2s;
}
#hg-btn:hover { background: #1e2d4a; box-shadow: 0 0 14px rgba(0,212,170,0.3); }
#hg-btn span {
    display: block; width: 22px; height: 2px;
    background: #00d4aa; border-radius: 2px; transition: all 0.3s ease;
}
#hg-btn.open span:nth-child(1) { transform: translateY(7px) rotate(45deg); }
#hg-btn.open span:nth-child(2) { opacity: 0; transform: scaleX(0); }
#hg-btn.open span:nth-child(3) { transform: translateY(-7px) rotate(-45deg); }

/* ── Sidebar panel ── */
#hg-panel {
    position: fixed; top: 0; left: 0;
    width: 300px; height: 100vh;
    background: #0d1526; border-right: 1px solid #1e2d4a;
    z-index: 9998; transform: translateX(-100%);
    transition: transform 0.35s cubic-bezier(.4,0,.2,1);
    overflow-y: auto; padding: 4.5rem 1.4rem 2rem;
}
#hg-panel.open { transform: translateX(0); }

#hg-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.55); z-index: 9997;
    backdrop-filter: blur(2px);
}
#hg-overlay.open { display: block; }

/* inputs inside panel */
.hg-label { font-size: 0.73rem; color: #94a3b8; margin-bottom: 4px; display: block; }
.hg-input {
    width: 100%; background: #131f35; border: 1px solid #1e3a5f;
    color: #e2e8f0; border-radius: 8px; padding: 0.48rem 0.7rem;
    margin-bottom: 12px; font-size: 0.83rem; font-family: 'DM Sans', sans-serif;
    outline: none; transition: border-color 0.2s;
}
.hg-input:focus { border-color: #00d4aa; }
.hg-divider { border: none; border-top: 1px solid #1e2d4a; margin: 0.8rem 0 1rem; }
.hg-section { font-size: 0.67rem; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; margin-bottom: 0.7rem; }
.hg-run-btn {
    width: 100%; padding: 0.7rem; margin-top: 0.5rem;
    background: linear-gradient(135deg,#00d4aa,#0099cc);
    color: #080e1a; font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 0.88rem; letter-spacing: 0.05em; text-transform: uppercase;
    border: none; border-radius: 10px; cursor: pointer; transition: all 0.2s;
}
.hg-run-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,212,170,0.35); }
.hg-disclaimer {
    margin-top: 1.5rem; padding: 0.8rem; background: #0a1220;
    border-radius: 8px; border: 1px solid #1e2d4a;
    font-size: 0.7rem; color: #475569; line-height: 1.6;
}

@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
</style>

<!-- Hamburger button -->
<div id="hg-btn" onclick="hgToggle()">
    <span></span><span></span><span></span>
</div>

<!-- Overlay -->
<div id="hg-overlay" onclick="hgClose()"></div>

<!-- Side panel -->
<div id="hg-panel">
    <div style="font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;
                background:linear-gradient(135deg,#00d4aa,#0099cc);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:2px;">
        🫀 Patient Input
    </div>
    <div style="font-size:0.67rem;color:#475569;letter-spacing:0.1em;
                text-transform:uppercase;margin-bottom:1rem;">Fill all fields · then predict</div>
    <hr class="hg-divider">

    <div class="hg-section">📊 Continuous Parameters</div>

    <label class="hg-label">Resting Blood Pressure (mmHg)</label>
    <input class="hg-input" type="number" id="i-trestbps" value="120" min="80" max="220">

    <label class="hg-label">Cholesterol (mg/dL)</label>
    <input class="hg-input" type="number" id="i-chol" value="240" min="100" max="600">

    <label class="hg-label">Max Heart Rate Achieved</label>
    <input class="hg-input" type="number" id="i-thalach" value="150" min="60" max="220">

    <hr class="hg-divider">
    <div class="hg-section">🔘 Categorical Parameters</div>

    <label class="hg-label">Chest Pain Type (cp)</label>
    <select class="hg-input" id="i-cp">
        <option value="0">0 — Typical Angina</option>
        <option value="1">1 — Atypical Angina</option>
        <option value="2">2 — Non-Anginal Pain</option>
        <option value="3">3 — Asymptomatic</option>
    </select>

    <label class="hg-label">Fasting Blood Sugar &gt; 120 mg/dL</label>
    <select class="hg-input" id="i-fbs">
        <option value="0">0 — No</option>
        <option value="1">1 — Yes</option>
    </select>

    <label class="hg-label">Resting ECG Result</label>
    <select class="hg-input" id="i-restecg">
        <option value="0">0 — Normal</option>
        <option value="1">1 — ST-T Wave Abnormality</option>
        <option value="2">2 — Left Ventricular Hypertrophy</option>
    </select>

    <label class="hg-label">Slope of ST Segment</label>
    <select class="hg-input" id="i-slope">
        <option value="0">0 — Upsloping</option>
        <option value="1">1 — Flat</option>
        <option value="2">2 — Downsloping</option>
    </select>

    <button class="hg-run-btn" onclick="hgSubmit()">⚡ Run Prediction</button>

    <div class="hg-disclaimer">
        ⚕️ For clinical decision support only.<br>Not a substitute for medical judgement.
    </div>
</div>

<script>
function hgToggle() {
    document.getElementById('hg-btn').classList.toggle('open');
    document.getElementById('hg-panel').classList.toggle('open');
    document.getElementById('hg-overlay').classList.toggle('open');
}
function hgClose() {
    ['hg-btn','hg-panel','hg-overlay'].forEach(id =>
        document.getElementById(id).classList.remove('open'));
}
function hgSubmit() {
    const p = new URLSearchParams({
        predict: '1',
        cp:       document.getElementById('i-cp').value,
        trestbps: document.getElementById('i-trestbps').value,
        chol:     document.getElementById('i-chol').value,
        fbs:      document.getElementById('i-fbs').value,
        restecg:  document.getElementById('i-restecg').value,
        thalach:  document.getElementById('i-thalach').value,
        slope:    document.getElementById('i-slope').value,
    });
    window.location.search = p.toString();
}
// Restore values from URL on load
window.addEventListener('load', () => {
    const p = new URLSearchParams(window.location.search);
    const map = {cp:'i-cp', trestbps:'i-trestbps', chol:'i-chol',
                 fbs:'i-fbs', restecg:'i-restecg', thalach:'i-thalach', slope:'i-slope'};
    Object.entries(map).forEach(([key, id]) => {
        if (p.has(key)) document.getElementById(id).value = p.get(key);
    });
});
</script>
""", unsafe_allow_html=True)


# ── Read URL params ─────────────────────────────────────────────────────────────
params     = st.query_params
do_predict = params.get("predict", "0") == "1"
cp         = int(params.get("cp",       0))
trestbps   = int(params.get("trestbps", 120))
chol       = int(params.get("chol",     240))
fbs        = int(params.get("fbs",      0))
restecg    = int(params.get("restecg",  0))
thalach    = int(params.get("thalach",  150))
slope      = int(params.get("slope",    0))


# ── Train model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "model", "heart5.xls")
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        np.random.seed(42); n = 303
        data = pd.DataFrame({
            "cp":np.random.randint(0,4,n),"trestbps":np.random.randint(94,200,n),
            "chol":np.random.randint(126,564,n),"fbs":np.random.randint(0,2,n),
            "restecg":np.random.randint(0,3,n),"thalach":np.random.randint(71,202,n),
            "slope":np.random.randint(0,3,n),"target":np.random.randint(0,2,n),
        })
    X = data.drop("target",axis=1)[["cp","trestbps","chol","fbs","restecg","thalach","slope"]]
    y = data["target"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(Xtr, ytr)
    return clf, sc, round(accuracy_score(yte, clf.predict(Xte))*100, 1)

model, scaler, accuracy = train_model()

def predict(cp,trestbps,chol,fbs,restecg,thalach,slope):
    X = scaler.transform([[cp,trestbps,chol,fbs,restecg,thalach,slope]])
    return int(model.predict(X)[0]), model.predict_proba(X)[0]


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;margin:0 0 1.6rem 3.8rem;'>
    <div>
        <div style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;
                    color:#f1f5f9;letter-spacing:-0.02em;'>Heart Disease Predictor</div>
        <div style='font-size:0.82rem;color:#475569;margin-top:3px;'>
            Random Forest Classifier · 7 Clinical Features</div>
    </div>
    <div style='text-align:right;'>
        <div style='font-family:DM Mono,monospace;font-size:1.5rem;font-weight:700;color:#00d4aa;'>{accuracy}%</div>
        <div style='font-size:0.68rem;color:#475569;letter-spacing:0.08em;text-transform:uppercase;'>Model Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not do_predict:
    st.markdown("""
    <div style='margin:0 0 1.2rem 3.8rem;background:#0d1a26;border:1px solid #1e3a5f;
                border-radius:10px;padding:0.65rem 1rem;font-size:0.82rem;color:#00d4aa;'>
        ☰ &nbsp; Click the <strong>three lines</strong> (top-left) to open the patient panel
    </div>""", unsafe_allow_html=True)


# ── Stat cards ──────────────────────────────────────────────────────────────────
def stat_card(col, icon, label, value, color="#94a3b8", sub=None):
    col.markdown(f"""
    <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:14px;padding:1.1rem 1.3rem;'>
        <div style='font-size:0.67rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#475569;margin-bottom:0.3rem;'>{icon} {label}</div>
        <div style='font-family:Syne,sans-serif;font-size:1.35rem;font-weight:700;color:{color};'>{value}</div>
        {f'<div style="font-size:0.68rem;color:#334155;margin-top:2px;">{sub}</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
stat_card(c1,"🩸","Blood Pressure",f"{trestbps} mmHg",
          "#ef4444" if trestbps>140 else "#f59e0b" if trestbps>120 else "#00d4aa",">140 = high")
stat_card(c2,"🧪","Cholesterol",f"{chol} mg/dL",
          "#ef4444" if chol>240 else "#f59e0b" if chol>200 else "#00d4aa",">240 = high")
stat_card(c3,"💓","Max Heart Rate",f"{thalach} bpm",
          "#ef4444" if thalach>180 else "#00d4aa" if thalach>100 else "#f59e0b")
stat_card(c4,"⚡","Chest Pain",
          {0:"Typical",1:"Atypical",2:"Non-Anginal",3:"Asymptomatic"}[cp],
          "#ef4444" if cp==3 else "#f59e0b" if cp==2 else "#00d4aa")

st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)


# ── Result + Chart ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1.5], gap="large")

with col_l:
    st.markdown("""<div style='font-family:Syne,sans-serif;font-size:0.72rem;font-weight:600;
        letter-spacing:0.12em;text-transform:uppercase;color:#475569;margin-bottom:1rem;'>
        Prediction Result</div>""", unsafe_allow_html=True)

    if do_predict:
        pred, proba = predict(cp,trestbps,chol,fbs,restecg,thalach,slope)
        pp   = proba[1]*100
        pos  = pred==1
        rc   = "#ef4444" if pos else "#00d4aa"
        bg   = "linear-gradient(90deg,#ef4444,#dc2626)" if pos else "linear-gradient(90deg,#00d4aa,#0099cc)"
        glow = "rgba(239,68,68,0.4)" if pos else "rgba(0,212,170,0.4)"
        lbl  = "HEART DISEASE DETECTED" if pos else "NO HEART DISEASE"

        st.markdown(f"""
        <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:16px;
                    padding:2rem;text-align:center;margin-bottom:1rem;'>
            <svg width="150" height="150" viewBox="0 0 160 160" style="margin-bottom:0.8rem;">
                <circle cx="80" cy="80" r="65" fill="none" stroke="#1e2d4a" stroke-width="12"/>
                <circle cx="80" cy="80" r="65" fill="none" stroke="{rc}" stroke-width="12"
                    stroke-dasharray="{408.4*proba[1]:.1f} 408.4" stroke-linecap="round"
                    transform="rotate(-90 80 80)" style="filter:drop-shadow(0 0 6px {rc});"/>
                <text x="80" y="72" text-anchor="middle"
                    style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;fill:{rc};">
                    {pp:.0f}%</text>
                <text x="80" y="92" text-anchor="middle"
                    style="font-family:DM Sans,sans-serif;font-size:10px;fill:#475569;letter-spacing:1px;">
                    PROBABILITY</text>
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
        <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:14px;padding:1rem 1.2rem;'>
            <div style='font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
                        color:#475569;margin-bottom:0.8rem;'>Probability Breakdown</div>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;'>
                <span style='font-size:0.82rem;color:#00d4aa;'>🟢 No Disease</span>
                <span style='font-family:DM Mono,monospace;font-size:0.9rem;
                             color:#00d4aa;font-weight:600;'>{proba[0]*100:.1f}%</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-size:0.82rem;color:#ef4444;'>🔴 Heart Disease</span>
                <span style='font-family:DM Mono,monospace;font-size:0.9rem;
                             color:#ef4444;font-weight:600;'>{proba[1]*100:.1f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0d1526;border:1px dashed #1e2d4a;border-radius:16px;
                    padding:3.5rem 2rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:1rem;opacity:0.2;'>🫀</div>
            <div style='font-size:0.85rem;color:#334155;'>
                Click ☰ top-left · fill data<br>
                then hit <strong style="color:#475569;">⚡ Run Prediction</strong>
            </div>
        </div>""", unsafe_allow_html=True)

with col_r:
    st.markdown("""<div style='font-family:Syne,sans-serif;font-size:0.72rem;font-weight:600;
        letter-spacing:0.12em;text-transform:uppercase;color:#475569;margin-bottom:1rem;'>
        Feature Importance</div>""", unsafe_allow_html=True)

    st.markdown("<div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:16px;padding:1.4rem;'>", unsafe_allow_html=True)

    fn  = ["Chest Pain","Rest. BP","Cholesterol","Fasting Sugar","Rest. ECG","Max HR","Slope"]
    imp = model.feature_importances_
    idx = np.argsort(imp)
    clrs= ["#00d4aa" if i==idx[-1] else "#0d6e8a" if i in idx[-3:] else "#1e3a5f" for i in range(len(imp))]

    fig,ax = plt.subplots(figsize=(7,4.5))
    fig.patch.set_facecolor("#0d1526"); ax.set_facecolor("#0d1526")
    ax.barh([fn[i] for i in idx], imp[idx], color=[clrs[i] for i in idx], edgecolor="none", height=0.55)
    for i,v in enumerate(imp[idx]):
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

st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
with st.expander("ℹ️  Model Details"):
    d1,d2 = st.columns(2)
    with d1:
        st.markdown(f"**Model:** Random Forest  \n**Accuracy:** `{accuracy}%`  \n**Split:** 80/20  \n**Scaler:** StandardScaler")
    with d2:
        st.markdown("**Fields:** `cp` · `trestbps` · `chol` · `fbs` · `restecg` · `thalach` · `slope`")
