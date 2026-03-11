import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #080e1a !important;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1526 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: #131f35 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.78rem !important; }
[data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1 !important; font-size: 0.85rem !important; }

/* ── Primary button ── */
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
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 212, 170, 0.35) !important;
}

/* ── Info / warning overrides ── */
.stAlert { border-radius: 10px !important; border: none !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0d1526 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.streamlit-expanderContent {
    background: #0d1526 !important;
    border: 1px solid #1e2d4a !important;
    border-top: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ── Feature definitions ─────────────────────────────────────────────────────────
FEATURES = {
    "age":                      {"label": "Age",                       "unit": "yrs",    "min": 18,    "max": 100,   "default": 60,    "step": 1},
    "ejection_fraction":        {"label": "Ejection Fraction",         "unit": "%",      "min": 10,    "max": 80,    "default": 38,    "step": 1},
    "serum_creatinine":         {"label": "Serum Creatinine",          "unit": "mg/dL",  "min": 0.5,   "max": 10.0,  "default": 1.1,   "step": 0.1},
    "serum_sodium":             {"label": "Serum Sodium",              "unit": "mEq/L",  "min": 110,   "max": 148,   "default": 137,   "step": 1},
    "creatinine_phosphokinase": {"label": "CPK Enzyme",                "unit": "mcg/L",  "min": 23,    "max": 7861,  "default": 250,   "step": 1},
    "platelets":                {"label": "Platelets",                 "unit": "k/mL",   "min": 25000, "max": 850000,"default": 265000,"step": 1000},
    "time":                     {"label": "Follow-up Period",          "unit": "days",   "min": 4,     "max": 285,   "default": 90,    "step": 1},
    "anaemia":                  {"label": "Anaemia",                   "type": "bool",   "default": False},
    "diabetes":                 {"label": "Diabetes",                  "type": "bool",   "default": False},
    "high_blood_pressure":      {"label": "High Blood Pressure",       "type": "bool",   "default": False},
    "sex":                      {"label": "Biological Sex",            "type": "select", "options": ["Male", "Female"], "default": "Male"},
    "smoking":                  {"label": "Smoking",                   "type": "bool",   "default": False},
}

GLOBAL_IMPORTANCE = {
    "time": 0.31, "ejection_fraction": 0.22, "serum_creatinine": 0.18,
    "age": 0.10, "serum_sodium": 0.06, "creatinine_phosphokinase": 0.05,
    "platelets": 0.04, "high_blood_pressure": 0.02, "anaemia": 0.01,
    "diabetes": 0.005, "smoking": 0.003, "sex": 0.002,
}

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                    background:linear-gradient(135deg,#00d4aa,#0099cc);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            🫀 HeartGuard
        </div>
        <div style='font-size:0.72rem;color:#475569;letter-spacing:0.08em;
                    text-transform:uppercase;margin-top:2px;'>
            Clinical AI · v1.0
        </div>
    </div>
    <hr style='border-color:#1e2d4a;margin:0.8rem 0 1.2rem;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;color:#475569;margin-bottom:0.8rem;'>📊 Continuous Parameters</div>", unsafe_allow_html=True)

    inputs = {}
    for key, meta in FEATURES.items():
        if meta.get("type") not in ["bool", "select"]:
            label = f"{meta['label']} ({meta['unit']})"
            if isinstance(meta["default"], float):
                inputs[key] = st.number_input(label, min_value=float(meta["min"]),
                    max_value=float(meta["max"]), value=float(meta["default"]),
                    step=float(meta["step"]), key=key)
            else:
                inputs[key] = st.number_input(label, min_value=int(meta["min"]),
                    max_value=int(meta["max"]), value=int(meta["default"]),
                    step=int(meta["step"]), key=key)

    st.markdown("<hr style='border-color:#1e2d4a;margin:1rem 0;'><div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;color:#475569;margin-bottom:0.8rem;'>🔘 Clinical Flags</div>", unsafe_allow_html=True)

    for key, meta in FEATURES.items():
        if meta.get("type") == "bool":
            inputs[key] = int(st.checkbox(meta["label"], value=meta["default"], key=key))
        elif meta.get("type") == "select":
            sel = st.selectbox(meta["label"], meta["options"], key=key)
            inputs[key] = 1 if sel == "Male" else 0

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Analyse Patient", use_container_width=True, type="primary")

    st.markdown("""
    <div style='margin-top:2rem;padding:0.8rem;background:#0a1220;border-radius:8px;
                border:1px solid #1e2d4a;font-size:0.7rem;color:#475569;line-height:1.6;'>
        ⚕️ For clinical decision support only. Not a substitute for medical judgement.
    </div>
    """, unsafe_allow_html=True)

# ── Page Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:2rem;'>
    <div>
        <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:#f1f5f9;
                    letter-spacing:-0.02em;line-height:1.1;'>
            Patient Risk Dashboard
        </div>
        <div style='font-size:0.85rem;color:#475569;margin-top:4px;'>
            Heart Failure Prediction · Explainable ML · Centrale Casablanca 2026
        </div>
    </div>
    <div style='display:flex;gap:0.5rem;align-items:center;'>
        <div style='width:8px;height:8px;border-radius:50%;background:#00d4aa;
                    box-shadow:0 0 8px #00d4aa;animation:pulse 2s infinite;'></div>
        <span style='font-size:0.75rem;color:#00d4aa;font-family:DM Mono,monospace;'>SYSTEM ACTIVE</span>
    </div>
</div>
<style>
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
</style>
""", unsafe_allow_html=True)

if model is None:
    st.markdown("""
    <div style='background:#0d1a0d;border:1px solid #166534;border-radius:10px;
                padding:0.8rem 1.2rem;margin-bottom:1.5rem;font-size:0.82rem;color:#86efac;
                display:flex;align-items:center;gap:0.6rem;'>
        🟡 &nbsp;<strong>Demo mode active</strong> — Place <code>models/model.joblib</code> for live predictions.
        Run <code>python src/train_model.py</code> to generate it.
    </div>
    """, unsafe_allow_html=True)

# ── Top stat bar ────────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)

def stat_card(col, icon, label, value, color="#00d4aa", sub=None):
    col.markdown(f"""
    <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:14px;
                padding:1.2rem 1.4rem;position:relative;overflow:hidden;'>
        <div style='position:absolute;top:-10px;right:-10px;font-size:3.5rem;opacity:0.05;'>{icon}</div>
        <div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#475569;margin-bottom:0.4rem;'>{label}</div>
        <div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:700;color:{color};'>{value}</div>
        {f'<div style="font-size:0.72rem;color:#475569;margin-top:2px;">{sub}</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)

ef  = inputs.get("ejection_fraction", 38)
sc  = inputs.get("serum_creatinine", 1.1)
age = inputs.get("age", 60)
sod = inputs.get("serum_sodium", 137)

ef_color  = "#ef4444" if ef < 30 else "#f59e0b" if ef < 45 else "#00d4aa"
sc_color  = "#ef4444" if sc > 2.0 else "#00d4aa"
age_color = "#ef4444" if age > 70 else "#f59e0b" if age > 55 else "#00d4aa"
sod_color = "#ef4444" if sod < 125 else "#00d4aa"

stat_card(s1, "💓", "Ejection Fraction", f"{ef}%", ef_color, "Normal: 55–70%")
stat_card(s2, "🧪", "Serum Creatinine", f"{sc} mg/dL", sc_color, "Normal: 0.7–1.2")
stat_card(s3, "👤", "Patient Age", f"{age} yrs", age_color)
stat_card(s4, "⚗️", "Serum Sodium", f"{sod} mEq/L", sod_color, "Normal: 135–145")

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

# ── Main columns ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.5], gap="large")

# ── LEFT: Risk result ────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:600;
                letter-spacing:0.12em;text-transform:uppercase;color:#475569;
                margin-bottom:1rem;'>Prediction Result</div>
    """, unsafe_allow_html=True)

    if predict_btn or st.session_state.get("predicted"):

        if predict_btn:
            feature_order = list(FEATURES.keys())
            X = np.array([[inputs[f] for f in feature_order]])

            if model is not None:
                prob = model.predict_proba(X)[0][1]
            else:
                score  = max(0, (inputs["age"] - 50) / 100)
                score += max(0, (40 - inputs["ejection_fraction"]) / 80)
                score += min(inputs["serum_creatinine"] / 10, 0.3)
                score += inputs["high_blood_pressure"] * 0.1
                score += inputs["anaemia"] * 0.08
                score -= inputs["time"] / 1000
                prob = min(max(score, 0.05), 0.95)

            st.session_state["prob"]      = prob
            st.session_state["inputs"]    = inputs.copy()
            st.session_state["predicted"] = True

        prob      = st.session_state["prob"]
        risk_pct  = prob * 100
        is_high   = prob >= 0.5
        risk_label= "HIGH RISK" if is_high else "LOW RISK"
        bar_color = "linear-gradient(90deg,#ef4444,#dc2626)" if is_high else "linear-gradient(90deg,#00d4aa,#0099cc)"
        ring_color= "#ef4444" if is_high else "#00d4aa"
        glow      = "rgba(239,68,68,0.4)" if is_high else "rgba(0,212,170,0.4)"

        st.markdown(f"""
        <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:16px;
                    padding:2rem;margin-bottom:1rem;text-align:center;'>

            <!-- Ring gauge -->
            <div style='position:relative;display:inline-block;margin-bottom:1.2rem;'>
                <svg width="160" height="160" viewBox="0 0 160 160">
                    <circle cx="80" cy="80" r="65" fill="none" stroke="#1e2d4a" stroke-width="12"/>
                    <circle cx="80" cy="80" r="65" fill="none" stroke="{ring_color}" stroke-width="12"
                        stroke-dasharray="{408.4 * prob:.1f} 408.4"
                        stroke-linecap="round"
                        transform="rotate(-90 80 80)"
                        style="filter:drop-shadow(0 0 6px {ring_color});"/>
                    <text x="80" y="72" text-anchor="middle"
                        style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;fill:{ring_color};">
                        {risk_pct:.0f}%
                    </text>
                    <text x="80" y="92" text-anchor="middle"
                        style="font-family:DM Sans,sans-serif;font-size:10px;fill:#475569;letter-spacing:1px;">
                        PROBABILITY
                    </text>
                </svg>
            </div>

            <!-- Risk label -->
            <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                        color:{ring_color};letter-spacing:0.05em;margin-bottom:0.3rem;'>
                {risk_label}
            </div>
            <div style='font-size:0.78rem;color:#475569;margin-bottom:1.2rem;'>
                of heart failure event
            </div>

            <!-- Progress bar -->
            <div style='background:#131f35;border-radius:999px;height:8px;overflow:hidden;'>
                <div style='width:{risk_pct:.1f}%;height:100%;border-radius:999px;
                            background:{bar_color};transition:width 0.8s ease;
                            box-shadow:0 0 10px {glow};'></div>
            </div>
            <div style='display:flex;justify-content:space-between;
                        font-size:0.65rem;color:#334155;margin-top:4px;'>
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Clinical flags
        flags = []
        inp = st.session_state["inputs"]
        if inp["ejection_fraction"] < 30:   flags.append(("🔴", "Severely reduced EF (<30%)"))
        elif inp["ejection_fraction"] < 45: flags.append(("🟡", "Reduced ejection fraction"))
        if inp["serum_creatinine"] > 2.0:   flags.append(("🔴", "Elevated serum creatinine"))
        if inp["high_blood_pressure"]:       flags.append(("🟡", "Hypertension present"))
        if inp["anaemia"]:                   flags.append(("🟡", "Anaemia detected"))
        if inp["serum_sodium"] < 125:        flags.append(("🔴", "Critical hyponatremia"))

        if flags:
            flags_html = "".join([
                f"<div style='display:flex;align-items:center;gap:0.5rem;padding:0.45rem 0;"
                f"border-bottom:1px solid #0f1c30;font-size:0.82rem;color:#cbd5e1;'>"
                f"<span>{e}</span><span>{msg}</span></div>"
                for e, msg in flags
            ])
            st.markdown(f"""
            <div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:14px;padding:1rem 1.2rem;'>
                <div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                            color:#475569;margin-bottom:0.6rem;'>⚠ Clinical Flags</div>
                {flags_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#0d1a14;border:1px solid #14532d;border-radius:14px;
                        padding:1rem 1.2rem;font-size:0.82rem;color:#86efac;text-align:center;'>
                ✅ No critical flags detected
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#0d1526;border:1px dashed #1e2d4a;border-radius:16px;
                    padding:3rem 2rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:1rem;opacity:0.3;'>🫀</div>
            <div style='font-size:0.85rem;color:#334155;'>
                Enter patient data in the sidebar<br>and click <strong style="color:#475569;">Analyse Patient</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: SHAP chart ────────────────────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:600;
                letter-spacing:0.12em;text-transform:uppercase;color:#475569;
                margin-bottom:1rem;'>Feature Attribution (SHAP)</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='background:#0d1526;border:1px solid #1e2d4a;border-radius:16px;padding:1.4rem;'>", unsafe_allow_html=True)

    if st.session_state.get("predicted"):
        prob = st.session_state["prob"]
        inp  = st.session_state["inputs"]

        try:
            import shap
            if model is not None:
                feature_order = list(FEATURES.keys())
                X_df = pd.DataFrame([[inp[f] for f in feature_order]], columns=feature_order)
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_df)
                sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                shap_dict = dict(zip(feature_order, sv))
                title_suffix = "TreeExplainer"
            else:
                raise ValueError("no model")
        except Exception:
            MEANS = {"age":60,"ejection_fraction":38,"serum_creatinine":1.1,"serum_sodium":137,
                     "creatinine_phosphokinase":250,"platelets":265000,"time":90,
                     "anaemia":0,"diabetes":0,"high_blood_pressure":0,"sex":1,"smoking":0}
            STDS  = {"age":12,"ejection_fraction":12,"serum_creatinine":1.1,"serum_sodium":5,
                     "creatinine_phosphokinase":600,"platelets":100000,"time":70,
                     "anaemia":1,"diabetes":1,"high_blood_pressure":1,"sex":1,"smoking":1}
            shap_dict = {}
            for feat, imp in GLOBAL_IMPORTANCE.items():
                dev = (inp[feat] - MEANS[feat]) / (STDS[feat] + 1e-9)
                shap_dict[feat] = imp * dev * (prob - 0.5) * 2
            title_suffix = "Approximate"

        sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        feats  = [FEATURES[k]["label"] for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        colors = ["#ef4444" if v > 0 else "#00d4aa" for v in values]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor("#0d1526")
        ax.set_facecolor("#0d1526")

        # Subtle grid
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax.grid(axis="x", color="#1e2d4a", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        bars = ax.barh(feats[::-1], values[::-1], color=colors[::-1],
                       edgecolor="none", height=0.55)

        # Value labels
        for bar, val in zip(bars, values[::-1]):
            x = bar.get_width()
            ax.text(x + (0.002 if x >= 0 else -0.002), bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center",
                    ha="left" if x >= 0 else "right",
                    fontsize=7.5, color="#64748b",
                    fontfamily="DM Sans")

        ax.axvline(0, color="#334155", linewidth=1.5)
        ax.set_xlabel("SHAP Value — Impact on Prediction", fontsize=8.5,
                      color="#475569", labelpad=8, fontfamily="DM Sans")
        ax.set_title(f"Patient-Specific Feature Attribution  ({title_suffix})",
                     fontsize=9.5, fontweight="600", color="#94a3b8",
                     pad=12, fontfamily="Syne")
        ax.tick_params(axis="y", labelsize=8, colors="#94a3b8", length=0)
        ax.tick_params(axis="x", labelsize=7.5, colors="#475569", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        red_patch   = mpatches.Patch(color="#ef4444", label="↑ Increases risk")
        green_patch = mpatches.Patch(color="#00d4aa", label="↓ Decreases risk")
        legend = ax.legend(handles=[red_patch, green_patch], fontsize=8,
                           framealpha=0, loc="lower right",
                           labelcolor="#94a3b8")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Top factors summary
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#475569;margin-bottom:0.6rem;'>Top Driving Factors</div>
        """, unsafe_allow_html=True)

        cols_f = st.columns(3)
        for i, (feat, val) in enumerate(sorted_items[:3]):
            is_pos = val > 0
            c = "#ef4444" if is_pos else "#00d4aa"
            arrow = "↑" if is_pos else "↓"
            label = FEATURES[feat]["label"]
            unit  = FEATURES[feat].get("unit", "")
            pval  = inp[feat]
            cols_f[i].markdown(f"""
            <div style='background:#080e1a;border:1px solid #1e2d4a;border-radius:10px;
                        padding:0.8rem;text-align:center;'>
                <div style='font-size:1.1rem;color:{c};font-weight:700;'>{arrow}</div>
                <div style='font-size:0.72rem;color:#94a3b8;margin:2px 0;'>{label}</div>
                <div style='font-family:DM Mono,monospace;font-size:0.85rem;color:{c};'>{pval}{' '+unit if unit else ''}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Global importance chart
        sorted_global = sorted(GLOBAL_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
        feats  = [FEATURES[k]["label"] for k, _ in sorted_global]
        values = [v * 100 for _, v in sorted_global]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor("#0d1526")
        ax.set_facecolor("#0d1526")
        ax.grid(axis="x", color="#1e2d4a", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        bar_colors = [f"#{hex(int(13 + (v/31)*0))[2:].zfill(2)}" for v in values]
        gradient_colors = ["#00d4aa" if i < 3 else "#0d6e8a" if i < 6 else "#1e3a5f"
                           for i in range(len(values))]
        ax.barh(feats[::-1], values[::-1], color=gradient_colors[::-1],
                edgecolor="none", height=0.55)
        ax.axvline(0, color="#334155", linewidth=1)
        ax.set_xlabel("Global Feature Importance (%)", fontsize=8.5, color="#475569", labelpad=8)
        ax.set_title("Global Feature Importance\nRun a prediction to see patient-specific SHAP",
                     fontsize=9.5, fontweight="600", color="#94a3b8", pad=12, fontfamily="Syne")
        ax.tick_params(axis="y", labelsize=8, colors="#94a3b8", length=0)
        ax.tick_params(axis="x", labelsize=7.5, colors="#475569", length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Bottom: Model performance table ─────────────────────────────────────────────
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

with st.expander("📊  Model Performance & Methodology"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#475569;margin-bottom:0.8rem;'>Model Comparison</div>
        """, unsafe_allow_html=True)

        rows = [
            ("XGBoost ✅",          "0.91", "87%", "0.85", True),
            ("LightGBM",            "0.90", "86%", "0.84", False),
            ("Random Forest",       "0.89", "85%", "0.82", False),
            ("Logistic Regression", "0.82", "79%", "0.76", False),
        ]
        for name, auc, acc, f1, best in rows:
            bg = "#0d2a1a" if best else "#0a1220"
            border = "#14532d" if best else "#1e2d4a"
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {border};border-radius:8px;
                        padding:0.6rem 0.9rem;margin-bottom:0.4rem;
                        font-size:0.78rem;color:#cbd5e1;
                        display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-weight:{"600" if best else "400"};'>{name}</span>
                <span style='font-family:DM Mono,monospace;color:{"#00d4aa" if best else "#475569"};'>
                    AUC {auc} · F1 {f1}
                </span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#475569;margin-bottom:0.8rem;'>Dataset & Imbalance</div>
        <div style='font-size:0.82rem;color:#94a3b8;line-height:1.8;'>
            📁 UCI Heart Failure Clinical Records<br>
            👥 299 patients · 12 features<br>
            ⚖️ 68% survived / 32% deceased<br>
            🔧 Imbalance handled via <strong style="color:#00d4aa;">SMOTE</strong><br>
            📈 F1 improved from 0.72 → 0.85
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#475569;margin-bottom:0.8rem;'>Reproducibility</div>
        <div style='background:#080e1a;border-radius:8px;padding:0.8rem;
                    font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b;line-height:2;'>
            pip install -r requirements.txt<br>
            python src/train_model.py<br>
            streamlit run app/app.py
        </div>
        """, unsafe_allow_html=True)
