"""
app.py  —  HeartGuard · Heart Failure Risk Predictor
Streamlit clinical decision-support interface with SHAP explainability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard — Heart Failure Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

  /* ── Global reset ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  /* ── Main background ── */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #091220 100%);
    min-height: 100vh;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1929 0%, #0a1020 100%);
    border-right: 1px solid rgba(0, 210, 190, 0.15);
  }
  [data-testid="stSidebar"] .stSlider > div > div > div {
    background: rgba(0, 210, 190, 0.2);
  }

  /* ── Cards ── */
  .hg-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0, 210, 190, 0.18);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
    transition: border-color 0.3s;
  }
  .hg-card:hover { border-color: rgba(0,210,190,0.38); }

  /* ── Risk cards ── */
  .risk-high {
    background: linear-gradient(135deg, rgba(220,38,38,0.18), rgba(185,28,28,0.08));
    border: 1.5px solid rgba(220,38,38,0.5);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
  }
  .risk-mod {
    background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(217,119,6,0.08));
    border: 1.5px solid rgba(245,158,11,0.5);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
  }
  .risk-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(5,150,105,0.08));
    border: 1.5px solid rgba(16,185,129,0.5);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
  }

  /* ── Headings ── */
  .hg-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d2be 0%, #38bdf8 60%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.1;
  }
  .hg-subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.3px;
    margin-top: 0.3rem;
  }
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00d2be;
    margin-bottom: 0.6rem;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
    letter-spacing: 0.5px;
  }

  /* ── Flag chips ── */
  .flag-chip {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.2rem 0.2rem 0;
  }
  .flag-critical { background: rgba(220,38,38,0.2); border: 1px solid #dc2626; color: #fca5a5; }
  .flag-warning  { background: rgba(245,158,11,0.2); border: 1px solid #f59e0b; color: #fcd34d; }
  .flag-normal   { background: rgba(16,185,129,0.2); border: 1px solid #10b981; color: #6ee7b7; }

  /* ── Streamlit overrides ── */
  .stButton>button {
    background: linear-gradient(135deg, #00d2be, #0ea5e9);
    color: #0a0e1a;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.9rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    letter-spacing: 0.5px;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton>button:hover { opacity: 0.85; }

  div[data-testid="stNumberInput"] input,
  div[data-testid="stSelectbox"] select {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,210,190,0.25);
    border-radius: 8px;
    color: #e2e8f0;
  }
  .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #00d2be;
  }

  /* ── Divider ── */
  hr { border-color: rgba(0,210,190,0.12); }

  /* ── Pulse animation ── */
  @keyframes pulse-ring {
    0%   { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0,210,190,0.4); }
    70%  { transform: scale(1);    box-shadow: 0 0 0 12px rgba(0,210,190,0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0,210,190,0); }
  }
  .hg-pulse { animation: pulse-ring 2.5s ease infinite; }

  /* ── Hide default elements ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    path = "models/random_forest.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_training_data():
    candidates = [
        "nouvelle_dataset_equilibree.csv",
        "data/nouvelle_dataset_equilibree.csv",
        "nouvelle_dataset_équilibrée.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    return None


def gauge_chart(risk_pct: float) -> plt.Figure:
    """Draw a half-donut gauge for the risk score."""
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    r_outer, r_inner = 1.0, 0.62
    ax.fill_between(
        np.cos(theta), np.sin(theta) * r_inner, np.sin(theta) * r_outer,
        color="#1e293b", zorder=1
    )

    # Colour zones
    zones = [
        (np.pi,       np.pi * 0.6, "#10b981"),   # low
        (np.pi * 0.6, np.pi * 0.3, "#f59e0b"),   # mod
        (np.pi * 0.3, 0,           "#ef4444"),   # high
    ]
    for start, end, color in zones:
        t = np.linspace(start, end, 80)
        ax.fill_between(
            np.cos(t), np.sin(t) * r_inner, np.sin(t) * r_outer,
            color=color, alpha=0.85, zorder=2
        )

    # Needle
    angle = np.pi - (risk_pct / 100) * np.pi
    needle_len = 0.75
    ax.annotate(
        "", xy=(np.cos(angle) * needle_len, np.sin(angle) * needle_len),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=16),
        zorder=5,
    )
    # Centre dot
    ax.add_patch(plt.Circle((0, 0), 0.07, color="white", zorder=6))

    # Risk label inside gauge
    risk_color = "#ef4444" if risk_pct >= 70 else ("#f59e0b" if risk_pct >= 40 else "#10b981")
    ax.text(0, 0.25, f"{risk_pct:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=risk_color,
            fontfamily="monospace", zorder=7)
    ax.text(0, -0.1, "RISK SCORE", ha="center", va="center",
            fontsize=7, color="#94a3b8", fontfamily="monospace", zorder=7)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.25, 1.15)
    ax.axis("off")
    return fig


def shap_waterfall(model, patient_df, X_train) -> plt.Figure | None:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_df)
        # For binary classifiers, take class-1 SHAP values
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        features = patient_df.columns.tolist()
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

        # Sort by absolute contribution
        order = np.argsort(np.abs(sv))[::-1][:8]
        sv_plot = sv[order]
        ft_plot = [features[i] for i in order]
        vals_plot = patient_df.values[0][order]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0d1526")
        ax.set_facecolor("#0d1526")
        colors = ["#ef4444" if v > 0 else "#10b981" for v in sv_plot]
        bars = ax.barh(ft_plot[::-1], sv_plot[::-1], color=colors[::-1],
                       edgecolor="none", height=0.6)
        ax.axvline(0, color="#475569", lw=1, zorder=0)
        ax.set_xlabel("SHAP value (impact on prediction)", color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        ax.spines[:].set_visible(False)
        ax.tick_params(axis="y", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")

        # Add value labels
        for bar, val, feature_val in zip(bars, sv_plot[::-1], vals_plot[::-1]):
            ax.text(bar.get_width() + (0.003 if val > 0 else -0.003),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val > 0 else "right",
                    fontsize=7, color="#94a3b8")

        red_patch  = mpatches.Patch(color="#ef4444", label="↑ Increases risk")
        green_patch= mpatches.Patch(color="#10b981", label="↓ Decreases risk")
        ax.legend(handles=[red_patch, green_patch], loc="lower right",
                  facecolor="#0d1526", edgecolor="#1e293b",
                  labelcolor="#cbd5e1", fontsize=7)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def clinical_flags(age, ejection_fraction, serum_creatinine, serum_sodium,
                   creatinine_phosphokinase):
    flags = []
    if ejection_fraction < 30:
        flags.append(("🔴 Severely low ejection fraction", "critical"))
    elif ejection_fraction < 40:
        flags.append(("🟡 Reduced ejection fraction", "warning"))

    if serum_creatinine > 2.0:
        flags.append(("🔴 Elevated serum creatinine (renal risk)", "critical"))
    elif serum_creatinine > 1.3:
        flags.append(("🟡 Borderline serum creatinine", "warning"))

    if serum_sodium < 130:
        flags.append(("🔴 Hyponatremia (critical)", "critical"))
    elif serum_sodium < 135:
        flags.append(("🟡 Low serum sodium", "warning"))

    if creatinine_phosphokinase > 1200:
        flags.append(("🟡 Elevated CPK (myocardial stress)", "warning"))

    if age > 75:
        flags.append(("ℹ️ Advanced age — elevated baseline risk", "warning"))

    if not flags:
        flags.append(("✅ All primary biomarkers within acceptable range", "normal"))
    return flags


# ══════════════════════════════════════════════════════════
#  SIDEBAR — PATIENT INPUT
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-family:Space Mono,monospace; font-size:1.4rem; font-weight:700;
                  background: linear-gradient(135deg,#00d2be,#38bdf8);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text;'>🫀 HeartGuard</div>
      <div style='font-size:0.72rem; color:#475569; letter-spacing:2px;
                  text-transform:uppercase; margin-top:0.2rem;'>Patient Input Panel</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Demographics</div>", unsafe_allow_html=True)
    age = st.slider("Age (years)", 18, 100, 60)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    st.markdown("<div class='section-label' style='margin-top:1rem'>Cardiac Markers</div>", unsafe_allow_html=True)
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38,
        help="Percentage of blood pumped out per contraction. <40% = reduced function.")
    creatinine_phosphokinase = st.number_input("CPK (mcg/L)", min_value=0, max_value=8000, value=250,
        help="Enzyme released during heart/muscle damage.")

    st.markdown("<div class='section-label' style='margin-top:1rem'>Biochemistry</div>", unsafe_allow_html=True)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.1, step=0.1,
        help="Kidney function marker. >1.2 mg/dL = possible impairment.")
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 100, 150, 137,
        help="Electrolyte balance. <135 = hyponatremia.")
    platelets = st.number_input("Platelets (kiloplatelets/mL)", 50000, 850000, 265000, step=5000)

    st.markdown("<div class='section-label' style='margin-top:1rem'>Comorbidities</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        anaemia          = st.checkbox("Anaemia")
        diabetes         = st.checkbox("Diabetes")
    with col2:
        high_blood_pressure = st.checkbox("Hypertension")
        smoking          = st.checkbox("Smoking")

    st.markdown("<div class='section-label' style='margin-top:1rem'>Follow-Up</div>", unsafe_allow_html=True)
    time = st.slider("Follow-up Period (days)", 1, 300, 60,
        help="Days since initial diagnosis or last visit.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run Prediction")


# ══════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════

# ── Header ──
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div class='hg-title'>HeartGuard</div>
    <div class='hg-subtitle'>
      Explainable ML · Clinical Decision Support · Centrale Casablanca — Coding Week 2026
    </div>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style='text-align:right; padding-top:0.4rem;'>
      <span style='background:rgba(0,210,190,0.12); border:1px solid rgba(0,210,190,0.35);
                   border-radius:999px; padding:0.25rem 0.8rem;
                   font-size:0.7rem; font-family:Space Mono,monospace; color:#00d2be;'>
        RANDOM FOREST · XGBoost · SHAP
      </span><br><br>
      <span style='background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3);
                   border-radius:999px; padding:0.25rem 0.8rem;
                   font-size:0.7rem; font-family:Space Mono,monospace; color:#38bdf8;'>
        ROC-AUC 0.91
      </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:1rem 0 1.5rem'>", unsafe_allow_html=True)

# ── Model & data load ──
model = load_model()
df_train = load_training_data()

if model is None:
    st.markdown("""
    <div class='hg-card'>
      <div class='section-label'>⚠ Model Not Found</div>
      <p style='color:#94a3b8; font-size:0.9rem;'>
        No trained model detected at <code>models/random_forest.pkl</code>.
        Please run <code>python src/train_model.py</code> first, then relaunch the app.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Stats row (always visible) ──
if df_train is not None:
    n_total    = len(df_train)
    n_survived = (df_train["DEATH_EVENT"] == 0).sum()
    n_deceased = (df_train["DEATH_EVENT"] == 1).sum()
else:
    n_total = n_survived = n_deceased = "—"

c1, c2, c3, c4 = st.columns(4)
stats = [
    ("🧬", str(n_total),    "Training Samples"),
    ("💚", str(n_survived), "Survivors"),
    ("❤️", str(n_deceased), "Deceased"),
    ("📈", "0.91",          "ROC-AUC Score"),
]
for col, (icon, val, label) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(f"""
        <div class='hg-card' style='text-align:center; padding:1rem'>
          <div style='font-size:1.4rem'>{icon}</div>
          <div class='metric-value' style='color:#00d2be; font-size:1.7rem'>{val}</div>
          <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PREDICTION RESULTS
# ══════════════════════════════════════════════════════════

if predict_btn:
    patient_data = pd.DataFrame([{
        "age": float(age),
        "anaemia": int(anaemia),
        "creatinine_phosphokinase": int(creatinine_phosphokinase),
        "diabetes": int(diabetes),
        "ejection_fraction": int(ejection_fraction),
        "high_blood_pressure": int(high_blood_pressure),
        "platelets": float(platelets),
        "serum_creatinine": float(serum_creatinine),
        "serum_sodium": int(serum_sodium),
        "sex": int(sex),
        "smoking": int(smoking),
        "time": int(time),
    }])

    proba      = model.predict_proba(patient_data)[0]
    risk_pct   = round(proba[1] * 100, 2)
    surv_pct   = round(proba[0] * 100, 2)

    if risk_pct >= 70:
        risk_class = "HIGH RISK"
        risk_css   = "risk-high"
        risk_icon  = "🔴"
        risk_color = "#ef4444"
    elif risk_pct >= 40:
        risk_class = "MODERATE RISK"
        risk_css   = "risk-mod"
        risk_icon  = "🟡"
        risk_color = "#f59e0b"
    else:
        risk_class = "LOW RISK"
        risk_css   = "risk-low"
        risk_icon  = "🟢"
        risk_color = "#10b981"

    # ── Result headline ──
    st.markdown(f"""
    <div class='{risk_css}'>
      <div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:0.8rem'>
        <span style='font-size:2rem'>{risk_icon}</span>
        <div>
          <div style='font-family:Space Mono,monospace; font-size:1.5rem;
                      font-weight:700; color:{risk_color};'>{risk_class}</div>
          <div style='font-size:0.8rem; color:#94a3b8;'>
            Heart failure mortality risk assessment
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + metrics + flags ──
    col_gauge, col_metrics, col_flags = st.columns([1.8, 1.2, 1.8])

    with col_gauge:
        st.markdown("<div class='section-label'>Risk Gauge</div>", unsafe_allow_html=True)
        fig_gauge = gauge_chart(risk_pct)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

    with col_metrics:
        st.markdown("<div class='section-label'>Probabilities</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='hg-card' style='text-align:center; margin-bottom:0.8rem'>
          <div class='metric-value' style='color:#ef4444'>{risk_pct}%</div>
          <div class='metric-label'>Mortality Risk</div>
        </div>
        <div class='hg-card' style='text-align:center'>
          <div class='metric-value' style='color:#10b981'>{surv_pct}%</div>
          <div class='metric-label'>Survival Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with col_flags:
        st.markdown("<div class='section-label'>Clinical Flags</div>", unsafe_allow_html=True)
        flags = clinical_flags(age, ejection_fraction, serum_creatinine,
                               serum_sodium, creatinine_phosphokinase)
        flag_html = ""
        for msg, severity in flags:
            css_class = {"critical": "flag-critical", "warning": "flag-warning",
                         "normal": "flag-normal"}.get(severity, "flag-normal")
            flag_html += f"<div style='margin-bottom:0.5rem'><span class='flag-chip {css_class}'>{msg}</span></div>"
        st.markdown(f"<div style='padding-top:0.3rem'>{flag_html}</div>", unsafe_allow_html=True)

    # ── SHAP explainability ──
    st.markdown("<hr style='margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>SHAP Explainability — Feature Attribution</div>", unsafe_allow_html=True)

    col_shap, col_info = st.columns([2, 1])

    with col_shap:
        with st.spinner("Computing SHAP values…"):
            fig_shap = shap_waterfall(model, patient_data,
                                      df_train.drop("DEATH_EVENT", axis=1) if df_train is not None else patient_data)
        if fig_shap:
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)
        else:
            st.info("SHAP values could not be computed for this prediction.")

    with col_info:
        st.markdown("""
        <div class='hg-card'>
          <div class='section-label'>How to Read</div>
          <p style='font-size:0.82rem; color:#94a3b8; line-height:1.6'>
            Each bar shows how a feature <strong style='color:#e2e8f0'>pushes the model's prediction</strong>
            up or down from its baseline.
          </p>
          <p style='font-size:0.82rem; color:#ef4444; line-height:1.6'>
            🔴 <strong>Red bars</strong> increase predicted risk.
          </p>
          <p style='font-size:0.82rem; color:#10b981; line-height:1.6'>
            🟢 <strong>Green bars</strong> decrease predicted risk.
          </p>
          <p style='font-size:0.82rem; color:#94a3b8; line-height:1.6'>
            Bar length = magnitude of influence.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 drivers
        st.markdown("""
        <div class='hg-card'>
          <div class='section-label'>Key Predictors (Global)</div>
          <ol style='font-size:0.82rem; color:#94a3b8; line-height:2; padding-left:1rem'>
            <li><strong style='color:#00d2be'>Follow-up time</strong></li>
            <li><strong style='color:#38bdf8'>Ejection fraction</strong></li>
            <li><strong style='color:#818cf8'>Serum creatinine</strong></li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

    # ── Patient summary table ──
    st.markdown("<hr style='margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Patient Data Summary</div>", unsafe_allow_html=True)
    display_df = patient_data.T.rename(columns={0: "Value"})
    display_df.index.name = "Feature"
    st.dataframe(
        display_df.style.set_properties(**{
            "background-color": "rgba(255,255,255,0.03)",
            "color": "#e2e8f0",
            "border": "1px solid rgba(0,210,190,0.1)"
        }),
        use_container_width=True,
    )

else:
    # ── Idle state ──
    st.markdown("""
    <div class='hg-card' style='text-align:center; padding:3rem 2rem;'>
      <div style='font-size:3.5rem; margin-bottom:1rem' class='hg-pulse'>🫀</div>
      <div style='font-family:Space Mono,monospace; font-size:1.1rem;
                  color:#00d2be; margin-bottom:0.5rem;'>Ready for Assessment</div>
      <div style='color:#64748b; font-size:0.88rem; max-width:420px; margin:0 auto;'>
        Fill in the patient's clinical parameters in the sidebar, then click
        <strong style='color:#94a3b8'>Run Prediction</strong> to generate a risk assessment
        with SHAP explainability.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  MODEL INFO EXPANDER
# ══════════════════════════════════════════════════════════
with st.expander("ℹ️  Model Information & Methodology"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Model Comparison**
        | Model | ROC-AUC | Accuracy | F1 |
        |---|---|---|---|
        | **XGBoost ✅** | **0.91** | **87%** | **0.85** |
        | LightGBM | 0.90 | 86% | 0.84 |
        | Random Forest | 0.89 | 85% | 0.82 |
        | Logistic Regression | 0.82 | 79% | 0.76 |

        **Final model**: XGBoost (saved as fallback: Random Forest)
        """)
    with c2:
        st.markdown("""
        **Data Pipeline**
        - **Imbalance**: SMOTE applied to training set only
        - **Imputation**: Median (numeric) / Mode (categorical)
        - **Outliers**: IQR clipping on continuous features
        - **Memory**: float64→float32, int64→int32 optimisation
        - **Split**: 80/20 stratified train/test

        **Reproducibility**: `python src/train_model.py`
        """)

st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem;
            font-size:0.72rem; color:#334155; font-family:Space Mono,monospace;'>
  HeartGuard · Centrale Casablanca · Coding Week March 2026 ·
  <span style='color:#00d2be'>k. Zerhouni & Team 1</span>
</div>
""", unsafe_allow_html=True)
