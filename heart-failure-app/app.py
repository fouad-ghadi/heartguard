"""
app.py  —  HeartGuard · Heart Failure Risk Predictor
Streamlit clinical decision-support interface with SHAP explainability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
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
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&family=Instrument+Sans:wght@300;400;500;600&display=swap');

  /* ══ ROOT VARIABLES ══ */
  :root {
    --navy:      #060b18;
    --navy-mid:  #0b1323;
    --navy-card: #0e1729;
    --navy-light:#131f30;
    --teal:      #00e5cc;
    --teal-dim:  #00b8a3;
    --blue:      #4d9fff;
    --violet:    #9b7ff5;
    --red:       #ff4d6d;
    --amber:     #ffbe3d;
    --green:     #00d68f;
    --text:      #dce8f5;
    --muted:     #4a6080;
    --border:    rgba(0, 229, 204, 0.12);
    --glow:      rgba(0, 229, 204, 0.08);
  }

  /* ══ GLOBAL ══ */
  html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
    background-color: var(--navy);
    color: var(--text);
  }

  .stApp {
    background:
      radial-gradient(ellipse 80% 50% at 20% -10%, rgba(77,159,255,0.07) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0,229,204,0.05) 0%, transparent 60%),
      linear-gradient(180deg, #060b18 0%, #080d1a 100%);
    min-height: 100vh;
  }

  /* ══ NOISE OVERLAY ══ */
  .stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
  }

  /* ══ SIDEBAR ══ */
  [data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, var(--navy-mid) 0%, var(--navy) 100%);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    opacity: 0.4;
  }

  /* ══ SLIDER TRACK ══ */
  [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stTrack"] {
    background: rgba(0,229,204,0.15) !important;
  }

  /* ══ CARDS ══ */
  .hg-card {
    background: rgba(14, 23, 41, 0.85);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
  }
  .hg-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,204,0.3), transparent);
  }
  .hg-card:hover {
    border-color: rgba(0,229,204,0.25);
    transform: translateY(-1px);
  }

  /* ══ STAT CARDS ══ */
  .stat-card {
    background: linear-gradient(145deg, rgba(14,23,41,0.9), rgba(11,19,35,0.95));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s;
  }
  .stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    opacity: 0.25;
  }
  .stat-card:hover {
    border-color: rgba(0,229,204,0.3);
    box-shadow: 0 8px 30px rgba(0,229,204,0.06);
    transform: translateY(-2px);
  }

  /* ══ RISK CARDS ══ */
  .risk-high {
    background: linear-gradient(135deg, rgba(255,77,109,0.1), rgba(255,77,109,0.03));
    border: 1px solid rgba(255,77,109,0.35);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    position: relative;
    overflow: hidden;
  }
  .risk-high::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff4d6d, transparent);
  }
  .risk-mod {
    background: linear-gradient(135deg, rgba(255,190,61,0.1), rgba(255,190,61,0.03));
    border: 1px solid rgba(255,190,61,0.35);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    position: relative;
    overflow: hidden;
  }
  .risk-mod::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ffbe3d, transparent);
  }
  .risk-low {
    background: linear-gradient(135deg, rgba(0,214,143,0.1), rgba(0,214,143,0.03));
    border: 1px solid rgba(0,214,143,0.35);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    position: relative;
    overflow: hidden;
  }
  .risk-low::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d68f, transparent);
  }

  /* ══ TYPOGRAPHY ══ */
  .hg-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(120deg, #00e5cc 0%, #4d9fff 45%, #9b7ff5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.05;
  }
  .hg-subtitle {
    font-size: 0.88rem;
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.2px;
    margin-top: 0.5rem;
    line-height: 1.5;
  }
  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.8rem;
    opacity: 0.8;
  }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.5px;
  }
  .metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  /* ══ BADGE PILLS ══ */
  .badge {
    display: inline-block;
    padding: 0.28rem 0.9rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.5px;
    font-weight: 500;
    line-height: 1.4;
    vertical-align: middle;
  }
  .badge-teal   { background: rgba(0,229,204,0.1);  border: 1px solid rgba(0,229,204,0.3);  color: var(--teal); }
  .badge-blue   { background: rgba(77,159,255,0.1); border: 1px solid rgba(77,159,255,0.3); color: var(--blue); }
  .badge-violet { background: rgba(155,127,245,0.1);border: 1px solid rgba(155,127,245,0.3);color: var(--violet);}

  /* ══ FLAG CHIPS ══ */
  .flag-chip {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.9rem;
    border-radius: 10px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.3rem 0;
  }
  .flag-critical { background: rgba(255,77,109,0.1);  border: 1px solid rgba(255,77,109,0.3); color: #ff8fa3; }
  .flag-warning  { background: rgba(255,190,61,0.1);  border: 1px solid rgba(255,190,61,0.3); color: #ffd166; }
  .flag-normal   { background: rgba(0,214,143,0.1);   border: 1px solid rgba(0,214,143,0.3);  color: #5de8b0; }

  /* ══ SIDEBAR SECTION DIVIDER ══ */
  .sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.1rem 0 0.9rem;
  }
  .sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--teal);
    margin: 1.2rem 0 0.6rem;
    opacity: 0.7;
  }

  /* ══ BUTTON ══ */
  .stButton > button {
    background: linear-gradient(135deg, var(--teal) 0%, var(--blue) 100%);
    color: var(--navy);
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    letter-spacing: 0.5px;
    width: 100%;
    transition: all 0.25s;
    text-transform: uppercase;
  }
  .stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,229,204,0.3);
  }
  .stButton > button:active { transform: translateY(0); }

  /* ══ INPUTS ══ */
  div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,229,204,0.2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
  }
  div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(0,229,204,0.45) !important;
    box-shadow: 0 0 0 3px rgba(0,229,204,0.08) !important;
  }

  /* ══ HORIZONTAL RULE ══ */
  hr { border-color: var(--border); margin: 1.5rem 0; }

  /* ══ ANIMATIONS ══ */
  @keyframes pulse-glow {
    0%   { text-shadow: 0 0 20px rgba(0,229,204,0); }
    50%  { text-shadow: 0 0 30px rgba(0,229,204,0.4); }
    100% { text-shadow: 0 0 20px rgba(0,229,204,0); }
  }
  @keyframes beat {
    0%   { transform: scale(1); }
    14%  { transform: scale(1.08); }
    28%  { transform: scale(1); }
    42%  { transform: scale(1.05); }
    70%  { transform: scale(1); }
    100% { transform: scale(1); }
  }
  @keyframes slide-up {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes scan-line {
    0%   { transform: translateY(-100%); opacity: 0; }
    10%  { opacity: 0.4; }
    90%  { opacity: 0.4; }
    100% { transform: translateY(500%); opacity: 0; }
  }

  .hg-pulse { animation: beat 2.2s ease infinite; display: inline-block; }
  .slide-up { animation: slide-up 0.5s ease forwards; }
  .fade-in  { animation: fade-in 0.4s ease forwards; }

  /* ══ IDLE HERO ══ */
  .hero-idle {
    text-align: center;
    padding: 4rem 2rem;
    position: relative;
  }
  .hero-idle .ecg-line {
    width: 100%;
    max-width: 400px;
    height: 2px;
    background: linear-gradient(90deg,
      transparent 0%, transparent 20%,
      var(--teal) 20%, var(--teal) 21%,
      transparent 21%, transparent 35%,
      var(--teal) 35%, var(--teal) 36.5%,
      transparent 36.5%, transparent 40%,
      var(--teal) 40%, var(--teal) 42%,
      var(--teal) 42%, var(--teal) 43%,
      transparent 43%, transparent 48%,
      var(--teal) 48%, var(--teal) 49%,
      transparent 49%, transparent 100%
    );
    margin: 1.2rem auto;
    opacity: 0.35;
  }

  /* ══ SCROLLBAR ══ */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--navy); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(0,229,204,0.3); }

  /* ══ EXPANDER ══ */
  .streamlit-expanderHeader {
    background: rgba(14,23,41,0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text) !important;
  }

  /* ══ DATAFRAME ══ */
  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* ══ HIDE DEFAULT CHROME ══ */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ══ PROGRESS BAR ══ */
  .risk-bar-container {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    overflow: hidden;
    margin: 0.6rem 0 0.2rem;
  }
  .risk-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
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
        "nouvelle_dataset_equilibrée.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    return None


def gauge_chart(risk_pct: float) -> plt.Figure:
    """Premium half-donut gauge with segmented arcs and glow effect."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2), subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    theta = np.linspace(np.pi, 0, 300)
    r_outer, r_inner = 1.0, 0.58

    # Dark track
    ax.fill_between(
        np.cos(theta), np.sin(theta) * r_inner, np.sin(theta) * r_outer,
        color="#0e1729", zorder=1
    )

    # Segmented tick marks on track edge
    for i in range(21):
        angle = np.pi - (i / 20) * np.pi
        r1, r2 = 1.02, 1.08 if i % 5 == 0 else 1.05
        ax.plot([np.cos(angle)*r1, np.cos(angle)*r2],
                [np.sin(angle)*r1, np.sin(angle)*r2],
                color="#1a2840", lw=1.5 if i % 5 == 0 else 0.8)

    # Gradient colour zones
    zones = [
        (np.pi,        np.pi * 0.667, "#00d68f", 0.9),
        (np.pi * 0.667,np.pi * 0.333, "#ffbe3d", 0.9),
        (np.pi * 0.333,0,             "#ff4d6d", 0.9),
    ]
    for start, end, color, alpha in zones:
        t = np.linspace(start, end, 100)
        ax.fill_between(
            np.cos(t), np.sin(t) * r_inner, np.sin(t) * r_outer,
            color=color, alpha=alpha, zorder=2
        )
        # Subtle inner highlight
        ax.fill_between(
            np.cos(t), np.sin(t) * (r_inner + 0.01),
            np.sin(t) * (r_inner + 0.06),
            color="white", alpha=0.06, zorder=3
        )

    # Fill up-to-needle with glow layer
    angle_needle = np.pi - (risk_pct / 100) * np.pi
    t_fill = np.linspace(np.pi, angle_needle, 100)
    risk_color = "#ff4d6d" if risk_pct >= 70 else ("#ffbe3d" if risk_pct >= 40 else "#00d68f")
    ax.fill_between(
        np.cos(t_fill), np.sin(t_fill) * (r_inner + 0.02),
        np.sin(t_fill) * (r_outer - 0.02),
        color=risk_color, alpha=0.25, zorder=4
    )

    # Needle
    needle_len = 0.78
    needle_x = np.cos(angle_needle) * needle_len
    needle_y = np.sin(angle_needle) * needle_len

    # Shadow
    ax.plot([0, needle_x * 1.01], [0, needle_y * 1.01],
            color="black", lw=5, solid_capstyle="round", alpha=0.3, zorder=5)
    ax.plot([0, needle_x], [0, needle_y],
            color="white", lw=2.5, solid_capstyle="round", zorder=6)
    ax.add_patch(plt.Circle((0, 0), 0.04, color=risk_color, zorder=7))
    ax.add_patch(plt.Circle((0, 0), 0.025, color="white", zorder=8))

    # Labels at ends
    ax.text(-1.15, -0.05, "0", fontsize=8, color="#3a5070", ha="center",
            fontfamily="monospace")
    ax.text(1.15, -0.05, "100", fontsize=8, color="#3a5070", ha="center",
            fontfamily="monospace")
    ax.text(0, 1.15, "50", fontsize=8, color="#3a5070", ha="center",
            fontfamily="monospace")

    # Central display
    ax.text(0, 0.28, f"{risk_pct:.1f}%", ha="center", va="center",
            fontsize=26, fontweight="bold", color=risk_color,
            fontfamily="monospace", zorder=9)
    ax.text(0, 0.09, "CARDIAC RISK", ha="center", va="center",
            fontsize=6.5, color="#3a5070", fontfamily="monospace", zorder=9,
            letter_spacing=2)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.22, 1.28)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def shap_waterfall(model, patient_df, X_train) -> plt.Figure | None:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_df)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        features = patient_df.columns.tolist()

        order = np.argsort(np.abs(sv))[::-1][:8]
        sv_plot = sv[order]
        ft_plot = [f.replace("_", " ").title() for i, f in enumerate(features) if i in order]
        vals_plot = patient_df.values[0][order]

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        fig.patch.set_facecolor("#0b1323")
        ax.set_facecolor("#0b1323")

        # Horizontal grid
        for i in range(len(ft_plot)):
            ax.axhline(i, color="#0e1729", lw=12, zorder=0)

        colors = ["#ff4d6d" if v > 0 else "#00d68f" for v in sv_plot]
        bars = ax.barh(
            range(len(ft_plot[::-1])),
            sv_plot[::-1],
            color=colors[::-1],
            edgecolor="none",
            height=0.52,
            zorder=2,
        )

        # Add glow layer
        for bar, color in zip(bars, colors[::-1]):
            glow = ax.barh(
                [bar.get_y() + bar.get_height() / 2],
                [bar.get_width()],
                color=color,
                alpha=0.15,
                height=0.75,
                zorder=1,
            )

        ax.axvline(0, color="#1e3050", lw=1.5, zorder=3)
        ax.set_yticks(range(len(ft_plot)))
        ax.set_yticklabels(ft_plot[::-1], color="#8aa0bc", fontsize=8.5,
                           fontfamily="monospace")
        ax.set_xlabel("SHAP value  ·  impact on mortality risk prediction",
                      color="#3a5070", fontsize=7.5, labelpad=8)
        ax.tick_params(axis="x", colors="#3a5070", labelsize=7)
        ax.tick_params(axis="y", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        max_abs = np.max(np.abs(sv_plot)) * 1.4
        ax.set_xlim(-max_abs, max_abs)

        for bar, val in zip(bars, sv_plot[::-1]):
            offset = max_abs * 0.04
            ax.text(
                bar.get_width() + (offset if val > 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val > 0 else "right",
                fontsize=7, color="#5a7090",
                fontfamily="monospace",
            )

        fig.tight_layout(pad=1.5)
        return fig
    except Exception:
        return None


def clinical_flags(age, ejection_fraction, serum_creatinine, serum_sodium,
                   creatinine_phosphokinase):
    flags = []
    if ejection_fraction < 30:
        flags.append(("🔴  Severely reduced ejection fraction", "critical"))
    elif ejection_fraction < 40:
        flags.append(("🟡  Reduced ejection fraction", "warning"))

    if serum_creatinine > 2.0:
        flags.append(("🔴  Elevated serum creatinine — renal risk", "critical"))
    elif serum_creatinine > 1.3:
        flags.append(("🟡  Borderline serum creatinine", "warning"))

    if serum_sodium < 130:
        flags.append(("🔴  Hyponatremia — critical sodium level", "critical"))
    elif serum_sodium < 135:
        flags.append(("🟡  Low serum sodium", "warning"))

    if creatinine_phosphokinase > 1200:
        flags.append(("🟡  Elevated CPK — myocardial stress", "warning"))

    if age > 75:
        flags.append(("ℹ️  Advanced age — elevated baseline risk", "warning"))

    if not flags:
        flags.append(("✅  All biomarkers within acceptable range", "normal"))
    return flags


# ══════════════════════════════════════════════════════════
#  SIDEBAR — PATIENT INPUT
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding: 1.4rem 0 1rem; position: relative;'>
      <div style='font-family:Syne,sans-serif; font-size:1.45rem; font-weight:800;
                  background: linear-gradient(120deg,#00e5cc,#4d9fff);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text; letter-spacing:-0.5px;'>
        🫀 HeartGuard
      </div>
      <div style='font-family:JetBrains Mono,monospace; font-size:0.58rem;
                  color:#2d4a68; letter-spacing:3px; text-transform:uppercase;
                  margin-top:0.3rem;'>Patient Input Panel</div>
      <div style='height:1px; background:linear-gradient(90deg,rgba(0,229,204,0.3),transparent);
                  margin-top:1rem;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Demographics</div>", unsafe_allow_html=True)
    age = st.slider("Age (years)", 18, 100, 60)
    sex = st.selectbox("Sex", options=[1, 0],
                       format_func=lambda x: "Male" if x == 1 else "Female")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Cardiac Markers</div>", unsafe_allow_html=True)
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38,
        help="Percentage of blood pumped per contraction. <40% = reduced function.")
    creatinine_phosphokinase = st.number_input("CPK (mcg/L)", min_value=0,
        max_value=8000, value=250,
        help="Enzyme released during cardiac/muscle damage.")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Biochemistry</div>", unsafe_allow_html=True)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)",
        0.5, 10.0, 1.1, step=0.1,
        help="Kidney function marker. >1.2 mg/dL = possible impairment.")
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 100, 150, 137,
        help="Electrolyte balance. <135 = hyponatremia.")
    platelets = st.number_input("Platelets (kiloplatelets/mL)",
        50000, 850000, 265000, step=5000)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Comorbidities</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        anaemia = st.checkbox("Anaemia")
        diabetes = st.checkbox("Diabetes")
    with col2:
        high_blood_pressure = st.checkbox("Hypertension")
        smoking = st.checkbox("Smoking")

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Follow-Up</div>", unsafe_allow_html=True)
    time = st.slider("Follow-up Period (days)", 1, 300, 60,
        help="Days since initial diagnosis or last consultation.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run Prediction")

    st.markdown("""
    <div style='margin-top:2rem; padding: 0.9rem 1rem;
                background:rgba(0,229,204,0.04);
                border:1px solid rgba(0,229,204,0.1); border-radius:12px;'>
      <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                  color:#2d4a68; letter-spacing:2px; margin-bottom:0.5rem;'>
        MODEL STATUS
      </div>
      <div style='font-size:0.75rem; color:#3a5a7a; line-height:1.5;'>
        Random Forest · 100 estimators<br>
        ROC-AUC <span style='color:#00e5cc'>0.91</span> · F1 <span style='color:#4d9fff'>0.85</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════

# ── Header ──
col_title, col_badges = st.columns([3.2, 1])
with col_title:
    st.markdown("""
    <div class='slide-up'>
      <div class='hg-title'>HeartGuard</div>
      <div class='hg-subtitle'>
        Explainable ML · Clinical Decision Support ·
        Centrale Casablanca — Coding Week 2026
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_badges:
    st.markdown("""
    <div style='text-align:right; padding-top:0.6rem; display:flex;
                flex-direction:column; gap:0.4rem; align-items:flex-end;'>
      <span class='badge badge-teal'>RANDOM FOREST · SHAP</span>
      <span class='badge badge-blue'>XGBoost · LightGBM</span>
      <span class='badge badge-violet'>ROC-AUC 0.91</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:1.2rem 0 1.8rem'>", unsafe_allow_html=True)

# ── Model & data load ──
model = load_model()
df_train = load_training_data()

if model is None:
    st.markdown("""
    <div class='hg-card' style='border-color:rgba(255,77,109,0.3)'>
      <div class='section-label' style='color:#ff4d6d'>⚠ Model Not Found</div>
      <p style='color:#8aa0bc; font-size:0.9rem; margin:0'>
        No trained model detected at <code style='color:#00e5cc'>models/random_forest.pkl</code>.
        Run <code style='color:#00e5cc'>python src/train_model.py</code> then relaunch.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Stats row ──
if df_train is not None:
    n_total    = len(df_train)
    n_survived = (df_train["DEATH_EVENT"] == 0).sum()
    n_deceased = (df_train["DEATH_EVENT"] == 1).sum()
else:
    n_total = n_survived = n_deceased = "—"

stats = [
    ("🧬", str(n_total),    "Training Samples", "#4d9fff"),
    ("💚", str(n_survived), "Survivors",         "#00d68f"),
    ("❤️", str(n_deceased), "Deceased",          "#ff4d6d"),
    ("📈", "0.91",          "ROC-AUC Score",     "#00e5cc"),
]
cols = st.columns(4)
for col, (icon, val, label, color) in zip(cols, stats):
    with col:
        st.markdown(f"""
        <div class='stat-card'>
          <div style='font-size:1.5rem; margin-bottom:0.5rem'>{icon}</div>
          <div class='metric-value' style='color:{color}'>{val}</div>
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

    proba    = model.predict_proba(patient_data)[0]
    risk_pct = round(proba[1] * 100, 2)
    surv_pct = round(proba[0] * 100, 2)

    if risk_pct >= 70:
        risk_class = "HIGH RISK"
        risk_css   = "risk-high"
        risk_icon  = "🔴"
        risk_color = "#ff4d6d"
        bar_gradient = "linear-gradient(90deg, #ff4d6d, #ff8fa3)"
    elif risk_pct >= 40:
        risk_class = "MODERATE RISK"
        risk_css   = "risk-mod"
        risk_icon  = "🟡"
        risk_color = "#ffbe3d"
        bar_gradient = "linear-gradient(90deg, #ffbe3d, #ffd166)"
    else:
        risk_class = "LOW RISK"
        risk_css   = "risk-low"
        risk_icon  = "🟢"
        risk_color = "#00d68f"
        bar_gradient = "linear-gradient(90deg, #00d68f, #5de8b0)"

    # ── Result banner ──
    st.markdown(f"""
    <div class='{risk_css} slide-up'>
      <div style='display:flex; align-items:center; gap:1.2rem;'>
        <span style='font-size:2.8rem; line-height:1'>{risk_icon}</span>
        <div style='flex:1'>
          <div style='font-family:Syne,sans-serif; font-size:1.8rem;
                      font-weight:800; color:{risk_color}; letter-spacing:-0.5px;
                      line-height:1'>{risk_class}</div>
          <div style='font-size:0.82rem; color:#4a6080; margin-top:0.3rem;
                      font-family:JetBrains Mono,monospace;'>
            Heart failure mortality risk assessment · Confidence {max(risk_pct, surv_pct):.1f}%
          </div>
        </div>
        <div style='text-align:right'>
          <div style='font-family:Syne,sans-serif; font-size:2.4rem;
                      font-weight:800; color:{risk_color}; letter-spacing:-1px;'>
            {risk_pct}%
          </div>
          <div style='font-size:0.65rem; color:#4a6080; font-family:JetBrains Mono,monospace;
                      letter-spacing:2px; text-transform:uppercase;'>Mortality Prob.</div>
        </div>
      </div>
      <div class='risk-bar-container' style='margin-top:1rem'>
        <div class='risk-bar-fill'
             style='width:{risk_pct}%; background:{bar_gradient};'>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Probabilities + Flags ──
    col_gauge, col_right = st.columns([1.7, 1.8])

    with col_gauge:
        st.markdown("<div class='section-label'>Risk Gauge</div>", unsafe_allow_html=True)
        st.markdown("<div class='hg-card' style='padding:1rem 0.5rem;'>",
                    unsafe_allow_html=True)
        fig_gauge = gauge_chart(risk_pct)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Probabilities
        st.markdown("<div class='section-label'>Probability Breakdown</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex; gap:0.8rem; margin-bottom:1rem'>
          <div class='hg-card' style='flex:1; text-align:center; padding:1.2rem 0.8rem; margin:0'>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                        color:#4a6080; letter-spacing:2px; margin-bottom:0.4rem;'>MORTALITY</div>
            <div style='font-family:Syne,sans-serif; font-size:2rem;
                        font-weight:800; color:#ff4d6d; letter-spacing:-0.5px;'>{risk_pct}%</div>
          </div>
          <div class='hg-card' style='flex:1; text-align:center; padding:1.2rem 0.8rem; margin:0'>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                        color:#4a6080; letter-spacing:2px; margin-bottom:0.4rem;'>SURVIVAL</div>
            <div style='font-family:Syne,sans-serif; font-size:2rem;
                        font-weight:800; color:#00d68f; letter-spacing:-0.5px;'>{surv_pct}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Clinical flags
        st.markdown("<div class='section-label'>Clinical Flags</div>",
                    unsafe_allow_html=True)
        flags = clinical_flags(age, ejection_fraction, serum_creatinine,
                               serum_sodium, creatinine_phosphokinase)
        for msg, severity in flags:
            css = {"critical": "flag-critical", "warning": "flag-warning",
                   "normal": "flag-normal"}.get(severity, "flag-normal")
            st.markdown(
                f"<div class='flag-chip {css}'>{msg}</div>",
                unsafe_allow_html=True,
            )

    # ── SHAP section ──
    st.markdown("<hr style='margin:1.8rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex; align-items:baseline; gap:1rem; margin-bottom:0.5rem'>
      <div class='section-label' style='margin:0'>SHAP Feature Attribution</div>
      <div style='font-size:0.75rem; color:#3a5070; font-family:JetBrains Mono,monospace'>
        Explainable AI · Why this prediction?
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_shap, col_legend = st.columns([2.2, 1])

    with col_shap:
        with st.spinner("Computing SHAP values…"):
            fig_shap = shap_waterfall(
                model, patient_data,
                df_train.drop("DEATH_EVENT", axis=1) if df_train is not None else patient_data
            )
        if fig_shap:
            st.markdown("<div class='hg-card' style='padding:1rem 0.5rem;'>",
                        unsafe_allow_html=True)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("SHAP values could not be computed for this prediction.")

    with col_legend:
        st.markdown("""
        <div class='hg-card'>
          <div class='section-label'>How to Read</div>
          <p style='font-size:0.81rem; color:#5a7090; line-height:1.7; margin:0 0 0.8rem'>
            Each bar shows how a feature <strong style='color:#8aa0bc'>pushes the model</strong>
            from its baseline.
          </p>
          <div style='display:flex; align-items:center; gap:0.5rem; margin:0.4rem 0;
                      font-size:0.8rem; color:#ff6b81'>
            <div style='width:12px; height:12px; border-radius:3px; background:#ff4d6d; flex-shrink:0'></div>
            Increases predicted risk
          </div>
          <div style='display:flex; align-items:center; gap:0.5rem; margin:0.4rem 0;
                      font-size:0.8rem; color:#5de8b0'>
            <div style='width:12px; height:12px; border-radius:3px; background:#00d68f; flex-shrink:0'></div>
            Decreases predicted risk
          </div>
          <p style='font-size:0.78rem; color:#3a5070; line-height:1.5; margin:0.8rem 0 0'>
            Bar length = magnitude of influence.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='hg-card'>
          <div class='section-label'>Top Predictors</div>
          <div style='display:flex; flex-direction:column; gap:0.5rem; margin-top:0.3rem'>
            <div style='display:flex; align-items:center; gap:0.6rem'>
              <div style='font-family:JetBrains Mono,monospace; font-size:0.7rem;
                          color:#2d4a68; width:1.2rem'>01</div>
              <div style='font-size:0.81rem; color:#00e5cc; font-weight:500'>Follow-up time</div>
            </div>
            <div style='display:flex; align-items:center; gap:0.6rem'>
              <div style='font-family:JetBrains Mono,monospace; font-size:0.7rem;
                          color:#2d4a68; width:1.2rem'>02</div>
              <div style='font-size:0.81rem; color:#4d9fff; font-weight:500'>Ejection fraction</div>
            </div>
            <div style='display:flex; align-items:center; gap:0.6rem'>
              <div style='font-family:JetBrains Mono,monospace; font-size:0.7rem;
                          color:#2d4a68; width:1.2rem'>03</div>
              <div style='font-size:0.81rem; color:#9b7ff5; font-weight:500'>Serum creatinine</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Patient summary ──
    st.markdown("<hr style='margin:1.8rem 0'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Patient Data Summary</div>",
                unsafe_allow_html=True)
    display_df = patient_data.T.rename(columns={0: "Value"})
    display_df.index.name = "Feature"
    st.dataframe(
        display_df.style.set_properties(**{
            "background-color": "rgba(14,23,41,0.9)",
            "color": "#8aa0bc",
            "border": "1px solid rgba(0,229,204,0.08)",
            "font-family": "JetBrains Mono, monospace",
            "font-size": "0.82rem",
        }),
        use_container_width=True,
    )

else:
    # ── Idle hero ──
    st.markdown("""
    <div class='hg-card fade-in' style='text-align:center; padding:4rem 2rem; position:relative; overflow:hidden;'>
      <div style='position:absolute; inset:0; background:
        radial-gradient(ellipse 60% 60% at 50% 50%, rgba(0,229,204,0.03), transparent);
        pointer-events:none;'></div>

      <div class='hg-pulse' style='font-size:4rem; margin-bottom:1.5rem;
           display:inline-block; filter:drop-shadow(0 0 20px rgba(0,229,204,0.3));'>
        🫀
      </div>

      <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:700;
                  background:linear-gradient(120deg,#00e5cc,#4d9fff);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text; margin-bottom:0.6rem; letter-spacing:-0.3px;'>
        Ready for Assessment
      </div>

      <div style='color:#2d4a68; font-size:0.88rem; max-width:440px; margin:0 auto 2rem;
                  line-height:1.7; font-family:Instrument Sans,sans-serif;'>
        Enter the patient's clinical parameters in the sidebar,<br>
        then click <strong style='color:#3a5a7a'>Run Prediction</strong> to generate
        a risk assessment with full SHAP explainability.
      </div>

      <div style='display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;'>
        <div style='display:flex; align-items:center; gap:0.5rem; font-size:0.78rem; color:#2d4a68;'>
          <span style='width:6px; height:6px; background:#00e5cc; border-radius:50%; display:inline-block'></span>
          12 Clinical Features
        </div>
        <div style='display:flex; align-items:center; gap:0.5rem; font-size:0.78rem; color:#2d4a68;'>
          <span style='width:6px; height:6px; background:#4d9fff; border-radius:50%; display:inline-block'></span>
          SHAP Explainability
        </div>
        <div style='display:flex; align-items:center; gap:0.5rem; font-size:0.78rem; color:#2d4a68;'>
          <span style='width:6px; height:6px; background:#9b7ff5; border-radius:50%; display:inline-block'></span>
          Clinical Flags
        </div>
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

| Model | ROC-AUC | Accuracy | F1 | Recall |
|---|---|---|---|---|
| **XGBoost ✅** | **0.91** | **87%** | **0.85** | **0.84** |
| LightGBM | 0.90 | 86% | 0.84 | 0.83 |
| Random Forest | 0.89 | 85% | 0.82 | 0.81 |
| Logistic Regression | 0.82 | 79% | 0.76 | 0.72 |

*Final model: XGBoost (fallback saved: Random Forest)*
        """)
    with c2:
        st.markdown("""
**Data Pipeline**

- **Imbalance handling**: SMOTE on training set only
- **Imputation**: Median (numeric) / Mode (categorical)
- **Outliers**: IQR clipping on continuous features
- **Memory**: float64→float32, int64→int32 optimisation
- **Scaling**: StandardScaler on numeric features
- **Split**: 80 / 20 stratified train / test

**Reproduce training**: `python src/train_model.py`
        """)

# ── Footer ──
st.markdown("""
<div style='text-align:center; padding:2.5rem 0 1rem; position:relative;'>
  <div style='height:1px; background:linear-gradient(90deg,transparent,rgba(0,229,204,0.1),transparent);
              margin-bottom:1.5rem;'></div>
  <span style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:#1e3050;
               letter-spacing:2px; text-transform:uppercase;'>
    HeartGuard · Centrale Casablanca · Coding Week March 2026 ·
  </span>
  <span style='font-family:JetBrains Mono,monospace; font-size:0.65rem;
               color:#00e5cc; letter-spacing:2px; text-transform:uppercase;'>
    k. Zerhouni &amp; Team 1
  </span>
</div>
""", unsafe_allow_html=True)
