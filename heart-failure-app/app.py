import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import math

st.set_page_config(page_title="HeartGuard AI · Clinical Dashboard", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--navy:#070d1a;--navy2:#0d1626;--slate:#111c30;--slate2:#172038;--border:#1e2d47;--teal:#00d4aa;--red:#ff4757;--amber:#ffa502;--text:#e8eef8;--muted:#6b7fa3;--white:#ffffff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--navy)!important;color:var(--text)!important}
.main{background-color:var(--navy)!important}.block-container{padding:1.5rem 2rem!important;max-width:1400px}
#MainMenu,footer,header{visibility:hidden}
.topbar{display:flex;align-items:center;justify-content:space-between;background:var(--slate);border:1px solid var(--border);border-radius:14px;padding:1rem 1.8rem;margin-bottom:1.5rem}
.topbar-logo{width:42px;height:42px;background:linear-gradient(135deg,#00d4aa,#0099ff);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;margin-right:1rem;float:left}
.topbar-title{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--white)}
.topbar-sub{font-size:.72rem;color:var(--muted);margin-top:1px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--teal);box-shadow:0 0 8px var(--teal);animation:pulse-dot 2s infinite;margin-right:6px}
@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.4}}
.sec-label{font-family:'Syne',sans-serif;font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.7rem;margin-top:.2rem}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:1.2rem}
.stat-card{background:var(--slate);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem;position:relative;overflow:hidden}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--ac,var(--teal))}
.stat-card .icon{font-size:1.3rem;margin-bottom:.3rem}
.stat-card .lbl{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.stat-card .val{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--white);line-height:1.1}
.badge{display:inline-block;font-size:.64rem;font-weight:600;padding:2px 8px;border-radius:20px;margin-top:.3rem}
.ok{background:rgba(0,212,170,.15);color:var(--teal)}.warn{background:rgba(255,165,2,.15);color:var(--amber)}.crit{background:rgba(255,71,87,.15);color:var(--red)}
.flag-list{display:flex;flex-direction:column;gap:.45rem;margin-top:.8rem}
.flag-item{display:flex;align-items:center;gap:.5rem;background:rgba(255,71,87,.08);border:1px solid rgba(255,71,87,.2);border-radius:8px;padding:.45rem .8rem;font-size:.78rem;color:#ffb3ba}
.info-table{width:100%;border-collapse:collapse;font-size:.82rem}
.info-table td{padding:.45rem .6rem;border-bottom:1px solid var(--border)}
.info-table td:first-child{color:var(--muted);width:48%}
.info-table td:last-child{color:var(--white);font-weight:500}
.empty-state{background:var(--slate);border:1px solid var(--border);border-radius:14px;padding:3rem 2rem;text-align:center;color:var(--muted)}
.disclaimer{background:rgba(255,165,2,.08);border:1px solid rgba(255,165,2,.25);border-radius:10px;padding:.7rem 1rem;font-size:.75rem;color:#ffd580;margin-top:1rem}
.driver-card{background:var(--slate2);border:1px solid var(--border);border-radius:12px;padding:1rem;text-align:center}
section[data-testid="stSidebar"]{background:var(--navy2)!important;border-right:1px solid var(--border)!important}
section[data-testid="stSidebar"] label{font-size:.76rem!important;color:var(--muted)!important;text-transform:uppercase!important;letter-spacing:.04em!important}
.stButton>button{background:linear-gradient(135deg,#00d4aa,#0099ff)!important;color:#000!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:.85rem!important;border:none!important;border-radius:10px!important;padding:.65rem 1.5rem!important;width:100%!important;letter-spacing:.04em!important}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

FEATURES = {
    "age":{"label":"Age","unit":"yrs","min":18,"max":100,"default":60,"step":1},
    "ejection_fraction":{"label":"Ejection Fraction","unit":"%","min":10,"max":80,"default":38,"step":1},
    "serum_creatinine":{"label":"Serum Creatinine","unit":"mg/dL","min":0.5,"max":10.0,"default":1.1,"step":0.1},
    "serum_sodium":{"label":"Serum Sodium","unit":"mEq/L","min":110,"max":148,"default":137,"step":1},
    "creatinine_phosphokinase":{"label":"CPK Enzyme","unit":"mcg/L","min":23,"max":7861,"default":250,"step":1},
    "platelets":{"label":"Platelets","unit":"k/mL","min":25000,"max":850000,"default":265000,"step":1000},
    "time":{"label":"Follow-up Period","unit":"days","min":4,"max":285,"default":90,"step":1},
    "anaemia":{"label":"Anaemia","type":"bool","default":False},
    "diabetes":{"label":"Diabetes","type":"bool","default":False},
    "high_blood_pressure":{"label":"Hypertension","type":"bool","default":False},
    "sex":{"label":"Sex","type":"select","options":["Male","Female"],"default":"Male"},
    "smoking":{"label":"Smoking","type":"bool","default":False},
}
FI = {"time":.31,"ejection_fraction":.22,"serum_creatinine":.18,"age":.10,"serum_sodium":.06,"creatinine_phosphokinase":.05,"platelets":.04,"high_blood_pressure":.02,"anaemia":.01,"diabetes":.005,"smoking":.003,"sex":.002}

ms_color = "#00d4aa" if model else "#ffa502"
ms_label = "Model Active" if model else "Demo Mode"
st.markdown(f"""<div class="topbar"><div style="display:flex;align-items:center"><div class="topbar-logo">🫀</div><div><div class="topbar-title">HeartGuard AI</div><div class="topbar-sub">Heart Failure Clinical Decision Support · Centrale Casablanca 2026</div></div></div><div style="display:flex;align-items:center;gap:2rem"><div style="font-size:.78rem;color:{ms_color}"><span class="dot" style="background:{ms_color};box-shadow:0 0 8px {ms_color}"></span>{ms_label}</div><div style="font-size:.78rem;color:#6b7fa3">Explainable ML · SHAP</div></div></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🩺 Patient Input")
    st.markdown('<div class="sec-label">Continuous Variables</div>', unsafe_allow_html=True)
    inputs = {}
    for key, meta in FEATURES.items():
        if meta.get("type") not in ["bool","select"]:
            lbl = f"{meta['label']} ({meta['unit']})"
            if isinstance(meta["default"], float):
                inputs[key] = st.number_input(lbl, min_value=float(meta["min"]), max_value=float(meta["max"]), value=float(meta["default"]), step=float(meta["step"]), key=key)
            else:
                inputs[key] = st.number_input(lbl, min_value=int(meta["min"]), max_value=int(meta["max"]), value=int(meta["default"]), step=int(meta["step"]), key=key)
    st.markdown("---")
    st.markdown('<div class="sec-label">Comorbidities & Profile</div>', unsafe_allow_html=True)
    for key, meta in FEATURES.items():
        if meta.get("type") == "bool":
            inputs[key] = int(st.checkbox(meta["label"], value=meta["default"], key=key))
        elif meta.get("type") == "select":
            inputs[key] = 1 if st.selectbox(meta["label"], meta["options"], key=key) == "Male" else 0
    st.markdown("---")
    predict_btn = st.button("⚡ Run Analysis", use_container_width=True)

left, right = st.columns([1.1, 1.9], gap="large")

with left:
    if predict_btn or st.session_state.get("predicted"):
        if predict_btn:
            X = np.array([[inputs[f] for f in FEATURES]])
            if model is not None:
                prob = model.predict_proba(X)[0][1]
            else:
                s  = max(0,(inputs["age"]-50)/100)+max(0,(40-inputs["ejection_fraction"])/80)
                s += min(inputs["serum_creatinine"]/10,.3)+inputs["high_blood_pressure"]*.1+inputs["anaemia"]*.08-inputs["time"]/1000
                prob = min(max(s,.05),.95)
            st.session_state.update({"prob":prob,"inputs":inputs.copy(),"predicted":True})

        prob=st.session_state["prob"]; inp=st.session_state["inputs"]
        risk_pct=prob*100; is_high=prob>=.5; color="#ff4757" if is_high else "#00d4aa"

        fig_g,ax_g=plt.subplots(figsize=(4,2.8),subplot_kw=dict(aspect="equal"))
        fig_g.patch.set_facecolor("#111c30"); ax_g.set_facecolor("#111c30")
        ax_g.add_patch(Wedge((.5,.18),.38,0,180,width=.10,facecolor="#1e2d47",edgecolor="none"))
        fa=prob*180
        ax_g.add_patch(Wedge((.5,.18),.38,0,fa,width=.10,facecolor=color,edgecolor="none",alpha=.9))
        rad=math.radians(180-fa); nx=.5+.28*math.cos(rad); ny=.18+.28*math.sin(rad)
        ax_g.plot([.5,nx],[.18,ny],color=color,linewidth=3,solid_capstyle="round",zorder=5)
        ax_g.add_patch(plt.Circle((.5,.18),.025,color=color,zorder=6))
        ax_g.text(.5,.44,f"{risk_pct:.1f}%",ha="center",va="center",fontsize=30,fontweight="bold",color=color)
        ax_g.text(.5,.27,"HIGH RISK" if is_high else "LOW RISK",ha="center",fontsize=9,color=color,fontweight="bold")
        ax_g.text(.13,.04,"LOW",ha="center",fontsize=7,color="#6b7fa3")
        ax_g.text(.87,.04,"HIGH",ha="center",fontsize=7,color="#6b7fa3")
        ax_g.text(.5,-.07,"Heart Failure Event Probability",ha="center",fontsize=7,color="#6b7fa3")
        ax_g.set_xlim(0,1); ax_g.set_ylim(-.18,.78); ax_g.axis("off")
        plt.tight_layout(pad=.2); st.pyplot(fig_g,use_container_width=True); plt.close(fig_g)

        ef=inp["ejection_fraction"]; sc=inp["serum_creatinine"]; sns=inp["serum_sodium"]; age=inp["age"]
        ef_cls="crit" if ef<30 else("warn" if ef<45 else"ok"); ef_lbl="Critical" if ef<30 else("Reduced" if ef<45 else"Normal")
        sc_cls="crit" if sc>3 else("warn" if sc>2 else"ok"); sc_lbl="Critical" if sc>3 else("Elevated" if sc>2 else"Normal")
        sns_cls="crit" if sns<130 else("warn" if sns<135 else"ok"); sns_lbl="Critical" if sns<130 else("Low" if sns<135 else"Normal")
        age_cls="warn" if age>70 else"ok"; age_lbl="High Risk Age" if age>70 else"Moderate"

        st.markdown(f"""<div class="stat-grid">
          <div class="stat-card" style="--ac:#00d4aa"><div class="icon">💓</div><div class="lbl">Ejection Fraction</div><div class="val">{ef}<span style="font-size:.85rem">%</span></div><span class="badge {ef_cls}">{ef_lbl}</span></div>
          <div class="stat-card" style="--ac:#0099ff"><div class="icon">🧪</div><div class="lbl">Serum Creatinine</div><div class="val">{sc}<span style="font-size:.85rem"> mg/dL</span></div><span class="badge {sc_cls}">{sc_lbl}</span></div>
          <div class="stat-card" style="--ac:#a78bfa"><div class="icon">⚗️</div><div class="lbl">Serum Sodium</div><div class="val">{sns}<span style="font-size:.85rem"> mEq</span></div><span class="badge {sns_cls}">{sns_lbl}</span></div>
          <div class="stat-card" style="--ac:#ffa502"><div class="icon">👤</div><div class="lbl">Age</div><div class="val">{age}<span style="font-size:.85rem"> yrs</span></div><span class="badge {age_cls}">{age_lbl}</span></div>
        </div>""", unsafe_allow_html=True)

        flags=[]
        if ef<30: flags.append("⚠ Severely reduced ejection fraction (<30%)")
        elif ef<45: flags.append("⚠ Reduced ejection fraction (<45%)")
        if sc>2: flags.append("⚠ Elevated creatinine — kidney concern")
        if sns<135: flags.append("⚠ Low serum sodium (hyponatremia)")
        if inp["high_blood_pressure"]: flags.append("⚠ Hypertension documented")
        if inp["anaemia"]: flags.append("⚠ Anaemia present")
        if flags:
            st.markdown('<div class="flag-list">'+"".join([f'<div class="flag-item">{f}</div>' for f in flags])+'</div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Patient Summary</div>',unsafe_allow_html=True)
        comorbidities=[k for k,v in{"Anaemia":inp["anaemia"],"Diabetes":inp["diabetes"],"Hypertension":inp["high_blood_pressure"],"Smoking":inp["smoking"]}.items() if v]
        st.markdown(f"""<table class="info-table">
          <tr><td>Sex</td><td>{"Male" if inp["sex"]==1 else "Female"}</td></tr>
          <tr><td>Follow-up</td><td>{inp["time"]} days</td></tr>
          <tr><td>CPK Enzyme</td><td>{inp["creatinine_phosphokinase"]} mcg/L</td></tr>
          <tr><td>Platelets</td><td>{inp["platelets"]:,} k/mL</td></tr>
          <tr><td>Comorbidities</td><td>{", ".join(comorbidities) or "None"}</td></tr>
        </table>""",unsafe_allow_html=True)
    else:
        st.markdown("""<div class="empty-state"><div style="font-size:3rem;margin-bottom:1rem">🫀</div><div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8eef8;margin-bottom:.5rem">Ready for Analysis</div><div style="font-size:.82rem">Enter patient data in the sidebar<br>then click <b style="color:#00d4aa">Run Analysis</b></div></div>""",unsafe_allow_html=True)

    st.markdown("""<div class="disclaimer">⚕ <b>Clinical Disclaimer:</b> This tool supports — never replaces — clinical judgement. Always validate with full patient history and specialist review.</div>""",unsafe_allow_html=True)

with right:
    st.markdown('<div class="sec-label">Explainability · SHAP Feature Attribution</div>',unsafe_allow_html=True)
    if st.session_state.get("predicted"):
        prob=st.session_state["prob"]; inp=st.session_state["inputs"]
        try:
            import shap
            if model is None: raise ValueError()
            explainer=shap.TreeExplainer(model)
            fo=list(FEATURES.keys()); X_df=pd.DataFrame([[inp[f] for f in fo]],columns=fo)
            sv=explainer.shap_values(X_df); sv=sv[1][0] if isinstance(sv,list) else sv[0]
            shap_dict=dict(zip(fo,sv)); shap_source="TreeExplainer SHAP"
        except Exception:
            MEANS={"age":60,"ejection_fraction":38,"serum_creatinine":1.1,"serum_sodium":137,"creatinine_phosphokinase":250,"platelets":265000,"time":90,"anaemia":0,"diabetes":0,"high_blood_pressure":0,"sex":1,"smoking":0}
            STDS={"age":12,"ejection_fraction":12,"serum_creatinine":1.1,"serum_sodium":5,"creatinine_phosphokinase":600,"platelets":100000,"time":70,"anaemia":1,"diabetes":1,"high_blood_pressure":1,"sex":1,"smoking":1}
            shap_dict={f:FI[f]*((inp[f]-MEANS[f])/(STDS[f]+1e-9))*(prob-.5)*2 for f in FEATURES}
            shap_source="Approximate SHAP (demo)"

        sorted_items=sorted(shap_dict.items(),key=lambda x:abs(x[1]),reverse=True)
        feats=[FEATURES[k]["label"] for k,_ in sorted_items]; values=[v for _,v in sorted_items]
        colors=["#ff4757" if v>0 else "#00d4aa" for v in values]

        fig,ax=plt.subplots(figsize=(8,5.8)); fig.patch.set_facecolor("#111c30"); ax.set_facecolor("#111c30")
        bars=ax.barh(feats,values,color=colors,edgecolor="none",height=.55)
        for bar,val in zip(bars,values):
            ax.text(val+(.002 if val>=0 else -.002),bar.get_y()+bar.get_height()/2,f"{val:+.3f}",va="center",ha="left" if val>=0 else "right",fontsize=7.5,color="#ff4757" if val>0 else "#00d4aa",fontweight="600")
        ax.invert_yaxis(); ax.axvline(0,color="#2a3f5f",linewidth=1.2,linestyle="--",alpha=.8)
        ax.set_xlabel("SHAP Value  (positive → ↑ risk  |  negative → ↓ risk)",fontsize=8.5,color="#6b7fa3",labelpad=8)
        ax.set_title(f"Patient-Specific Feature Impact  ·  {shap_source}",fontsize=10,fontweight="bold",color="#e8eef8",pad=12)
        ax.tick_params(axis="x",colors="#6b7fa3",labelsize=8); ax.tick_params(axis="y",colors="#e8eef8",labelsize=9,length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.legend(handles=[mpatches.Patch(color="#ff4757",label="↑ Increases risk"),mpatches.Patch(color="#00d4aa",label="↓ Decreases risk")],fontsize=8,framealpha=0,labelcolor="#e8eef8",loc="lower right")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

        st.markdown('<div class="sec-label" style="margin-top:1rem">Top Clinical Drivers</div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        for col,(feat,val),rank in zip([c1,c2,c3],sorted_items[:3],["1st","2nd","3rd"]):
            clr="#ff4757" if val>0 else "#00d4aa"; unit=FEATURES[feat].get("unit","")
            with col:
                st.markdown(f"""<div class="driver-card"><div style="font-size:.64rem;color:#6b7fa3;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">{rank} Driver</div><div style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;color:#e8eef8;margin-bottom:.3rem">{FEATURES[feat]['label']}</div><div style="font-size:1.1rem;font-weight:700;color:{clr}">{inp[feat]}{" "+unit if unit else ""}</div><div style="font-size:.72rem;color:{clr};margin-top:.2rem">{"↑ Increases" if val>0 else "↓ Decreases"} risk</div></div>""",unsafe_allow_html=True)
    else:
        sorted_g=sorted(FI.items(),key=lambda x:x[1],reverse=True)
        feats_g=[FEATURES[k]["label"] for k,_ in sorted_g]; vals_g=[v for _,v in sorted_g]
        fig,ax=plt.subplots(figsize=(8,5.8)); fig.patch.set_facecolor("#111c30"); ax.set_facecolor("#111c30")
        ax.barh(feats_g[::-1],vals_g[::-1],color="#1e3d6e",edgecolor="none",height=.55)
        ax.set_xlabel("Global Feature Importance",fontsize=8.5,color="#6b7fa3",labelpad=8)
        ax.set_title("Global Feature Importance · Run analysis for patient-specific SHAP",fontsize=10,fontweight="bold",color="#e8eef8",pad=12)
        ax.tick_params(axis="y",colors="#e8eef8",labelsize=9,length=0); ax.tick_params(axis="x",colors="#6b7fa3",labelsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    with st.expander("📊 Model Performance & Methodology"):
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Model Comparison**")
            st.dataframe(pd.DataFrame({"Model":["XGBoost ✅","LightGBM","Random Forest","Logistic Reg."],"ROC-AUC":[.91,.90,.89,.82],"Accuracy":["87%","86%","85%","79%"],"F1":[.85,.84,.82,.76]}),use_container_width=True,hide_index=True)
        with c2:
            st.markdown("**Methodology**")
            st.markdown("- **Dataset:** UCI Heart Failure (299 patients)\n- **Imbalance:** SMOTE oversampling\n- **Explainability:** SHAP TreeExplainer\n- **Validation:** Stratified K-Fold CV\n- **Best model:** XGBoost (ROC-AUC 0.91)")
