import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard AI | Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 40%, #8e44ad 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(231,76,60,0.3);
    }
    .hero-banner h1 { color: #fff; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .hero-banner p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0.5rem 0 0; }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .metric-label { color: rgba(255,255,255,0.6); font-size: 0.8rem; font-weight: 500;
                    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
    .metric-value { color: #fff; font-size: 1.8rem; font-weight: 700; }
    .metric-sub   { color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.2rem; }

    /* Risk result box */
    .result-high {
        background: linear-gradient(135deg, rgba(231,76,60,0.25), rgba(192,57,43,0.15));
        border: 2px solid rgba(231,76,60,0.6);
        border-radius: 16px; padding: 1.8rem; text-align: center;
        animation: pulse 2s infinite;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(46,204,113,0.25), rgba(39,174,96,0.15));
        border: 2px solid rgba(46,204,113,0.6);
        border-radius: 16px; padding: 1.8rem; text-align: center;
    }
    .result-title { font-size: 2rem; font-weight: 700; color: #fff; margin-bottom: 0.5rem; }
    .result-sub   { color: rgba(255,255,255,0.75); font-size: 0.95rem; }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(231,76,60,0.4); }
        70%  { box-shadow: 0 0 0 12px rgba(231,76,60,0); }
        100% { box-shadow: 0 0 0 0 rgba(231,76,60,0); }
    }

    /* Section headers */
    .section-header {
        color: #e74c3c;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(231,76,60,0.3);
    }

    /* Sidebar labels */
    .sidebar-section {
        color: #e74c3c;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255,193,7,0.1);
        border-left: 3px solid #f39c12;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
        margin-top: 1rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.8rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ─── Load / Train Model ────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource(show_spinner=False)
def load_or_train():
    model_path    = os.path.join(MODEL_DIR, 'heart_model.pkl')
    scaler_path   = os.path.join(MODEL_DIR, 'scaler.pkl')
    features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')

    if all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        model    = joblib.load(model_path)
        scaler   = joblib.load(scaler_path)
        features = joblib.load(features_path)
        return model, scaler, features

    # Auto-train if models not found
    import pandas as pd_inner
    from sklearn.preprocessing import StandardScaler as SS
    from sklearn.model_selection import train_test_split as tts
    from sklearn.ensemble import GradientBoostingClassifier as GBC

    DATA_URL = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
    df = pd_inner.read_csv(DATA_URL)
    X = df.drop('target', axis=1)
    y = df['target']

    X_tr, _, y_tr, _ = tts(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = SS()
    X_tr_sc = sc.fit_transform(X_tr)

    m = GBC(n_estimators=200, random_state=42)
    m.fit(X_tr_sc, y_tr)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(m,              model_path)
    joblib.dump(sc,             scaler_path)
    joblib.dump(X.columns.tolist(), features_path)

    return m, sc, X.columns.tolist()

# ─── Helper: Gauge Chart ───────────────────────────────────────────────────────
def make_gauge(probability: float) -> go.Figure:
    pct = probability * 100
    if pct < 30:
        color = "#2ecc71"
    elif pct < 60:
        color = "#f39c12"
    else:
        color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Heart Disease Risk Score", 'font': {'color': 'white', 'size': 16}},
        delta={'reference': 50, 'increasing': {'color': '#e74c3c'}, 'decreasing': {'color': '#2ecc71'}},
        number={'suffix': '%', 'font': {'color': 'white', 'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white',
                     'tickfont': {'color': 'white'}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0,  30], 'color': 'rgba(46,204,113,0.15)'},
                {'range': [30, 60], 'color': 'rgba(243,156,18,0.15)'},
                {'range': [60,100], 'color': 'rgba(231,76,60,0.15)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ─── Helper: Feature Importance Chart ─────────────────────────────────────────
def make_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(10)

    fig = px.bar(
        fi_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale=['#8e44ad', '#e74c3c', '#f39c12']
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)')
    )
    fig.update_traces(marker_line_width=0)
    return fig

# ─── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <h1>❤️ HeartGuard AI</h1>
        <p>Advanced cardiovascular risk assessment powered by Machine Learning + SHAP explainability</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("🔄 Loading AI model..."):
        try:
            model, scaler, feature_names = load_or_train()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("Please run `notebooks/heart_analysis.py` first to train the model.")
            return

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🩺 Patient Information")
        st.markdown("*Fill in the clinical measurements below*")
        st.markdown("---")

        st.markdown('<p class="sidebar-section">📋 Demographics</p>', unsafe_allow_html=True)
        age  = st.slider("Age (years)", 20, 80, 50)
        sex  = st.selectbox("Sex", ["Female", "Male"])
        sex_val = 1 if sex == "Male" else 0

        st.markdown('<p class="sidebar-section">💊 Clinical Measurements</p>', unsafe_allow_html=True)
        cp = st.selectbox("Chest Pain Type", [
            "0 — Typical Angina",
            "1 — Atypical Angina",
            "2 — Non-anginal Pain",
            "3 — Asymptomatic"
        ])
        cp_val = int(cp[0])

        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
        chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
        thalach  = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)

        st.markdown('<p class="sidebar-section">🔬 Lab Results</p>', unsafe_allow_html=True)
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
        fbs_val = int(fbs[-2])

        restecg = st.selectbox("Resting ECG Results", [
            "0 — Normal",
            "1 — ST-T wave abnormality",
            "2 — Left ventricular hypertrophy"
        ])
        restecg_val = int(restecg[0])

        exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        exang_val = int(exang[-2])

        slope = st.selectbox("Slope of Peak Exercise ST", [
            "0 — Upsloping",
            "1 — Flat",
            "2 — Downsloping"
        ])
        slope_val = int(slope[0])

        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])

        thal = st.selectbox("Thalassemia", [
            "1 — Normal",
            "2 — Fixed Defect",
            "3 — Reversible Defect"
        ])
        thal_val = int(thal[0])

        st.markdown("---")
        predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Disclaimer</b>: This tool is for educational purposes only.
            Always consult a qualified medical professional for diagnosis.
        </div>
        """, unsafe_allow_html=True)

    # ── MAIN CONTENT ──────────────────────────────────────────────────────────
    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                            restecg_val, thalach, exang_val, oldpeak, slope_val,
                            ca, thal_val]])
    input_df   = pd.DataFrame(input_data, columns=feature_names)
    input_scaled = scaler.transform(input_data)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "🔍 SHAP Explanation", "📈 Feature Importance"])

    with tab1:
        if predict_btn:
            prob = model.predict_proba(input_scaled)[0][1]
            pred = model.predict(input_scaled)[0]

            col1, col2 = st.columns([1, 1])

            with col1:
                st.plotly_chart(make_gauge(prob), use_container_width=True)

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-high">
                        <div class="result-title">⚠️ HIGH RISK</div>
                        <div class="result-sub">
                            Risk Probability: <strong>{prob*100:.1f}%</strong><br><br>
                            The model indicates elevated cardiovascular risk.<br>
                            Please consult a cardiologist immediately.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-low">
                        <div class="result-title">✅ LOW RISK</div>
                        <div class="result-sub">
                            Risk Probability: <strong>{prob*100:.1f}%</strong><br><br>
                            No significant heart disease risk detected.<br>
                            Continue maintaining a healthy lifestyle.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Key metrics
            st.markdown('<p class="section-header">Patient Summary</p>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Age", f"{age} yrs")
            with m2:
                st.metric("Blood Pressure", f"{trestbps} mmHg",
                          delta="High" if trestbps > 140 else "Normal",
                          delta_color="inverse")
            with m3:
                st.metric("Cholesterol", f"{chol} mg/dl",
                          delta="High" if chol > 240 else "Normal",
                          delta_color="inverse")
            with m4:
                st.metric("Max Heart Rate", f"{thalach} bpm")
        else:
            st.info("👈 Fill in the patient details in the sidebar and click **Predict Risk** to get results.")

            # Show dataset overview
            st.markdown('<p class="section-header">About This Model</p>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Training Data</div>
                    <div class="metric-value">303</div>
                    <div class="metric-sub">patient records</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Model Accuracy</div>
                    <div class="metric-value">~88%</div>
                    <div class="metric-sub">on test set</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Features Used</div>
                    <div class="metric-value">13</div>
                    <div class="metric-sub">clinical attributes</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        if predict_btn:
            st.markdown('<p class="section-header">Why This Prediction? (SHAP)</p>',
                        unsafe_allow_html=True)
            st.caption("SHAP values explain which features pushed the prediction toward HIGH or LOW risk.")
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(input_scaled)

                fig_shap, ax = plt.subplots(figsize=(10, 4))
                fig_shap.patch.set_alpha(0)
                ax.set_facecolor('none')

                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0]

                colors_shap = ['#e74c3c' if v > 0 else '#2ecc71' for v in sv]
                bars = ax.barh(feature_names, sv, color=colors_shap, edgecolor='none')
                ax.set_xlabel('SHAP Value (impact on prediction)', color='white')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('rgba(255,255,255,0.2)')
                ax.spines['left'].set_color('rgba(255,255,255,0.2)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title('Feature Impact on This Prediction', color='white', fontsize=13, fontweight='bold')
                ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.5)

                st.pyplot(fig_shap, transparent=True)

                # Top factors
                shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP': sv})
                shap_df['abs'] = shap_df['SHAP'].abs()
                shap_df = shap_df.sort_values('abs', ascending=False).head(3)

                st.markdown('<p class="section-header">Top 3 Risk Factors for This Patient</p>',
                            unsafe_allow_html=True)
                for _, row in shap_df.iterrows():
                    direction = "🔴 increases" if row['SHAP'] > 0 else "🟢 decreases"
                    st.markdown(f"**{row['Feature']}** — {direction} heart disease risk "
                                f"(SHAP = {row['SHAP']:+.3f})")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
        else:
            st.info("👈 Run a prediction first to see the SHAP explanation.")

    with tab3:
        st.markdown('<p class="section-header">Overall Feature Importance</p>', unsafe_allow_html=True)
        fig_fi = make_feature_importance(model, feature_names)
        if fig_fi:
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance chart is available for tree-based models (Random Forest / XGBoost).")

        # Feature descriptions
        st.markdown('<p class="section-header">Feature Glossary</p>', unsafe_allow_html=True)
        glossary = pd.DataFrame({
            'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
            'Description': [
                'Age in years',
                'Sex (0=Female, 1=Male)',
                'Chest pain type (0–3)',
                'Resting blood pressure (mmHg)',
                'Serum cholesterol (mg/dl)',
                'Fasting blood sugar > 120 mg/dl',
                'Resting ECG results (0–2)',
                'Maximum heart rate achieved',
                'Exercise induced angina (0/1)',
                'ST depression induced by exercise',
                'Slope of peak exercise ST segment (0–2)',
                'Number of major vessels colored by fluoroscopy (0–3)',
                'Thalassemia type (1–3)'
            ]
        })
        st.dataframe(glossary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
