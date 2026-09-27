"""
Streamlit Application: Explainable AI-Based Rockfall Risk Prediction & Early Warning System.
Designed for College PBL, Viva, Presentations, and Academic Demonstrations.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_preprocessing import (
    load_and_validate_data,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    TARGET_COLUMN,
    RAW_TARGET_COLUMN
)
from src.explainability import RockfallExplainer, plot_global_shap_bar, plot_local_shap_waterfall

# Page Configuration
st.set_page_config(
    page_title="Rockfall Risk AI System",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Geotechnical Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 18px;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .status-card-stable {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #065F46;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .status-card-risk {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #991B1B;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_metadata():
    """Loads serialized pipeline, candidate models, and execution metadata."""
    model_path = os.path.join("model", "rockfall_model.pkl")
    meta_path = os.path.join("model", "model_metadata.json")
    all_models_path = os.path.join("model", "all_models.pkl")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None, None

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    all_models = joblib.load(all_models_path) if os.path.exists(all_models_path) else None
    return model, metadata, all_models


@st.cache_data
def load_dataset_cached():
    """Loads slope stability dataset for exploration and background samples."""
    csv_path = os.path.join("dataset", "slope_stability_dataset.csv")
    if not os.path.exists(csv_path):
        return None
    from src.data_preprocessing import standardize_columns, create_risk_target
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')
    df = standardize_columns(df)
    df = create_risk_target(df, threshold=1.0)
    return df


# Load Core Artifacts
model, metadata, all_models = load_model_and_metadata()
dataset_df = load_dataset_cached()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/mine-cart.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Interface Page:",
    [
        "🏠 Dashboard",
        "⚠️ Risk Prediction",
        "🔍 Explainable AI",
        "📊 Model Comparison",
        "📈 Risk Analysis",
        "🔄 What-If Analysis",
        "🗂 Dataset Explorer",
        "ℹ️ Project Information"
    ]
)

# Global Prototype Disclaimer on Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Educational Prototype**  
    *Not certified for operational mine safety.*  
    Uses slope-stability data as a proxy for rockfall-risk assessment.
    """
)

if metadata:
    st.sidebar.caption(f"**Active Model:** {metadata.get('selected_model', 'N/A')}")
    st.sidebar.caption(f"**Test F1-Score:** {metadata.get('selected_model_metrics', {}).get('f1_score', 'N/A')}")


# ==============================================================================
# 1. DASHBOARD
# ==============================================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-title">Explainable AI-Based Rockfall Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Decision-Support and Early-Warning Prototype for Open-Pit Mine Slope Stability</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <b>Academic Disclaimer:</b> This project is an educational decision-support prototype that uses slope-stability data as a proxy for rockfall-risk assessment. It is not a certified operational mine-safety system.
    </div>
    """, unsafe_allow_html=True)

    if metadata is None:
        st.error("Model artifacts not found! Please execute `python src/train_model.py` to generate model files.")
        st.stop()

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Dataset Records</div>
            <div class="metric-value">{metadata['dataset_statistics']['total_records']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Input Features</div>
            <div class="metric-value">{len(metadata['features'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Optimal Model</div>
            <div class="metric-value">{metadata['selected_model']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Validation F1-Score</div>
            <div class="metric-value">{metadata['selected_model_metrics']['f1_score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architectural Workflow Diagram
    st.subheader("System Architecture & Workflow")
    workflow_col1, workflow_col2 = st.columns([3, 2])
    with workflow_col1:
        st.markdown("""
        ```mermaid
        flowchart LR
            A["Slope Conditions<br/>(6 Geo + 1 Support)"] --> B["Preprocessing Pipeline<br/>(Scaling + Imputation + OHE)"]
            B --> C["Trained Model<br/>(Optimal Classifier)"]
            C --> D["Risk Classification<br/>(STABLE vs RISK)"]
            C --> E["Predicted Probability<br/>(Calibrated Risk %)"]
            C --> F["SHAP Explainer<br/>(Global & Local XAI)"]
            D --> G["Early Warning Status<br/>(🟢 / 🔴 Visual Alert)"]
        ```
        """, unsafe_allow_html=True)

    with workflow_col2:
        st.markdown("### Key Technical Highlights")
        st.markdown(f"""
        * **Zero Data Leakage:** Preprocessors strictly fitted on the 80% training partition ({metadata['dataset_statistics']['training_samples']} samples).
        * **Imbalance Handling:** Stratified sampling and cost-sensitive weighting tuned to detect rare unstable slopes ({metadata['dataset_statistics']['risk_distribution']['risk_percentage']}% positive incidence).
        * **Explainable AI:** SHAP attributions highlight critical destabilizing factors such as pore water pressure and low cohesion.
        * **Model Validation:** Tested across 4 algorithms with exact real metrics.
        """)

    st.markdown("---")
    st.subheader("Quick System Verification")
    st.info("Navigate to the **⚠️ Risk Prediction** tab in the sidebar to simulate slope conditions, view early-warning alerts, and inspect SHAP explanations.")


# ==============================================================================
# 2. RISK PREDICTION
# ==============================================================================
elif page == "⚠️ Risk Prediction":
    st.markdown('<div class="main-title">Slope Risk Prediction & Early Warning</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter geotechnical slope conditions to predict stability class and early-warning alert.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <b>Early-Warning Style Risk Indicator</b> — Educational Prototype (Not for operational mine safety).
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Trained model is not available. Please run `python src/train_model.py` first.")
        st.stop()

    st.markdown("### 1. Geotechnical & Environmental Inputs")

    # Form with columns for inputs
    col1, col2 = st.columns(2)
    with col1:
        unit_weight = st.slider(
            "Unit Weight (kN/m³)",
            min_value=15.0, max_value=25.0, value=20.0, step=0.1,
            help="Bulk unit weight of the slope rock/soil mass."
        )
        cohesion = st.slider(
            "Cohesion (kPa)",
            min_value=5.0, max_value=50.0, value=25.0, step=0.5,
            help="Internal shear strength holding particles together independent of normal stress."
        )
        friction_angle = st.slider(
            "Internal Friction Angle (°)",
            min_value=20.0, max_value=45.0, value=32.0, step=0.5,
            help="Angle of internal shearing resistance."
        )
        reinforcement_type = st.selectbox(
            "Reinforcement Measure",
            options=FEATURE_METADATA['reinforcement_type']['categories'],
            index=3,
            help="Engineered stabilization measure applied to the slope face."
        )

    with col2:
        slope_angle = st.slider(
            "Slope Angle (°)",
            min_value=10.0, max_value=60.0, value=35.0, step=0.5,
            help="Inclination of the bench or slope face from horizontal."
        )
        slope_height = st.slider(
            "Slope Height (m)",
            min_value=5.0, max_value=50.0, value=25.0, step=0.5,
            help="Vertical bench/slope face height."
        )
        pore_pressure = st.slider(
            "Pore Water Pressure Ratio (r_u)",
            min_value=0.0, max_value=1.0, value=0.30, step=0.01,
            help="Ratio of groundwater pore pressure to total vertical overburden stress."
        )

    # Optional model selector if multiple models are available
    if all_models:
        selected_model_name = st.selectbox(
            "Active Inference Classifier:",
            options=list(all_models.keys()),
            index=list(all_models.keys()).index(metadata['selected_model']) if metadata['selected_model'] in all_models else 0,
            help="Select which validated machine learning model to execute."
        )
        active_pipeline = all_models[selected_model_name]
    else:
        active_pipeline = model
        selected_model_name = metadata['selected_model']

    predict_btn = st.button("🚨 Predict Risk & Early Warning", type="primary", use_container_width=True)

    input_df = pd.DataFrame([{
        'unit_weight': unit_weight,
        'cohesion': cohesion,
        'friction_angle': friction_angle,
        'slope_angle': slope_angle,
        'slope_height': slope_height,
        'pore_water_pressure_ratio': pore_pressure,
        'reinforcement_type': reinforcement_type
    }])

    # Execute Prediction
    if predict_btn or 'last_prediction' in st.session_state:
        # Cache current input in session
        if predict_btn:
            pred_class = int(active_pipeline.predict(input_df)[0])
            prob_risk = float(active_pipeline.predict_proba(input_df)[0][1])
            st.session_state['last_prediction'] = {
                'class': pred_class,
                'probability': prob_risk,
                'inputs': input_df,
                'model_name': selected_model_name
            }
        else:
            cached = st.session_state['last_prediction']
            pred_class = cached['class']
            prob_risk = cached['probability']
            selected_model_name = cached['model_name']

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("2. Prediction & Risk Status Card")

        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            if pred_class == 0:
                st.markdown(f"""
                <div class="status-card-stable">
                    <h1 style="margin:0; font-size: 3rem;">🟢 STABLE</h1>
                    <h3 style="margin-top:10px; margin-bottom: 5px;">Risk Class: Stable</h3>
                    <p style="font-size: 1.1rem; margin: 5px 0;"><b>Predicted Probability:</b> {prob_risk*100:.2f}%</p>
                    <p style="font-size: 0.95rem; color: #047857; margin: 0;"><b>Model:</b> {selected_model_name}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card-risk">
                    <h1 style="margin:0; font-size: 3rem;">🔴 RISK / UNSTABLE</h1>
                    <h3 style="margin-top:10px; margin-bottom: 5px;">Risk Class: Risk / Unstable</h3>
                    <p style="font-size: 1.1rem; margin: 5px 0;"><b>Predicted Probability:</b> {prob_risk*100:.2f}%</p>
                    <p style="font-size: 0.95rem; color: #B91C1C; margin: 0;"><b>Model:</b> {selected_model_name}</p>
                </div>
                """, unsafe_allow_html=True)

            st.caption("Scientific interpretation: The model predicts the risk class based on the available input data. Predicted Probability reflects statistical classifier confidence, not physical certainty.")

        with res_col2:
            # Probability Gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_risk * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Probability of Instability", 'font': {'size': 16}},
                number={'suffix': "%", 'valueformat': ".2f"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#EF4444" if pred_class == 1 else "#10B981"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 50], 'color': '#ECFDF5'},
                        {'range': [50, 100], 'color': '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)


# ==============================================================================
# 3. EXPLAINABLE AI (SHAP)
# ==============================================================================
elif page == "🔍 Explainable AI":
    st.markdown('<div class="main-title">Explainable AI — SHAP Interpretability</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Understand the global and local feature contributions driving model predictions.</div>', unsafe_allow_html=True)

    if model is None or dataset_df is None:
        st.error("Required model or dataset artifacts not found.")
        st.stop()

    tab1, tab2 = st.tabs(["🌐 Global Feature Importance", "🎯 Local Prediction Explanation"])

    with tab1:
        st.subheader("Global Model Explanations")
        st.markdown(
            "Global explanations quantify which slope features exert the greatest overall influence "
            "across the entire dataset. In geotechnical terms, shear strength (cohesion and friction angle) "
            "and driving shear stresses (slope geometry and water pressure) dominate stability."
        )

        global_records = metadata.get("global_shap_importance", [])
        if global_records:
            global_df = pd.DataFrame(global_records).sort_values("Mean |SHAP Value|", ascending=True)
            fig_global = plot_global_shap_bar(global_df)
            st.plotly_chart(fig_global, use_container_width=True)

            st.markdown("#### Feature Importance Ranking")
            st.dataframe(
                global_df.sort_values("Mean |SHAP Value|", ascending=False).reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.info("Precomputed global SHAP records not found in metadata.")

    with tab2:
        st.subheader("Local Instance Explanation")
        st.markdown(
            "Inspect how individual geotechnical properties of a specific slope push the prediction "
            "toward **STABLE** (green) or **RISK / UNSTABLE** (red)."
        )

        # Check for current input from Risk Prediction or provide default
        if 'last_prediction' in st.session_state:
            active_df = st.session_state['last_prediction']['inputs']
            st.success("Explaining the most recent prediction made on the **Risk Prediction** page.")
        else:
            st.info("Displaying explanation for default baseline slope conditions. Enter custom parameters in Risk Prediction to update.")
            active_df = pd.DataFrame([{
                'unit_weight': 21.0,
                'cohesion': 15.0,
                'friction_angle': 26.0,
                'slope_angle': 45.0,
                'slope_height': 35.0,
                'pore_water_pressure_ratio': 0.65,
                'reinforcement_type': 'Soil Nailing'
            }])

        st.write("**Current Input Conditions:**")
        st.dataframe(active_df, use_container_width=True)

        with st.spinner("Computing local SHAP attributions..."):
            explainer = RockfallExplainer(model, background_data=dataset_df[ALL_FEATURES])
            local_expl = explainer.explain_instance(active_df)

        fig_local = plot_local_shap_waterfall(local_expl)
        st.plotly_chart(fig_local, use_container_width=True)

        st.markdown("#### Detailed Feature Contributions")
        attr_df = pd.DataFrame(local_expl["attributions"])[
            ["feature", "shap_value", "direction", "scientific_interpretation"]
        ]
        attr_df.columns = ["Feature", "SHAP Contribution", "Impact Direction", "Scientific Interpretation"]
        st.dataframe(attr_df, use_container_width=True)

        st.caption("Language Rule: 'This feature contributed to the model prediction.' Explanations establish statistical correlation within the model, not physical causality.")


# ==============================================================================
# 4. MODEL COMPARISON
# ==============================================================================
elif page == "📊 Model Comparison":
    st.markdown('<div class="main-title">Empirical Model Comparison & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comparison of all 4 candidate classifiers evaluated on the hold-out test set.</div>', unsafe_allow_html=True)

    if metadata is None:
        st.error("Metadata not found. Please train models first.")
        st.stop()

    evals = metadata.get("all_model_evaluations", {})
    from src.evaluate_model import build_comparison_dataframe
    comp_df = build_comparison_dataframe(evals)

    st.subheader("1. Test Performance Metrics (Actual Calculated Values)")
    st.markdown("All values represent exact execution on the 20% hold-out test set (2,000 samples).")

    # Display comparison table
    st.dataframe(
        comp_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], color="#D1FAE5"),
        use_container_width=True
    )

    # Highlight Selected Model
    st.success(f"**Selected Model: {metadata['selected_model']}** — {metadata['selection_rationale']}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("2. Multi-Metric Visual Comparison")

    # Bar chart comparison
    melted_df = comp_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], var_name="Metric", value_name="Score")
    fig_bar = px.bar(
        melted_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="<b>Classification Performance by Metric</b>",
        text_auto=".3f",
        template="plotly_white"
    )
    fig_bar.update_layout(yaxis=dict(range=[0.5, 1.05]), height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("3. Confusion Matrix Breakdown")

    cm_cols = st.columns(4)
    for i, (m_name, m_data) in enumerate(evals.items()):
        with cm_cols[i % 4]:
            cm = np.array(m_data["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Stable (0)", "Risk (1)"],
                y=["Stable (0)", "Risk (1)"],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"<b>{m_name}</b>"
            )
            fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(f"TN: {cm[0,0]} | FP: {cm[0,1]}<br>FN: {cm[1,0]} | TP: {cm[1,1]}", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("4. Receiver Operating Characteristic (ROC) Curves")

    fig_roc = go.Figure()
    for m_name, m_data in evals.items():
        curve = m_data.get("roc_curve", {})
        if curve:
            fig_roc.add_trace(go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode='lines',
                name=f"{m_name} (AUC = {m_data['roc_auc']:.4f})"
            ))
    # Diagonal random baseline
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name="Random Classifier"
    ))
    fig_roc.update_layout(
        title="<b>Holdout Test Set ROC Curves</b>",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_roc, use_container_width=True)


# ==============================================================================
# 5. RISK ANALYSIS
# ==============================================================================
elif page == "📈 Risk Analysis":
    st.markdown('<div class="main-title">Geotechnical Risk & Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Exploration of Factor of Safety and parameter distributions across stability regimes.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <b>Data Boundary Notice:</b> The visualizations below analyze the static training dataset distribution. They are distinct from live real-time mine sensor monitoring.
    </div>
    """, unsafe_allow_html=True)

    if dataset_df is None:
        st.error("Dataset not loaded.")
        st.stop()

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.subheader("1. Factor of Safety (FS) Distribution")
        fig_fs = px.histogram(
            dataset_df,
            x="factor_of_safety",
            nbins=40,
            color="risk",
            color_discrete_map={0: "#10B981", 1: "#EF4444"},
            labels={"factor_of_safety": "Factor of Safety (FS)", "risk": "Risk Target"},
            title="<b>Factor of Safety Histogram (Threshold = 1.0)</b>",
            template="plotly_white"
        )
        fig_fs.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="FS Critical Limit = 1.0")
        fig_fs.update_layout(height=380)
        st.plotly_chart(fig_fs, use_container_width=True)

    with r_col2:
        st.subheader("2. Class Balance (Stable vs Risk)")
        class_counts = dataset_df["risk"].value_counts().rename({0: "STABLE (FS >= 1.0)", 1: "RISK / UNSTABLE (FS < 1.0)"})
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            color=class_counts.index,
            color_discrete_map={
                "STABLE (FS >= 1.0)": "#10B981",
                "RISK / UNSTABLE (FS < 1.0)": "#EF4444"
            },
            hole=0.4,
            title="<b>Dataset Ground Truth Class Proportion</b>"
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("3. Feature vs. Stability Boxplots")

    sel_feat = st.selectbox(
        "Select Feature for Distribution Breakdown:",
        options=NUMERICAL_FEATURES,
        format_func=lambda x: FEATURE_METADATA[x]["display_name"]
    )

    fig_box = px.box(
        dataset_df,
        x="risk",
        y=sel_feat,
        color="risk",
        color_discrete_map={0: "#10B981", 1: "#EF4444"},
        labels={"risk": "Class (0 = Stable, 1 = Unstable)", sel_feat: FEATURE_METADATA[sel_feat]["display_name"]},
        title=f"<b>{FEATURE_METADATA[sel_feat]['display_name']} Distribution by Risk Class</b>",
        template="plotly_white"
    )
    fig_box.update_layout(height=400)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("4. Feature Correlation Matrix")
    corr_cols = NUMERICAL_FEATURES + ["factor_of_safety", "risk"]
    corr_matrix = dataset_df[corr_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="<b>Pearson Correlation Matrix</b>",
        aspect="auto"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# 6. WHAT-IF ANALYSIS
# ==============================================================================
elif page == "🔄 What-If Analysis":
    st.markdown('<div class="main-title">Model-Based What-If Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluate model sensitivity by comparing baseline slope parameters against modified scenarios.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <b>Notice:</b> This analysis shows how the trained model responds to changed inputs. It is not a physical simulation and does not establish causality.
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Trained model is not available.")
        st.stop()

    st.markdown("### Set Original Conditions vs. Modified Conditions")
    w_col1, w_col2 = st.columns(2)

    with w_col1:
        st.markdown("#### 🟢 Scenario A (Original Conditions)")
        a_unit_weight = st.slider("Original Unit Weight (kN/m³)", 15.0, 25.0, 20.0, 0.1, key="a_uw")
        a_cohesion = st.slider("Original Cohesion (kPa)", 5.0, 50.0, 20.0, 0.5, key="a_coh")
        a_friction = st.slider("Original Friction Angle (°)", 20.0, 45.0, 30.0, 0.5, key="a_fric")
        a_slope_angle = st.slider("Original Slope Angle (°)", 10.0, 60.0, 40.0, 0.5, key="a_sa")
        a_slope_height = st.slider("Original Slope Height (m)", 5.0, 50.0, 30.0, 0.5, key="a_sh")
        a_pore_water = st.slider("Original Pore Water Ratio", 0.0, 1.0, 0.50, 0.01, key="a_pw")
        a_reinf = st.selectbox("Original Reinforcement", FEATURE_METADATA['reinforcement_type']['categories'], index=0, key="a_rf")

    with w_col2:
        st.markdown("#### 🟡 Scenario B (Modified Conditions)")
        b_unit_weight = st.slider("Modified Unit Weight (kN/m³)", 15.0, 25.0, 20.0, 0.1, key="b_uw")
        b_cohesion = st.slider("Modified Cohesion (kPa)", 5.0, 50.0, 35.0, 0.5, key="b_coh")
        b_friction = st.slider("Modified Friction Angle (°)", 20.0, 45.0, 35.0, 0.5, key="b_fric")
        b_slope_angle = st.slider("Modified Slope Angle (°)", 10.0, 60.0, 30.0, 0.5, key="b_sa")
        b_slope_height = st.slider("Modified Slope Height (m)", 5.0, 50.0, 20.0, 0.5, key="b_sh")
        b_pore_water = st.slider("Modified Pore Water Ratio", 0.0, 1.0, 0.10, 0.01, key="b_pw")
        b_reinf = st.selectbox("Modified Reinforcement", FEATURE_METADATA['reinforcement_type']['categories'], index=2, key="b_rf")

    df_a = pd.DataFrame([{
        'unit_weight': a_unit_weight, 'cohesion': a_cohesion, 'friction_angle': a_friction,
        'slope_angle': a_slope_angle, 'slope_height': a_slope_height,
        'pore_water_pressure_ratio': a_pore_water, 'reinforcement_type': a_reinf
    }])

    df_b = pd.DataFrame([{
        'unit_weight': b_unit_weight, 'cohesion': b_cohesion, 'friction_angle': b_friction,
        'slope_angle': b_slope_angle, 'slope_height': b_slope_height,
        'pore_water_pressure_ratio': b_pore_water, 'reinforcement_type': b_reinf
    }])

    pred_a = int(model.predict(df_a)[0])
    prob_a = float(model.predict_proba(df_a)[0][1])

    pred_b = int(model.predict(df_b)[0])
    prob_b = float(model.predict_proba(df_b)[0][1])

    st.markdown("---")
    st.subheader("Comparison of What-If Outcomes")

    res_a_col, res_b_col, res_delta_col = st.columns(3)

    with res_a_col:
        st.markdown("### Original Outcome (A)")
        if pred_a == 0:
            st.success(f"**Class:** STABLE  \n**Predicted Probability:** {prob_a*100:.2f}%")
        else:
            st.error(f"**Class:** RISK / UNSTABLE  \n**Predicted Probability:** {prob_a*100:.2f}%")

    with res_b_col:
        st.markdown("### Modified Outcome (B)")
        if pred_b == 0:
            st.success(f"**Class:** STABLE  \n**Predicted Probability:** {prob_b*100:.2f}%")
        else:
            st.error(f"**Class:** RISK / UNSTABLE  \n**Predicted Probability:** {prob_b*100:.2f}%")

    with res_delta_col:
        delta_prob = (prob_b - prob_a) * 100
        st.markdown("### Sensitivity Delta")
        status_change = "Unchanged" if pred_a == pred_b else f"{'STABLE' if pred_a == 0 else 'RISK'} ➔ {'STABLE' if pred_b == 0 else 'RISK'}"
        st.metric(
            label="Probability Delta",
            value=f"{delta_prob:+.2f}%",
            delta=f"{delta_prob:+.2f}%",
            delta_color="inverse"
        )
        st.write(f"**Prediction Shift:** {status_change}")

    # Summary of Changed Features Table
    st.markdown("#### Input Differences")
    diff_records = []
    for feat in ALL_FEATURES:
        val_a = df_a.iloc[0][feat]
        val_b = df_b.iloc[0][feat]
        if val_a != val_b:
            diff_records.append({
                "Feature": FEATURE_METADATA[feat]["display_name"],
                "Original (A)": val_a,
                "Modified (B)": val_b,
                "Net Change": f"{val_b - val_a:+.2f}" if isinstance(val_a, (int, float)) else "Category Changed"
            })

    if diff_records:
        st.dataframe(pd.DataFrame(diff_records), use_container_width=True)
    else:
        st.info("Both scenarios have identical parameter values. Adjust sliders above to evaluate sensitivity.")


# ==============================================================================
# 7. DATASET EXPLORER
# ==============================================================================
elif page == "🗂 Dataset Explorer":
    st.markdown('<div class="main-title">Slope Stability Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Inspect records, data types, missing values, and descriptive statistics.</div>', unsafe_allow_html=True)

    if dataset_df is None:
        st.error("Dataset not found at `dataset/slope_stability_dataset.csv`.")
        st.stop()

    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    e_col1.metric("Total Records", f"{len(dataset_df):,}")
    e_col2.metric("Total Columns", len(dataset_df.columns))
    e_col3.metric("Missing Values", dataset_df.isnull().sum().sum())
    e_col4.metric("Duplicates", dataset_df.duplicated().sum())

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("1. Sample Dataset Records")

    # Filters
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        sel_class = st.selectbox("Filter by Class:", ["All", "STABLE (FS >= 1.0)", "RISK / UNSTABLE (FS < 1.0)"])
    with f_col2:
        sel_reinf = st.selectbox("Filter by Reinforcement:", ["All"] + FEATURE_METADATA['reinforcement_type']['categories'])

    filtered_df = dataset_df.copy()
    if sel_class == "STABLE (FS >= 1.0)":
        filtered_df = filtered_df[filtered_df["risk"] == 0]
    elif sel_class == "RISK / UNSTABLE (FS < 1.0)":
        filtered_df = filtered_df[filtered_df["risk"] == 1]

    if sel_reinf != "All":
        filtered_df = filtered_df[filtered_df["reinforcement_type"] == sel_reinf]

    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.caption(f"Showing top {min(100, len(filtered_df))} of {len(filtered_df):,} filtered records.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("2. Summary Descriptive Statistics")
    st.dataframe(dataset_df.describe().T.style.format("{:.3f}"), use_container_width=True)


# ==============================================================================
# 8. PROJECT INFORMATION
# ==============================================================================
elif page == "ℹ️ Project Information":
    st.markdown('<div class="main-title">Project Documentation & Technical Specifications</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comprehensive overview for PBL demonstration, viva review, and academic presentation.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Problem Statement
    In surface mining operations, pit wall instability and rockfall occurrences present severe risks 
    to operational personnel, heavy equipment, and resource extraction continuity. Traditional limit-equilibrium 
    and finite-element numerical analyses require substantial computational overhead. This project 
    implements a rapid, explainable machine-learning screening prototype capable of assessing 
    slope failure risk and attributing decisive geotechnical factors.

    ### Proposed Solution
    An end-to-end Explainable AI (XAI) pipeline that takes multi-parametric geotechnical slope attributes 
    (density, cohesion, internal friction, slope geometry, groundwater pore pressures, and engineered supports) 
    to classify slope risk and compute calibrated probability of failure. Integrated SHAP explanations 
    provide transparent attributions for both local instances and global patterns.

    ### Academic Objectives
    1. **Data Preprocessing & Validation:** Standardize raw variables, handle missing data, and apply zero-leakage scaling and encoding.
    2. **Model Training & Comparison:** Evaluate 4 diverse supervised learning architectures (Logistic Regression, Support Vector Machine, Random Forest, and XGBoost).
    3. **Imbalance-Aware Evaluation:** Prioritize Recall, Precision, and F1-score to mitigate critical false negatives in risk screening.
    4. **Explainable AI Integration:** Deploy SHAP (SHapley Additive exPlanations) for both global feature importance and single-instance local transparency.
    5. **Interactive What-If Simulation:** Enable engineers to observe model sensitivity to changes in bench slope angles, water drainage, or support measures.

    ### Technology Stack
    * **Language:** Python 3.12
    * **Data Manipulation:** Pandas, NumPy
    * **Machine Learning:** Scikit-learn, XGBoost
    * **Explainability:** SHAP (TreeExplainer & KernelExplainer)
    * **Interactive Visualization:** Plotly Express & Graph Objects
    * **Web Application:** Streamlit
    * **Model Serialization:** Joblib

    ### Project Limitations & Scientific Boundaries
    * **Proxy Dataset:** The model was trained on a standardized slope-stability simulation dataset and NOT verified historical rockfall event logs or live geotechnical telemetry.
    * **Statistical Nature:** ML predictions reflect probabilistic correlations learned from the dataset and do not replace physical limit-equilibrium calculations or certified rock mechanics instrumentation.
    * **No Causality:** SHAP values represent statistical attribution toward model log-odds/probability, not physical causality.

    ---
    ### Official Disclaimer
    > **Disclaimer:** This application is an educational machine-learning prototype developed for academic purposes. It uses a public slope-stability dataset as a proxy for risk assessment and has not been validated using site-specific operational mine data. Predictions must not be used as a substitute for professional geotechnical assessment or certified mine-safety systems.
    """)
