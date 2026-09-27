"""
Explainable AI (XAI) Module using SHAP.
Generates global feature importance and local instance-level risk attribution
with scientifically sound interpretations for geotechnical decision support.
Supports both Tree-based models (Random Forest, XGBoost) and Kernel-based models (SVM, Logistic Regression).
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from sklearn.pipeline import Pipeline


class RockfallExplainer:
    """
    SHAP-based Explainability engine for the Rockfall Risk Prediction Pipeline.
    Automatically detects model type to utilize TreeExplainer or KernelExplainer.
    """

    def __init__(self, pipeline: Pipeline, background_data: Optional[pd.DataFrame] = None):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps['preprocessor']
        self.classifier = pipeline.named_steps['classifier']
        self.feature_names = self._extract_feature_names()
        self.background_data = background_data
        self._global_importance_df = None

        # Build appropriate explainer
        self.is_tree = any(
            t in type(self.classifier).__name__.lower()
            for t in ['tree', 'forest', 'xgb', 'gradientboost']
        )

        if background_data is not None:
            # Transform sample for background
            sample_bg = background_data.sample(min(len(background_data), 35), random_state=42)
            self.X_bg_trans = self.preprocessor.transform(sample_bg)
        else:
            self.X_bg_trans = np.zeros((1, len(self.feature_names)))

        if self.is_tree:
            self.explainer = shap.TreeExplainer(self.classifier)
        else:
            # KernelExplainer on classifier's predict_proba
            self.explainer = shap.KernelExplainer(self.classifier.predict_proba, self.X_bg_trans)

        if background_data is not None:
            self._compute_global_importance(background_data)

    def _extract_feature_names(self) -> List[str]:
        try:
            raw_names = self.preprocessor.get_feature_names_out()
            clean_names = []
            for name in raw_names:
                name_str = str(name)
                if 'num__' in name_str:
                    clean = name_str.replace('num__', '')
                    pretty = {
                        'unit_weight': 'Unit Weight (kN/m³)',
                        'cohesion': 'Cohesion (kPa)',
                        'friction_angle': 'Internal Friction Angle (°)',
                        'slope_angle': 'Slope Angle (°)',
                        'slope_height': 'Slope Height (m)',
                        'pore_water_pressure_ratio': 'Pore Water Pressure Ratio'
                    }.get(clean, clean)
                    clean_names.append(pretty)
                elif 'cat__' in name_str:
                    clean = name_str.replace('cat__reinforcement_type_', 'Reinforcement: ')
                    clean_names.append(clean)
                else:
                    clean_names.append(name_str)
            return clean_names
        except Exception:
            return [
                'Unit Weight (kN/m³)', 'Cohesion (kPa)', 'Internal Friction Angle (°)',
                'Slope Angle (°)', 'Slope Height (m)', 'Pore Water Pressure Ratio',
                'Reinforcement: Drainage', 'Reinforcement: Geosynthetics',
                'Reinforcement: Retaining Wall', 'Reinforcement: Soil Nailing'
            ]

    def _compute_global_importance(self, X_sample: pd.DataFrame, max_samples: int = 50):
        sample_df = X_sample.sample(min(len(X_sample), max_samples), random_state=42)
        X_trans = self.preprocessor.transform(sample_df)

        shap_vals = self.explainer.shap_values(X_trans)

        if isinstance(shap_vals, list):
            vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif len(shap_vals.shape) == 3:
            vals = shap_vals[:, :, 1]  # positive risk class
        else:
            vals = shap_vals

        mean_abs_shap = np.mean(np.abs(vals), axis=0)

        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean |SHAP Value|': mean_abs_shap
        }).sort_values(by='Mean |SHAP Value|', ascending=True).reset_index(drop=True)

        self._global_importance_df = df

    def get_global_importance(self) -> pd.DataFrame:
        if self._global_importance_df is None and self.background_data is not None:
            self._compute_global_importance(self.background_data)
        return self._global_importance_df

    def explain_instance(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes local SHAP attribution for a single user-provided slope condition instance.
        """
        X_trans = self.preprocessor.transform(input_df)

        if self.is_tree:
            explanation = self.explainer(X_trans)
            shap_vals = explanation.values[0]
            if len(shap_vals.shape) == 2:
                shap_vals = shap_vals[:, 1]
            base_val = float(explanation.base_values[0]) if hasattr(explanation.base_values, '__len__') else float(explanation.base_values)
        else:
            raw_shap = self.explainer.shap_values(X_trans)
            if isinstance(raw_shap, list):
                shap_vals = raw_shap[1][0]
            elif len(raw_shap.shape) == 3:
                shap_vals = raw_shap[0, :, 1]
            else:
                shap_vals = raw_shap[0]
            base_val = float(self.explainer.expected_value[1]) if isinstance(self.explainer.expected_value, (list, np.ndarray)) else float(self.explainer.expected_value)

        attributions = []
        for name, val in zip(self.feature_names, shap_vals):
            direction = "Risk Increasing" if val > 0.0001 else "Risk Decreasing" if val < -0.0001 else "Neutral"
            attributions.append({
                "feature": name,
                "shap_value": float(val),
                "abs_shap": abs(float(val)),
                "direction": direction,
                "scientific_interpretation": (
                    f"This feature contributed to increasing the predicted risk ({val:+.4f})."
                    if val > 0 else
                    f"This feature contributed to decreasing the predicted risk ({val:.4f})."
                )
            })

        attributions_sorted = sorted(attributions, key=lambda x: x["abs_shap"], reverse=True)

        return {
            "base_value": base_val,
            "attributions": attributions_sorted,
            "top_risk_increasers": [a for a in attributions_sorted if a["shap_value"] > 0][:3],
            "top_risk_reducers": [a for a in attributions_sorted if a["shap_value"] < 0][:3],
            "feature_names": self.feature_names,
            "shap_values": [float(v) for v in shap_vals]
        }


def plot_global_shap_bar(importance_df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive Plotly horizontal bar chart of global feature importance.
    """
    fig = go.Figure(go.Bar(
        x=importance_df['Mean |SHAP Value|'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Mean |SHAP Value|'],
            colorscale='Blues',
            showscale=False
        ),
        hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="<b>Global SHAP Feature Importance</b><br><sup>Average magnitude of model feature impact across slope stability dataset</sup>",
        xaxis_title="Mean |SHAP Value| (Impact on Model Prediction)",
        yaxis_title="Geotechnical Features",
        template="plotly_white",
        height=450,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


def plot_local_shap_waterfall(explanation_dict: Dict[str, Any]) -> go.Figure:
    """
    Renders an interactive divergence bar chart for local prediction attribution.
    """
    attrs = explanation_dict["attributions"]
    df = pd.DataFrame(attrs)
    df_top = df.head(8).iloc[::-1]

    colors = ['#DC2626' if v > 0 else '#16A34A' for v in df_top['shap_value']]

    fig = go.Figure(go.Bar(
        x=df_top['shap_value'],
        y=df_top['feature'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in df_top['shap_value']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="<b>Local SHAP Feature Attribution</b><br><sup>Red = Contributed toward Risk | Green = Contributed toward Stability</sup>",
        xaxis_title="SHAP Value (Contribution to Risk Prediction)",
        yaxis_title="",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig
