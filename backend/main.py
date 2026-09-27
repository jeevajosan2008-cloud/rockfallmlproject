"""
FastAPI Backend Service for Rockfall Risk Prediction and Explainable AI.
Exposes REST endpoints for prediction, SHAP attribution, what-if simulation, and dataset exploration.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.data_preprocessing import (
    load_and_validate_data,
    ALL_FEATURES,
    FEATURE_METADATA
)
from src.explainability import RockfallExplainer

# Initialize FastAPI App
app = FastAPI(
    title="Rockfall Risk Prediction & Explainable AI API",
    description="Backend microservice providing machine learning inference, calibrated probabilities, and SHAP explainability for open-pit mine slope stability assessment.",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models and Metadata on Startup
MODEL_PATH = os.path.join("model", "rockfall_model.pkl")
ALL_MODELS_PATH = os.path.join("model", "all_models.pkl")
METADATA_PATH = os.path.join("model", "model_metadata.json")
DATASET_PATH = os.path.join("dataset", "slope_stability_dataset.csv")

try:
    primary_pipeline = joblib.load(MODEL_PATH)
    all_models = joblib.load(ALL_MODELS_PATH) if os.path.exists(ALL_MODELS_PATH) else {
        "SVM": primary_pipeline
    }
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("Backend: ML models and metadata loaded successfully.")
except Exception as e:
    print(f"Backend Warning: Could not pre-load model artifacts: {e}")
    primary_pipeline = None
    all_models = {}
    metadata = {}

# Lazy-loaded SHAP Explainer
explainer_cache: Dict[str, RockfallExplainer] = {}


def get_explainer(model_name: str = "SVM") -> RockfallExplainer:
    if model_name not in explainer_cache:
        target_model = all_models.get(model_name, primary_pipeline)
        if target_model is None:
            raise HTTPException(status_code=500, detail="Target model not available.")
        # Load background data
        df = load_and_validate_data(DATASET_PATH)[ALL_FEATURES]
        explainer_cache[model_name] = RockfallExplainer(target_model, background_data=df)
    return explainer_cache[model_name]


# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------
class SlopeConditions(BaseModel):
    unit_weight: float = Field(20.0, ge=15.0, le=25.0, description="Bulk unit weight in kN/m³")
    cohesion: float = Field(25.0, ge=5.0, le=50.0, description="Cohesion in kPa")
    friction_angle: float = Field(32.0, ge=20.0, le=45.0, description="Internal friction angle in degrees")
    slope_angle: float = Field(35.0, ge=10.0, le=60.0, description="Slope inclination in degrees")
    slope_height: float = Field(25.0, ge=5.0, le=50.0, description="Slope height in meters")
    pore_water_pressure_ratio: float = Field(0.30, ge=0.0, le=1.0, description="Pore pressure ratio r_u")
    reinforcement_type: str = Field("Soil Nailing", description="Support measure (Drainage, Geosynthetics, Retaining Wall, Soil Nailing)")
    model_name: Optional[str] = Field("SVM", description="Inference classifier to execute")


class PredictionResult(BaseModel):
    prediction_class: str
    risk_code: int
    predicted_probability_percent: float
    model_used: str
    status_indicator: str
    geotechnical_interpretation: str
    disclaimer: str


class WhatIfRequest(BaseModel):
    scenario_a: SlopeConditions
    scenario_b: SlopeConditions


# ------------------------------------------------------------------------------
# REST API Endpoints
# ------------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint indicating model readiness."""
    return {
        "status": "healthy",
        "models_loaded": list(all_models.keys()),
        "selected_model": metadata.get("selected_model", "SVM"),
        "version": "1.0.0"
    }


@app.get("/api/metadata", tags=["Model Information"])
def get_metadata():
    """Returns training metadata, dataset statistics, and empirical validation metrics."""
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return metadata


@app.get("/api/models", tags=["Model Information"])
def get_model_benchmark():
    """Returns performance metrics and confusion matrices for all evaluated models."""
    evals = metadata.get("all_model_evaluations", {})
    return {
        "selected_model": metadata.get("selected_model", "SVM"),
        "models": evals
    }


@app.post("/api/predict", response_model=PredictionResult, tags=["Inference"])
def predict_risk(inputs: SlopeConditions):
    """
    Executes model inference on user-provided slope conditions.
    Returns binary classification, calibrated failure probability, and visual indicator.
    """
    target_pipeline = all_models.get(inputs.model_name, primary_pipeline)
    if target_pipeline is None:
        raise HTTPException(status_code=500, detail="Inference pipeline not initialized.")

    input_df = pd.DataFrame([{
        "unit_weight": inputs.unit_weight,
        "cohesion": inputs.cohesion,
        "friction_angle": inputs.friction_angle,
        "slope_angle": inputs.slope_angle,
        "slope_height": inputs.slope_height,
        "pore_water_pressure_ratio": inputs.pore_water_pressure_ratio,
        "reinforcement_type": inputs.reinforcement_type
    }])

    pred_code = int(target_pipeline.predict(input_df)[0])
    prob = float(target_pipeline.predict_proba(input_df)[0][1])

    is_risk = pred_code == 1
    return PredictionResult(
        prediction_class="RISK / UNSTABLE" if is_risk else "STABLE",
        risk_code=pred_code,
        predicted_probability_percent=round(prob * 100, 2),
        model_used=inputs.model_name or metadata.get("selected_model", "SVM"),
        status_indicator="🔴 RISK / UNSTABLE" if is_risk else "🟢 STABLE",
        geotechnical_interpretation=(
            "The model predicts critical instability conditions based on adverse shear stress and pore pressure."
            if is_risk else
            "The model predicts stable equilibrium governed by adequate resisting shear strength."
        ),
        disclaimer="Educational Prototype — Not for operational mine safety."
    )


@app.post("/api/explain", tags=["Explainable AI"])
def explain_prediction(inputs: SlopeConditions):
    """
    Computes local SHAP attributions decomposing the prediction into feature contributions.
    """
    input_df = pd.DataFrame([{
        "unit_weight": inputs.unit_weight,
        "cohesion": inputs.cohesion,
        "friction_angle": inputs.friction_angle,
        "slope_angle": inputs.slope_angle,
        "slope_height": inputs.slope_height,
        "pore_water_pressure_ratio": inputs.pore_water_pressure_ratio,
        "reinforcement_type": inputs.reinforcement_type
    }])

    try:
        explainer = get_explainer(inputs.model_name or "SVM")
        explanation = explainer.explain_instance(input_df)
        return {
            "model_used": inputs.model_name or "SVM",
            "base_value": explanation["base_value"],
            "attributions": explanation["attributions"],
            "top_risk_increasers": explanation["top_risk_increasers"],
            "top_risk_reducers": explanation["top_risk_reducers"],
            "scientific_note": "This feature contributed to the model prediction (correlative attribution, not physical causation)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")


@app.post("/api/whatif", tags=["Sensitivity Analysis"])
def what_if_analysis(request: WhatIfRequest):
    """
    Compares two slope scenarios and quantifies the probability delta and prediction shifts.
    """
    target_pipeline = primary_pipeline
    if target_pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    df_a = pd.DataFrame([request.scenario_a.model_dump()])
    df_b = pd.DataFrame([request.scenario_b.model_dump()])

    prob_a = float(target_pipeline.predict_proba(df_a)[0][1]) * 100
    pred_a = int(target_pipeline.predict(df_a)[0])

    prob_b = float(target_pipeline.predict_proba(df_b)[0][1]) * 100
    pred_b = int(target_pipeline.predict(df_b)[0])

    delta = prob_b - prob_a

    return {
        "scenario_a": {
            "prediction": "RISK / UNSTABLE" if pred_a == 1 else "STABLE",
            "probability_percent": round(prob_a, 2)
        },
        "scenario_b": {
            "prediction": "RISK / UNSTABLE" if pred_b == 1 else "STABLE",
            "probability_percent": round(prob_b, 2)
        },
        "probability_delta_percent": round(delta, 2),
        "prediction_shift": "Unchanged" if pred_a == pred_b else f"{'STABLE' if pred_a == 0 else 'RISK'} -> {'STABLE' if pred_b == 0 else 'RISK'}"
    }


@app.get("/api/dataset/sample", tags=["Dataset"])
def get_dataset_sample(limit: int = Query(50, ge=1, le=500)):
    """Returns sample records from the slope stability dataset."""
    df = load_and_validate_data(DATASET_PATH)
    sample_records = df.head(limit).to_dict(orient="records")
    return {
        "total_records": len(df),
        "returned_records": len(sample_records),
        "data": sample_records
    }


# Mount Static Frontend Files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "FastAPI Backend is running. Open /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
