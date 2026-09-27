"""
Automated Test Suite for Rockfall Prediction System.
Tests data preprocessing, model inference, probability bounds, and SHAP explainability.
"""

import os
import joblib
import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    load_and_validate_data,
    standardize_columns,
    create_risk_target,
    build_preprocessor,
    ALL_FEATURES,
    TARGET_COLUMN
)
from src.explainability import RockfallExplainer


@pytest.fixture
def dataset_path():
    return os.path.join("dataset", "slope_stability_dataset.csv")


@pytest.fixture
def trained_model():
    model_path = os.path.join("model", "rockfall_model.pkl")
    assert os.path.exists(model_path), "Trained model rockfall_model.pkl must exist."
    return joblib.load(model_path)


def test_data_loading_and_structure(dataset_path):
    """Verifies dataset loads with 10,000 rows and standardized headers."""
    df = load_and_validate_data(dataset_path)
    assert len(df) == 10000
    assert df.isnull().sum().sum() == 0
    for feat in ALL_FEATURES:
        assert feat in df.columns


def test_target_creation(dataset_path):
    """Verifies target binary mapping: FS < 1.0 -> 1, FS >= 1.0 -> 0."""
    df = load_and_validate_data(dataset_path)
    df = create_risk_target(df, threshold=1.0)
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}
    # Check threshold logic
    risk_mask = df["factor_of_safety"] < 1.0
    assert (df.loc[risk_mask, TARGET_COLUMN] == 1).all()
    assert (df.loc[~risk_mask, TARGET_COLUMN] == 0).all()


def test_preprocessor_pipeline():
    """Verifies ColumnTransformer correctly transforms numerical and categorical inputs."""
    sample_df = pd.DataFrame([{
        "unit_weight": 20.0,
        "cohesion": 25.0,
        "friction_angle": 32.0,
        "slope_angle": 35.0,
        "slope_height": 25.0,
        "pore_water_pressure_ratio": 0.3,
        "reinforcement_type": "Soil Nailing"
    }])
    preprocessor = build_preprocessor()
    preprocessor.fit(sample_df)
    transformed = preprocessor.transform(sample_df)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] == 10  # 6 numerical + 4 one-hot categories


def test_model_inference_and_probabilities(trained_model):
    """Tests model prediction and calibrated probability bounds."""
    # Test safe conditions
    safe_input = pd.DataFrame([{
        "unit_weight": 18.0,
        "cohesion": 45.0,
        "friction_angle": 42.0,
        "slope_angle": 15.0,
        "slope_height": 10.0,
        "pore_water_pressure_ratio": 0.05,
        "reinforcement_type": "Retaining Wall"
    }])
    pred_safe = trained_model.predict(safe_input)[0]
    prob_safe = trained_model.predict_proba(safe_input)[0][1]

    assert pred_safe in [0, 1]
    assert 0.0 <= prob_safe <= 1.0
    assert pred_safe == 0
    assert prob_safe < 0.20

    # Test hazardous conditions
    hazardous_input = pd.DataFrame([{
        "unit_weight": 24.5,
        "cohesion": 6.0,
        "friction_angle": 21.0,
        "slope_angle": 58.0,
        "slope_height": 48.0,
        "pore_water_pressure_ratio": 0.90,
        "reinforcement_type": "Drainage"
    }])
    pred_hazard = trained_model.predict(hazardous_input)[0]
    prob_hazard = trained_model.predict_proba(hazardous_input)[0][1]

    assert pred_hazard in [0, 1]
    assert 0.0 <= prob_hazard <= 1.0
    assert pred_hazard == 1
    assert prob_hazard > 0.80


def test_shap_explanation(trained_model, dataset_path):
    """Verifies SHAP explainer correctly generates local attributions."""
    df = load_and_validate_data(dataset_path)[ALL_FEATURES]
    explainer = RockfallExplainer(trained_model, background_data=df.iloc[:20])

    sample = df.iloc[0:1]
    explanation = explainer.explain_instance(sample)

    assert "base_value" in explanation
    assert "attributions" in explanation
    assert len(explanation["attributions"]) == 10
    assert "top_risk_increasers" in explanation
    assert "top_risk_reducers" in explanation
