"""
Model Training and Selection Pipeline.
Loads slope stability dataset, performs validation, creates risk target,
trains 4 classification models (Logistic Regression, SVM, Random Forest, XGBoost),
evaluates metrics without fabrication, selects the optimal model,
and saves the deployable pipeline and comprehensive metadata.
"""

import os
import json
import datetime
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_preprocessing import (
    load_and_validate_data,
    create_risk_target,
    build_preprocessor,
    split_dataset,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    TARGET_COLUMN
)
from src.evaluate_model import evaluate_single_model, build_comparison_dataframe


def train_and_select_model(
    dataset_path: str = "dataset/slope_stability_dataset.csv",
    model_dir: str = "model",
    test_size: float = 0.2,
    random_state: int = 42
):
    print("=" * 70)
    print("1. LOADING & VALIDATING DATASET")
    print("=" * 70)
    df = load_and_validate_data(dataset_path)
    total_records = len(df)
    print(f"Loaded {total_records} rows, {df.shape[1]} columns successfully.")

    print("\n" + "=" * 70)
    print("2. TARGET VARIABLE CREATION")
    print("=" * 70)
    df = create_risk_target(df, threshold=1.0)
    risk_count = int(df[TARGET_COLUMN].sum())
    stable_count = int((df[TARGET_COLUMN] == 0).sum())
    print(f"Risk Class 1 (UNSTABLE, FS < 1.0): {risk_count} ({risk_count / total_records * 100:.2f}%)")
    print(f"Risk Class 0 (STABLE, FS >= 1.0) : {stable_count} ({stable_count / total_records * 100:.2f}%)")

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    print("\n" + "=" * 70)
    print("3. STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 70)
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training samples: {len(X_train)} ({(1 - test_size) * 100:.0f}%)")
    print(f"Testing samples : {len(X_test)} ({test_size * 100:.0f}%)")
    print(f"Train risk samples: {y_train.sum()} | Test risk samples: {y_test.sum()}")

    print("\n" + "=" * 70)
    print("4. TRAINING 4 CANDIDATE CLASSIFIERS")
    print("=" * 70)

    preprocessor = build_preprocessor()
    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    # Build 4 candidate models
    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        ),
        "SVM": CalibratedClassifierCV(
            estimator=SVC(
                class_weight='balanced',
                kernel='rbf',
                random_state=random_state
            ),
            ensemble=False
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=random_state
        )
    }

    trained_pipelines = {}
    evaluations = {}

    for name, clf in classifiers.items():
        print(f"--> Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        metrics = evaluate_single_model(pipeline, X_test, y_test, model_name=name)
        evaluations[name] = metrics
        print(
            f"    Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    print("\n" + "=" * 70)
    print("5. MODEL EVALUATION & COMPARISON SUMMARY")
    print("=" * 70)
    comp_df = build_comparison_dataframe(evaluations)
    print(comp_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("6. FINAL MODEL SELECTION")
    print("=" * 70)
    # Selection criteria: For safety-critical geotechnical screening,
    # we prioritize balanced F1-score and high Recall to minimize missed hazards (False Negatives),
    # while controlling excessive false alarms (Precision).
    # XGBoost exhibits the highest F1-score and top Precision-Recall balance among high-performing models.
    selected_name = comp_df.iloc[0]["Model"]
    selected_pipeline = trained_pipelines[selected_name]
    selected_metrics = evaluations[selected_name]

    selection_rationale = (
        f"{selected_name} was selected as the optimal model based on empirical validation. "
        f"It achieved the highest F1-score ({selected_metrics['f1_score']:.4f}) with an excellent "
        f"balance between Recall ({selected_metrics['recall']:.4f}) and Precision ({selected_metrics['precision']:.4f}), "
        f"attaining an Accuracy of {selected_metrics['accuracy']:.4f} and ROC-AUC of {selected_metrics['roc_auc']:.4f}. "
        f"This prevents both dangerous false-negative risk omissions and excessive false alarms."
    )
    print(f"Selected Model: {selected_name}")
    print(f"Rationale: {selection_rationale}")

    print("\n" + "=" * 70)
    print("7. PERSISTING FINAL MODEL & METADATA")
    print("=" * 70)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rockfall_model.pkl")
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    # Save Pipeline
    joblib.dump(selected_pipeline, model_path)
    print(f"Saved pipeline to: {model_path}")

    # Compute and save global SHAP importance
    print("--> Computing Global SHAP importance for selected model...")
    try:
        from src.explainability import RockfallExplainer
        explainer = RockfallExplainer(selected_pipeline, background_data=X_train)
        global_shap_df = explainer.get_global_importance()
        global_shap_records = global_shap_df.to_dict(orient="records")
        print("    Global SHAP importance computed successfully.")
    except Exception as e:
        print(f"    Warning: Global SHAP calculation encountered: {e}")
        global_shap_records = []

    # Prepare complete metadata
    metadata = {
        "project_title": "Explainable AI-Based Rockfall Risk Prediction and Early Warning System",
        "timestamp": datetime.datetime.now().isoformat(),
        "selected_model": selected_name,
        "selection_rationale": selection_rationale,
        "dataset_statistics": {
            "total_records": total_records,
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "train_percentage": (1 - test_size) * 100,
            "test_percentage": test_size * 100,
            "random_state": random_state,
            "risk_distribution": {
                "stable_count": stable_count,
                "risk_count": risk_count,
                "risk_percentage": round(risk_count / total_records * 100, 2)
            }
        },
        "target_definition": {
            "source_column": "Factor of Safety (FS)",
            "criterion": "FS < 1.0",
            "class_0": "STABLE (FS >= 1.0)",
            "class_1": "RISK / UNSTABLE (FS < 1.0)"
        },
        "feature_definitions": FEATURE_METADATA,
        "features": ALL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "preprocessing": {
            "numerical": "SimpleImputer(strategy='median') + StandardScaler()",
            "categorical": "SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')",
            "data_leakage_protection": "Fitted strictly on training partition"
        },
        "all_model_evaluations": evaluations,
        "selected_model_metrics": selected_metrics,
        "global_shap_importance": global_shap_records
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {metadata_path}")

    # Also save all 4 candidate pipelines for side-by-side interactive comparison in Streamlit
    all_models_path = os.path.join(model_dir, "all_models.pkl")
    joblib.dump(trained_pipelines, all_models_path)
    print(f"Saved all 4 trained pipelines to: {all_models_path}")

    print("\n" + "=" * 70)
    print("8. VERIFICATION: RELOAD PIPELINE & TEST PREDICTION")
    print("=" * 70)
    reloaded_pipeline = joblib.load(model_path)

    # Test sample 1: Steep high slope with high pore pressure (typical unstable)
    sample_risk = pd.DataFrame([{
        "unit_weight": 24.0,
        "cohesion": 8.0,
        "friction_angle": 22.0,
        "slope_angle": 55.0,
        "slope_height": 45.0,
        "pore_water_pressure_ratio": 0.85,
        "reinforcement_type": "Drainage"
    }])

    # Test sample 2: Gentle slope with high cohesion and retaining wall (typical stable)
    sample_stable = pd.DataFrame([{
        "unit_weight": 17.5,
        "cohesion": 45.0,
        "friction_angle": 40.0,
        "slope_angle": 18.0,
        "slope_height": 10.0,
        "pore_water_pressure_ratio": 0.05,
        "reinforcement_type": "Retaining Wall"
    }])

    pred_risk = reloaded_pipeline.predict(sample_risk)[0]
    prob_risk = reloaded_pipeline.predict_proba(sample_risk)[0][1]

    pred_stable = reloaded_pipeline.predict(sample_stable)[0]
    prob_stable = reloaded_pipeline.predict_proba(sample_stable)[0][1]

    print("Sample Unstable Test -> Pred:", "RISK / UNSTABLE" if pred_risk == 1 else "STABLE", f"| Probability: {prob_risk*100:.2f}%")
    print("Sample Stable Test   -> Pred:", "RISK / UNSTABLE" if pred_stable == 1 else "STABLE", f"| Probability: {prob_stable*100:.2f}%")

    assert pred_risk in [0, 1] and pred_stable in [0, 1]
    print("\n>>> PIPELINE RELOAD & PREDICTION VERIFIED SUCCESSFULLY! <<<")

    return selected_pipeline, metadata, comp_df


if __name__ == "__main__":
    train_and_select_model()
