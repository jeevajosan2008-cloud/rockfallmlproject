"""
Data Preprocessing Pipeline for Rockfall Risk Prediction.
Handles data loading, validation, feature standardization, target generation,
and reproducible preprocessing pipelines (imputation, scaling, one-hot encoding).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Feature Definitions and Metadata
FEATURE_METADATA = {
    'unit_weight': {
        'display_name': 'Unit Weight (kN/m³)',
        'unit': 'kN/m³',
        'type': 'numerical',
        'min': 15.0,
        'max': 25.0,
        'default': 20.0,
        'description': 'Bulk unit weight of the rock/soil mass forming the slope.'
    },
    'cohesion': {
        'display_name': 'Cohesion (kPa)',
        'unit': 'kPa',
        'type': 'numerical',
        'min': 5.0,
        'max': 50.0,
        'default': 25.0,
        'description': 'Internal shear strength holding rock/soil particles together independent of stress.'
    },
    'friction_angle': {
        'display_name': 'Internal Friction Angle (°)',
        'unit': '°',
        'type': 'numerical',
        'min': 20.0,
        'max': 45.0,
        'default': 32.0,
        'description': 'Angle of shearing resistance of the slope material.'
    },
    'slope_angle': {
        'display_name': 'Slope Angle (°)',
        'unit': '°',
        'type': 'numerical',
        'min': 10.0,
        'max': 60.0,
        'default': 35.0,
        'description': 'Angle of the slope face relative to the horizontal plane.'
    },
    'slope_height': {
        'display_name': 'Slope Height (m)',
        'unit': 'm',
        'type': 'numerical',
        'min': 5.0,
        'max': 50.0,
        'default': 25.0,
        'description': 'Vertical height difference from the slope crest to toe.'
    },
    'pore_water_pressure_ratio': {
        'display_name': 'Pore Water Pressure Ratio (r_u)',
        'unit': 'ratio [0-1]',
        'type': 'numerical',
        'min': 0.0,
        'max': 1.0,
        'default': 0.3,
        'description': 'Ratio of groundwater pore pressure to total overburden vertical stress.'
    },
    'reinforcement_type': {
        'display_name': 'Reinforcement Type',
        'unit': 'category',
        'type': 'categorical',
        'categories': ['Drainage', 'Geosynthetics', 'Retaining Wall', 'Soil Nailing'],
        'default': 'Soil Nailing',
        'description': 'Engineered slope stabilization or support measure applied.'
    }
}

NUMERICAL_FEATURES = [
    'unit_weight',
    'cohesion',
    'friction_angle',
    'slope_angle',
    'slope_height',
    'pore_water_pressure_ratio'
]

CATEGORICAL_FEATURES = ['reinforcement_type']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
RAW_TARGET_COLUMN = 'factor_of_safety'
TARGET_COLUMN = 'risk'


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize raw column names across character encoding variations.
    """
    rename_rules = {}
    for col in df.columns:
        col_clean = str(col).strip()
        if 'Unit Weight' in col_clean:
            rename_rules[col] = 'unit_weight'
        elif 'Cohesion' in col_clean:
            rename_rules[col] = 'cohesion'
        elif 'Internal Friction' in col_clean or 'Friction Angle' in col_clean:
            rename_rules[col] = 'friction_angle'
        elif 'Slope Angle' in col_clean:
            rename_rules[col] = 'slope_angle'
        elif 'Slope Height' in col_clean:
            rename_rules[col] = 'slope_height'
        elif 'Pore Water' in col_clean:
            rename_rules[col] = 'pore_water_pressure_ratio'
        elif 'Reinforcement Type' in col_clean:
            rename_rules[col] = 'reinforcement_type'
        elif 'Reinforcement Numeric' in col_clean:
            rename_rules[col] = 'reinforcement_numeric'
        elif 'Factor of Safety' in col_clean or col_clean == 'FS':
            rename_rules[col] = 'factor_of_safety'
    return df.rename(columns=rename_rules)


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from CSV, standardizes column headers, and verifies structural integrity.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at expected path: {filepath}")

    # Read CSV supporting UTF-8 with fallback
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')

    df = standardize_columns(df)

    # Validate essential columns
    required = set(ALL_FEATURES + [RAW_TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def create_risk_target(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Creates binary classification target based on Factor of Safety (FS).
    FS < 1.0  => RISK / UNSTABLE (1)
    FS >= 1.0 => STABLE (0)
    """
    df = df.copy()
    if RAW_TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{RAW_TARGET_COLUMN}' missing from dataframe.")

    df[TARGET_COLUMN] = (df[RAW_TARGET_COLUMN] < threshold).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
    Ensures zero data leakage by computing statistics solely on training data.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            categories=[FEATURE_METADATA['reinforcement_type']['categories']],
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def split_dataset(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target using stratified sampling to preserve rare-class proportion.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Retrieves human-readable transformed feature names from fitted ColumnTransformer.
    """
    names = []
    # Numerical
    names.extend(NUMERICAL_FEATURES)
    # Categorical
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    # Clean prefixes
    clean_cat_names = [c.replace('reinforcement_type_', 'Reinforcement: ') for c in cat_names]
    names.extend(clean_cat_names)
    return names
