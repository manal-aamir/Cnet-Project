from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PreprocessedData:
    """Container for preprocessed training and testing data."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    scaler: StandardScaler
    feature_names: List[str]
    class_names: List[str]


def load_and_preprocess(csv_path: str, test_size: float = 0.2, random_state: int = 42) -> PreprocessedData:
    """
    Load CSV data, preprocess features, and split into train/test sets.
    
    Args:
        csv_path: Path to the CSV dataset
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        PreprocessedData object containing train/test splits and encoders
    """
    print(f"\nLoading dataset: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    
    # Identify target column
    target_col = _infer_target_column(df)
    print(f"Target column: '{target_col}'")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert non-numeric features to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = _coerce_feature(X[col])
    
    feature_names = X.columns.tolist()
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()
    
    print(f"Classes: {class_names}")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return PreprocessedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        scaler=scaler,
        feature_names=feature_names,
        class_names=class_names
    )


def _infer_target_column(df: pd.DataFrame) -> str:
    """Infer the target column name from the dataframe."""
    possible_names = ['Class', 'class', 'Label', 'label', 'Attack type', 'attack_type']
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"Could not infer target column. Available columns: {df.columns.tolist()}")


def _coerce_feature(series: pd.Series) -> pd.Series:
    """Convert object-type series to numeric."""
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        le = LabelEncoder()
        return le.fit_transform(series.astype(str))
