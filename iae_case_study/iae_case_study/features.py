import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def build_feature_pipeline(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler and return standardized feature matrix and the scaler.
    This accepts only numeric columns; non-numeric columns must be removed before calling.
    """
    numeric_df = df.select_dtypes(include=["number"]).fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_df.values)
    return X, scaler
