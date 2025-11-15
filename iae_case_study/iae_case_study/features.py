import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def build_feature_pipeline(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler and return standardized feature matrix and the scaler.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    return X, scaler
