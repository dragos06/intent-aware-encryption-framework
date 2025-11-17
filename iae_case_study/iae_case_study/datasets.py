import pandas as pd
from pathlib import Path
import numpy as np

def load_cicids2017(flow_csv_path: str) -> pd.DataFrame:
    """
    Loads CICIDS2017 flow-based traffic dataset.
    Keeps commonly-used numeric columns; cleans NaNs/infs.
    """
    df = pd.read_csv(flow_csv_path)

    # Remove invalid or infinite values and drop pure-empty rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    # Normalize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    return df
