import numpy as np
import pandas as pd
from typing import Tuple
from .config import cfg

def generate_features(n_samples: int = None, random_seed: int = None) -> pd.DataFrame:
    """Small synthetic generator retained only for lightweight debugging."""
    if n_samples is None:
        n_samples = cfg.n_samples
    rng = np.random.RandomState(random_seed or cfg.random_seed)
    n = n_samples

    # lightweight synthetic flow-like features
    flow_duration = rng.exponential(scale=1.0, size=n) * 1000.0  # ms
    total_fwd_pkts = rng.poisson(lam=10, size=n)
    total_bwd_pkts = rng.poisson(lam=8, size=n)
    total_bytes = rng.exponential(scale=2000.0, size=n)

    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2025-01-01", periods=n, freq="s"),
        "flow_duration": flow_duration,
        "total_fwd_pkts": total_fwd_pkts,
        "total_bwd_pkts": total_bwd_pkts,
        "total_bytes": total_bytes,
        "total_length_fwd": total_bytes * 0.6,
        "total_length_bwd": total_bytes * 0.4,
        "data_size_kb": total_bytes / 1024.0,
        "intent_label": ["BENIGN"] * n
    })
    return df
