import numpy as np
import pandas as pd
from typing import Tuple
from .config import cfg

def generate_features(n_samples: int = None, random_seed: int = None) -> pd.DataFrame:
    """Generate semi-realistic features for IoT-ish events."""
    if n_samples is None:
        n_samples = cfg.n_samples
    rng = np.random.RandomState(random_seed or cfg.random_seed)
    n = n_samples

    time_of_day = rng.randint(0, 24, size=n)
    device_trust = rng.beta(a=2.0, b=5.0, size=n)
    data_size_kb = rng.exponential(scale=50.0, size=n)
    base_priority = rng.rand(n)
    priority_score = np.clip(base_priority + (time_of_day >= 6) * 0.05, 0.0, 1.0)
    anomaly_score = rng.beta(a=1.5, b=8.0, size=n)
    spikes_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    anomaly_score[spikes_idx] += rng.rand(len(spikes_idx)) * 0.8
    anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
    sensor_variability = rng.beta(a=2.0, b=4.0, size=n)

    df = pd.DataFrame({
        "time_of_day": time_of_day,
        "device_trust": device_trust,
        "data_size_kb": data_size_kb,
        "priority_score": priority_score,
        "anomaly_score": anomaly_score,
        "sensor_variability": sensor_variability
    })
    return df
