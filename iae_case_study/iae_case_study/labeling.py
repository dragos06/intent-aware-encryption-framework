import numpy as np
from typing import Sequence
from .config import cfg

def label_intents(df, intent_set: Sequence[str] = None, intent_prior: Sequence[float] = None, noise_prob=None, rng_seed=None):
    """Rule-based + probabilistic labeling to simulate intents."""
    intent_set = intent_set or cfg.intent_set
    intent_prior = intent_prior or cfg.intent_prior
    noise_prob = cfg.label_noise_prob if noise_prob is None else noise_prob
    rng = np.random.RandomState(rng_seed or cfg.random_seed)
    labels = []

    for _, row in df.iterrows():
        scores = {intent: 0.0 for intent in intent_set}
        if row["anomaly_score"] > 0.7 or (row["priority_score"] > 0.85 and row["device_trust"] < 0.3):
            scores["emergency"] += 5.0
        if row["data_size_kb"] > 200 and row["device_trust"] > 0.4:
            scores["confidential-transfer"] += 4.0
        if row["data_size_kb"] < 50 and row["priority_score"] < 0.6:
            scores["routine-report"] += 3.0
        if 0.4 < row["anomaly_score"] <= 0.7 and row["sensor_variability"] > 0.5:
            scores["diagnostic"] += 2.5
        if row["anomaly_score"] < 0.3 and row["priority_score"] < 0.5 and 50 < row["data_size_kb"] <= 300:
            scores["maintenance"] += 1.8

        total = sum(scores.values())
        if total == 0.0:
            label = rng.choice(intent_set, p=intent_prior)
        else:
            intents = list(scores.keys())
            vals = np.array([scores[k] for k in intents])
            probs = np.exp(vals) / np.sum(np.exp(vals))
            label = rng.choice(intents, p=probs)

        if rng.rand() < noise_prob:
            alt = rng.choice([x for x in intent_set if x != label])
            label = alt

        labels.append(label)
    return np.array(labels)
