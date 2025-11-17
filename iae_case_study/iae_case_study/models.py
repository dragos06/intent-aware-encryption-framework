from typing import Dict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .config import cfg
import numpy as np
import pandas as pd

def prepare_dataset_with_anomaly(df: pd.DataFrame, feature_cols: list = None, random_seed: int = None) -> pd.DataFrame:
    """
    Compute deterministic device_trust, priority_score, and anomaly_score (0..1).
    - device_trust: heuristic inverse of normalized total_packets (higher packets -> lower trust)
    - priority_score: normalized sensitivity_level (higher sensitivity -> higher priority)
    - anomaly_score: IsolationForest trained on inferred benign subset; anomaly score scaled 0..1
    Returns df with added columns: device_trust, priority_score, anomaly_score.
    """
    rng_seed = random_seed or cfg.random_seed
    df = df.copy()

    # device_trust: inverse of total_packets (normalized)
    if "total_packets" in df.columns:
        pkt = df["total_packets"].astype(float).fillna(0.0)
        pkt_norm = (pkt - pkt.min()) / (pkt.max() - pkt.min() + 1e-9)
        df["device_trust"] = 1.0 - pkt_norm  # higher packets => lower trust
    else:
        df["device_trust"] = 0.5

    # priority_score: based on sensitivity_level and data_size_kb
    if "sensitivity_level" in df.columns:
        sl = df["sensitivity_level"].astype(float).fillna(1.0)
        size = df.get("data_size_kb", pd.Series(0.0, index=df.index)).astype(float)
        size_norm = (size - size.min()) / (size.max() - size.min() + 1e-9)
        df["priority_score"] = (sl - sl.min()) / (sl.max() - sl.min() + 1e-9) * 0.7 + size_norm * 0.3
    else:
        df["priority_score"] = 0.5

    # Anomaly detection using IsolationForest trained on "benign" labeled flows (if present)
    feat_cols = feature_cols or ["flow_duration", "total_packets", "data_size_kb"]
    available_feats = [c for c in feat_cols if c in df.columns]
    feat_matrix = df[available_feats].fillna(0.0).values

    # Train IsolationForest on presumed benign subset (intent_label contains 'BENIGN')
    rng = np.random.RandomState(rng_seed)
    if "intent_label" in df.columns:
        benign_mask = df["intent_label"].str.upper().str.contains("BENIGN", na=False)
        if benign_mask.sum() >= 10:
            base_train = feat_matrix[benign_mask.values]
        else:
            base_train = feat_matrix  # fallback
    else:
        base_train = feat_matrix

    iso = IsolationForest(random_state=rng_seed, contamination="auto")
    try:
        iso.fit(base_train)
        # decision_function: higher => more normal, negative => anomalous
        scores = iso.decision_function(feat_matrix)
        # convert to anomaly score in 0..1 where 1 is most anomalous
        # normalize so that min(decision_function) -> 1, max -> 0
        s_min, s_max = float(scores.min()), float(scores.max())
        if s_max - s_min > 0:
            anomaly_score = 1.0 - (scores - s_min) / (s_max - s_min)
        else:
            anomaly_score = (scores * 0.0)
        df["anomaly_score"] = anomaly_score
    except Exception:
        df["anomaly_score"] = 0.0

    return df


def train_classifiers(X_train, y_train, params=None):
    """Train RF, LR, and MLP with lightweight gridsearch."""
    params = params or {}
    models = {}

    rf = RandomForestClassifier(random_state=cfg.random_seed)
    rf_grid = GridSearchCV(rf, params.get("rf_params", {"n_estimators": [50]}), cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    models["random_forest"] = rf_grid.best_estimator_

    lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
    lr_grid = GridSearchCV(lr, params.get("lr_params", {"C": [1.0]}), cv=3, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    models["logistic_regression"] = lr_grid.best_estimator_

    mlp = MLPClassifier(random_state=cfg.random_seed, max_iter=300)
    mlp_grid = GridSearchCV(mlp, params.get("mlp_params", {"hidden_layer_sizes": [(50,)]}), cv=3, n_jobs=-1)
    mlp_grid.fit(X_train, y_train)
    models["mlp"] = mlp_grid.best_estimator_

    return models

def evaluate_classifier(estimator, X_test, y_test, labels):
    y_pred = estimator.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }
