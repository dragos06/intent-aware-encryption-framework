from typing import Tuple, Dict, Any
from .config import cfg
import numpy as np

def policy_static_strong(event_row: Dict[str, Any]) -> Tuple[str, Dict]:
    return "high", {}

def policy_context_only(event_row: Dict[str, Any], thresholds: Dict = None) -> Tuple[str, Dict]:
    thresholds = thresholds or {"anomaly": 0.6, "data_size_kb": 150, "priority": 0.8}
    if event_row["anomaly_score"] >= thresholds["anomaly"]:
        return "high", {}
    if event_row["data_size_kb"] >= thresholds["data_size_kb"] and event_row["priority_score"] >= thresholds["priority"]:
        return "medium", {}
    return "low", {}

def policy_rule_based_intent_mapping(event_row: Dict[str, Any]) -> Tuple[str, Dict]:
    mapping = {
        "emergency": "high",
        "confidential-transfer": "medium",
        "routine-report": "low",
        "diagnostic": "low",
        "maintenance": "low"
    }
    intent = event_row.get("intent", None)
    return mapping.get(intent, "low"), {}

def policy_iae_ml(event_row: Dict[str, Any], estimator, prob_thresholds: Dict = None) -> Tuple[str, Dict]:
    prob_thresholds = prob_thresholds or {"emergency": 0.5, "confidential-transfer": 0.45}
    X_vec = event_row["_X_vec"].reshape(1, -1)
    probs = None
    try:
        probs = estimator.predict_proba(X_vec)[0]
        classes = estimator.classes_
    except Exception:
        pred = estimator.predict(X_vec)[0]
        classes = estimator.classes_
        probs = np.array([1.0 if c == pred else 0.0 for c in classes])

    prob_map = {c: p for c, p in zip(classes, probs)}
    if prob_map.get("emergency", 0.0) >= prob_thresholds["emergency"]:
        return "high", prob_map
    if prob_map.get("confidential-transfer", 0.0) >= prob_thresholds["confidential-transfer"]:
        return "medium", prob_map
    top_intent = max(prob_map.items(), key=lambda kv: kv[1])[0]
    return "low", prob_map
