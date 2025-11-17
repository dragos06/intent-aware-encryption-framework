import numpy as np
import pandas as pd
from typing import Dict
from .utils import ensure_dir, now_str, save_json
from .config import cfg
from .encryption import SimulatedEncryptor, estimate_energy
from .policies import (
    policy_static_strong,
    policy_context_only,
    policy_rule_based_intent_mapping,
    policy_iae_ml,
)
from .models import prepare_dataset_with_anomaly
import logging

logger = logging.getLogger(__name__)

def assemble_profile_map():
    return {k: SimulatedEncryptor(v) for k, v in cfg.encryption_profiles.items()}

def run_experiments(df_events: pd.DataFrame, X_matrix: np.ndarray, y_labels: np.ndarray, models: Dict, outdir: str):
    ensure_dir(outdir)
    profile_map = assemble_profile_map()

    # attach _X_vec for model-based policies
    df_events["_X_vec"] = list(X_matrix)

    policies = {
        "STATIC_STRONG": lambda row: policy_static_strong(row),
        "CONTEXT_ONLY": lambda row: policy_context_only(row),
        "RULE_BASED_INTENT_MAPPING": lambda row: policy_rule_based_intent_mapping(row),
        "ABE_SIM": lambda row: ("abe_sim", {})
    }

    # Add IAE policies per trained model
    for model_name, est in models.items():
        policies[f"IAE_{model_name}"] = (lambda est_=est: (lambda row: policy_iae_ml(row, est_)))()

    rng = np.random.RandomState(cfg.random_seed)
    n_sample_events = min(200, len(df_events))
    sample_idx = rng.choice(len(df_events), size=n_sample_events, replace=False)

    results = []
    for policy_name, policy_callable in policies.items():
        latencies = []
        energies = []
        total_ops = 0
        successful_ops = 0
        clf_true = []
        clf_pred = []

        for idx in sample_idx:
            row = df_events.iloc[idx]
            action, meta = policy_callable(row)
            profile_key = action if action in profile_map else action
            if profile_key not in profile_map:
                profile_key = "high"
            encryptor = profile_map[profile_key]

            for _ in range(cfg.n_runs_per_policy):
                payload = f"EVENT:{idx}:size={row.get('data_size_kb', 0.0):.2f}"
                try:
                    _, elapsed_ms = encryptor.encrypt(payload)
                except Exception as e:
                    logger.exception("Encryption error: %s", e)
                    elapsed_ms = cfg.encryption_profiles.get(profile_key).base_latency_ms
                energy = estimate_energy(elapsed_ms, cfg.encryption_profiles.get(profile_key).cpu_coeff, cfg.energy_unit_per_ms)
                latencies.append(elapsed_ms)
                energies.append(energy)
                total_ops += 1
                successful_ops += 1

            if policy_name.startswith("IAE_") and isinstance(meta, dict) and meta:
                # meta contains probability map
                pred_intent = max(meta.items(), key=lambda kv: kv[1])[0]
                clf_pred.append(pred_intent)
                clf_true.append(row["intent"])

        avg_latency = float(np.mean(latencies)) if latencies else float("nan")
        median_latency = float(np.median(latencies)) if latencies else float("nan")
        avg_energy = float(np.mean(energies)) if energies else float("nan")
        throughput = float(successful_ops / (np.sum(latencies)/1000.0)) if latencies and np.sum(latencies) > 0 else float("nan")

        summary = {
            "policy": policy_name,
            "avg_latency_ms": avg_latency,
            "median_latency_ms": median_latency,
            "avg_energy_units": avg_energy,
            "throughput_ops_per_sec": throughput,
            "total_ops": int(total_ops),
            "successful_ops": int(successful_ops),
            "n_sample_events": int(n_sample_events)
        }

        if clf_true:
            # compute simple classification metrics for the IAE policy run
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            acc = float(accuracy_score(clf_true, clf_pred))
            prec, rec, f1, _ = precision_recall_fscore_support(clf_true, clf_pred, average="weighted")
            summary.update({
                "clf_acc": acc,
                "clf_precision": float(prec),
                "clf_recall": float(rec),
                "clf_f1": float(f1)
            })

        results.append(summary)
        # Save raw per-policy traces
        df_trace = pd.DataFrame({"latency_ms": latencies, "energy_units": energies})
        trace_path = f"{outdir}/policy_{policy_name}_{now_str()}.csv"
        df_trace.to_csv(trace_path, index=False)

    summary_df = pd.DataFrame(results)
    summary_path = f"{outdir}/summary_results_{now_str()}.csv"
    summary_df.to_csv(summary_path, index=False)
    save_json(summary_df.to_dict(orient="records"), f"{outdir}/summary_results_{now_str()}.json")
    return summary_df, results
