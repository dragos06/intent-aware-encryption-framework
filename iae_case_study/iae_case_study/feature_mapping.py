import pandas as pd
import numpy as np

def map_cicids_to_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CIC-IDS flow table into a compact event record useful for intent-aware
    processing. Handles messy column names automatically.
    """

    # clean column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # auto-detect label column
    label_candidates = ["Label", "class", "Class", "Attack", "Attack_type", "attack_cat"]

    label_col = None
    for c in df.columns:
        cn = c.strip().lower()
        if cn in [lc.lower() for lc in label_candidates]:
            label_col = c
            break

    if label_col is None:
        raise ValueError(
            "No label column found. Columns present:\n" + str(list(df.columns))
        )

    # drop rows missing labels
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    events = pd.DataFrame()

    # artificial timestamp (monotonic)
    events["timestamp"] = pd.date_range(start="2025-01-01", periods=len(df), freq="s")

    # extract numeric features safely
    get = lambda name: df.get(name, pd.Series(0.0, index=df.index)).astype(float)

    events["flow_duration"] = get("Flow_Duration")
    events["total_fwd_pkts"] = get("Total_Fwd_Packets")
    events["total_bwd_pkts"] = get("Total_Backward_Packets")
    events["total_packets"] = events["total_fwd_pkts"] + events["total_bwd_pkts"]

    events["total_length_fwd"] = get("Total_Length_of_Fwd_Packets")
    events["total_length_bwd"] = get("Total_Length_of_Bwd_Packets")
    events["total_bytes"] = events["total_length_fwd"] + events["total_length_bwd"]

    # KB size
    events["data_size_kb"] = events["total_bytes"] / 1024.0

    # use original label (cleaned)
    events["intent_label"] = df[label_col].astype(str)

    # simple sensitivity heuristic
    events["sensitivity_level"] = events["intent_label"].apply(
        lambda x: 1 if "BENIGN" in x.upper() else 3
    )

    return events
