import numpy as np
from typing import Sequence
from .config import cfg
import pandas as pd

def map_labels_to_intents(label_series: pd.Series) -> np.ndarray:
    """
    Deterministic mapping from CIC-IDS Label strings to the research intent taxonomy.
    This mapping is heuristic but deterministic and documentable:
      - BENIGN -> routine-report
      - DoS / DDoS / HULK / Slowloris / 'DoS' in label -> emergency
      - PortScan / Port Scan / Scan -> diagnostic
      - Infiltration / Bot / BruteForce / FTP / Web Attack / SQL Injection -> confidential-transfer
      - otherwise -> maintenance
    Returns an array of intent strings matching cfg.intent_set.
    """
    def _map_label(lbl: str) -> str:
        s = str(lbl).upper()
        if "BENIGN" in s:
            return "routine-report"
        if any(k in s for k in ("DOS", "DDOS", "HULK", "SLOWLORIS", "HTTP_DOS")):
            return "emergency"
        if any(k in s for k in ("PORTSCAN", "SCAN", "SCANNING")):
            return "diagnostic"
        if any(k in s for k in ("INFILTRATION", "BOT", "BRUTE", "FTP", "WEB", "SQL", "EXFIL")):
            return "confidential-transfer"
        return "maintenance"

    mapped = label_series.apply(_map_label).values
    return mapped
