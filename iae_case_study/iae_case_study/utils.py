import os
import json
import logging
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_str():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level=level
    )
