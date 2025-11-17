import argparse
import glob
import logging
import os
import sys

import pandas as pd

from .datasets import load_cicids2017
from .feature_mapping import map_cicids_to_events
from .utils import setup_logging, ensure_dir, now_str, save_json
from .features import build_feature_pipeline
from .labeling import map_labels_to_intents
from .models import prepare_dataset_with_anomaly, train_classifiers, evaluate_classifier
from .evaluation import run_experiments
from .config import cfg

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="IAE case study runner")
    parser.add_argument("--outdir", type=str, default=f"iae_results_{now_str()}", help="Output directory")
    parser.add_argument("--seed", type=int, default=cfg.random_seed, help="Random seed")
    parser.add_argument("--cic_folder", type=str, required=True, help="Path to folder containing CIC-IDS CSV files")
    parser.add_argument("--cic_limit", type=int, default=None, help="Max rows per CSV (optional)")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def load_cic_folder(folder_path: str, limit: int = None) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in folder {folder_path}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} CSV files in {folder_path}")
    df_list = []
    for f in csv_files:
        df_raw = load_cicids2017(f)
        if limit:
            df_raw = df_raw.sample(n=min(len(df_raw), limit), random_state=cfg.random_seed)
        df_events = map_cicids_to_events(df_raw)
        df_list.append(df_events)

    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total events loaded: {len(df)}")
    return df


def main():
    args = parse_args()
    setup_logging(getattr(logging, args.loglevel.upper(), logging.INFO))
    logger.info("Starting IAE case study")

    cfg.random_seed = args.seed
    ensure_dir(args.outdir)

    # Load CICIDS dataset
    df = load_cic_folder(args.cic_folder, limit=args.cic_limit)
    df["intent"] = map_labels_to_intents(df["intent_label"])
    df = prepare_dataset_with_anomaly(df, random_seed=cfg.random_seed)

    logger.info("Unique intent labels: %s", df["intent"].unique())

    # Build feature matrix
    feature_cols = [c for c in df.columns if c not in ("intent", "timestamp", "intent_label")]
    X, scaler = build_feature_pipeline(df[feature_cols])
    y = df["intent"].values

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y
    )

    # Train classifiers
    models = train_classifiers(X_train, y_train)

    # Evaluate classifiers
    classifier_reports = {}
    for name, est in models.items():
        report = evaluate_classifier(est, X_test, y_test, labels=list(cfg.intent_set))
        classifier_reports[name] = report

    save_json(classifier_reports, os.path.join(args.outdir, "classifier_report.json"))

    # Run IAE experiments
    df_events = df.reset_index(drop=True)
    summary_df, results = run_experiments(df_events, X, y, models, args.outdir)

    logger.info("Completed experiments. Summary saved to %s", args.outdir)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
