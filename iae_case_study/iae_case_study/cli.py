import argparse
import logging
from .utils import setup_logging, ensure_dir, now_str
from .data import generate_features
from .labeling import label_intents
from .features import build_feature_pipeline
from .models import train_classifiers, evaluate_classifier
from .evaluation import run_experiments
from .config import cfg
from .utils import save_json
import pandas as pd
import os

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="IAE modular case study runner")
    parser.add_argument("--outdir", type=str, default=f"iae_results_{now_str()}", help="Output directory")
    parser.add_argument("--seed", type=int, default=cfg.random_seed, help="Random seed")
    parser.add_argument("--n_samples", type=int, default=cfg.n_samples, help="Number of synthetic samples")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Logging level")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(getattr(logging, args.loglevel.upper(), logging.INFO))
    logger.info("Starting IAE case study")
    # update cfg minimally
    cfg.random_seed = args.seed
    cfg.n_samples = args.n_samples

    ensure_dir(args.outdir)
    # Generate data
    df = generate_features(n_samples=cfg.n_samples, random_seed=cfg.random_seed)
    # Label
    labels = label_intents(df)
    df["intent"] = labels

    # Build features
    X, scaler = build_feature_pipeline(df.drop(columns=["intent"]))
    y = labels

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y)

    # Train models
    models = train_classifiers(X_train, y_train)

    # Evaluate classifiers and save report
    classifier_reports = {}
    for name, est in models.items():
        report = evaluate_classifier(est, X_test, y_test, labels=cfg.intent_set)
        classifier_reports[name] = report

    save_json(classifier_reports, os.path.join(args.outdir, "classifier_report.json"))

    # Prepare df_events for evaluation (use original df)
    df_events = df.reset_index(drop=True)

    # Run experiments
    summary_df, results = run_experiments(df_events, X, y, models, args.outdir)

    logger.info("Completed experiments. Summary saved to %s", args.outdir)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
