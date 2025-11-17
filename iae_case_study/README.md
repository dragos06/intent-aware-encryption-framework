# Intent-Aware Encryption (IAE) — Modular Case Study

This repository implements the IAE case study with reproducible experiments.

## Features

- Loads real CICIDS2017 CSV datasets.
- Maps attack labels to research intents (`emergency`, `confidential-transfer`, `routine-report`, `diagnostic`, `maintenance`).
- Builds numeric feature pipeline with scaling.
- Trains ML classifiers: Random Forest, Logistic Regression, MLP.
- Implements multiple policies: static, context-only, rule-based, ML-driven (IAE).
- Simulates encryption latency & energy using `cryptography.Fernet`.
- Evaluates latency, energy, throughput, and classification metrics.
- Outputs JSON and CSV summaries per run.

## Requirements

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows
pip install -r requirements.txt
