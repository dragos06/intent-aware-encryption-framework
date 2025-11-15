# Intent-Aware Encryption (IAE) — Modular Case Study

This repository contains a modular implementation of the IAE case study:
- synthetic data generation
- rule-based intent labeling
- feature pipeline and ML models (RF, LR, MLP)
- multiple policy simulations including IAE-driven policies
- encryption simulation using `cryptography.Fernet`
- evaluation & outputs (CSV/JSON)

## Quickstart

1. Create virtualenv and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
