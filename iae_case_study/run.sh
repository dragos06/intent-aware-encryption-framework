#!/usr/bin/env bash
# Run IAE experiments on CICIDS CSV folder
# Make sure to adjust paths to your CSV folder and results folder

CIC_FOLDER="./iae_case_study/data/cic_ids/"
OUTDIR="./results/cicids_run"

mkdir -p "$OUTDIR"
python -m iae_case_study.cli --cic_folder "$CIC_FOLDER" --outdir "$OUTDIR" --seed 42 --loglevel INFO