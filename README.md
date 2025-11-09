# IoTID20 DoS Reproduction

This project re-implements the baseline pipeline from *“Anomaly Detection IDS for Detecting DoS Attacks in IoT Networks Based on Machine Learning Algorithms”* (Sensors 2024, Altulaihan et al.) so the experiments can be reproduced locally.

## Repository Layout

- `src/cnetids/` – reusable library code (loading, preprocessing, feature selection, training, evaluation).
- `scripts/run_experiments.py` – CLI driver that orchestrates the full experimental sweep.
- `artifacts/` – default output location for logs, metrics tables, and serialized feature sets.

## Environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

Download IoTID20 from the [BoT-IoT/IoTID20 repository](https://www.kaggle.com/datasets) or the original authors’ URL, then:

1. Extract the CSVs locally.
2. Merge the relevant DoS and normal traffic files if needed.
3. Provide the merged CSV to the CLI via `--data-path`.

The script assumes a binary column named `label` that contains strings (`DoS` and `Normal`). You can override column names and label values via flags.

## Running the Experiments

```
python scripts/run_experiments.py \
  --data-path /path/to/IoTID20.csv \
  --label-column label \
  --positive-class DoS \
  --drops flow_id src_ip dst_ip timestamp attack_subcategory attack_type
```

Outputs:

- `artifacts/metrics.csv` – Accuracy/Precision/Recall/F1 plus timing per model/feature-selection combo.
- `artifacts/confusion_matrices.json` – Confusion matrices keyed by experiment id.
- `artifacts/feature_sets.json` – Feature indices selected by GA and CFS.

Add `--max-ga-features` and `--max-cfs-features` to match the counts reported in the paper (13).

## Extending

- Adjust GA hyperparameters (`--ga-population`, `--ga-generations`, etc.) to trade accuracy for runtime.
- Change the classifier registry inside `cnetids.models` to evaluate additional algorithms.
- Integrate alternative datasets by supplying a different CSV and tweaking preprocessing options.
# Cnet-Project
