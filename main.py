import os
import pandas as pd
from src.preprocessing import load_and_clean_iotid20
from src.feature_selection import cfs_select, ga_select
from src.models import get_classifiers
from src.evaluate import run_experiment


def main():
    # Paths
    raw_csv = 'data/IoT Network Intrusion Dataset.csv'  # IoTID20 dataset
    cleaned_csv = 'data/iotid20_filtered.csv'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Preprocessing
    df = load_and_clean_iotid20(raw_csv, cleaned_csv)
    X = df.drop(columns=['Target'])
    y = df['Target']

    # Step 2: Feature Selection
    cfs_feats = cfs_select(X, y, max_features=13)
    ga_feats = ga_select(X, y, max_features=13)
    # Save feature lists
    pd.Series(cfs_feats).to_csv(f'{output_dir}/cfs_features.csv', index=False)
    pd.Series(ga_feats).to_csv(f'{output_dir}/ga_features.csv', index=False)

    # Step 3: Classifiers
    classifiers = get_classifiers()
    feature_sets = {
        'All': list(X.columns),
        'CFS': cfs_feats,
        'GA': ga_feats
    }

    # Step 4: Experiment/Evaluation
    results, conf_matrices = run_experiment(X, y, feature_sets, classifiers, output_dir=output_dir)
    print('Experiment complete. Results saved to outputs/.')

if __name__ == '__main__':
    main()

