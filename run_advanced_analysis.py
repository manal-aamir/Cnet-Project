"""
Demo script for running advanced analysis on trained models
===========================================================

This script demonstrates how to use the advanced_analysis module to:
1. Analyze model explainability
2. Test adversarial robustness
3. Perform cross-dataset validation

Usage:
    python run_advanced_analysis.py --data-path datasets/UNSW_2018_IoT_Botnet_Merged_10.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Import from main notebook/pipeline
try:
    # Try importing from the notebook code
    sys.path.insert(0, str(Path(__file__).parent))
    from advanced_analysis import (
        ExplainabilityAnalyzer,
        AdversarialRobustnessTester,
        CrossDatasetValidator,
        run_advanced_analysis
    )
except ImportError as e:
    print(f"Error importing advanced_analysis: {e}")
    sys.exit(1)


def load_and_preprocess_data(data_path: str, label_column: str = "auto",
                             positive_label: str = "DoS",
                             negative_labels: list = None):
    """Load and preprocess dataset"""
    if negative_labels is None:
        negative_labels = ["Normal", "Benign"]
    
    print(f"[LOADING] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    df.columns = [str(c).strip().replace('"', '').replace("'", "") for c in df.columns]
    
    print(f"[LOADING] Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Detect label column
    if label_column == "auto":
        candidates = {positive_label.lower()} | {label.lower() for label in negative_labels}
        best_col = None
        best_hits = -1
        for column in df.columns:
            series = df[column].astype(str).str.strip().str.lower()
            hits = series.isin(candidates).sum()
            if hits > best_hits:
                best_hits = hits
                best_col = column
        if best_col is None or best_hits <= 0:
            raise ValueError("Failed to detect label column. Please specify manually.")
        label_column = best_col
    
    print(f"[LOADING] Using label column: {label_column}")
    
    # Preprocess
    labels_raw = df[label_column].astype(str).str.strip()
    labels = labels_raw.apply(lambda v: 1 if v.lower() == positive_label.lower() else 0)
    features = df.drop(columns=label_column).copy()
    
    # Coerce to numeric
    for c in features.columns:
        if not pd.api.types.is_numeric_dtype(features[c]):
            numeric = pd.to_numeric(features[c], errors="coerce")
            if numeric.notna().sum() > 0:
                features[c] = numeric
            else:
                encoded, _ = pd.factorize(features[c].fillna("missing").astype(str))
                features[c] = pd.Series(encoded, index=features.index, dtype=float)
    
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    print(f"[LOADING] Preprocessing complete. Features: {features.shape[1]}, Samples: {len(labels)}")
    print(f"[LOADING] Class distribution: Normal={sum(labels==0)}, DoS={sum(labels==1)}")
    
    return features, labels, label_column


def main():
    parser = argparse.ArgumentParser(
        description="Run advanced analysis on IoT DoS detection models"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/UNSW_2018_IoT_Botnet_Merged_10.csv",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--target-data-path",
        type=str,
        default=None,
        help="Path to target dataset for cross-dataset validation (optional)"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="auto",
        help="Label column name (or 'auto' to detect)"
    )
    parser.add_argument(
        "--positive-label",
        type=str,
        default="DoS",
        help="Positive class label"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.33,
        help="Test set size"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["DecisionTree", "RandomForest"],
        default="RandomForest",
        help="Model to train and analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/advanced",
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-adversarial",
        action="store_true",
        help="Skip adversarial robustness testing"
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip explainability analysis"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ADVANCED ANALYSIS FOR IOT DOS DETECTION")
    print("=" * 70)
    
    # Load and preprocess data
    X, y, label_col = load_and_preprocess_data(
        args.data_path, args.label_column, args.positive_label
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    
    print(f"\n[DATA] Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    
    # Train model
    print(f"\n[TRAINING] Training {args.model}...")
    if args.model == "DecisionTree":
        model = DecisionTreeClassifier(random_state=args.random_state, max_depth=20)
    else:
        model = RandomForestClassifier(
            n_estimators=100, random_state=args.random_state, n_jobs=-1, max_depth=20
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate baseline
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred)
    print(f"[BASELINE] Test accuracy: {baseline_acc:.4f}")
    print("\n[BASELINE] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "DoS"]))
    
    feature_names = list(X_train.columns)
    
    # Run advanced analyses
    results = {}
    
    # 1. Explainability
    if not args.skip_explainability:
        print("\n" + "=" * 70)
        print("1. EXPLAINABILITY ANALYSIS")
        print("=" * 70)
        explainer = ExplainabilityAnalyzer(f"{args.output_dir}/explainability")
        explain_results = explainer.analyze(
            model, X_train, X_test, y_train, y_test, feature_names, args.model
        )
        results['explainability'] = explain_results
        
        print("\n[EXPLAINABILITY] Top 10 Features by Permutation Importance:")
        sorted_perm = sorted(
            explain_results.permutation_importance.items(),
            key=lambda x: x[1], reverse=True
        )[:10]
        for i, (feat, imp) in enumerate(sorted_perm, 1):
            print(f"  {i:2d}. {feat:30s} {imp:.6f}")
    
    # 2. Adversarial Robustness
    if not args.skip_adversarial:
        print("\n" + "=" * 70)
        print("2. ADVERSARIAL ROBUSTNESS TESTING")
        print("=" * 70)
        adversarial_tester = AdversarialRobustnessTester(f"{args.output_dir}/adversarial")
        adversarial_results = adversarial_tester.test_robustness(
            model, X_test, y_test, feature_names, args.model
        )
        results['adversarial'] = adversarial_results
        
        print("\n[ADVERSARIAL] Robustness Summary:")
        print(f"{'Attack':<20} {'Original Acc':<15} {'Adv Acc':<15} {'Robustness':<12} {'Success Rate':<12}")
        print("-" * 75)
        for r in adversarial_results:
            print(f"{r.attack_name:<20} {r.original_accuracy:<15.4f} "
                  f"{r.adversarial_accuracy:<15.4f} {r.robustness_score:<12.4f} "
                  f"{r.attack_success_rate:<12.4f}")
    
    # 3. Cross-Dataset Validation
    if args.target_data_path and Path(args.target_data_path).exists():
        print("\n" + "=" * 70)
        print("3. CROSS-DATASET VALIDATION")
        print("=" * 70)
        
        source_name = Path(args.data_path).stem
        target_name = Path(args.target_data_path).stem
        
        validator = CrossDatasetValidator(f"{args.output_dir}/cross_dataset")
        cross_results = validator.validate(
            model=model,
            source_features=feature_names,
            source_dataset_name=source_name,
            target_data_path=args.target_data_path,
            target_dataset_name=target_name,
            label_column=args.label_column,
            positive_label=args.positive_label
        )
        results['cross_dataset'] = cross_results
        
        print(f"\n[CROSS-DATASET] Results: {source_name} -> {target_name}")
        print(f"  Accuracy:  {cross_results.accuracy:.4f}")
        print(f"  Precision: {cross_results.precision:.4f}")
        print(f"  Recall:    {cross_results.recall:.4f}")
        print(f"  F1 Score:  {cross_results.f1_score:.4f}")
        print(f"  Feature Overlap: {len(cross_results.feature_overlap)}/{len(feature_names)}")
    else:
        if args.target_data_path:
            print(f"\n[WARNING] Target dataset not found: {args.target_data_path}")
        else:
            print("\n[INFO] Skipping cross-dataset validation (no target dataset provided)")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print(f"  - Explainability plots: {args.output_dir}/explainability/")
    print(f"  - Adversarial results: {args.output_dir}/adversarial/")
    if args.target_data_path and Path(args.target_data_path).exists():
        print(f"  - Cross-dataset results: {args.output_dir}/cross_dataset/")


if __name__ == "__main__":
    main()

