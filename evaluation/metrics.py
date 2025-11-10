import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    classification_report: str


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str]
) -> Tuple[pd.DataFrame, Dict[str, ModelMetrics]]:
    """
    Evaluate all models on test data and compute metrics.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
    
    Returns:
        Tuple of (metrics_dataframe, detailed_metrics_dict)
    """
    print("Evaluating models on test data...")
    
    results = []
    detailed_metrics = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        
        detailed_metrics[name] = ModelMetrics(
            model_name=name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm,
            classification_report=report
        )
        
        print(f"  {name}: Accuracy = {accuracy:.4f}")
    
    metrics_df = pd.DataFrame(results)
    print()
    return metrics_df, detailed_metrics


def save_results(
    dataset_name: str,
    metrics_df: pd.DataFrame,
    detailed_metrics: Dict[str, ModelMetrics],
    output_dir: str = "outputs"
):
    """
    Save evaluation results to files.
    
    Args:
        dataset_name: Name of the dataset
        metrics_df: DataFrame with summary metrics
        detailed_metrics: Dictionary with detailed metrics per model
        output_dir: Directory to save outputs
    """
    print(f"Saving results to '{output_dir}/'...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save summary metrics CSV
    metrics_path = os.path.join(output_dir, f"{dataset_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")
    
    # Save detailed reports and confusion matrices for each model
    for name, metrics in detailed_metrics.items():
        # Classification report
        report_path = os.path.join(output_dir, f"{dataset_name}_{name}_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(metrics.classification_report)
        
        # Confusion matrix
        cm_path = os.path.join(output_dir, f"{dataset_name}_{name}_confusion_matrix.csv")
        pd.DataFrame(metrics.confusion_matrix).to_csv(cm_path, index=False)
    
    print(f"  Saved: {len(detailed_metrics)} detailed reports")
    
    # Create performance comparison chart
    plot_path = os.path.join(output_dir, f"{dataset_name}_performance.png")
    _plot_performance(metrics_df, dataset_name, plot_path)
    print(f"  Saved: {plot_path}\n")


def _plot_performance(metrics_df: pd.DataFrame, dataset_name: str, save_path: str):
    """Create and save performance comparison chart."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.2
    
    plt.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    plt.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    plt.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    plt.bar(x + 1.5*width, metrics_df['F1'], width, label='F1', alpha=0.8)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance Comparison - {dataset_name}', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics_df['Model'], rotation=45, ha='right')
    plt.ylim(0.85, 1.02)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_summary(metrics_df: pd.DataFrame):
    """Print formatted summary table."""
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)
    print()
