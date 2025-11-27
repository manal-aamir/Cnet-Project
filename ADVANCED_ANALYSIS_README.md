# Advanced Analysis Module

This module extends the base paper implementation with three advanced features:

1. **Explainability** - Understand model decisions
2. **Adversarial Robustness** - Test model resilience to attacks
3. **Cross-Dataset Validation** - Evaluate generalization across datasets

## Installation

Install additional dependencies:

```bash
pip install shap adversarial-robustness-toolbox
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Complete Analysis

```bash
python run_advanced_analysis.py \
    --data-path datasets/UNSW_2018_IoT_Botnet_Merged_10.csv \
    --model RandomForest \
    --output-dir outputs/advanced
```

### Cross-Dataset Validation

```bash
python run_advanced_analysis.py \
    --data-path datasets/UNSW_2018_IoT_Botnet_Merged_10.csv \
    --target-data-path datasets/UNSW_2018_IoT_Botnet_Dataset_7.csv \
    --model DecisionTree
```

## Features

### 1. Explainability

Analyzes model decisions using multiple methods:

- **Feature Importance**: Extracts importance from tree-based models
- **Permutation Importance**: Measures feature importance by shuffling
- **SHAP Values**: Explains individual predictions using SHAP

**Usage:**

```python
from advanced_analysis import ExplainabilityAnalyzer

explainer = ExplainabilityAnalyzer("outputs/explainability")
results = explainer.analyze(
    model, X_train, X_test, y_train, y_test, 
    feature_names, "DecisionTree"
)

# Access results
print(results.feature_importance)
print(results.permutation_importance)
```

**Outputs:**
- `feature_importance_{model_name}.png` - Feature importance comparison plot
- `shap_summary_{model_name}.png` - SHAP summary plot (if SHAP available)

### 2. Adversarial Robustness

Tests model resilience against adversarial attacks:

- **FGSM (Fast Gradient Sign Method)**: Fast adversarial attack
- **PGD (Projected Gradient Descent)**: Iterative attack
- **Random Noise**: Baseline robustness test
- **Custom Attacks**: Simplified attacks for tree-based models

**Usage:**

```python
from advanced_analysis import AdversarialRobustnessTester

tester = AdversarialRobustnessTester("outputs/adversarial")
results = tester.test_robustness(
    model, X_test, y_test, feature_names, "RandomForest"
)

# View results
for r in results:
    print(f"{r.attack_name}: {r.robustness_score:.4f}")
    print(f"  Original Acc: {r.original_accuracy:.4f}")
    print(f"  Adversarial Acc: {r.adversarial_accuracy:.4f}")
    print(f"  Attack Success Rate: {r.attack_success_rate:.4f}")
```

**Outputs:**
- `adversarial_results_{model_name}.json` - Detailed results
- `adversarial_summary_{model_name}.csv` - Summary table

**Metrics:**
- `robustness_score`: Ratio of adversarial to original accuracy
- `attack_success_rate`: Percentage of successful attacks
- `perturbation_stats`: Statistics on adversarial perturbations

### 3. Cross-Dataset Validation

Tests model generalization across different datasets:

- Automatically detects and aligns features
- Handles missing features
- Evaluates performance drop

**Usage:**

```python
from advanced_analysis import CrossDatasetValidator

validator = CrossDatasetValidator("outputs/cross_dataset")
results = validator.validate(
    model=model,
    source_features=feature_names,
    source_dataset_name="UNSW_Merged_10",
    target_data_path="datasets/UNSW_Dataset_7.csv",
    target_dataset_name="UNSW_Dataset_7",
    label_column="auto",
    positive_label="DoS"
)

print(f"Cross-dataset Accuracy: {results.accuracy:.4f}")
print(f"Feature Overlap: {len(results.feature_overlap)}/{len(feature_names)}")
```

**Outputs:**
- `cross_dataset_{source}_to_{target}.json` - Full results
- `confusion_matrix_{source}_to_{target}.csv` - Confusion matrix

## Complete Example

```python
from advanced_analysis import run_advanced_analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Run all analyses
results = run_advanced_analysis(
    model=model,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=list(X_train.columns),
    model_name="RandomForest",
    output_dir="outputs/advanced"
)

# Access results
explainability = results['explainability']
adversarial = results['adversarial']
```

## Command Line Options

```bash
python run_advanced_analysis.py --help
```

**Key Options:**
- `--data-path`: Path to training dataset
- `--target-data-path`: Path to target dataset (for cross-dataset validation)
- `--model`: Model type (DecisionTree or RandomForest)
- `--output-dir`: Output directory for results
- `--skip-adversarial`: Skip adversarial testing
- `--skip-explainability`: Skip explainability analysis

## Research Applications

### Explainability Research
- Identify most important features for DoS detection
- Understand model decision boundaries
- Compare feature importance across models

### Adversarial Robustness Research
- Evaluate model vulnerability to attacks
- Test robustness of different architectures
- Develop defense mechanisms

### Cross-Dataset Validation Research
- Measure generalization across IoT environments
- Identify dataset-specific biases
- Evaluate transfer learning potential

## Output Structure

```
outputs/advanced/
├── explainability/
│   ├── feature_importance_DecisionTree.png
│   ├── shap_summary_RandomForest.png
│   └── ...
├── adversarial/
│   ├── adversarial_results_RandomForest.json
│   ├── adversarial_summary_RandomForest.csv
│   └── ...
└── cross_dataset/
    ├── cross_dataset_UNSW_Merged_10_to_UNSW_Dataset_7.json
    ├── confusion_matrix_UNSW_Merged_10_to_UNSW_Dataset_7.csv
    └── ...
```

## Notes

- **SHAP**: Requires `shap` package. Falls back gracefully if unavailable.
- **ART**: Requires `adversarial-robustness-toolbox`. Uses custom implementations if unavailable.
- **Performance**: Adversarial attacks can be slow. Consider using smaller test sets.
- **Memory**: SHAP can be memory-intensive for large datasets. Uses sampling by default.

## Troubleshooting

### SHAP Import Error
```bash
pip install shap
```

### ART Import Error
```bash
pip install adversarial-robustness-toolbox
```

### Memory Issues
- Reduce sample sizes in `advanced_analysis.py`
- Use smaller test sets
- Process datasets in batches

### Feature Mismatch (Cross-Dataset)
- Ensure datasets have similar feature sets
- Check feature names match
- Missing features are filled with zeros

## Citation

If you use this module in your research, please cite the base paper:

```
Altulaihan, E., Almaiah, M. A., & Aljughaiman, A. (2024). 
Anomaly Detection IDS for Detecting DoS Attacks in IoT Networks 
Based on Machine Learning Algorithms. Sensors, 24(2), 713.
```

