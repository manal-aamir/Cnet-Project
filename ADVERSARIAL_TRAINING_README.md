# Adversarial Training for Robust Models

## Problem Identified

The adversarial robustness testing revealed that models are vulnerable to adversarial attacks:

- **DecisionTree**: Accuracy dropped from **99.94%** → **55.1%** (44.9% attack success rate)
- **RandomForest**: Accuracy dropped from **99.96%** → **86.8%** (13.2% attack success rate)  
- **KNN**: Accuracy dropped from **99.71%** → **91.9%** (8% attack success rate)

This indicates that while models achieve high accuracy on clean data, they are vulnerable to carefully crafted adversarial perturbations.

## Solution: Adversarial Training

We've implemented **adversarial training** to improve model robustness. This technique:

1. **Generates adversarial examples** during training
2. **Mixes adversarial examples** with original training data
3. **Iteratively retrains** the model to learn robust decision boundaries
4. **Compares robustness** before and after training

## Implementation

### New Components

1. **`AdversarialTrainer` class** (`advanced_analysis.py`)
   - `train_adversarial_model()`: Trains robust models using adversarial examples
   - `compare_robustness()`: Compares original vs robust model performance
   - `_generate_adversarial_batch()`: Generates adversarial examples using FGSM-like approach

2. **`train_robust_models()` function** (`advanced_analysis.py`)
   - Loads all saved models
   - Trains robust versions of each model
   - Saves robust models and comparison results

3. **`train_robust_models.py` script**
   - Standalone script to train robust models
   - Easy to run: `python train_robust_models.py`

## Usage

### Option 1: Run the standalone script

```bash
python train_robust_models.py
```

### Option 2: Use programmatically

```python
from advanced_analysis import train_robust_models

results = train_robust_models(
    data_dir="data",
    output_dir="outputs",
    test_size=0.33,
    random_state=42,
    adv_ratio=0.3,      # 30% adversarial examples in training
    eps=0.1,            # Perturbation magnitude
    n_iterations=3      # Number of adversarial training iterations
)
```

### Option 3: Train individual models

```python
from advanced_analysis import AdversarialTrainer
from sklearn.base import clone

# Load original model
original_model = joblib.load("outputs/model_All_DecisionTree.pkl")

# Initialize trainer
trainer = AdversarialTrainer("outputs/adversarial_training")

# Train robust version
robust_model, stats = trainer.train_adversarial_model(
    base_model=original_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    model_name="All_DecisionTree",
    adv_ratio=0.3,
    eps=0.1,
    n_iterations=3
)
```

## Parameters

- **`adv_ratio`** (default: 0.3): Ratio of adversarial examples to mix with training data (0.0-1.0)
  - Higher values = more adversarial examples, potentially more robust but slower training
  - Lower values = fewer adversarial examples, faster training but potentially less robust

- **`eps`** (default: 0.1): Perturbation magnitude for adversarial examples
  - Higher values = larger perturbations, more challenging but may hurt clean accuracy
  - Lower values = smaller perturbations, less challenging but preserves clean accuracy better

- **`n_iterations`** (default: 3): Number of adversarial training iterations
  - More iterations = potentially more robust but longer training time
  - Each iteration generates new adversarial examples based on current model

## Output Files

After running adversarial training, you'll find:

```
outputs/adversarial_training/
├── robust_model_All_DecisionTree.pkl          # Robust model
├── training_stats_All_DecisionTree.json        # Training statistics
├── robustness_comparison_All_DecisionTree.json # Before/after comparison
├── robust_model_All_KNN.pkl
├── training_stats_All_KNN.json
└── ...
```

### Training Statistics JSON

```json
{
  "baseline_accuracy": 0.9994,
  "iterations": [
    {
      "iteration": 1,
      "test_accuracy": 0.9985,
      "adversarial_accuracy": 0.8920,
      "robustness_score": 0.8934
    },
    ...
  ],
  "final_accuracy": 0.9982,
  "robustness_improvement": -0.0012,
  "comparison": {
    "original": {
      "baseline": 0.9994,
      "eps_0.05": 0.8234,
      "eps_0.1": 0.5510,
      ...
    },
    "robust": {
      "baseline": 0.9982,
      "eps_0.05": 0.9123,
      "eps_0.1": 0.7845,
      ...
    },
    "improvements": {
      "eps_0.05": 0.0889,
      "eps_0.1": 0.2335,
      ...
    }
  }
}
```

## Expected Improvements

After adversarial training, you should see:

1. **Higher adversarial accuracy**: Models should maintain better accuracy under attack
2. **Better robustness scores**: Adversarial accuracy / clean accuracy ratio should improve
3. **Lower attack success rates**: Fewer adversarial examples should successfully fool the model

### Example Improvement

**Before adversarial training:**
- Clean accuracy: 99.94%
- Adversarial accuracy (eps=0.1): 55.1%
- Robustness score: 0.551

**After adversarial training:**
- Clean accuracy: 99.82% (slight drop)
- Adversarial accuracy (eps=0.1): 78.5% (significant improvement)
- Robustness score: 0.786 (42% improvement)

## Trade-offs

Adversarial training typically involves a **trade-off**:

- ✅ **Improved robustness** against adversarial attacks
- ✅ **Better generalization** to perturbed inputs
- ⚠️ **Slight drop in clean accuracy** (usually < 1%)
- ⚠️ **Longer training time** (due to adversarial example generation)

## Best Practices

1. **Start with default parameters** and adjust based on results
2. **Monitor both clean and adversarial accuracy** during training
3. **Compare robustness across different eps values** to ensure improvement
4. **Use robust models for deployment** if security is a concern
5. **Keep original models** for comparison and fallback

## Next Steps

1. Run `python train_robust_models.py` to train robust versions of all models
2. Review the comparison results in `robustness_comparison_*.json`
3. Evaluate whether the robustness improvement justifies the slight clean accuracy drop
4. Use robust models for production deployment if improved security is needed

## References

- Adversarial training is based on the concept introduced in "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2015)
- The implementation uses FGSM-like attacks adapted for tree-based models
- Iterative adversarial training follows the PGD training approach

