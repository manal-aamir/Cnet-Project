# IoTID20 DoS Detection (per Sensors 2024 base paper)# WSN DoS Attack Detection using Ensemble Machine Learning



This repository implements the exact pipeline described in:A clean, modular implementation of Wireless Sensor Network DoS attack detection using ensemble machine learning (Hard vs. Soft Voting), based on the paper:



Altulaihan, E., Almaiah, M. A., & Aljughaiman, A. (2024). Anomaly Detection IDS for Detecting DoS Attacks in IoT Networks Based on Machine Learning Algorithms. Sensors, 24(2), 713. https://doi.org/10.3390/s24020713> **Al Sukkar & Al-Sharaeh (2024)**, *Enhancing Security in Wireless Sensor Networks: A Machine Learning-based DoS Attack Detection*, Sensors 2024, 24(2), 713. [https://www.mdpi.com/1424-8220/24/2/713](https://www.mdpi.com/1424-8220/24/2/713)



- Input: IoTID20 Dataset---

- Step 1: Data Preprocessing (normalize, handle missing, clean noise)

- Step 2: Feature Selection (CFS and GA)## Project Structure

- Step 3: Train Classifiers (DT, RF, KNN, SVM)

- Step 4: Evaluate (accuracy, train/test time, compare CFS vs GA)```

- Output: Best classifier + feature selection combo (DT + GA).

├── preprocessing/

## Structure│   ├── __init__.py

- src/: All code modules│   └── data_loader.py          # Data loading, encoding, scaling

- data/: Place IoTID20 CSVs here├── training/

- outputs/: All experiment results│   ├── __init__.py

- notebooks/: results.ipynb for visualizations│   └── models.py                # Model builders (RF, SVC, LR, Ensembles)

├── evaluation/

## Usage│   ├── __init__.py

(Instructions will be added after implementation)│   └── metrics.py               # Evaluation metrics & reporting

│
├── outputs/                     # Results directory (auto-created)
│
├── main.py                      # Main executable script
├── results.ipynb                # Results analysis notebook
├── requirements.txt             # Python dependencies
├── WSN-DS.csv                   # Dataset 1 (place here)
├── WSNBFSFdataset.csv          # Dataset 2 (place here)
└── README.md                    # This file
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone or download the repository
cd Cnet-Project

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies (already done if you ran pip install earlier)
pip install -r requirements.txt
```

### 2. Download Datasets

Download the following datasets and place them in the project root:

- **WSN-DS.csv** - [Download from Kaggle](https://www.kaggle.com/datasets/WSN-DS)
- **WSNBFSFdataset.csv** - [Download from Kaggle](https://www.kaggle.com/datasets/WSNBFSF)

```bash
# Your project root should have:
# WSN-DS.csv
# WSNBFSFdataset.csv
```

---

## Training Models

### Train on Both Datasets (Recommended)

```bash
python main.py
```

This will:
1. **Preprocess** both datasets (label encoding, feature scaling, 80/20 split)
2. **Train** 5 models per dataset:
   - RandomForest
   - SVC (RBF kernel)
   - LogisticRegression
   - **EnsembleHard** (majority voting)
   - **EnsembleSoft** (probability averaging)
3. **Evaluate** each model (accuracy, precision, recall, F1)
4. **Save** results to `outputs/`

### Train on Single Dataset

```bash
# WSN-DS only
python main.py --dataset WSN-DS

# WSN-BFSF only
python main.py --dataset WSN-BFSF
```

### Custom Output Directory

```bash
python main.py -o results/experiment1/
```

---

## Evaluation & Results

### Where to Find Results

All outputs are saved to `outputs/` (or your custom directory):

```
outputs/
├── WSN-DS_metrics.csv                    # Summary metrics table
├── WSN-DS_performance.png                 # Performance bar chart
├── WSN-DS_RandomForest_report.txt         # Detailed classification report
├── WSN-DS_RandomForest_confusion_matrix.csv
├── WSN-DS_SVC_report.txt
├── WSN-DS_SVC_confusion_matrix.csv
├── WSN-DS_LogisticRegression_report.txt
├── WSN-DS_LogisticRegression_confusion_matrix.csv
├── WSN-DS_EnsembleHard_report.txt
├── WSN-DS_EnsembleHard_confusion_matrix.csv
├── WSN-DS_EnsembleSoft_report.txt
├── WSN-DS_EnsembleSoft_confusion_matrix.csv
└── ... (same structure for WSN-BFSF)
```

### View Results

**Summary Metrics (CSV)**
```bash
cat outputs/WSN-DS_metrics.csv
```

**Performance Plot**
```bash
open outputs/WSN-DS_performance.png  # macOS
xdg-open outputs/WSN-DS_performance.png  # Linux
```

**Detailed Classification Report**
```bash
cat outputs/WSN-DS_EnsembleSoft_report.txt
```

---

## Expected Results

### WSN-DS Dataset
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **EnsembleSoft** | **~98.12%** | ~0.98 | ~0.98 | ~0.98 |
| **EnsembleHard** | **~97.97%** | ~0.98 | ~0.98 | ~0.98 |
| RandomForest | ~97.5% | ~0.97 | ~0.97 | ~0.97 |
| SVC | ~96.8% | ~0.97 | ~0.97 | ~0.97 |
| LogisticRegression | ~95.2% | ~0.95 | ~0.95 | ~0.95 |

### WSN-BFSF Dataset
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **EnsembleSoft** | **~100%** | ~1.0 | ~1.0 | ~1.0 |
| **EnsembleHard** | **~99.967%** | ~1.0 | ~1.0 | ~1.0 |

*Note: Results may vary slightly based on dataset versions and hardware.*

---

## Advanced Usage

### Interactive Results Analysis

For detailed analysis and visualization:

```bash
jupyter notebook results.ipynb
```

The notebook provides:
- Performance comparison charts
- Confusion matrix visualizations
- Detailed classification reports
- Ensemble method comparisons

### Python API Usage

You can also use the modules programmatically:

```python
from preprocessing import load_and_preprocess
from training import build_models, train_models
from evaluation import evaluate_models, save_results

# Load data
data = load_and_preprocess("WSN-DS.csv")

# Build and train models
models = build_models()
trained = train_models(models, data.X_train, data.y_train)

# Evaluate
metrics_df, details = evaluate_models(
    trained, data.X_test, data.y_test, data.class_names
)

# Save results
save_results("WSN-DS", metrics_df, details)
```

---

## Model Configuration

### Base Models

**RandomForest**
- `n_estimators=100`
- `random_state=42`
- `n_jobs=-1` (parallel processing)

**SVC**
- `kernel='rbf'`
- `gamma='scale'`
- `probability=True` (required for soft voting)
- `random_state=42`

**LogisticRegression**
- `max_iter=1000`
- `random_state=42`
- `n_jobs=-1`

### Ensemble Methods

**Hard Voting**: Majority class prediction
- Each model votes for a class
- Most frequent class wins

**Soft Voting**: Probability averaging
- Each model outputs class probabilities
- Probabilities are averaged
- Class with highest average wins

---

## Key Features

- **Clean Architecture**: Modular design with separate preprocessing, training, and evaluation  
- **Reproducible**: Fixed random seed (42) for consistent results  
- **Multi-class Support**: Handles multiple attack types via LabelEncoder  
- **Standardized Metrics**: Accuracy, weighted precision/recall/F1, confusion matrices  
- **Auto Scaling**: StandardScaler for feature normalization  
- **Comprehensive Reports**: Detailed classification reports per model  
- **Visualizations**: Performance comparison bar charts  
- **Error Handling**: Graceful failures with informative messages  

---

## Troubleshooting

**Problem:** `FileNotFoundError: Dataset not found`  
**Solution:** Download datasets and place in project root. Check filenames match exactly: `WSN-DS.csv` and `WSNBFSFdataset.csv`

**Problem:** `ModuleNotFoundError: No module named 'sklearn'`  
**Solution:** Install dependencies: `pip install -r requirements.txt`

**Problem:** Pipeline runs slowly  
**Solution:** 
- Reduce `n_estimators` in RandomForest (edit `training/models.py`)
- Use smaller test_size in `preprocessing/data_loader.py`

**Problem:** Memory error  
**Solution:**
- Process datasets one at a time: `python main.py --dataset WSN-DS`
- Sample the data before training (modify `preprocessing/data_loader.py`)

---
