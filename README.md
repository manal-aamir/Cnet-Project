# IoTID20 DoS Detection using Machine Learning

A complete implementation of the methodology described in:

**Altulaihan, E., Almaiah, M. A., & Aljughaiman, A. (2024).** *Anomaly Detection IDS for Detecting DoS Attacks in IoT Networks Based on Machine Learning Algorithms.* Sensors, 24(2), 713. https://doi.org/10.3390/s24020713

---

## Overview

This project implements an Intrusion Detection System (IDS) for detecting Denial of Service (DoS) attacks in IoT networks using the **IoTID20 dataset**. The implementation follows the exact methodology from the base paper:

1. **Data Preprocessing**: Filter DoS and Normal classes, drop identifiers, encode labels
2. **Feature Selection**: Compare Correlation-Based Feature Selection (CFS) vs Genetic Algorithm (GA)
3. **Model Training**: Train Decision Tree, Random Forest, KNN, and SVM classifiers
4. **Evaluation**: Compare performance across feature selection methods and classifiers
5. **Result**: Identify optimal combination (Decision Tree + GA with 13 features)

### Key Results
- Accuracy: 99.96% with Decision Tree + GA
- Feature Reduction: 83% (77 → 13 features)
- Training Time: 0.029s (74% faster than using all features)
- F1 Score: 99.97% (near-perfect classification)

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/manal-aamir/Cnet-Project.git
cd Cnet-Project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **IoTID20 dataset** and place it in the `data/` directory:

- **Dataset Name**: `IoT Network Intrusion Dataset.csv`
- **Source**: IoTID20 - IoT Network Intrusion Dataset
- **Size**: ~294 MB (625,783 samples, 86 features)

```bash
# Your data directory should contain:
data/IoT Network Intrusion Dataset.csv
```

---

## Running the Pipeline

### Execute Complete Pipeline

```bash
# Activate virtual environment
source venv/bin/activate

# Run the complete pipeline
python main.py
```

### Single-File Runner (UNSW IoT Botnet)

If you prefer a self-contained script that embeds all logic and uses the repository's copy of `UNSW_2018_IoT_Botnet_Merged_10.csv`, run:

```
python paper_baseline.py --output-dir artifacts_unsw
```

Flags like `--disable-ga`, `--disable-cfs`, `--positive-label`, `--test-size`, and GA hyperparameters are also available. Metrics/confusion matrices/feature sets are written under the chosen output directory, and the console prints the same comparison table.

```bash
# Activate virtual environment
source venv/bin/activate

**Expected Runtime**: 5-10 minutes

**Pipeline Steps**:
1. **Preprocessing** (~30 seconds)
   - Filters dataset to DoS (59,391) and Normal (40,073) classes
   - Drops identifiers (Flow_ID, IPs, Ports, Timestamp, etc.)
   - Encodes target: DoS=1, Normal=0
   - Saves to `data/iotid20_filtered.csv` (99,464 samples × 78 columns)

2. **Feature Selection** (~2-5 minutes)
   - **CFS**: Correlation-based selection → 13 features
   - **GA**: Genetic algorithm optimization → 13 features
   - Saves feature lists to `outputs/cfs_features.csv` and `outputs/ga_features.csv`

3. **Model Training** (~3-5 minutes)
   - Trains 12 models: 4 classifiers × 3 feature sets (All, CFS, GA)
   - Uses 67/33 train/test split (stratified)
   - Saves trained models to `outputs/model_*.pkl`

4. **Evaluation** (automatic)
   - Computes accuracy, precision, recall, F1 score
   - Measures training and testing time
   - Generates confusion matrices
   - Saves all metrics to `outputs/metrics.csv`

---

## Viewing Results

### Option 1: Jupyter Notebook (Recommended)

```bash
# Launch Jupyter
jupyter notebook notebooks/results.ipynb

```

**Notebook Features**:
- Performance metrics comparison (bar charts)
- Confusion matrices heatmaps
- Training/testing time analysis
- Summary tables with rankings
- Feature selection comparison

### Option 2: Command Line

```bash
# View metrics CSV
cat outputs/metrics.csv

# View best performer
head -n 2 outputs/metrics.csv | tail -n 1

# Check confusion matrix for Decision Tree + GA
cat outputs/confmat_GA_DecisionTree.csv
```
---

## Results Summary

### Best Performing Model

**Decision Tree + GA (13 features)**
- **Accuracy**: 99.96%
- **Precision**: 99.98%
- **Recall**: 99.95%
- **F1 Score**: 99.97%
- **Training Time**: 0.029s
- **Testing Time**: 0.001s

**Confusion Matrix**:
```
              Predicted
              Normal    DoS
Actual Normal  13,221     3
       DoS         10 19,590
```
Only 13 misclassifications out of 32,824 test samples!

### All Results Ranking

| Rank | Classifier | Feature Set | Features | Accuracy | F1 Score |
|------|-----------|-------------|----------|----------|----------|
| 1 | DecisionTree | GA | 13 | 99.96% | 99.97% |
| 1 | RandomForest | GA | 13 | 99.96% | 99.97% |
| 3 | RandomForest | All | 77 | 99.95% | 99.95% |
| 4 | DecisionTree | All | 77 | 99.94% | 99.95% |
| 5 | KNN | GA | 13 | 99.91% | 99.92% |

**Key Finding**: GA feature selection with Decision Tree achieves the best performance with 83% fewer features!

---

## Repository Contents

### Source Code

- **`src/preprocessing.py`**: Data loading, filtering (DoS/Normal), identifier removal, label encoding
- **`src/feature_selection.py`**: CFS (correlation-based) and GA (genetic algorithm) implementations
- **`src/models.py`**: Classifier factory for Decision Tree, Random Forest, KNN, SVM
- **`src/evaluate.py`**: Training harness with metrics computation and model serialization

### Main Scripts

- **`main.py`**: Pipeline orchestration (preprocessing → feature selection → training → evaluation)
- **`notebooks/results.ipynb`**: Interactive visualization and analysis notebook

### Documentation

- **`README.md`**: This file - quick start and usage guide
- **`REPORT.md`**: Comprehensive implementation report with detailed analysis
- **`requirements.txt`**: Python dependencies

---

## Technical Details

### Dataset Preprocessing

**Original IoTID20**:
- 625,783 samples, 86 features
- Multiple attack types (DoS, Mirai, Scan, MITM)

**Filtered for This Study**:
- 99,464 samples (DoS: 59,391 | Normal: 40,073)
- 77 numerical features (dropped: Flow_ID, Src_IP, Src_Port, Dst_IP, Dst_Port, Timestamp, Label, Cat, Sub_Cat)
- Binary classification: DoS (1) vs Normal (0)

### Feature Selection Methods

#### CFS (Correlation-Based Feature Selection)
- Ranks features by correlation with target
- Removes redundant features (high inter-feature correlation)
- Selects top 13 features

**CFS Selected Features**:
1. ACK_Flag_Cnt
2. Bwd_Pkt_Len_Min
3. Fwd_Pkt_Len_Min
4. Fwd_Pkt_Len_Std
5. Fwd_PSH_Flags
6. Fwd_URG_Flags
7. Bwd_Pkts/s
8. Pkt_Len_Std
9. Flow_IAT_Min
10. Fwd_Byts/b_Avg
11. Fwd_Pkts/b_Avg
12. Fwd_Blk_Rate_Avg
13. Bwd_Byts/b_Avg

#### GA (Genetic Algorithm)
- Binary chromosome (1 = feature selected, 0 = excluded)
- Population: 20, Generations: 50
- Fitness: 5-fold CV accuracy with Decision Tree
- Single-point crossover, bit-flip mutation
- Selects top 13 features from best chromosome

**GA Selected Features**:
1. Fwd_Pkt_Len_Max
2. Bwd_Pkt_Len_Max
3. Bwd_Pkt_Len_Min
4. Fwd_IAT_Std
5. ACK_Flag_Cnt
6. URG_Flag_Cnt
7. ECE_Flag_Cnt
8. Fwd_Pkts/b_Avg
9. Bwd_Blk_Rate_Avg
10. Subflow_Fwd_Byts
11. Active_Min
12. Idle_Max
13. Idle_Min

### Model Configuration

**Classifiers**:
- **Decision Tree**: `DecisionTreeClassifier(random_state=42)`
- **Random Forest**: `RandomForestClassifier(n_estimators=100, random_state=42)`
- **KNN**: `KNeighborsClassifier(n_neighbors=5)`
- **SVM**: `SVC(kernel='rbf', random_state=42)`

**Train/Test Split**: 67% / 33% (stratified by class)

**Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, Training Time, Testing Time

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

**Python Version**: 3.8 or higher

---

## Alignment with Base Paper

This implementation reproduces the methodology from:

> Altulaihan, E., Almaiah, M. A., & Aljughaiman, A. (2024). "Anomaly Detection IDS for Detecting DoS Attacks in IoT Networks Based on Machine Learning Algorithms." *Sensors*, 24(2), 713.

### Methodology Alignment

| Paper Component | Our Implementation | Status |
|----------------|-------------------|--------|
| Dataset | IoTID20 (DoS + Normal) | Identical |
| Preprocessing | Drop identifiers, encode labels | 100% aligned |
| Feature Selection | CFS + GA (13 features each) | 100% aligned |
| Classifiers | DT, RF, KNN, SVM | 100% aligned |
| Train/Test Split | 67/33 stratified | 100% aligned |
| Best Result | DT + GA | Confirmed |

### Performance Comparison

| Metric | Base Paper | Our Implementation |
|--------|-----------|-------------------|
| Best Classifier | Decision Tree + GA | Decision Tree + GA (Match) |
| Accuracy | ~99.5-99.8% | **99.96%** (Match/Exceed) |
| Feature Count | 13 (GA) | 13 (GA) (Match) |
| Feature Reduction | 83% | 83% (Match) |

**Conclusion**: Our implementation successfully reproduces and validates the base paper's findings!

---

## Acknowledgments

- **Base Paper**: Altulaihan et al. (2024) - Sensors, 24(2), 713
- **Dataset**: IoTID20 - IoT Network Intrusion Dataset
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn

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
