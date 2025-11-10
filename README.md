# IoTID20 DoS Detection using Machine Learning

A complete implementation of the methodology described in:

**Altulaihan, E., Almaiah, M. A., & Aljughaiman, A. (2024).** *Anomaly Detection IDS for Detecting DoS Attacks in IoT Networks Based on Machine Learning Algorithms.* Sensors, 24(2), 713. https://doi.org/10.3390/s24020713

**Repository**: https://github.com/manal-aamir/Cnet-Project.git

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

## Project Structure

```
Cnet-Project/
├── data/
│   ├── IoT Network Intrusion Dataset.csv    # Raw IoTID20 dataset (place here)
│   └── iotid20_filtered.csv                 # Preprocessed dataset (auto-generated)
├── src/
│   ├── preprocessing.py                     # Data loading & cleaning
│   ├── feature_selection.py                 # CFS & GA implementations
│   ├── models.py                            # Classifier factory (DT, RF, KNN, SVM)
│   └── evaluate.py                          # Training & evaluation harness
├── outputs/
│   ├── metrics.csv                          # Performance metrics (all experiments)
│   ├── cfs_features.csv                     # CFS selected features
│   ├── ga_features.csv                      # GA selected features
│   ├── confmat_*.csv                        # Confusion matrices (12 files)
│   └── model_*.pkl                          # Trained models (12 files)
├── notebooks/
│   └── results.ipynb                        # Visualizations & analysis
├── main.py                                  # Main pipeline script
├── requirements.txt                         # Python dependencies
├── REPORT.md                                # Comprehensive implementation report
└── README.md                                # This file
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
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

**Output**:
```
Saved cleaned IoTID20 to data/iotid20_filtered.csv with shape (99464, 78)
[CFS feature selection running...]
[GA feature selection running...]
[Training 12 models...]
Experiment complete. Results saved to outputs/.
```

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

### Option 3: Read the Report

For comprehensive analysis, see **`REPORT.md`** which includes:
- Detailed methodology comparison with base paper
- Complete results analysis
- Feature selection insights
- Deployment recommendations
- Future work suggestions

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

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{altulaihan2024anomaly,
  title={Anomaly Detection IDS for Detecting DoS Attacks in IoT Networks Based on Machine Learning Algorithms},
  author={Altulaihan, Esra and Almaiah, Mohammed Amin and Aljughaiman, Ahmed},
  journal={Sensors},
  volume={24},
  number={2},
  pages={713},
  year={2024},
  publisher={MDPI},
  doi={10.3390/s24020713},
  url={https://www.mdpi.com/1424-8220/24/2/713}
}
```

---

## Troubleshooting

### Common Issues

**1. Dataset not found**
```
FileNotFoundError: data/IoT Network Intrusion Dataset.csv
```
**Solution**: Download IoTID20 dataset and place in `data/` directory with exact filename.

**2. Memory error during GA**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce GA parameters in `src/feature_selection.py`:
```python
pop_size=10  # default: 20
n_gen=25     # default: 50
```

**3. Slow training**
```
GA taking too long...
```
**Solution**: GA with full dataset (~100K samples) takes 3-5 minutes. This is normal. For testing, you can use a data sample:
```python
# In main.py after loading data
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)
```

**4. Jupyter kernel issues**
```
Kernel died unexpectedly
```
**Solution**: Install jupyter in the virtual environment:
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=cnet-venv
```

---

## Contributing

This is a research implementation for educational purposes. For improvements or bug reports, please open an issue or submit a pull request.

---

## License

This project is provided for educational and research purposes. Please refer to the original paper for citation requirements.

---

## Acknowledgments

- **Base Paper**: Altulaihan et al. (2024) - Sensors, 24(2), 713
- **Dataset**: IoTID20 - IoT Network Intrusion Dataset
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn

---

## Contact

For questions about this implementation, please refer to the repository issues or the base paper documentation.

**Last Updated**: November 11, 2025
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
**Last Updated**: November 11, 2025
