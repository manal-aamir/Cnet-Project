"""
Advanced Analysis Module for IoT DoS Detection
===============================================

This module extends the base paper implementation with:
1. Explainability: SHAP values, feature importance, permutation importance
2. Adversarial Robustness: FGSM, PGD attacks, adversarial training
3. Cross-Dataset Validation: Test models on different datasets

Author: Extended Implementation
Date: 2024
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import joblib

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not available. Install with: pip install shap")

try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import SklearnClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("[WARNING] Adversarial Robustness Toolbox (ART) not available. Install with: pip install adversarial-robustness-toolbox")


@dataclass
class ExplainabilityResults:
    """Results from explainability analysis"""
    feature_importance: Dict[str, float]
    permutation_importance: Dict[str, float]
    shap_values: Optional[np.ndarray] = None
    shap_base_value: Optional[float] = None
    summary_plot_path: Optional[str] = None
    feature_importance_plot_path: Optional[str] = None


@dataclass
class AdversarialResults:
    """Results from adversarial robustness testing"""
    attack_name:setstr
    original_accuracy: float
    adversarial_accuracy: float
    robustness_score: float
    num_adversarial_samples: int
    perturbation_stats: Dict[str, float]
    attack_success_rate: float


@dataclass
class CrossDatasetResults:
    """Results from cross-dataset validation"""
    source_dataset: str
    target_dataset: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    feature_overlap: List[str]
    performance_drop: float


class ExplainabilityAnalyzer:
    """Analyze model explainability using multiple methods"""
    
    def __init__(self, output_dir: str = "outputs/explainability"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, model: BaseEstimator, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series, feature_names: List[str],
                model_name: str = "model") -> ExplainabilityResults:
        """
        Comprehensive explainability analysis
        
        Args:
            model: Trained sklearn model
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            feature_names: List of feature names
            model_name: Name for saving outputs
        
        Returns:
            ExplainabilityResults object
        """
        print(f"\n[EXPLAINABILITY] Analyzing {model_name}...")
        
        # 1. Feature Importance (for tree-based models)
        feature_importance = self._get_feature_importance(model, feature_names)
        
        # 2. Permutation Importance
        perm_importance = self._get_permutation_importance(
            model, X_test, y_test, feature_names
        )
        
        # 3. SHAP Values (if available)
        shap_values = None
        shap_base_value = None
        shap_plot_path = None
        
        if SHAP_AVAILABLE:
            try:
                shap_results = self._compute_shap_values(
                    model, X_train, X_test, feature_names, model_name
                )
                shap_values = shap_results['values']
                shap_base_value = shap_results['base_value']
                shap_plot_path = shap_results['plot_path']
            except Exception as e:
                print(f"[WARNING] SHAP computation failed: {e}")
        
        # 4. Create visualization
        plot_path = self._plot_feature_importance(
            feature_importance, perm_importance, feature_names, model_name
        )
        
        return ExplainabilityResults(
            feature_importance=feature_importance,
            permutation_importance=perm_importance,
            shap_values=shap_values,
            shap_base_value=shap_base_value,
            summary_plot_path=shap_plot_path,
            feature_importance_plot_path=str(plot_path) if plot_path else None
        )
    
    def _get_feature_importance(self, model: BaseEstimator, 
                                feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from tree-based models"""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for name, imp in zip(feature_names, importances):
                importance_dict[name] = float(imp)
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
            for name, c in zip(feature_names, coef):
                importance_dict[name] = float(c)
        else:
            # Default: uniform importance
            for name in feature_names:
                importance_dict[name] = 1.0 / len(feature_names)
        
        return importance_dict
    
    def _get_permutation_importance(self, model: BaseEstimator, X_test: pd.DataFrame,
                                   y_test: pd.Series, feature_names: List[str],
                                   n_repeats: int = 10, random_state: int = 42) -> Dict[str, float]:
        """Compute permutation importance"""
        print(f"[EXPLAINABILITY] Computing permutation importance (n_repeats={n_repeats})...")
        
        result = permutation_importance(
            model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        
        perm_dict = {}
        for name, imp in zip(feature_names, result.importances_mean):
            perm_dict[name] = float(imp)
        
        return perm_dict
    
    def _compute_shap_values(self, model: BaseEstimator, X_train: pd.DataFrame,
                            X_test: pd.DataFrame, feature_names: List[str],
                            model_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """Compute SHAP values using TreeExplainer or KernelExplainer"""
        print(f"[EXPLAINABILITY] Computing SHAP values (sampling {sample_size} instances)...")
        
        # Sample for efficiency
        X_test_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
        X_train_sample = X_train.sample(min(100, len(X_train)), random_state=42)
        
        # Choose explainer based on model type
        model_type = type(model).__name__
        
        if 'Tree' in model_type or 'Forest' in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
        else:
            # Use KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
            shap_values = explainer.shap_values(X_test_sample)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        base_value = float(explainer.expected_value)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[1] if len(base_value) > 1 else base_value[0])
        
        # Create summary plot
        plot_path = self.output_dir / f"shap_summary_{model_name}.png"
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            print("[WARNING] matplotlib not available, skipping SHAP plot generation")
            plot_path = None
        
        return {
            'values': shap_values,
            'base_value': base_value,
            'plot_path': str(plot_path)
        }
    
    def _plot_feature_importance(self, feat_imp: Dict[str, float],
                                 perm_imp: Dict[str, float], feature_names: List[str],
                                 model_name: str) -> Optional[Path]:
        """Create feature importance comparison plot"""
        if not MATPLOTLIB_AVAILABLE:
            print("[WARNING] matplotlib not available, skipping feature importance plot")
            return None
        
        # Sort by permutation importance
        sorted_features = sorted(perm_imp.items(), key=lambda x: x[1], reverse=True)[:15]
        features = [f[0] for f in sorted_features]
        perm_values = [f[1] for f in sorted_features]
        feat_values = [feat_imp.get(f, 0) for f in features]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Permutation Importance
        ax1.barh(range(len(features)), perm_values, color='steelblue')
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Permutation Importance', fontsize=12)
        ax1.set_title('Permutation Importance (Top 15)', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Feature Importance
        ax2.barh(range(len(features)), feat_values, color='coral')
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features)
        ax2.set_xlabel('Feature Importance', fontsize=12)
        ax2.set_title('Model Feature Importance (Top 15)', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"feature_importance_{model_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path


class AdversarialRobustnessTester:
    """Test and improve adversarial robustness of models"""
    
    def __init__(self, output_dir: str = "outputs/adversarial"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_robustness(self, model: BaseEstimator, X_test: pd.DataFrame,
                       y_test: pd.Series, feature_names: List[str],
                       model_name: str = "model",
                       use_art: bool = True) -> List[AdversarialResults]:
        """
        Test model against various adversarial attacks
        
        Args:
            model: Trained sklearn model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            model_name: Name for saving outputs
            use_art: Whether to use ART library (if available)
        
        Returns:
            List of AdversarialResults
        """
        print(f"\n[ADVERSARIAL] Testing robustness of {model_name}...")
        
        results = []
        
        # Get baseline accuracy
        y_pred = model.predict(X_test)
        baseline_acc = accuracy_score(y_test, y_pred)
        print(f"[ADVERSARIAL] Baseline accuracy: {baseline_acc:.4f}")
        
        # Sample for efficiency (adversarial attacks can be slow)
        sample_size = min(1000, len(X_test))
        # Sample indices once and use for both X and y to ensure alignment
        sampled_indices = X_test.sample(sample_size, random_state=42).index
        X_sample = X_test.loc[sampled_indices].values
        y_sample = y_test.loc[sampled_indices].values
        
        if use_art and ART_AVAILABLE:
            # Use ART library for sophisticated attacks
            results.extend(self._test_with_art(model, X_sample, y_sample, 
                                              feature_names, model_name, baseline_acc))
        else:
            # Use custom implementations
            results.extend(self._test_custom_attacks(model, X_sample, y_sample,
                                                   X_test, feature_names, model_name, baseline_acc))
        
        # Save results
        self._save_results(results, model_name)
        
        return results
    
    def _test_with_art(self, model: BaseEstimator, X_test: np.ndarray,
                      y_test: np.ndarray, feature_names: List[str],
                      model_name: str, baseline_acc: float) -> List[AdversarialResults]:
        """Test using Adversarial Robustness Toolbox"""
        results = []
        
        # Wrap model for ART
        try:
            art_classifier = SklearnClassifier(model=model, clip_values=(0, np.inf))
        except Exception as e:
            print(f"[WARNING] Could not wrap model for ART: {e}")
            return results
        
        # Normalize features for attacks
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test)
        art_classifier.set_params(clip_values=(X_scaled.min(), X_scaled.max()))
        
        # Test FGSM attack
        try:
            print("[ADVERSARIAL] Testing FGSM attack...")
            attack_fgsm = FastGradientMethod(estimator=art_classifier, eps=0.1)
            X_adv_fgsm = attack_fgsm.generate(x=X_scaled)
            
            y_pred_adv = art_classifier.predict(X_adv_fgsm)
            adv_acc = accuracy_score(y_test, y_pred_adv)
            
            perturbation = np.mean(np.abs(X_adv_fgsm - X_scaled))
            
            results.append(AdversarialResults(
                attack_name="FGSM",
                original_accuracy=baseline_acc,
                adversarial_accuracy=adv_acc,
                robustness_score=adv_acc / baseline_acc if baseline_acc > 0 else 0,
                num_adversarial_samples=len(X_adv_fgsm),
                perturbation_stats={
                    'mean': float(np.mean(np.abs(X_adv_fgsm - X_scaled))),
                    'std': float(np.std(np.abs(X_adv_fgsm - X_scaled))),
                    'max': float(np.max(np.abs(X_adv_fgsm - X_scaled)))
                },
                attack_success_rate=1 - adv_acc
            ))
        except Exception as e:
            print(f"[WARNING] FGSM attack failed: {e}")
        
        # Test PGD attack
        try:
            print("[ADVERSARIAL] Testing PGD attack...")
            attack_pgd = ProjectedGradientDescent(
                estimator=art_classifier, eps=0.1, eps_step=0.01, max_iter=10
            )
            X_adv_pgd = attack_pgd.generate(x=X_scaled)
            
            y_pred_adv = art_classifier.predict(X_adv_pgd)
            adv_acc = accuracy_score(y_test, y_pred_adv)
            
            results.append(AdversarialResults(
                attack_name="PGD",
                original_accuracy=baseline_acc,
                adversarial_accuracy=adv_acc,
                robustness_score=adv_acc / baseline_acc if baseline_acc > 0 else 0,
                num_adversarial_samples=len(X_adv_pgd),
                perturbation_stats={
                    'mean': float(np.mean(np.abs(X_adv_pgd - X_scaled))),
                    'std': float(np.std(np.abs(X_adv_pgd - X_scaled))),
                    'max': float(np.max(np.abs(X_adv_pgd - X_scaled)))
                },
                attack_success_rate=1 - adv_acc
            ))
        except Exception as e:
            print(f"[WARNING] PGD attack failed: {e}")
        
        return results
    
    def _test_custom_attacks(self, model: BaseEstimator, X_sample: np.ndarray,
                           y_sample: np.ndarray, X_full: pd.DataFrame,
                           feature_names: List[str], model_name: str,
                           baseline_acc: float) -> List[AdversarialResults]:
        """Custom adversarial attack implementations"""
        results = []
        
        # Simple FGSM-like attack
        print("[ADVERSARIAL] Testing custom FGSM-like attack...")
        try:
            X_adv = self._fgsm_attack(model, X_sample, y_sample, eps=0.1)
            y_pred_adv = model.predict(X_adv)
            adv_acc = accuracy_score(y_sample, y_pred_adv)
            
            perturbation = np.mean(np.abs(X_adv - X_sample))
            
            results.append(AdversarialResults(
                attack_name="Custom_FGSM",
                original_accuracy=baseline_acc,
                adversarial_accuracy=adv_acc,
                robustness_score=adv_acc / baseline_acc if baseline_acc > 0 else 0,
                num_adversarial_samples=len(X_adv),
                perturbation_stats={
                    'mean': float(np.mean(np.abs(X_adv - X_sample))),
                    'std': float(np.std(np.abs(X_adv - X_sample))),
                    'max': float(np.max(np.abs(X_adv - X_sample)))
                },
                attack_success_rate=1 - adv_acc
            ))
        except Exception as e:
            print(f"[WARNING] Custom FGSM attack failed: {e}")
        
        # Random noise attack (baseline)
        print("[ADVERSARIAL] Testing random noise attack...")
        try:
            noise_scale = 0.1 * np.std(X_sample, axis=0)
            X_noisy = X_sample + np.random.normal(0, noise_scale, X_sample.shape)
            X_noisy = np.clip(X_noisy, 0, np.inf)  # Ensure non-negative
            
            y_pred_noisy = model.predict(X_noisy)
            noisy_acc = accuracy_score(y_sample, y_pred_noisy)
            
            results.append(AdversarialResults(
                attack_name="Random_Noise",
                original_accuracy=baseline_acc,
                adversarial_accuracy=noisy_acc,
                robustness_score=noisy_acc / baseline_acc if baseline_acc > 0 else 0,
                num_adversarial_samples=len(X_noisy),
                perturbation_stats={
                    'mean': float(np.mean(np.abs(X_noisy - X_sample))),
                    'std': float(np.std(np.abs(X_noisy - X_sample))),
                    'max': float(np.max(np.abs(X_noisy - X_sample)))
                },
                attack_success_rate=1 - noisy_acc
            ))
        except Exception as e:
            print(f"[WARNING] Random noise attack failed: {e}")
        
        return results
    
    def _fgsm_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                    eps: float = 0.1) -> np.ndarray:
        """Fast Gradient Sign Method attack (simplified for sklearn tree-based models)"""
        # For tree-based models, use feature importance to guide perturbations
        X_adv = X.copy().astype(float)
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.ones(X.shape[1]) / X.shape[1]
        
        # Normalize importances
        importances = np.abs(importances)
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        # Apply perturbations based on feature importance
        for i in range(len(X)):
            x = X[i].copy()
            pred = model.predict_proba(x.reshape(1, -1))[0]
            
            # Find features that, when perturbed, decrease confidence in true class
            true_class = y[i]
            perturbations = np.zeros_like(x)
            
            # Try perturbing each feature
            for j in range(len(x)):
                # Perturbation direction: try both positive and negative
                for direction in [-1, 1]:
                    perturbation = np.zeros_like(x)
                    # Scale perturbation by feature importance
                    perturbation[j] = direction * eps * importances[j] * np.std(X[:, j]) if np.std(X[:, j]) > 0 else direction * eps * importances[j]
                    
                    x_pert = x + perturbation
                    x_pert = np.clip(x_pert, 0, np.inf)  # Ensure non-negative
                    
                    try:
                        pred_pert = model.predict_proba(x_pert.reshape(1, -1))[0]
                        # If perturbation decreases confidence in true class, keep it
                        if pred_pert[true_class] < pred[true_class]:
                            perturbations += perturbation
                            break  # Use first successful direction
                    except:
                        continue
            
            # Apply accumulated perturbations
            X_adv[i] = np.clip(x + perturbations, 0, np.inf)
        
        return X_adv
    
    def _save_results(self, results: List[AdversarialResults], model_name: str):
        """Save adversarial test results"""
        results_dict = [asdict(r) for r in results]
        
        # Convert numpy arrays to lists for JSON serialization
        for r in results_dict:
            if 'confusion_matrix' in r and isinstance(r['confusion_matrix'], np.ndarray):
                r['confusion_matrix'] = r['confusion_matrix'].tolist()
        
        output_path = self.output_dir / f"adversarial_results_{model_name}.json"
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Also save as CSV summary
        summary_data = []
        for r in results:
            summary_data.append({
                'attack': r.attack_name,
                'original_accuracy': r.original_accuracy,
                'adversarial_accuracy': r.adversarial_accuracy,
                'robustness_score': r.robustness_score,
                'attack_success_rate': r.attack_success_rate,
                'mean_perturbation': r.perturbation_stats['mean'],
                'max_perturbation': r.perturbation_stats['max']
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / f"adversarial_summary_{model_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[ADVERSARIAL] Results saved to {csv_path}")


class CrossDatasetValidator:
    """Validate models across different datasets"""
    
    def __init__(self, output_dir: str = "outputs/cross_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self, model: BaseEstimator, source_features: List[str],
                source_dataset_name: str, target_data_path: str,
                target_dataset_name: str, label_column: str = "auto",
                positive_label: str = "DoS",
                negative_labels: List[str] = None) -> CrossDatasetResults:
        """
        Test a model trained on one dataset against another dataset
        
        Args:
            model: Trained model
            source_features: Features used during training
            source_dataset_name: Name of source dataset
            target_data_path: Path to target dataset CSV
            target_dataset_name: Name of target dataset
            label_column: Label column name (or 'auto' to detect)
            positive_label: Positive class label
            negative_labels: Negative class labels
        
        Returns:
            CrossDatasetResults object
        """
        print(f"\n[CROSS-DATASET] Validating {source_dataset_name} -> {target_dataset_name}...")
        
        if negative_labels is None:
            negative_labels = ["Normal", "Benign"]
        
        # Load and preprocess target dataset
        target_df = pd.read_csv(target_data_path, low_memory=False, on_bad_lines="skip")
        target_df.columns = [str(c).strip().replace('"', '').replace("'", "") 
                            for c in target_df.columns]
        
        # Detect label column if needed
        if label_column == "auto":
            label_column = self._detect_label_column(
                target_df, positive_label, negative_labels
            )
        
        # Preprocess target data
        target_preprocessed = self._preprocess_dataframe(
            target_df, label_column, positive_label
        )
        
        # Find feature overlap
        target_features = list(target_preprocessed.features.columns)
        feature_overlap = [f for f in source_features if f in target_features]
        
        print(f"[CROSS-DATASET] Feature overlap: {len(feature_overlap)}/{len(source_features)}")
        
        if len(feature_overlap) == 0:
            raise ValueError("No overlapping features between source and target datasets!")
        
        # Prepare data with overlapping features only
        X_target = target_preprocessed.features[feature_overlap]
        y_target = target_preprocessed.labels
        
        # Handle missing features (fill with 0 or mean)
        missing_features = [f for f in source_features if f not in feature_overlap]
        if missing_features:
            print(f"[CROSS-DATASET] Warning: {len(missing_features)} features missing, filling with 0")
            for feat in missing_features:
                X_target[feat] = 0.0
        
        # Reorder to match source feature order
        X_target = X_target[source_features]
        
        # Evaluate model
        y_pred = model.predict(X_target)
        accuracy = accuracy_score(y_target, y_pred)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_target, y_pred, zero_division=0)
        recall = recall_score(y_target, y_pred, zero_division=0)
        f1 = f1_score(y_target, y_pred, zero_division=0)
        cm = confusion_matrix(y_target, y_pred)
        
        # Estimate performance drop (would need source accuracy for comparison)
        # For now, we'll use a placeholder
        performance_drop = 0.0  # Could be calculated if source accuracy is provided
        
        results = CrossDatasetResults(
            source_dataset=source_dataset_name,
            target_dataset=target_dataset_name,
            model_name=type(model).__name__,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            feature_overlap=feature_overlap,
            performance_drop=performance_drop
        )
        
        # Save results
        self._save_results(results, source_dataset_name, target_dataset_name)
        
        return results
    
    def _detect_label_column(self, df: pd.DataFrame, positive: str,
                            negatives: List[str]) -> str:
        """Detect label column in dataset"""
        candidates = {positive.lower()} | {label.lower() for label in negatives}
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
        
        print(f"[CROSS-DATASET] Detected label column: {best_col}")
        return best_col
    
    def _preprocess_dataframe(self, df: pd.DataFrame, label_column: str,
                             positive_label: str) -> Any:
        """Preprocess dataframe similar to main pipeline"""
        labels_raw = df[label_column].astype(str).str.strip()
        labels = labels_raw.apply(
            lambda v: 1 if v.lower() == positive_label.lower() else 0
        )
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
        
        return type('PreprocessResult', (), {
            'features': features,
            'labels': labels
        })()
    
    def _save_results(self, results: CrossDatasetResults, source_name: str, target_name: str):
        """Save cross-dataset validation results"""
        results_dict = asdict(results)
        results_dict['confusion_matrix'] = results.confusion_matrix.tolist()
        
        output_path = self.output_dir / f"cross_dataset_{source_name}_to_{target_name}.json"
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save confusion matrix
        cm_path = self.output_dir / f"confusion_matrix_{source_name}_to_{target_name}.csv"
        pd.DataFrame(results.confusion_matrix).to_csv(cm_path, index=False)
        
        print(f"[CROSS-DATASET] Results saved to {output_path}")


def run_advanced_analysis(model: BaseEstimator, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series, feature_names: List[str],
                          model_name: str = "model", output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Run all advanced analyses on a pre-trained model.
    
    This function implements three main features:
    1. Explainability: SHAP, feature importance, permutation importance
    2. Adversarial Robustness: FGSM, PGD attacks
    3. Cross-Dataset Validation: (handled separately in main function)
    
    Args:
        model: Pre-trained sklearn model (loaded from outputs/)
        X_train: Training features (for SHAP background)
        X_test: Test features (for analysis)
        y_train: Training labels
        y_test: Test labels
        feature_names: List of feature names
        model_name: Name for saving outputs
        output_dir: Base output directory
    
    Returns:
        Dictionary containing all analysis results
    """
    print(f"\n{'='*60}")
    print(f"ADVANCED ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    results = {}
    
    # ========================================================================
    # 1. EXPLAINABILITY ANALYSIS
    # ========================================================================
    # - Feature Importance: Extracts from tree-based models
    # - Permutation Importance: Measures by shuffling features
    # - SHAP Values: Explains individual predictions (if SHAP available)
    # ========================================================================
    print("\n[1/2] Running Explainability Analysis...")
    explainer = ExplainabilityAnalyzer(f"{output_dir}/explainability")
    explain_results = explainer.analyze(
        model, X_train, X_test, y_train, y_test, feature_names, model_name
    )
    results['explainability'] = explain_results
    
    # ========================================================================
    # 2. ADVERSARIAL ROBUSTNESS TESTING
    # ========================================================================
    # - FGSM Attack: Fast Gradient Sign Method
    # - PGD Attack: Projected Gradient Descent (if ART available)
    # - Custom Attacks: Simplified attacks for tree-based models
    # ========================================================================
    print("\n[2/2] Running Adversarial Robustness Testing...")
    adversarial_tester = AdversarialRobustnessTester(f"{output_dir}/adversarial")
    adversarial_results = adversarial_tester.test_robustness(
        model, X_test, y_test, feature_names, model_name
    )
    results['adversarial'] = adversarial_results
    
    return results


def load_models_from_outputs(output_dir: str = "outputs") -> Dict[str, BaseEstimator]:
    """
    Load all pre-trained models from the outputs directory
    
    Args:
        output_dir: Directory containing model files
    
    Returns:
        Dictionary mapping model names to loaded models
    """
    models = {}
    output_path = Path(output_dir)
    
    # Pattern: model_{FeatureSet}_{Classifier}.pkl
    model_files = list(output_path.glob("model_*.pkl"))
    
    print(f"[LOADING] Found {len(model_files)} model files in {output_dir}")
    
    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            # Extract model name from filename: model_All_DecisionTree.pkl -> All_DecisionTree
            model_name = model_file.stem.replace("model_", "")
            models[model_name] = model
            print(f"[LOADING] Loaded: {model_name}")
        except Exception as e:
            print(f"[WARNING] Failed to load {model_file.name}: {e}")
    
    return models


def load_data_from_data_folder(data_dir: str = "data", 
                                data_file: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from the data folder
    
    Args:
        data_dir: Directory containing data files
        data_file: Specific file to load (if None, tries to find appropriate file)
    
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    data_path = Path(data_dir)
    
    # If no specific file, try to find the filtered/preprocessed one first
    if data_file is None:
        # Prefer filtered/cleaned version
        if (data_path / "iotid20_filtered.csv").exists():
            data_file = "iotid20_filtered.csv"
        elif (data_path / "IoT Network Intrusion Dataset.csv").exists():
            data_file = "IoT Network Intrusion Dataset.csv"
        else:
            # Find any CSV file
            csv_files = list(data_path.glob("*.csv"))
            if csv_files:
                data_file = csv_files[0].name
            else:
                raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    file_path = data_path / data_file
    print(f"[LOADING] Loading data from {file_path}...")
    
    df = pd.read_csv(file_path, low_memory=False, on_bad_lines="skip")
    df.columns = [str(c).strip().replace('"', '').replace("'", "") for c in df.columns]
    
    print(f"[LOADING] Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Detect label column
    label_candidates = ['Target', 'Label', 'Cat', 'Class']
    label_col = None
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        # Try to find column with binary values
        for col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                label_col = col
                break
    
    if label_col is None:
        raise ValueError("Could not detect label column. Please ensure data has 'Target', 'Label', or 'Cat' column.")
    
    print(f"[LOADING] Using label column: {label_col}")
    
    # Extract labels and features
    labels_raw = df[label_col]
    
    # Convert to binary if needed
    if labels_raw.dtype == 'object' or labels_raw.dtype.name == 'category':
        # Map DoS/Attack to 1, Normal/Benign to 0
        labels = labels_raw.astype(str).str.strip().str.lower().apply(
            lambda x: 1 if any(term in x for term in ['dos', 'attack', 'malicious']) else 0
        )
    else:
        labels = labels_raw
    
    features = df.drop(columns=[label_col])
    
    # Ensure numeric features
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            numeric = pd.to_numeric(features[col], errors="coerce")
            if numeric.notna().sum() > 0:
                features[col] = numeric
            else:
                encoded, _ = pd.factorize(features[col].fillna("missing").astype(str))
                features[col] = pd.Series(encoded, index=features.index, dtype=float)
    
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    print(f"[LOADING] Features: {features.shape[1]}, Samples: {len(labels)}")
    print(f"[LOADING] Class distribution: Normal={sum(labels==0)}, DoS={sum(labels==1)}")
    
    return features, labels


def load_feature_sets(output_dir: str = "outputs") -> Dict[str, List[str]]:
    """
    Load feature sets from outputs directory
    
    Args:
        output_dir: Directory containing feature selection results
    
    Returns:
        Dictionary mapping feature set names to feature lists
    """
    feature_sets = {}
    output_path = Path(output_dir)
    
    # Load CFS features
    cfs_path = output_path / "cfs_features.csv"
    if cfs_path.exists():
        cfs_features = pd.read_csv(cfs_path, header=None)[0].tolist()
        feature_sets['CFS'] = cfs_features
        print(f"[LOADING] Loaded CFS features: {len(cfs_features)} features")
    
    # Load GA features
    ga_path = output_path / "ga_features.csv"
    if ga_path.exists():
        ga_features = pd.read_csv(ga_path, header=None)[0].tolist()
        feature_sets['GA'] = ga_features
        print(f"[LOADING] Loaded GA features: {len(ga_features)} features")
    
    return feature_sets


def load_metrics_from_outputs(output_dir: str = "outputs") -> pd.DataFrame:
    """
    Load evaluation metrics from outputs directory
    
    Args:
        output_dir: Directory containing metrics.csv
    
    Returns:
        DataFrame with metrics
    """
    metrics_path = Path(output_dir) / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        print(f"[LOADING] Loaded metrics for {len(metrics)} model configurations")
        return metrics
    else:
        print(f"[WARNING] metrics.csv not found in {output_dir}")
        return pd.DataFrame()


def run_advanced_analysis_on_saved_models(data_dir: str = "data",
                                          output_dir: str = "outputs",
                                          data_file: str = None,
                                          test_size: float = 0.33,
                                          random_state: int = 42,
                                          skip_models: List[str] = None) -> Dict[str, Any]:
    """
    Run advanced analysis on all pre-trained models from outputs folder.
    
    NOTE: This function does NOT retrain models. It loads existing trained models
    from the outputs/ folder and runs advanced analysis on them.
    
    The three main features implemented:
    1. Explainability: SHAP, feature importance, permutation importance
    2. Adversarial Robustness: FGSM, PGD attacks
    3. Cross-Dataset Validation: Test models on different datasets
    
    Args:
        data_dir: Directory containing data files
        output_dir: Directory containing models and results (default: "outputs")
        data_file: Specific data file to use (if None, auto-detects)
        test_size: Test set size for splitting data (must match original training: 0.33)
        random_state: Random state for reproducibility (must match original: 42)
        skip_models: List of model names to skip (e.g., ['All_SVM', 'CFS_KNN'])
    
    Returns:
        Dictionary containing all analysis results for each model
    """
    print("=" * 70)
    print("ADVANCED ANALYSIS ON SAVED MODELS")
    print("=" * 70)
    print("NOTE: Using pre-trained models from outputs/ - NO RETRAINING")
    print("=" * 70)
    
    if skip_models is None:
        skip_models = []
    
    # Load existing models (NO RETRAINING)
    print("\n[1/4] Loading pre-trained models from outputs/...")
    models = load_models_from_outputs(output_dir)
    if not models:
        raise ValueError(f"No models found in {output_dir}/. Please train models first using main.py")
    
    # Load existing feature sets
    print("\n[2/4] Loading feature sets from outputs/...")
    feature_sets = load_feature_sets(output_dir)
    
    # Load existing metrics for baseline comparison
    print("\n[3/4] Loading existing evaluation results from metrics.csv...")
    metrics_df = load_metrics_from_outputs(output_dir)
    
    # Load data for analysis (using same split as original training)
    print("\n[4/4] Loading data for analysis...")
    X, y = load_data_from_data_folder(data_dir, data_file)
    
    # Split data using SAME parameters as original training to ensure consistency
    # Original training used: test_size=0.33, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"\n[DATA] Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    print(f"[MODELS] Loaded {len(models)} pre-trained models")
    
    all_results = {}
    
    # ========================================================================
    # RUN ADVANCED ANALYSIS ON EACH SAVED MODEL
    # ========================================================================
    # For each pre-trained model, run:
    # 1. Explainability Analysis
    # 2. Adversarial Robustness Testing
    # 3. Cross-Dataset Validation (if multiple datasets available)
    # ========================================================================
    
    for model_name, model in models.items():
        if model_name in skip_models:
            print(f"\n[SKIPPING] {model_name}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")
        
        # Extract feature set and classifier from model name
        # Format: FeatureSet_Classifier (e.g., "All_DecisionTree", "CFS_RandomForest")
        parts = model_name.split('_', 1)
        if len(parts) == 2:
            feature_set_name, classifier_name = parts
        else:
            feature_set_name = "All"
            classifier_name = parts[0]
        
        # Get appropriate features
        if feature_set_name in feature_sets:
            selected_features = feature_sets[feature_set_name]
            # Filter to only features that exist in the data
            selected_features = [f for f in selected_features if f in X.columns]
        elif feature_set_name == "All":
            selected_features = list(X.columns)
        else:
            print(f"[WARNING] Feature set '{feature_set_name}' not found, using all features")
            selected_features = list(X.columns)
        
        # Prepare data with selected features
        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]
        
        print(f"[FEATURES] Using {len(selected_features)} features from '{feature_set_name}' set")
        
        # Get baseline metrics if available
        if not metrics_df.empty:
            model_metrics = metrics_df[
                (metrics_df['FeatureSet'] == feature_set_name) & 
                (metrics_df['Classifier'] == classifier_name)
            ]
            if not model_metrics.empty:
                row = model_metrics.iloc[0]
                print(f"[BASELINE] Accuracy: {row['Accuracy']:.4f}, "
                      f"Precision: {row['Precision']:.4f}, "
                      f"Recall: {row['Recall']:.4f}, "
                      f"F1: {row['F1']:.4f}")
        
        # Run advanced analysis on the pre-trained model
        # This includes:
        # 1. Explainability (SHAP, feature importance, permutation importance)
        # 2. Adversarial Robustness (FGSM, PGD attacks)
        try:
            results = run_advanced_analysis(
                model=model,  # Using pre-trained model (NO RETRAINING)
                X_train=X_train_subset,
                X_test=X_test_subset,
                y_train=y_train,
                y_test=y_test,
                feature_names=selected_features,
                model_name=model_name,
                output_dir=output_dir
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"[ERROR] Failed to analyze {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========================================================================
    # 3. CROSS-DATASET VALIDATION
    # ========================================================================
    # Test pre-trained models on different datasets to evaluate generalization
    # ========================================================================
    data_path = Path(data_dir)
    data_files = list(data_path.glob("*.csv"))
    if len(data_files) > 1:
        print(f"\n{'='*70}")
        print("3. CROSS-DATASET VALIDATION")
        print(f"{'='*70}")
        print("Testing pre-trained models on different datasets...")
        
        # Use first file as source, others as targets
        source_file = data_files[0]
        target_files = data_files[1:]
        
        for target_file in target_files:
            print(f"\n[CROSS-DATASET] Testing models on {target_file.name}...")
            try:
                # Load target dataset
                X_target, y_target = load_data_from_data_folder(data_dir, target_file.name)
                
                validator = CrossDatasetValidator(f"{output_dir}/cross_dataset")
                
                # Test each pre-trained model on the target dataset
                for model_name, model in models.items():
                    if model_name in skip_models:
                        continue
                    
                    # Extract feature set name
                    parts = model_name.split('_', 1)
                    if len(parts) == 2:
                        feature_set_name, _ = parts
                    else:
                        feature_set_name = "All"
                    
                    # Get appropriate features
                    if feature_set_name in feature_sets:
                        selected_features = feature_sets[feature_set_name]
                        selected_features = [f for f in selected_features if f in X.columns]
                    elif feature_set_name == "All":
                        selected_features = list(X.columns)
                    else:
                        selected_features = list(X.columns)
                    
                    try:
                        cross_results = validator.validate(
                            model=model,  # Using pre-trained model
                            source_features=selected_features,
                            source_dataset_name=source_file.stem,
                            target_data_path=str(target_file),
                            target_dataset_name=target_file.stem,
                            label_column="auto",
                            positive_label="DoS"
                        )
                        if model_name not in all_results:
                            all_results[model_name] = {}
                        all_results[model_name]['cross_dataset'] = cross_results
                    except Exception as e:
                        print(f"[WARNING] Cross-dataset validation failed for {model_name}: {e}")
            except Exception as e:
                print(f"[WARNING] Failed to load target dataset {target_file.name}: {e}")
    else:
        print(f"\n[INFO] Only one dataset found. Skipping cross-dataset validation.")
        print(f"      Add more datasets to {data_dir}/ for cross-dataset testing.")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Explainability: {output_dir}/explainability/")
    print(f"  - Adversarial: {output_dir}/adversarial/")
    print(f"  - Cross-dataset: {output_dir}/cross_dataset/")
    
    return all_results


if __name__ == "__main__":
    """
    Advanced Analysis Module for IoT DoS Detection
    ===============================================
    
    This module extends the base paper implementation with three advanced features:
    
    1. EXPLAINABILITY
       - SHAP values for model interpretability
       - Feature importance extraction
       - Permutation importance analysis
    
    2. ADVERSARIAL ROBUSTNESS
       - FGSM (Fast Gradient Sign Method) attacks
       - PGD (Projected Gradient Descent) attacks
       - Custom adversarial attack implementations
    
    3. CROSS-DATASET VALIDATION
       - Tests pre-trained models on different datasets
       - Evaluates generalization across IoT environments
       - Measures performance degradation
    
    IMPORTANT: This script uses PRE-TRAINED models from outputs/ folder.
    It does NOT retrain models. Models must be trained first using main.py.
    
    Usage:
        python advanced_analysis.py
    """
    print("=" * 70)
    print("Advanced Analysis Module for IoT DoS Detection")
    print("=" * 70)
    print("\nThis module provides:")
    print("  1. Explainability: SHAP, feature importance, permutation importance")
    print("  2. Adversarial Robustness: FGSM, PGD attacks")
    print("  3. Cross-Dataset Validation: Test models on different datasets")
    print("\n" + "=" * 70)
    print("Using PRE-TRAINED models from outputs/ folder")
    print("(Models are NOT retrained - they must exist from main.py)")
    print("=" * 70)
    
    # Run analysis on all saved models
    try:
        results = run_advanced_analysis_on_saved_models(
            data_dir="data",
            output_dir="outputs",
            test_size=0.33,      # Must match original training
            random_state=42      # Must match original training
        )
        print("\n" + "=" * 70)
        print("✓ ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved to outputs/")
        print(f"  - Explainability: outputs/explainability/")
        print(f"  - Adversarial: outputs/adversarial/")
        print(f"  - Cross-dataset: outputs/cross_dataset/")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("TROUBLESHOOTING:")
        print("=" * 70)
        print("1. Ensure models exist in outputs/ folder")
        print("   Run: python main.py")
        print("2. Ensure data exists in data/ folder")
        print("3. Check that feature sets exist: outputs/cfs_features.csv, outputs/ga_features.csv")
        print("\nYou can also use this module programmatically:")
        print("  from advanced_analysis import run_advanced_analysis_on_saved_models")
        print("  results = run_advanced_analysis_on_saved_models()")

