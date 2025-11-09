#!/usr/bin/env python3
"""WSN IDS reproduction runner with verbose print statements."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    ROOT_DIR / "WSN-DS.csv",
    ROOT_DIR / "WSNBFSFdataset.csv",
]

@dataclass
class PreprocessResult:
    features: pd.DataFrame
    labels: pd.Series
    label_column: str
    encoded_columns: List[str]


@dataclass
class SelectionResult:
    selected_features: List[str]
    support_mask: np.ndarray


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    confusion: np.ndarray


class GeneticFeatureSelector:
    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        population_size=30,
        generations=25,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_features=13,
        scoring="f1",
        cv_splits=3,
        random_state=42,
        penalty=0.01,
    ):
        self.estimator = estimator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_features = max_features
        self.scoring = scoring
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.penalty = penalty
        self.selected_features_: List[str] = []

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> SelectionResult:
        rng = np.random.default_rng(self.random_state)
        matrix = np.asarray(features.values, dtype=float)
        n_features = matrix.shape[1]
        population = self._init_population(n_features, rng)
        best_mask, best_score = population[0], -math.inf

        for gen in range(self.generations):
            scores = self._evaluate_population(population, matrix, labels)
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > best_score:
                best_score, best_mask = scores[best_idx], population[best_idx].copy()
            population = self._breed(population, scores, rng)
            if gen % 5 == 0:
                print(f"[INFO] GA generation {gen}/{self.generations} — best score: {best_score:.4f}")

        mask = best_mask.astype(bool)
        self.selected_features_ = list(features.columns[mask])
        print(f"[INFO] GA selected {len(self.selected_features_)} features.")
        return SelectionResult(self.selected_features_, mask)

    def _init_population(self, n_features, rng):
        prob = min(0.9, self.max_features / float(n_features)) if n_features else 0.5
        pop = rng.random((self.population_size, n_features)) < prob
        pop[np.sum(pop, axis=1) == 0] = rng.random(n_features) < prob
        return pop.astype(bool)

    def _evaluate_population(self, population, matrix, labels):
        scores = np.zeros(population.shape[0])
        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        for i, mask in enumerate(population):
            if not mask.any():
                continue
            subset = matrix[:, mask]
            try:
                estimator = clone(self.estimator)
                cv_scores = cross_val_score(
                    estimator,
                    subset,
                    labels,
                    cv=skf,
                    scoring=self.scoring,
                    n_jobs=1,
                )
                score = float(np.mean(cv_scores))
            except Exception:
                score = 0.0
            if self.max_features and mask.sum() > self.max_features:
                score -= self.penalty * (mask.sum() - self.max_features)
            scores[i] = score
        return scores

    def _breed(self, population, fitness, rng):
        parents = self._tournament(population, fitness, rng)
        offspring = []
        for i in range(0, len(parents), 2):
            a, b = parents[i], parents[(i + 1) % len(parents)]
            if rng.random() < self.crossover_rate:
                p = rng.integers(1, a.size - 1)
                c1 = np.concatenate([a[:p], b[p:]])
                c2 = np.concatenate([b[:p], a[p:]])
            else:
                c1, c2 = a.copy(), b.copy()
            offspring.extend([self._mutate(c1, rng), self._mutate(c2, rng)])
        return np.array(offspring[: self.population_size], dtype=bool)

    def _tournament(self, population, fitness, rng):
        selected = []
        for _ in range(len(population)):
            i1, i2 = rng.choice(len(population), size=2, replace=False)
            selected.append(population[i1 if fitness[i1] >= fitness[i2] else i2])
        return selected

    def _mutate(self, chrom, rng):
        mask = rng.random(chrom.shape[0]) < self.mutation_rate
        chrom = chrom.copy()
        chrom[mask] = ~chrom[mask]
        if not chrom.any():
            chrom[rng.integers(0, chrom.shape[0])] = True
        return chrom


class CorrelationFeatureSelector:
    def __init__(self, *, max_features=13, min_target_correlation=0.01, redundancy_threshold=0.7):
        self.max_features = max_features
        self.min_target_correlation = min_target_correlation
        self.redundancy_threshold = redundancy_threshold
        self.selected_features_: List[str] = []

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> SelectionResult:
        correlations = []
        labels_numeric = pd.to_numeric(labels, errors="coerce")
        for c in features.columns:
            s = features[c]
            corr = abs(s.corr(labels_numeric)) if s.std() > 0 else 0.0
            correlations.append((c, corr))
        correlations.sort(key=lambda x: x[1], reverse=True)

        selected = []
        for c, corr in correlations:
            if corr < self.min_target_correlation:
                continue
            if any(abs(features[c].corr(features[sel])) >= self.redundancy_threshold for sel in selected):
                continue
            selected.append(c)
            if len(selected) >= self.max_features:
                break
        mask = features.columns.isin(selected)
        self.selected_features_ = selected
        print(f"[INFO] CFS selected {len(selected)} features.")
        return SelectionResult(selected, mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train WSN IDS models with GA/CFS feature selection while printing study context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_paths = [str(p) for p in DEFAULT_DATASETS if p.exists()]
    parser.add_argument(
        "--data-paths",
        nargs="+",
        default=default_paths,
        help="One or more CSV files to evaluate sequentially.",
    )
    parser.add_argument(
        "--label-column",
        default="auto",
        help="Name of the label column. If auto, the script attempts to detect it.",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help="Label that should be considered an attack. If omitted, any value not in --negative-labels is treated as positive.",
    )
    parser.add_argument(
        "--negative-labels",
        nargs="+",
        default=["Normal", "normal", "Benign", "benign", "0"],
        help="Labels that should be treated as benign/normal.",
    )
    parser.add_argument("--test-size", type=float, default=0.33, help="Hold-out test size.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--disable-ga", action="store_true", help="Skip GA feature selection.")
    parser.add_argument("--disable-cfs", action="store_true", help="Skip CFS feature selection.")
    parser.add_argument("--max-ga-features", type=int, default=13)
    parser.add_argument("--max-cfs-features", type=int, default=13)
    parser.add_argument("--ga-population", type=int, default=30)
    parser.add_argument("--ga-generations", type=int, default=25)
    parser.add_argument("--ga-mutation", type=float, default=0.1)
    parser.add_argument("--ga-crossover", type=float, default=0.8)
    parser.add_argument("--output-dir", default="artifacts_wsn", help="Directory for optional future outputs.")
    return parser.parse_args()



def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    df.columns = [str(c).strip().replace('"', '').replace("'", "") for c in df.columns]
    print(f"[INFO] Loaded {path} -> {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def detect_label_column(df: pd.DataFrame, *, positive: Optional[str], negatives: Sequence[str]) -> str:
    candidates = {label.lower() for label in negatives}
    if positive:
        candidates.add(positive.lower())
    best_col: Optional[str] = None
    best_hits = -1
    for column in df.columns:
        series = df[column].astype(str).str.strip().str.lower()
        hits = series.isin(candidates).sum()
        if hits > best_hits:
            best_hits = hits
            best_col = column
    if best_col is None or best_hits <= 0:
        raise ValueError("Failed to detect label column automatically.")
    print(f"[INFO] Detected label column: {best_col}")
    return best_col


def coerce_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        return numeric
    encoded, _ = pd.factorize(series.fillna("missing").astype(str))
    return pd.Series(encoded, index=series.index, dtype=float)


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    label_column: str,
    positive_label: Optional[str],
    negative_labels: Sequence[str],
) -> PreprocessResult:
    labels_raw = df[label_column].astype(str).str.strip()
    negatives = {label.lower() for label in negative_labels}
    positives = {positive_label.lower()} if positive_label else set()

    def encode_label(value: str) -> int:
        lv = value.lower()
        if lv in negatives:
            return 0
        if positives:
            return 1 if lv in positives else 0
        return 1

    labels = labels_raw.apply(encode_label)
    features = df.drop(columns=[label_column]).copy()

    encoded_cols: List[str] = []
    for c in features.columns:
        if not pd.api.types.is_numeric_dtype(features[c]):
            features[c] = coerce_numeric(features[c])
            encoded_cols.append(c)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"[INFO] Encoded {len(encoded_cols)} non-numeric columns.")
    positives_count = int(labels.sum())
    print(
        f"[INFO] Label distribution -> attacks: {positives_count:,}, normal: {labels.size - positives_count:,}"
    )
    return PreprocessResult(features=features, labels=labels, label_column=label_column, encoded_columns=encoded_cols)


def evaluate_model(model, X_train, X_test, y_train, y_test) -> EvaluationResult:
    t0 = perf_counter()
    model.fit(X_train, y_train)
    train_time = perf_counter() - t0
    t1 = perf_counter()
    preds = model.predict(X_test)
    test_time = perf_counter() - t1

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "train_time": train_time,
        "test_time": test_time,
    }
    cm = confusion_matrix(y_test, preds, labels=[1, 0])
    return EvaluationResult(metrics=metrics, confusion=cm)


def run_experiment(data_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    print(f"\n[INFO] ===== Starting experiment for {data_path} =====")
    df = load_dataset(data_path)
    label_column = args.label_column
    if label_column == "auto":
        label_column = detect_label_column(df, positive=args.positive_label, negatives=args.negative_labels)

    pre = preprocess_dataframe(
        df,
        label_column=label_column,
        positive_label=args.positive_label,
        negative_labels=args.negative_labels,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        pre.features,
        pre.labels,
        test_size=args.test_size,
        stratify=pre.labels,
        random_state=args.random_state,
    )

    classifiers = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel="rbf", gamma="scale"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    feature_sets = {"All": list(pre.features.columns)}

    if not args.disable_ga:
        print("\n[INFO] Running GA feature selection...")
        ga = GeneticFeatureSelector(
            DecisionTreeClassifier(random_state=42),
            population_size=args.ga_population,
            generations=args.ga_generations,
            mutation_rate=args.ga_mutation,
            crossover_rate=args.ga_crossover,
            max_features=args.max_ga_features,
        )
        selection = ga.fit(X_train, y_train)
        feature_sets["GA"] = selection.selected_features

    if not args.disable_cfs:
        print("\n[INFO] Running CFS feature selection...")
        cfs = CorrelationFeatureSelector(max_features=args.max_cfs_features)
        selection = cfs.fit(X_train, y_train)
        feature_sets["CFS"] = selection.selected_features

    rows = []
    for selector_name, feats in feature_sets.items():
        Xtr, Xte = X_train[feats], X_test[feats]
        for clf_name, clf in classifiers.items():
            print(f"[INFO] Training {clf_name} with {selector_name} feature set ({len(feats)} columns)...")
            result = evaluate_model(clone(clf), Xtr, Xte, y_train, y_test)
            rows.append(
                {
                    "Selector": selector_name,
                    "Classifier": clf_name,
                    "n_features": len(feats),
                    **result.metrics,
                }
            )
    df_results = pd.DataFrame(rows)
    print("\n[INFO] Final Evaluation Results:")
    print(tabulate(df_results, headers="keys", tablefmt="github", floatfmt=".4f"))
    return df_results


def main() -> None:
    args = parse_args()
    if not args.data_paths:
        raise SystemExit("No data paths provided and defaults missing. Please supply --data-paths.")

    aggregated_results = []
    for path_str in args.data_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"[WARN] Skipping missing dataset: {path}")
            continue
        df_results = run_experiment(path, args)
        df_results.insert(0, "Dataset", path.name)
        aggregated_results.append(df_results)

    if aggregated_results:
        combined = pd.concat(aggregated_results, ignore_index=True)
        print("\n[INFO] Combined results across all datasets:")
        print(tabulate(combined, headers="keys", tablefmt="github", floatfmt=".4f"))
    else:
        print("[WARN] No datasets were processed; nothing to report.")


if __name__ == "__main__":
    main()
