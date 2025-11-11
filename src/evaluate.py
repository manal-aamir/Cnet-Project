import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

def run_experiment(X, y, feature_sets, classifiers, test_size=0.33, random_state=42, output_dir=None):
    results = []
    conf_matrices = {}
    for feat_name, feats in feature_sets.items():
        X_sel = X[feats]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y, test_size=test_size, stratify=y, random_state=random_state)
        for clf_name, clf in classifiers.items():
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            y_pred = clf.predict(X_test)
            test_time = time.perf_counter() - t1
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            results.append({
                'FeatureSet': feat_name,
                'Classifier': clf_name,
                'n_features': len(feats),
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'TrainTime': train_time,
                'TestTime': test_time
            })
            conf_matrices[(feat_name, clf_name)] = cm
            # Save confusion matrix and model
            if output_dir:
                cm_path = f"{output_dir}/confmat_{feat_name}_{clf_name}.csv"
                pd.DataFrame(cm).to_csv(cm_path, index=False)
                model_path = f"{output_dir}/model_{feat_name}_{clf_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(clf, f)
    df_results = pd.DataFrame(results)
    if output_dir:
        df_results.to_csv(f"{output_dir}/metrics.csv", index=False)
    return df_results, conf_matrices

