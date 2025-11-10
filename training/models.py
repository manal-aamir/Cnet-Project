from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def build_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Build all ML models including base models and ensembles.
    
    Args:
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary of model_name -> model_instance (unfitted)
    """
    # Base models with configurations from the paper
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    
    svc = SVC(
        kernel='rbf',
        gamma='scale',
        probability=True,  # Required for soft voting
        random_state=random_state
    )
    
    lr = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1
    )
    
    models = {
        'RandomForest': rf,
        'SVC': svc,
        'LogisticRegression': lr
    }
    
    # Ensemble models
    ensemble_hard = VotingClassifier(
        estimators=[('rf', rf), ('svc', svc), ('lr', lr)],
        voting='hard'
    )
    
    ensemble_soft = VotingClassifier(
        estimators=[('rf', rf), ('svc', svc), ('lr', lr)],
        voting='soft'
    )
    
    models['EnsembleHard'] = ensemble_hard
    models['EnsembleSoft'] = ensemble_soft
    
    return models


def train_models(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Train all models on the training data.
    
    Args:
        models: Dictionary of model_name -> untrained model
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Dictionary of model_name -> trained model
    """
    print(f"\nTraining {len(models)} models...")
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"  Training {name}...", end=' ')
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print("Done")
        except Exception as e:
            print(f"Failed: {e}")
    
    print(f"Successfully trained {len(trained_models)} models\n")
    return trained_models
