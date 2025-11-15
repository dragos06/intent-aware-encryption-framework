from typing import Dict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .config import cfg
import numpy as np

def train_classifiers(X_train, y_train, params=None):
    """Train RF, LR, and MLP with lightweight gridsearch."""
    params = params or {}
    models = {}

    rf = RandomForestClassifier(random_state=cfg.random_seed)
    rf_grid = GridSearchCV(rf, params.get("rf_params", {"n_estimators": [50]}), cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    models["random_forest"] = rf_grid.best_estimator_

    lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
    lr_grid = GridSearchCV(lr, params.get("lr_params", {"C": [1.0]}), cv=3, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    models["logistic_regression"] = lr_grid.best_estimator_

    mlp = MLPClassifier(random_state=cfg.random_seed, max_iter=300)
    mlp_grid = GridSearchCV(mlp, params.get("mlp_params", {"hidden_layer_sizes": [(50,)]}), cv=3, n_jobs=-1)
    mlp_grid.fit(X_train, y_train)
    models["mlp"] = mlp_grid.best_estimator_

    return models

def evaluate_classifier(estimator, X_test, y_test, labels):
    y_pred = estimator.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }
