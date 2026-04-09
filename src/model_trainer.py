"""
model_trainer.py
================
Trains and evaluates two complementary models:

  1. IsolationForest  — UNSUPERVISED anomaly detection
       • No labels required during training
       • Scores each connection; low score → anomaly
       • Mirrors production systems where labels are unavailable

  2. RandomForestClassifier — SUPERVISED attack classification
       • Trained on labelled data (binary: normal vs attack)
       • Also provides class probability (confidence score)
       • Feature importance reveals WHICH features expose each attack

Both models are serialised with joblib so the detector can reload
them without retraining.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score
)


# ─────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────

def train_isolation_forest(X_train: np.ndarray,
                            contamination: float = 0.10,
                            n_estimators: int = 100) -> IsolationForest:
    """
    Train Isolation Forest.

    contamination : expected proportion of anomalies in training data.
                    Set to the known attack ratio, or leave at 0.10
                    when that ratio is unknown (real-world default).
    """
    print("\n[+] ─── Training Isolation Forest ─────────────────────")
    print(f"    n_estimators={n_estimators}  contamination={contamination}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples='auto',
        bootstrap=False,
        random_state=42,
        n_jobs=-1          # use all CPU cores
    )
    model.fit(X_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/isolation_forest.pkl')
    print("[+] Isolation Forest saved → models/isolation_forest.pkl")
    return model


def train_random_forest(X_train: np.ndarray,
                         y_train: np.ndarray,
                         n_estimators: int = 150) -> RandomForestClassifier:
    """
    Train Random Forest Classifier.

    class_weight='balanced' compensates for the U2R / R2L minority classes.
    max_depth=None lets trees grow fully (high accuracy on KDD / simulation).
    """
    print("\n[+] ─── Training Random Forest ────────────────────────")
    print(f"    n_estimators={n_estimators}  class_weight=balanced")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/random_forest.pkl')
    print("[+] Random Forest saved → models/random_forest.pkl")
    return model


# ─────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────

def evaluate_random_forest(rf_model, X_test, y_test):
    """
    Compute and print full classification metrics for the RF model.

    Returns
    -------
    y_pred  : predicted binary labels
    cm      : confusion matrix (2×2 ndarray)
    metrics : dict of scalar scores
    """
    y_pred   = rf_model.predict(X_test)
    y_proba  = rf_model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print("  Random Forest — Evaluation Results")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\n  Confusion Matrix:")
    print(f"             Predicted Normal  Predicted Attack")
    print(f"  Actual Normal    {cm[0,0]:>8,}        {cm[0,1]:>8,}")
    print(f"  Actual Attack    {cm[1,0]:>8,}        {cm[1,1]:>8,}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['Normal', 'Attack'],
                                 zero_division=0))

    metrics = dict(accuracy=acc, precision=prec,
                   recall=rec, f1=f1, roc_auc=auc)
    return y_pred, cm, metrics


def evaluate_isolation_forest(if_model, X_test, y_test):
    """
    Convert Isolation Forest scores → binary predictions and
    compute basic detection metrics (no retraining needed).

    Isolation Forest uses predict():
       1  → normal
      -1  → anomaly (attack)

    We remap to {0: normal, 1: attack} for consistency.
    """
    raw_pred = if_model.predict(X_test)   # 1 or -1
    y_pred   = (raw_pred == -1).astype(int)
    scores   = if_model.score_samples(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print("  Isolation Forest — Evaluation Results")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Note: IF is unsupervised — performance depends on")
    print(f"  contamination parameter, not labelled training data.")

    return y_pred, scores, cm


def get_feature_importance(rf_model, feature_cols) -> pd.DataFrame:
    """Return sorted feature importance DataFrame."""
    df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return df


def load_models():
    """Reload saved models from disk (used by detector.py)."""
    paths = {
        'isolation_forest': 'models/isolation_forest.pkl',
        'random_forest':    'models/random_forest.pkl',
        'scaler':           'models/scaler.pkl',
        'encoders':         'models/encoders.pkl',
    }
    loaded = {}
    for name, path in paths.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
            print(f"[+] Loaded {name} from {path}")
        else:
            print(f"[!] WARNING: {path} not found")
    return loaded
