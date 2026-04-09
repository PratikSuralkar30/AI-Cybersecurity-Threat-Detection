"""
preprocessor.py
===============
Cleans the raw network traffic DataFrame and prepares it for ML:
  1. Map raw labels → binary (is_attack) + category (attack_category)
  2. Encode categorical columns with LabelEncoder
  3. Normalize all numeric features with StandardScaler
  4. Stratified train / test split (80 / 20)
  5. Return everything the rest of the pipeline needs
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

from data_loader import map_attack_category

# Columns that are textual / categorical in the raw data
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Columns to exclude when building the feature matrix
LABEL_COLS = ['label', 'attack_category', 'is_attack', 'difficulty_level']


def preprocess(df: pd.DataFrame,
               sample_size: int = None,
               test_size: float = 0.20,
               random_state: int = 42):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df           : raw DataFrame from data_loader
    sample_size  : if set, randomly subsample before processing
                   (useful to speed up experiments)
    test_size    : fraction reserved for evaluation
    random_state : reproducibility seed

    Returns
    -------
    X_train, X_test  : scaled feature arrays
    y_train, y_test  : binary labels (0=normal, 1=attack)
    scaler           : fitted StandardScaler (needed for detector.py)
    feature_cols     : list of feature column names
    df_clean         : processed DataFrame (needed for visualiser)
    encoders         : dict of {col: fitted LabelEncoder}
    """
    print("\n[+] ─── Preprocessing ─────────────────────────────────")

    # ── Optional downsampling ────────────────────────────────────
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"[+] Sampled {sample_size:,} rows from full dataset")

    df = df.copy()

    # ── Clean label column ───────────────────────────────────────
    df['label'] = (df['label']
                   .astype(str)
                   .str.replace('.', '', regex=False)
                   .str.strip()
                   .str.lower())

    # ── Map to high-level attack category ───────────────────────
    df['attack_category'] = df['label'].apply(map_attack_category)

    # ── Binary target ────────────────────────────────────────────
    df['is_attack'] = (df['label'] != 'normal').astype(np.int8)

    # ── Remove exact duplicates ──────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[+] Dropped {before - len(df):,} duplicate rows")

    # ── Handle missing values ────────────────────────────────────
    null_counts = df.isnull().sum().sum()
    if null_counts:
        df.fillna(0, inplace=True)
        print(f"[!] Filled {null_counts} null values with 0")
    else:
        print("[+] No null values found")

    # ── Encode categorical columns ───────────────────────────────
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    print(f"[+] Encoded categorical columns: {CATEGORICAL_COLS}")

    # ── Build feature matrix ─────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y = df['is_attack'].values

    # ── Normalize with StandardScaler ───────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler so detector.py can reuse it
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    print("[+] Scaler + encoders saved to models/")

    # ── Train / test split (stratified) ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ── Summary ─────────────────────────────────────────────────
    attack_pct = y.mean() * 100
    print(f"[+] Features: {len(feature_cols)}")
    print(f"[+] Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
    print(f"[+] Attack ratio in full set: {attack_pct:.1f}%")
    print(f"[+] Attack distribution:\n"
          f"    {df['attack_category'].value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols, df, encoders
