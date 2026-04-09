"""
feature_engineer.py
===================
Optional feature-selection step that ranks features by importance
and returns a reduced feature set.

We provide two strategies:
  A) SelectKBest  — statistical test (chi2 or f_classif)
  B) RandomForest importance — tree-based ranking

This module is optional.  main.py uses all 41 features by default,
but calling select_features() can trim the set to the top-k most
discriminative features, which speeds up training with minimal loss
in accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier


def select_features_kbest(X_train, y_train, X_test,
                           feature_cols, k=20):
    """
    Reduce to top-k features using ANOVA F-statistic.

    Parameters
    ----------
    X_train, y_train  : training data
    X_test            : test data (same transform applied)
    feature_cols      : list of current feature names
    k                 : number of top features to keep

    Returns
    -------
    X_train_sel, X_test_sel, selected_cols, selector
    """
    print(f"\n[+] SelectKBest → keeping top {k} of {len(feature_cols)} features")

    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    mask = selector.get_support()
    selected_cols = [c for c, m in zip(feature_cols, mask) if m]

    scores = pd.DataFrame({
        'feature': feature_cols,
        'score': selector.scores_,
        'selected': mask
    }).sort_values('score', ascending=False)

    print("[+] Top 10 features by F-score:")
    print(scores[scores['selected']].head(10).to_string(index=False))

    return X_train_sel, X_test_sel, selected_cols, selector


def get_rf_feature_importance(rf_model, feature_cols, top_n=20):
    """
    Return a sorted DataFrame of Random Forest feature importances.

    Parameters
    ----------
    rf_model     : trained RandomForestClassifier
    feature_cols : list of feature names
    top_n        : how many top features to highlight

    Returns
    -------
    pd.DataFrame with columns ['feature', 'importance', 'rank']
    """
    importance = rf_model.feature_importances_
    df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    print(f"\n[+] Top {top_n} features by RF importance:")
    print(df.head(top_n).to_string(index=False))

    return df


def correlation_heatmap_data(df_clean, feature_cols, top_n=15):
    """
    Compute correlation matrix for the top-n most important numeric
    columns + the is_attack target. Used by visualizer.py.

    Returns a square DataFrame suitable for seaborn heatmap.
    """
    cols = feature_cols[:top_n] + ['is_attack']
    cols = [c for c in cols if c in df_clean.columns]
    return df_clean[cols].corr()
