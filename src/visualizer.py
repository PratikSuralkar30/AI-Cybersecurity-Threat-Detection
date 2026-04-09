"""
visualizer.py
=============
All plotting functions for the project dashboard.

Charts generated
----------------
1.  confusion_matrix.png         — RF detection accuracy
2.  feature_importance.png       — Top-15 RF features
3.  attack_distribution.png      — Pie chart of attack categories
4.  anomaly_scores.png           — IF scores scatter (red/green)
5.  roc_curve.png                — ROC-AUC curve
6.  alert_timeline.png           — Alert volume over simulated time
7.  severity_breakdown.png       — Bar chart of alert severity
8.  correlation_heatmap.png      — Feature correlation with target

All images saved to outputs/images/ at 150 dpi.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend (safe in scripts)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc

os.makedirs('outputs/images', exist_ok=True)

# ── Style ──────────────────────────────────────────────────────
sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams.update({
    'figure.facecolor':  '#1a1a2e',
    'axes.facecolor':    '#16213e',
    'axes.edgecolor':    '#0f3460',
    'axes.labelcolor':   '#e0e0e0',
    'xtick.color':       '#e0e0e0',
    'ytick.color':       '#e0e0e0',
    'text.color':        '#e0e0e0',
    'grid.color':        '#0f3460',
    'grid.alpha':        0.5,
    'font.family':       'monospace',
})

ATTACK_COLORS = {
    'normal': '#2ecc71',
    'DoS':    '#e74c3c',
    'Probe':  '#3498db',
    'R2L':    '#f39c12',
    'U2R':    '#9b59b6',
    'other':  '#95a5a6',
}

SEV_COLORS = {
    'CRITICAL': '#e74c3c',
    'HIGH':     '#e67e22',
    'MEDIUM':   '#f1c40f',
    'LOW':      '#3498db',
    'NONE':     '#2ecc71',
}


def _save(name: str):
    path = f'outputs/images/{name}'
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=plt.rcParams['figure.facecolor'])
    plt.close()
    print(f"[+] Saved → {path}")


# ─────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray,
                           title: str = 'Random Forest — Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels  = ['Normal', 'Attack']

    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='#0f3460',
                annot_kws={'size': 14, 'weight': 'bold'},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Actual Label',    fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

    # Annotate TP / TN / FP / FN
    for (r, c), val in np.ndenumerate(cm):
        tag = {(0, 0): 'TN', (0, 1): 'FP',
               (1, 0): 'FN', (1, 1): 'TP'}.get((r, c), '')
        ax.text(c + 0.5, r + 0.75, tag,
                ha='center', va='center',
                fontsize=9, color='#aaaaaa')

    _save('confusion_matrix.png')


# ─────────────────────────────────────────────
# 2. Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    top = importance_df.head(top_n).copy()
    top = top.sort_values('importance')            # horizontal bar: bottom = least

    cmap   = plt.cm.get_cmap('YlOrRd', top_n)
    colors = [cmap(i / top_n) for i in range(top_n)]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top['feature'], top['importance'],
                   color=colors, edgecolor='none', height=0.7)

    # Value labels
    for bar, val in zip(bars, top['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8, color='#e0e0e0')

    ax.set_title(f'Top {top_n} Feature Importances — Random Forest',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Gini Importance Score', fontsize=11)
    ax.set_xlim(0, top['importance'].max() * 1.18)

    _save('feature_importance.png')


# ─────────────────────────────────────────────
# 3. Attack Category Distribution
# ─────────────────────────────────────────────

def plot_attack_distribution(df_clean: pd.DataFrame):
    counts = df_clean['attack_category'].value_counts()
    labels = counts.index.tolist()
    colors = [ATTACK_COLORS.get(l, '#95a5a6') for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        wedgeprops={'edgecolor': '#1a1a2e', 'linewidth': 2},
        pctdistance=0.82
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color('white')
    ax1.set_title('Attack Category Distribution', fontsize=13,
                   fontweight='bold', color='#e0e0e0')

    # Bar chart
    bars = ax2.bar(labels, counts.values, color=colors,
                   edgecolor='#1a1a2e', linewidth=1.5, width=0.6)
    ax2.set_title('Connection Count by Category', fontsize=13,
                   fontweight='bold', color='#e0e0e0')
    ax2.set_ylabel('Number of Connections', fontsize=11)
    ax2.set_xlabel('Attack Category', fontsize=11)

    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{val:,}', ha='center', va='bottom',
                 fontsize=9, color='#e0e0e0')

    fig.suptitle('Network Traffic Analysis', fontsize=15,
                  fontweight='bold', color='#e0e0e0', y=1.01)
    _save('attack_distribution.png')


# ─────────────────────────────────────────────
# 4. Anomaly Score Distribution
# ─────────────────────────────────────────────

def plot_anomaly_scores(if_scores: np.ndarray, y_true: np.ndarray,
                         threshold: float = -0.1, sample_n: int = 2000):
    """
    Scatter plot of Isolation Forest anomaly scores.
    Green = normal traffic, Red = attacks.
    Dashed threshold line shows the detection boundary.
    """
    if len(if_scores) > sample_n:
        idx = np.random.choice(len(if_scores), sample_n, replace=False)
        scores = if_scores[idx]
        labels = y_true[idx]
    else:
        scores = if_scores
        labels = y_true

    colors = np.where(labels == 1, '#e74c3c', '#2ecc71')
    alphas = np.where(labels == 1, 0.7, 0.4)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=False)

    # Top: scatter
    for i, (s, c, a) in enumerate(zip(scores, colors, alphas)):
        ax1.scatter(i, s, c=c, alpha=float(a), s=8, linewidths=0)

    ax1.axhline(y=threshold, color='#f1c40f', linestyle='--',
                linewidth=1.5, label=f'Threshold ({threshold})')
    ax1.set_ylabel('Anomaly Score', fontsize=11)
    ax1.set_title('Isolation Forest Anomaly Scores\n'
                   '(Red = Attack  |  Green = Normal)',
                   fontsize=13, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='Attack'),
        mpatches.Patch(color='#2ecc71', label='Normal'),
        mpatches.Patch(color='#f1c40f', label=f'Threshold={threshold}'),
    ]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=9)

    # Bottom: histogram of scores
    attack_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    ax2.hist(normal_scores, bins=50, color='#2ecc71', alpha=0.6,
             label='Normal', density=True)
    ax2.hist(attack_scores, bins=50, color='#e74c3c', alpha=0.6,
             label='Attack', density=True)
    ax2.axvline(x=threshold, color='#f1c40f', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Anomaly Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend(fontsize=9)

    _save('anomaly_scores.png')


# ─────────────────────────────────────────────
# 5. ROC Curve
# ─────────────────────────────────────────────

def plot_roc_curve(rf_model, X_test: np.ndarray, y_test: np.ndarray):
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#3498db', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#7f8c8d', lw=1,
            linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#3498db')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)',       fontsize=11)
    ax.set_title('ROC Curve — Random Forest Classifier',
                  fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)

    _save('roc_curve.png')
    return roc_auc


# ─────────────────────────────────────────────
# 6. Alert Timeline
# ─────────────────────────────────────────────

def plot_alert_timeline(alert_df: pd.DataFrame):
    """
    Simulate a 24-hour alert timeline from the alert log.
    Groups alerts into 30-minute buckets and plots severity stacks.
    """
    if alert_df.empty:
        print("[!] No alerts to plot in timeline.")
        return

    df = alert_df.copy()
    # Use connection_id as a proxy for time if timestamps are identical
    df['bucket'] = (df['connection_id'] // 50).astype(int)

    pivot = (df.groupby(['bucket', 'severity'])
               .size()
               .unstack(fill_value=0))

    # Ensure all severity levels present
    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if sev not in pivot.columns:
            pivot[sev] = 0
    pivot = pivot[['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']]

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(pivot))

    for sev in pivot.columns:
        color = SEV_COLORS[sev]
        ax.bar(pivot.index, pivot[sev], bottom=bottom,
               color=color, label=sev, width=0.8, alpha=0.9)
        bottom += pivot[sev].values

    ax.set_title('Alert Volume Over Detection Window (Stacked by Severity)',
                  fontsize=13, fontweight='bold')
    ax.set_xlabel('Batch Window', fontsize=11)
    ax.set_ylabel('Number of Alerts',  fontsize=11)
    ax.legend(title='Severity', bbox_to_anchor=(1.01, 1), loc='upper left',
               fontsize=9)

    _save('alert_timeline.png')


# ─────────────────────────────────────────────
# 7. Severity Breakdown
# ─────────────────────────────────────────────

def plot_severity_breakdown(alert_df: pd.DataFrame):
    if alert_df.empty:
        print("[!] No alerts to plot.")
        return

    counts = alert_df['severity'].value_counts()
    order  = [s for s in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NONE']
               if s in counts.index]
    counts = counts.reindex(order)
    colors = [SEV_COLORS[s] for s in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=colors, edgecolor='#1a1a2e',
                  linewidth=1.5, width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.values.max() * 0.01,
                f'{val:,}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#e0e0e0')

    ax.set_title('Detected Threats — Severity Breakdown',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xlabel('Severity Level', fontsize=11)

    _save('severity_breakdown.png')


# ─────────────────────────────────────────────
# 8. Correlation Heatmap
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df_clean: pd.DataFrame,
                              feature_cols: list, top_n: int = 12):
    """
    Heatmap of feature correlations with the binary target (is_attack).
    Uses only numeric columns.
    """
    cols = [c for c in feature_cols[:top_n] if c in df_clean.columns]
    cols_with_target = cols + ['is_attack']
    cols_with_target = [c for c in cols_with_target if c in df_clean.columns]

    corr = df_clean[cols_with_target].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)          # hide diagonal

    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', linewidths=0.5,
                annot_kws={'size': 8}, ax=ax,
                linecolor='#1a1a2e')

    ax.set_title('Feature Correlation Matrix\n'
                  '(is_attack = target column)',
                  fontsize=13, fontweight='bold')

    _save('correlation_heatmap.png')


# ─────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────

def generate_all_charts(rf_model, if_model, X_test, y_test,
                         if_scores, importance_df,
                         df_clean, feature_cols, alert_df):
    """Run every chart function in one call from main.py."""
    from sklearn.metrics import confusion_matrix as cm_fn
    y_pred = rf_model.predict(X_test)
    cm = cm_fn(y_test, y_pred)

    print("\n[+] ─── Generating Dashboard Charts ───────────────────")
    plot_confusion_matrix(cm)
    plot_feature_importance(importance_df)
    plot_attack_distribution(df_clean)
    plot_anomaly_scores(if_scores, y_test)
    plot_roc_curve(rf_model, X_test, y_test)
    plot_alert_timeline(alert_df)
    plot_severity_breakdown(alert_df)
    plot_correlation_heatmap(df_clean, feature_cols)
    print("[+] All 8 charts saved to outputs/images/")
