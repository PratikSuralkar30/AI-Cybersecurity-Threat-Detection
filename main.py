"""
main.py
=======
AI-Powered Cybersecurity Threat Detection System
─────────────────────────────────────────────────
Entry point.  Runs the full pipeline end-to-end:

  1.  Load / simulate dataset
  2.  Preprocess & engineer features
  3.  Train Isolation Forest  (unsupervised)
  4.  Train Random Forest     (supervised)
  5.  Evaluate both models
  6.  Run threat detection + alert generation
  7.  Simulate real-time streaming detection
  8.  Generate all 8 dashboard charts
  9.  Print alert report

Usage
─────
  python main.py                   # full run (synthetic data)
  python main.py --sample 20000    # quick run with 20k rows
  python main.py --data data/kddcup.data.gz   # real KDD dataset
"""

import sys
import os
import argparse
import time
import numpy as np

# Fix Unicode printing on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Add src/ to path so imports work from project root ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader      import load_dataset, generate_simulation_dataset
from preprocessor     import preprocess
from feature_engineer import get_rf_feature_importance
from model_trainer    import (train_isolation_forest, train_random_forest,
                               evaluate_random_forest, evaluate_isolation_forest)
from detector         import ThreatDetector
from alert_generator  import generate_alert_report
from visualizer       import generate_all_charts


# ─────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='AI-Powered Cybersecurity Threat Detection System')
    p.add_argument('--data',   default=None,
                   help='Path to KDD Cup .gz file (omit = synthetic sim)')
    p.add_argument('--sample', type=int, default=50_000,
                   help='Number of connections to use (default 50000)')
    p.add_argument('--stream-batch', type=int, default=100,
                   help='Batch size for streaming simulation (default 100)')
    p.add_argument('--no-charts', action='store_true',
                   help='Skip chart generation (faster)')
    return p.parse_args()


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def run_pipeline(args):
    t_start = time.time()

    _banner()

    # ── 1. Load data ─────────────────────────────────────────────
    print("\n[STEP 1/8] Data Loading")
    print("─" * 55)
    if args.data and os.path.exists(args.data):
        df = load_dataset(args.data)
    else:
        df = generate_simulation_dataset(n_samples=args.sample,
                                          attack_ratio=0.60)

    # ── 2. Preprocess ─────────────────────────────────────────────
    print("\n[STEP 2/8] Preprocessing & Feature Engineering")
    print("─" * 55)
    (X_train, X_test, y_train, y_test,
     scaler, feature_cols, df_clean, encoders) = preprocess(
         df, sample_size=args.sample)

    # ── 3. Train Isolation Forest ─────────────────────────────────
    print("\n[STEP 3/8] Training Isolation Forest (unsupervised)")
    print("─" * 55)
    attack_ratio = y_train.mean()
    if_model = train_isolation_forest(
        X_train,
        contamination=min(attack_ratio, 0.499),   # must be < 0.5
        n_estimators=100
    )

    # ── 4. Train Random Forest ────────────────────────────────────
    print("\n[STEP 4/8] Training Random Forest (supervised)")
    print("─" * 55)
    rf_model = train_random_forest(X_train, y_train, n_estimators=150)

    # ── 5. Evaluate ───────────────────────────────────────────────
    print("\n[STEP 5/8] Model Evaluation")
    print("─" * 55)
    y_pred_rf, cm_rf, metrics_rf = evaluate_random_forest(
        rf_model, X_test, y_test)

    y_pred_if, if_scores_test, cm_if = evaluate_isolation_forest(
        if_model, X_test, y_test)

    # ── 6. Feature importance ─────────────────────────────────────
    importance_df = get_rf_feature_importance(rf_model, feature_cols)

    # ── 7. Threat Detection — batch mode ──────────────────────────
    print("\n[STEP 6/8] Running Threat Detection (batch mode)")
    print("─" * 55)
    detector = ThreatDetector(if_model, rf_model, scaler)

    # Inverse-transform the scaled test set back to raw values
    X_test_raw = scaler.inverse_transform(X_test)

    # Detect on up to first 1000 connections
    batch_x = X_test_raw[:1000]
    batch_results = detector.detect(
        batch_x,
        connection_ids=range(len(batch_x)),
        verbose=True
    )

    # ── 8. Streaming simulation ───────────────────────────────────
    print("\n[STEP 7/8] Live Stream Simulation")
    print("─" * 55)
    stream_results = detector.stream_detect(
        X_test_raw[:500],
        batch_size=args.stream_batch
    )

    # Get alert log for charts
    alert_df = detector.get_alert_summary()

    # ── 9. Charts ─────────────────────────────────────────────────
    if not args.no_charts:
        print("\n[STEP 8/8] Generating Dashboard Charts")
        print("─" * 55)
        generate_all_charts(
            rf_model=rf_model,
            if_model=if_model,
            X_test=X_test,
            y_test=y_test,
            if_scores=if_scores_test,
            importance_df=importance_df,
            df_clean=df_clean,
            feature_cols=feature_cols,
            alert_df=alert_df
        )
    else:
        print("\n[STEP 8/8] Chart generation skipped (--no-charts)")

    # ── 10. Alert report ─────────────────────────────────────────
    generate_alert_report()

    # ── Final summary ─────────────────────────────────────────────
    elapsed = time.time() - t_start
    _print_final_summary(metrics_rf, detector.alert_count, elapsed,
                          not args.no_charts)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║    AI-Powered Cybersecurity Threat Detection System      ║
║    ─────────────────────────────────────────────────     ║
║    Models  : Isolation Forest + Random Forest            ║
║    Dataset : KDD Cup 1999 / Synthetic Simulation         ║
║    Author  : Your Name                                   ║
╚══════════════════════════════════════════════════════════╝
""")


def _print_final_summary(metrics: dict, alert_count: int,
                          elapsed: float, charts: bool):
    print("""
╔══════════════════════════════════════════════════════════╗
║                  FINAL SUMMARY                           ║
╠══════════════════════════════════════════════════════════╣""")
    print(f"║  RF Accuracy    : {metrics['accuracy']*100:6.2f}%                           ║")
    print(f"║  RF Precision   : {metrics['precision']*100:6.2f}%                           ║")
    print(f"║  RF Recall      : {metrics['recall']*100:6.2f}%                           ║")
    print(f"║  RF F1 Score    : {metrics['f1']*100:6.2f}%                           ║")
    print(f"║  ROC-AUC        : {metrics['roc_auc']:.4f}                              ║")
    print(f"║  Total Alerts   : {alert_count:<6,}                              ║")
    print(f"║  Charts saved   : {'YES' if charts else 'NO':<6}  (outputs/images/)           ║")
    print(f"║  Alert log      : outputs/alerts_log.csv                  ║")
    print(f"║  Alert report   : outputs/alert_report.txt                ║")
    print(f"║  Runtime        : {elapsed:.1f}s                                  ║")
    print("""╚══════════════════════════════════════════════════════════╝

  GitHub proof checklist:
  ✓ outputs/images/confusion_matrix.png
  ✓ outputs/images/feature_importance.png
  ✓ outputs/images/attack_distribution.png
  ✓ outputs/images/anomaly_scores.png
  ✓ outputs/images/roc_curve.png
  ✓ outputs/images/alert_timeline.png
  ✓ outputs/images/severity_breakdown.png
  ✓ outputs/images/correlation_heatmap.png
  ✓ outputs/alerts_log.csv
  ✓ outputs/alert_report.txt
  ✓ models/isolation_forest.pkl
  ✓ models/random_forest.pkl
""")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    run_pipeline(parse_args())
