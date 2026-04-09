"""
detector.py
===========
The real-time threat detection pipeline.

ThreatDetector takes a batch of raw (unscaled) network connections,
runs them through both models, assigns severity levels, and writes
every detected threat to an append-only CSV alert log.

Severity mapping (mirrors CVSS severity bands):
  CRITICAL  → DoS (service disruption)
  HIGH      → U2R (privilege escalation)
  MEDIUM    → R2L (unauthorized access)
  LOW       → Probe (reconnaissance)
  NONE      → Normal traffic
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

ALERT_LOG_PATH = 'outputs/alerts_log.csv'
ALERT_COLUMNS  = [
    'timestamp', 'connection_id',
    'if_score', 'if_flag',
    'rf_prediction', 'rf_confidence',
    'severity', 'alert_message'
]

SEVERITY = {
    'DoS':    'CRITICAL',
    'U2R':    'HIGH',
    'R2L':    'MEDIUM',
    'Probe':  'LOW',
    'normal': 'NONE',
    'other':  'LOW',
}

ALERT_MSG = {
    'CRITICAL': '🔴 CRITICAL — Possible Denial-of-Service attack detected',
    'HIGH':     '🟠 HIGH     — Privilege escalation attempt detected',
    'MEDIUM':   '🟡 MEDIUM   — Unauthorized remote-access attempt detected',
    'LOW':      '🔵 LOW      — Network reconnaissance / port scan detected',
    'NONE':     '🟢 NONE     — Normal traffic',
}


class ThreatDetector:
    """
    End-to-end threat detection pipeline.

    Parameters
    ----------
    isolation_forest  : trained IsolationForest
    random_forest     : trained RandomForestClassifier
    scaler            : fitted StandardScaler from preprocessor
    log_path          : path to CSV alert log file
    """

    def __init__(self, isolation_forest, random_forest, scaler,
                 log_path: str = ALERT_LOG_PATH):
        self.if_model  = isolation_forest
        self.rf_model  = random_forest
        self.scaler    = scaler
        self.log_path  = log_path
        self.alert_count = 0

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Create / overwrite alert log with header
        pd.DataFrame(columns=ALERT_COLUMNS).to_csv(log_path, index=False)
        print(f"[+] Alert log initialised → {log_path}")

    # ─────────────────────────────────────────
    # Main detection method
    # ─────────────────────────────────────────

    def detect(self,
               X_raw: np.ndarray,
               connection_ids=None,
               verbose: bool = True) -> pd.DataFrame:
        """
        Run the full detection pipeline on a batch of connections.

        Parameters
        ----------
        X_raw          : shape (n, n_features) UNSCALED feature matrix
        connection_ids : optional list of IDs for logging
        verbose        : print alert summary to console

        Returns
        -------
        pd.DataFrame with per-connection predictions and scores
        """
        n = len(X_raw)
        ids = (list(connection_ids) if connection_ids is not None
               else list(range(self.alert_count, self.alert_count + n)))

        # ── Normalize ────────────────────────────────────────────
        X = self.scaler.transform(X_raw)

        # ── Isolation Forest ─────────────────────────────────────
        if_raw_pred = self.if_model.predict(X)        # 1 or -1
        if_flag     = (if_raw_pred == -1).astype(int) # 1 = anomaly
        if_scores   = self.if_model.score_samples(X)

        # ── Random Forest ────────────────────────────────────────
        rf_pred     = self.rf_model.predict(X)        # 0 or 1
        rf_proba    = self.rf_model.predict_proba(X)[:, 1]  # P(attack)

        # ── Build result frame ───────────────────────────────────
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results = pd.DataFrame({
            'timestamp':     ts,
            'connection_id': ids,
            'if_score':      if_scores.round(4),
            'if_flag':       if_flag,
            'rf_prediction': rf_pred,
            'rf_confidence': rf_proba.round(4),
        })

        # ── Assign severity (RF drives the decision) ─────────────
        # RF predicts 0/1 — we map 1 → HIGH by default;
        # caller can pass a category override for richer display
        results['severity'] = results['rf_prediction'].apply(
            lambda x: 'HIGH' if x == 1 else 'NONE'
        )

        # Double-flag with IF: if IF also flags → upgrade to CRITICAL
        results.loc[
            (results['rf_prediction'] == 1) & (results['if_flag'] == 1),
            'severity'
        ] = 'CRITICAL'

        results['alert_message'] = results['severity'].map(ALERT_MSG)

        # ── Log only threats ─────────────────────────────────────
        threats = results[results['rf_prediction'] == 1]
        if len(threats):
            self._log_alerts(threats)
            self.alert_count += len(threats)

        # ── Console summary ──────────────────────────────────────
        if verbose:
            n_threat  = (results['rf_prediction'] == 1).sum()
            n_normal  = n - n_threat
            n_crit    = (results['severity'] == 'CRITICAL').sum()
            print(f"\n  ┌── Detection Results ({n} connections) ────────────")
            print(f"  │  Normal    : {n_normal:>5,}")
            print(f"  │  Threats   : {n_threat:>5,}")
            print(f"  │  CRITICAL  : {n_crit:>5,}  (flagged by BOTH models)")
            print(f"  │  Avg IF score (threats): "
                  f"{threats['if_score'].mean():.4f}" if len(threats) else
                  f"  │  Avg IF score (threats): N/A")
            print(f"  └────────────────────────────────────────────────")

        return results

    # ─────────────────────────────────────────
    # Simulate real-time stream
    # ─────────────────────────────────────────

    def stream_detect(self,
                      X_raw: np.ndarray,
                      batch_size: int = 100,
                      verbose: bool = True) -> pd.DataFrame:
        """
        Simulate a real-time detection stream by processing X_raw
        in sequential batches of `batch_size` connections.

        This mirrors how a production IDS processes live packet captures:
        connections arrive continuously and are scored in micro-batches.

        Returns
        -------
        Combined pd.DataFrame of all batch results
        """
        print(f"\n[+] ─── Live Stream Simulation ─────────────────────")
        print(f"    Total: {len(X_raw):,} connections  |  "
              f"Batch size: {batch_size}")
        print(f"    {'─'*45}")

        all_results = []
        n_batches = (len(X_raw) + batch_size - 1) // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end   = min(start + batch_size, len(X_raw))
            batch = X_raw[start:end]

            print(f"  Batch {i+1:>3}/{n_batches}  "
                  f"[conn {start:>6,}–{end-1:>6,}]", end='  ')

            result = self.detect(
                batch,
                connection_ids=range(start, end),
                verbose=False  # suppress per-batch verbose
            )
            all_results.append(result)

            n_t = (result['rf_prediction'] == 1).sum()
            bar = '█' * min(n_t, 30)
            print(f"threats={n_t:>4}  {bar}")

        combined = pd.concat(all_results, ignore_index=True)
        total_threats = (combined['rf_prediction'] == 1).sum()
        total_critical = (combined['severity'] == 'CRITICAL').sum()

        print(f"\n  ┌── Stream Complete ──────────────────────────────")
        print(f"  │  Connections processed : {len(combined):>7,}")
        print(f"  │  Total threats detected: {total_threats:>7,}  "
              f"({total_threats/len(combined)*100:.1f}%)")
        print(f"  │  CRITICAL alerts       : {total_critical:>7,}")
        print(f"  │  Alert log             : {self.log_path}")
        print(f"  └────────────────────────────────────────────────")

        return combined

    # ─────────────────────────────────────────
    # Alert logging
    # ─────────────────────────────────────────

    def _log_alerts(self, threats: pd.DataFrame):
        """Append threat rows to the CSV alert log."""
        log_df = threats[ALERT_COLUMNS].copy()
        log_df.to_csv(self.log_path, mode='a', header=False, index=False)

    def get_alert_summary(self) -> pd.DataFrame:
        """Load and return the current alert log as a DataFrame."""
        if os.path.exists(self.log_path):
            df = pd.read_csv(self.log_path)
            print(f"\n[+] Alert Log Summary ({len(df)} total alerts):")
            if len(df):
                print(df['severity'].value_counts().to_string())
            return df
        print("[!] Alert log not found.")
        return pd.DataFrame()
