"""
alert_generator.py
==================
Standalone alert-generation utilities.

This module is separate from detector.py so it can be used independently
(e.g., in notebooks or by downstream reporting tools) without needing
the full ThreatDetector class loaded.

Functions
---------
generate_alert_report()  — pretty-print + CSV summary of logged alerts
assign_severity()        — category → severity string mapping
format_alert_line()      — format one alert for console display
print_live_alert()       — coloured console alert for a single connection
"""

import os
import pandas as pd
from datetime import datetime


# ANSI colour codes for terminal output
class C:
    RED    = '\033[91m'
    ORANGE = '\033[33m'
    YELLOW = '\033[93m'
    BLUE   = '\033[94m'
    GREEN  = '\033[92m'
    BOLD   = '\033[1m'
    RESET  = '\033[0m'


SEVERITY_COLOUR = {
    'CRITICAL': C.RED,
    'HIGH':     C.ORANGE,
    'MEDIUM':   C.YELLOW,
    'LOW':      C.BLUE,
    'NONE':     C.GREEN,
}

CATEGORY_TO_SEVERITY = {
    'DoS':    'CRITICAL',
    'U2R':    'HIGH',
    'R2L':    'MEDIUM',
    'Probe':  'LOW',
    'normal': 'NONE',
}


def assign_severity(attack_category: str) -> str:
    """Map attack category to severity string."""
    return CATEGORY_TO_SEVERITY.get(attack_category, 'LOW')


def format_alert_line(connection_id, severity, if_score, confidence,
                       timestamp=None) -> str:
    """Return a single formatted alert line."""
    ts  = timestamp or datetime.now().strftime('%H:%M:%S')
    col = SEVERITY_COLOUR.get(severity, C.RESET)
    return (f"{col}[{ts}] [{severity:<8}] "
            f"Conn #{connection_id:<6}  "
            f"IF_score={if_score:+.3f}  "
            f"Confidence={confidence:.2%}{C.RESET}")


def print_live_alert(connection_id, severity, if_score,
                     confidence, attack_category='unknown'):
    """Print a single live alert to the console (colour-coded)."""
    line = format_alert_line(connection_id, severity, if_score, confidence)
    detail = f"  → Attack type: {attack_category}"
    print(line)
    if severity != 'NONE':
        print(f"{SEVERITY_COLOUR.get(severity,'')}{detail}{C.RESET}")


def generate_alert_report(log_path: str = 'outputs/alerts_log.csv',
                            output_path: str = 'outputs/alert_report.txt'):
    """
    Read the alert CSV log and produce:
      1. A printed summary table in the console
      2. A plain-text report file saved to disk

    Returns
    -------
    pd.DataFrame  — the full alert log
    """
    if not os.path.exists(log_path):
        print(f"[!] Alert log not found: {log_path}")
        return pd.DataFrame()

    df = pd.read_csv(log_path)
    if df.empty:
        print("[!] Alert log is empty — no threats were detected.")
        return df

    total         = len(df)
    by_severity   = df['severity'].value_counts()
    by_hour       = (pd.to_datetime(df['timestamp'])
                     .dt.hour.value_counts()
                     .sort_index())
    avg_if        = df['if_score'].mean()
    avg_conf      = df['rf_confidence'].mean()

    report_lines = [
        "=" * 60,
        "  AI Cybersecurity Threat Detection — Alert Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"  Total alerts logged  : {total:,}",
        f"  Avg anomaly score    : {avg_if:.4f}",
        f"  Avg RF confidence   : {avg_conf:.2%}",
        "",
        "  Alerts by severity:",
    ]
    for sev, count in by_severity.items():
        bar = '█' * min(count // max(total // 30, 1), 30)
        pct = count / total * 100
        report_lines.append(f"    {sev:<10} {count:>6,}  ({pct:5.1f}%)  {bar}")

    report_lines += [
        "",
        "  Alerts by hour:",
    ]
    for hr, count in by_hour.items():
        report_lines.append(f"    {hr:02d}:00   {count:>6,} alerts")

    report_lines += [
        "",
        "  Top 5 most suspicious connections (by IF score):",
    ]
    top5 = df.nsmallest(5, 'if_score')[
        ['connection_id', 'severity', 'if_score', 'rf_confidence', 'timestamp']
    ]
    report_lines.append(top5.to_string(index=False))

    report_text = '\n'.join(report_lines)

    # Print to console
    print('\n' + report_text)

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[+] Report saved → {output_path}")

    return df
