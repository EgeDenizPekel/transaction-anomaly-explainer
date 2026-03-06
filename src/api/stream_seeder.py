"""
Background stream seeder for demo purposes.

Replays data/streaming/simulated_stream.parquet at ~2 tx/s (configurable via
SEEDER_TX_INTERVAL env var), looping indefinitely.

For each transaction it scores directly against the model (no HTTP round-trip)
and pushes records into app.state.recent_transactions so the dashboard
TransactionFeed stays live.

After each complete batch (1000 transactions) it:
  - Computes F1 (predicted vs isFraud ground truth)
  - Runs a drift check on the batch's feature rows
  - Appends a BatchMetric to app.state.batch_metrics_history
  - Appends a DriftEvent to app.state.drift_history

This is synthetic concept drift demo data. See README for details.
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.model_registry import registry
from src.models.shap_utils import compute_shap_values, top_k_features

log = logging.getLogger(__name__)

STREAM_PATH = ROOT / "data" / "streaming" / "simulated_stream.parquet"
TX_INTERVAL_S = float(os.environ.get("SEEDER_TX_INTERVAL", "0.1"))

MISSING = -999.0


def _alert_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def _score_row(row: pd.Series, feature_cols: list[str]) -> tuple[float, bool, pd.DataFrame]:
    """Score a single stream row. Returns (anomaly_score, is_flagged, feature_df)."""
    feat = {col: MISSING for col in feature_cols}
    for col in feature_cols:
        if col in row.index and pd.notna(row[col]):
            feat[col] = float(row[col])
    X = pd.DataFrame([feat])
    score = float(registry.model.predict_proba(X)[0, 1])
    return score, score >= registry.threshold, X


def _run_seeder(app) -> None:
    """Main seeder loop. Runs in a daemon thread."""
    if not STREAM_PATH.exists():
        log.warning(f"Stream parquet not found at {STREAM_PATH} - seeder disabled")
        return

    # Wait until feature_cols are set (startup may not be done yet)
    for _ in range(20):
        if registry.feature_cols:
            break
        time.sleep(1)
    if not registry.feature_cols:
        log.warning("Seeder: registry.feature_cols never set - seeder disabled")
        return

    stream_df = pd.read_parquet(STREAM_PATH)
    log.info(f"Stream seeder loaded {len(stream_df)} rows, TX_INTERVAL={TX_INTERVAL_S}s")

    loop = 0
    while True:
        loop += 1
        log.info(f"Stream seeder loop {loop}")

        for batch_id in sorted(stream_df["batch_id"].unique()):
            batch = stream_df[stream_df["batch_id"] == batch_id].reset_index(drop=True)
            is_post_drift = bool(batch["is_post_drift"].iloc[0])

            feature_cols = registry.feature_cols
            y_true: list[int] = []
            y_pred: list[int] = []
            feature_rows: list[dict] = []

            for idx, row in batch.iterrows():
                try:
                    score, is_flagged, X = _score_row(row, feature_cols)
                    alert = _alert_level(score)

                    y_true.append(int(row.get("isFraud", 0)))
                    y_pred.append(1 if is_flagged else 0)

                    feat_row = {col: float(X[col].iloc[0]) for col in feature_cols}
                    feature_rows.append(feat_row)

                    # Compute SHAP for flagged transactions (only ~2-3% of stream)
                    top_feats = []
                    if is_flagged:
                        try:
                            shap_vals = compute_shap_values(registry.explainer, X)[0]
                            top_feats = top_k_features(shap_vals, feature_cols, X.values[0], k=3)
                        except Exception as e:
                            log.warning(f"Seeder SHAP error row {idx}: {e}")

                    app.state.recent_transactions.appendleft({
                        "transaction_id": f"stream_b{batch_id}_{idx}",
                        "anomaly_score": round(score, 4),
                        "is_flagged": is_flagged,
                        "alert_level": alert,
                        "transaction_amt": float(row["TransactionAmt"]) if "TransactionAmt" in row.index and pd.notna(row["TransactionAmt"]) else None,
                        "timestamp_dt": float(row["TransactionDT"]) if "TransactionDT" in row.index and pd.notna(row["TransactionDT"]) else None,
                        "top_features": top_feats,
                        "explanation": None,
                    })
                    app.state.n_scored += 1
                    if is_flagged:
                        app.state.n_flagged += 1

                except Exception as e:
                    log.warning(f"Seeder scoring error row {idx}: {e}")

                time.sleep(TX_INTERVAL_S)

            # Batch F1
            if y_true:
                from sklearn.metrics import f1_score
                batch_f1 = float(f1_score(y_true, y_pred, zero_division=0))
                fraud_rate = float(np.mean(y_true))
            else:
                batch_f1 = 0.0
                fraud_rate = 0.0

            # Batch drift check
            drift_result: dict = {"drift_detected": False, "drifted_features": [], "psi_scores": {}}
            monitor = getattr(app.state, "drift_monitor", None)
            if monitor is not None and feature_rows:
                try:
                    batch_df = pd.DataFrame(feature_rows)
                    drift_result = monitor.check_batch(batch_df)
                    if drift_result.get("drift_detected"):
                        app.state.drift_state.update({
                            "drift_detected": drift_result["drift_detected"],
                            "drifted_features": drift_result["drifted_features"],
                            "psi_scores": drift_result["psi_scores"],
                            "last_checked": datetime.now(timezone.utc).isoformat(),
                        })
                    app.state.drift_history.append({
                        "batch_id": int(batch_id),
                        "is_post_drift": is_post_drift,
                        "drift_detected": drift_result.get("drift_detected", False),
                        "drifted_features": drift_result.get("drifted_features", []),
                        "psi_scores": drift_result.get("psi_scores", {}),
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception as e:
                    log.warning(f"Seeder drift check error batch {batch_id}: {e}")

            app.state.batch_metrics_history.append({
                "batch_id": int(batch_id),
                "is_post_drift": is_post_drift,
                "f1": round(batch_f1, 4),
                "fraud_rate": round(fraud_rate, 4),
                "n_transactions": len(batch),
                "drift_detected": drift_result.get("drift_detected", False),
                "drifted_features": drift_result.get("drifted_features", []),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            log.info(
                f"Batch {batch_id} done: F1={batch_f1:.3f} fraud_rate={fraud_rate:.3f} "
                f"post_drift={is_post_drift} drift={drift_result.get('drift_detected', False)}"
            )


def start_seeder(app) -> threading.Thread:
    """Start the stream seeder as a daemon thread. Returns the thread."""
    t = threading.Thread(target=_run_seeder, args=(app,), daemon=True, name="stream-seeder")
    t.start()
    log.info("Stream seeder thread started")
    return t
