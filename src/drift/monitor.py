"""
Evidently-based drift monitor using PSI (Population Stability Index).

Monitors feature distributions batch-by-batch against a reference dataset.
Flags drift when PSI > psi_threshold on any monitored feature.

PSI interpretation:
    < 0.1   No significant change
    0.1-0.2 Moderate change - monitor closely
    > 0.2   Significant change - drift alert

Uses Evidently 0.4.x ColumnDriftMetric with PSI stattest per feature.
For the drift demo, reference data is the validation set.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

PSI_THRESHOLD = 0.2


class DriftMonitor:
    """
    Per-batch drift monitor.

    Args:
        reference_data: Baseline distribution (e.g. validation set).
        features:       Feature names to monitor.
        psi_threshold:  PSI threshold above which drift is flagged.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        features: list[str],
        psi_threshold: float = PSI_THRESHOLD,
    ):
        self.reference = reference_data[features].copy()
        self.features = features
        self.psi_threshold = psi_threshold

    def check_batch(self, current_batch: pd.DataFrame) -> dict:
        """
        Run PSI drift detection on one batch vs. the reference.

        Returns:
            {
                "drift_detected":   bool,
                "drifted_features": list[str],
                "psi_scores":       {feature: float},
                "n_drifted":        int,
            }
        """
        from evidently.metrics import ColumnDriftMetric
        from evidently.report import Report

        current = current_batch[self.features].copy()
        metrics = [
            ColumnDriftMetric(
                column_name=f,
                stattest="psi",
                stattest_threshold=self.psi_threshold,
            )
            for f in self.features
        ]

        try:
            report = Report(metrics=metrics)
            report.run(reference_data=self.reference, current_data=current)
            result = report.as_dict()
        except Exception as e:
            log.warning(f"Evidently error: {e}")
            return {
                "drift_detected": False,
                "drifted_features": [],
                "psi_scores": {f: float("nan") for f in self.features},
                "n_drifted": 0,
            }

        psi_scores = {}
        drifted = []
        for entry in result["metrics"]:
            r = entry["result"]
            col = r["column_name"]
            score = float(r.get("drift_score", 0.0))
            detected = bool(r.get("drift_detected", False))
            psi_scores[col] = round(score, 4)
            if detected:
                drifted.append(col)

        return {
            "drift_detected": len(drifted) > 0,
            "drifted_features": drifted,
            "psi_scores": psi_scores,
            "n_drifted": len(drifted),
        }

    def check_stream(self, stream: pd.DataFrame) -> list[dict]:
        """
        Run batch-by-batch drift detection over the full stream.

        Args:
            stream: DataFrame with 'batch_id' and 'is_post_drift' columns.

        Returns:
            List of dicts (one per batch) with check_batch() results
            plus 'batch_id' and 'is_post_drift'.
        """
        results = []
        for batch_id in sorted(stream["batch_id"].unique()):
            batch = stream[stream["batch_id"] == batch_id]
            is_post = bool(batch["is_post_drift"].iloc[0])
            log.info(f"Checking batch {batch_id} (post_drift={is_post}) ...")
            report = self.check_batch(batch)
            report["batch_id"] = int(batch_id)
            report["is_post_drift"] = is_post
            results.append(report)
        return results
