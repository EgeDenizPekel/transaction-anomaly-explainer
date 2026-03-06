"""
Model performance and drift monitoring endpoints.

GET /metrics      - Current model metrics + aggregate scoring stats.
GET /drift-status - Latest PSI values per feature + drift flag.
"""

from fastapi import APIRouter, HTTPException, Request

from src.api.model_registry import registry
from src.api.schemas import (
    BatchMetric, CalibrationBin, CalibrationResponse, DriftEvent,
    DriftStatusResponse, ExplanationDriftEvent, MetricsResponse,
)

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request):
    """
    Return current model metadata and aggregate scoring statistics
    since the API started.
    """
    state = request.app.state
    n_scored = state.n_scored
    n_flagged = state.n_flagged
    flag_rate = round(n_flagged / n_scored, 4) if n_scored > 0 else 0.0

    cal = getattr(state, "calibration", None)
    return MetricsResponse(
        model_version=registry.version,
        val_roc_auc=registry.val_roc_auc,
        val_brier_score=round(cal["brier_score"], 4) if cal else None,
        n_transactions_scored=n_scored,
        n_flagged=n_flagged,
        flag_rate=flag_rate,
        drift_detected=state.drift_state.get("drift_detected", False),
    )


@router.get("/drift-status", response_model=DriftStatusResponse)
def get_drift_status(request: Request):
    """
    Return the latest drift monitoring result.

    PSI is computed over the last 1,000 scored transactions vs. the validation
    set reference distribution. Refreshed every DRIFT_CHECK_INTERVAL transactions.
    PSI > 0.2 on any monitored feature triggers drift_detected=True.
    """
    state = request.app.state
    ds = state.drift_state

    return DriftStatusResponse(
        drift_detected=ds.get("drift_detected", False),
        drifted_features=ds.get("drifted_features", []),
        psi_scores=ds.get("psi_scores", {}),
        last_checked=ds.get("last_checked"),
        n_transactions_since_last_check=state.n_since_drift_check,
    )


@router.get("/batch-metrics", response_model=list[BatchMetric])
def get_batch_metrics(request: Request):
    """
    Return per-batch F1 and fraud rate from the stream seeder.
    Ordered by batch_id ascending for charting.
    """
    history = list(request.app.state.batch_metrics_history)
    history.sort(key=lambda x: x["batch_id"])
    return [BatchMetric(**b) for b in history]


@router.get("/drift-history", response_model=list[DriftEvent])
def get_drift_history(request: Request):
    """
    Return per-batch drift events from the stream seeder.
    Ordered by batch_id ascending.
    """
    history = list(request.app.state.drift_history)
    history.sort(key=lambda x: x["batch_id"])
    return [DriftEvent(**e) for e in history]


@router.get("/calibration", response_model=CalibrationResponse)
def get_calibration(request: Request):
    """
    Return model probability calibration metrics computed from the validation set.

    Brier score measures mean squared error of predicted probabilities.
    baseline_brier is the naive 'always predict class prior' score.
    A model with Brier score below baseline_brier is useful on calibration.
    """
    cal = getattr(request.app.state, "calibration", None)
    if cal is None:
        raise HTTPException(
            status_code=503,
            detail="Calibration data not available (val parquet may not have isFraud column).",
        )
    return CalibrationResponse(
        brier_score=cal["brier_score"],
        baseline_brier=cal["baseline_brier"],
        n_val_samples=cal["n_val_samples"],
        reliability_diagram=[CalibrationBin(**b) for b in cal["reliability_diagram"]],
    )


@router.get("/explanation-drift", response_model=list[ExplanationDriftEvent])
def get_explanation_drift(request: Request):
    """
    Return per-batch top SHAP feature frequency counts.

    Tracks which features dominate the model's explanations (top-3 SHAP) per
    batch. A shift in dominant features post-drift indicates that the explanation
    layer is tracking a different causal structure than pre-drift.
    """
    history = list(request.app.state.batch_metrics_history)
    history.sort(key=lambda x: x["batch_id"])
    return [
        ExplanationDriftEvent(
            batch_id=b["batch_id"],
            is_post_drift=b["is_post_drift"],
            top_feature_counts=b.get("top_feature_counts", {}),
            n_flagged=b.get("n_flagged_batch", 0),
            recorded_at=b["recorded_at"],
        )
        for b in history
    ]
