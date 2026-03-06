"""
Model performance and drift monitoring endpoints.

GET /metrics      - Current model metrics + aggregate scoring stats.
GET /drift-status - Latest PSI values per feature + drift flag.
"""

from fastapi import APIRouter, Request

from src.api.model_registry import registry
from src.api.schemas import BatchMetric, DriftEvent, DriftStatusResponse, MetricsResponse

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

    return MetricsResponse(
        model_version=registry.version,
        val_roc_auc=registry.val_roc_auc,
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
