"""
Admin endpoints.

POST /retrain - Trigger model retraining in the background using the
               simulated post-drift stream data.

Design: Uses FastAPI BackgroundTasks (not Celery). The retrain job
runs in the same process as the API. Status is tracked via an in-memory
dict. The API continues serving requests during retraining.

Demo note: In this implementation, retraining uses the simulated stream
(data/streaming/simulated_stream.parquet, post-drift portion) as new data.
In a production system, new data would come from analyst-reviewed transactions
with confirmed labels.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.api.model_registry import registry
from src.api.schemas import RetrainResponse
from src.drift.retrain_trigger import get_status, retrain

log = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])

STREAM_PATH = ROOT / "data" / "streaming" / "simulated_stream.parquet"
VAL_PATH = ROOT / "data" / "processed" / "features_val.parquet"

# In-memory retrain status (separate from retrain_trigger._status so the
# API controls the lifecycle)
_retrain_status: dict = {"running": False}


@router.post("/retrain", response_model=RetrainResponse)
def trigger_retrain(background_tasks: BackgroundTasks, request: Request):
    """
    Trigger a background retraining job.

    Uses the post-drift portion of the simulated stream as new labeled data,
    combined with a 50K subsample of the original training set. If the
    retrained model's val AUC does not degrade by more than 0.005, it is
    promoted and the serving model is hot-swapped without restarting the API.

    Returns immediately. Poll GET /metrics or check logs for retrain completion.
    """
    if _retrain_status["running"]:
        return RetrainResponse(
            status="already_running",
            message="A retrain job is already in progress.",
        )

    if not STREAM_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Stream file not found: {STREAM_PATH}. Run notebooks/04_drift_analysis.ipynb first.",
        )

    background_tasks.add_task(_retrain_background, request.app.state)
    return RetrainResponse(
        status="started",
        message="Retraining started in background. Check logs for progress.",
    )


@router.get("/retrain/status")
def retrain_status():
    """Return the current retrain job status."""
    return _retrain_status.copy()


def _retrain_background(app_state) -> None:
    """Background task: retrain, evaluate, hot-swap if promoted."""
    global _retrain_status
    _retrain_status["running"] = True
    _retrain_status["error"] = None

    try:
        log.info("Loading post-drift stream for retraining ...")
        stream = pd.read_parquet(STREAM_PATH)
        new_data = stream[stream["is_post_drift"]].copy()
        val_df = pd.read_parquet(VAL_PATH)

        log.info(f"Post-drift rows: {len(new_data):,} | fraud rate: {new_data['isFraud'].mean():.3f}")
        log.info("Starting retrain ...")

        result = retrain(new_data=new_data, val_df=val_df)

        _retrain_status.update({
            "running": False,
            "auc_before": result["auc_before"],
            "auc_after": result["auc_after"],
            "auc_delta": result["auc_delta"],
            "promoted": result["promoted"],
            "new_version": result["new_version"],
        })

        if result["promoted"]:
            registry.hot_swap(
                new_model=result["model"],
                new_threshold=result["threshold"],
                new_version=str(result["new_version"]),
            )
            log.info(f"Model hot-swapped to version {result['new_version']}")
        else:
            log.info("Retrain complete - model not promoted (AUC delta below threshold)")

    except Exception as e:
        log.error(f"Retrain failed: {e}", exc_info=True)
        _retrain_status.update({"running": False, "error": str(e)})
