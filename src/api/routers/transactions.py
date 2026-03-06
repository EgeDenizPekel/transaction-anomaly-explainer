"""
Transaction scoring endpoints.

POST /score   - Score a single transaction, compute SHAP, optionally generate LLM explanation.
GET  /transactions - Return recent scored transactions (most recent first).
"""

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.api.feature_store import feature_store
from src.api.model_registry import registry
from src.api.schemas import ExplainRequest, ExplainResponse, ScoreResponse, TopFeature, TransactionRecord, TransactionRequest
from src.explainability.llm_explainer import generate_explanation
from src.models.shap_utils import compute_shap_values, top_k_features

log = logging.getLogger(__name__)

router = APIRouter(tags=["transactions"])

MISSING = -999.0


def _build_feature_row(
    features_raw: dict,
    temporal: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame with all model features.

    Priority (later overwrites earlier):
      1. -999 baseline for all features (LightGBM native missing)
      2. Numeric values from the request features dict
      3. Computed temporal features from the card state store

    Categorical string values are dropped (model expects encoded numerics).
    TransactionAmt_log and time features are derived if TransactionAmt / TransactionDT present.
    """
    row: dict[str, float] = {f: MISSING for f in feature_cols}

    # Overlay request features (numeric only)
    for k, v in features_raw.items():
        if k in row and v is not None:
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass  # skip non-numeric (e.g. ProductCD="W" - not encoded yet)

    # Derived features from raw values
    amt = features_raw.get("TransactionAmt")
    if amt is not None:
        try:
            row["TransactionAmt_log"] = math.log1p(float(amt))
        except (TypeError, ValueError):
            pass

    dt = features_raw.get("TransactionDT")
    if dt is not None:
        try:
            dt_f = float(dt)
            row["hour_of_day"] = float(int(dt_f % 86400) // 3600)
            row["day_of_week"] = float(int(dt_f // 86400) % 7)
        except (TypeError, ValueError):
            pass

    # Overlay temporal features (highest priority - computed from state store)
    for k, v in temporal.items():
        if k in row:
            row[k] = v

    return pd.DataFrame([row])[feature_cols]


@router.post("/score", response_model=ScoreResponse)
def score_transaction(request_body: TransactionRequest, request: Request):
    """
    Score a single transaction and return anomaly score, SHAP attribution,
    and an optional LLM-generated analyst explanation.

    Features are passed as a dict keyed by model feature names. Missing features
    default to -999. Temporal features (zscore, velocity, is_new_device) are
    computed from the in-memory card state store if not explicitly provided.
    """
    t0 = time.perf_counter()

    features_raw = request_body.features

    # Extract card identity for feature store
    card_id = int(features_raw.get("card1", 0) or 0)
    amount = float(features_raw.get("TransactionAmt", 0.0) or 0.0)
    device_info = features_raw.get("DeviceInfo") or features_raw.get("_DeviceInfo_raw")
    if isinstance(device_info, float) and math.isnan(device_info):
        device_info = None
    transaction_dt = float(features_raw.get("TransactionDT", time.time()) or time.time())

    # Compute temporal features BEFORE updating state (prevents leakage)
    temporal = feature_store.compute_features(card_id, amount, device_info, transaction_dt)

    # Build model input
    feature_cols = registry.feature_cols
    X = _build_feature_row(features_raw, temporal, feature_cols)

    # Score
    score = float(registry.model.predict_proba(X)[0, 1])
    is_flagged = score >= registry.threshold
    alert_level = _alert_level(score)

    # SHAP + explanation for flagged transactions only
    top_feats: list[dict] = []
    explanation: str | None = None

    if is_flagged:
        try:
            shap_vals = compute_shap_values(registry.explainer, X)[0]
            top_feats = top_k_features(shap_vals, feature_cols, X.values[0], k=3)
            if request_body.generate_explanation:
                explanation = generate_explanation(score, top_feats, prompt_version="v2")
        except Exception as e:
            log.warning(f"SHAP/LLM error for {request_body.transaction_id}: {e}")

    # Update card state store AFTER scoring
    feature_store.update(card_id, amount, device_info, transaction_dt)

    # Record for GET /transactions and drift monitoring
    app_state = request.app.state
    record = {
        "transaction_id": request_body.transaction_id,
        "anomaly_score": round(score, 4),
        "is_flagged": is_flagged,
        "alert_level": alert_level,
        "transaction_amt": amount if amount else None,
        "timestamp_dt": transaction_dt,
        "top_features": top_feats,
        "explanation": explanation,
    }
    feature_row = {k: float(X[k].iloc[0]) for k in feature_cols}

    app_state.recent_transactions.appendleft(record)
    app_state.recent_feature_rows.appendleft(feature_row)
    app_state.n_scored += 1
    if is_flagged:
        app_state.n_flagged += 1

    # Trigger background drift check every DRIFT_CHECK_INTERVAL transactions
    app_state.n_since_drift_check += 1
    if app_state.n_since_drift_check >= app_state.drift_check_interval:
        app_state.n_since_drift_check = 0
        _run_drift_check(app_state)

    latency_ms = (time.perf_counter() - t0) * 1000

    return ScoreResponse(
        transaction_id=request_body.transaction_id,
        anomaly_score=round(score, 4),
        is_flagged=is_flagged,
        alert_level=alert_level,
        top_features=[TopFeature(**f) for f in top_feats],
        explanation=explanation,
        model_version=registry.version,
        latency_ms=round(latency_ms, 1),
    )


@router.post("/explain", response_model=ExplainResponse)
def explain_transaction(body: ExplainRequest):
    """
    Generate an LLM explanation for a flagged transaction given its SHAP top features.
    Calls the constrained v2 prompt which cites only the provided SHAP features.
    """
    top_feats = [f.model_dump() for f in body.top_features]
    explanation = generate_explanation(body.anomaly_score, top_feats, prompt_version="v2")
    return ExplainResponse(transaction_id=body.transaction_id, explanation=explanation)


@router.get("/transactions", response_model=list[TransactionRecord])
def get_transactions(limit: int = 50, request: Request = None):
    """Return the most recent scored transactions, newest first."""
    records = list(request.app.state.recent_transactions)[:limit]
    return [TransactionRecord(**r) for r in records]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alert_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def _run_drift_check(app_state) -> None:
    """Run PSI drift check on recent feature rows. Updates app_state.drift_state."""
    monitor = getattr(app_state, "drift_monitor", None)
    if monitor is None:
        return
    rows = list(app_state.recent_feature_rows)
    if len(rows) < 100:
        return
    try:
        import pandas as pd
        current = pd.DataFrame(rows)
        result = monitor.check_batch(current)
        from datetime import datetime, timezone
        app_state.drift_state.update({
            "drift_detected": result["drift_detected"],
            "drifted_features": result["drifted_features"],
            "psi_scores": result["psi_scores"],
            "last_checked": datetime.now(timezone.utc).isoformat(),
        })
        if result["drift_detected"]:
            log.warning(f"Drift detected! Features: {result['drifted_features']}")
    except Exception as e:
        log.warning(f"Drift check failed: {e}")
