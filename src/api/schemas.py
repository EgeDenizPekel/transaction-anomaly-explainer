"""
Pydantic request/response models for the Transaction Anomaly Explainer API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# POST /score
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id: str
    features: dict[str, Any] = Field(
        description=(
            "Feature dict keyed by model feature names. "
            "Missing features default to -999 (LightGBM native missing). "
            "Temporal features (TransactionAmt_zscore, is_new_device, txn_velocity_1h) "
            "are computed from the in-memory card state store if not provided."
        )
    )
    generate_explanation: bool = Field(
        default=True,
        description="Whether to call the LLM for a natural-language summary of model attribution signals. "
                    "Set False to get a score-only response with lower latency.",
    )
    compute_counterfactuals: bool = Field(
        default=False,
        description="If True, compute single-feature counterfactuals for flagged transactions. "
                    "Adds latency (~30-100ms per flagged transaction).",
    )
    compute_stability: bool = Field(
        default=False,
        description="If True, compute SHAP attribution stability score for flagged transactions. "
                    "Adds latency (~50-200ms per flagged transaction).",
    )


class TopFeature(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class CounterfactualFeature(BaseModel):
    feature: str
    current_value: float
    counterfactual_value: float
    score_after: float
    pct_change: float  # % change from current to counterfactual value


class ScoreResponse(BaseModel):
    transaction_id: str
    anomaly_score: float
    is_flagged: bool
    alert_level: str          # HIGH | MEDIUM | LOW
    top_features: list[TopFeature]
    explanation: str | None
    model_version: str
    latency_ms: float
    stability_score: float | None = None
    counterfactuals: list[CounterfactualFeature] = []


# ---------------------------------------------------------------------------
# GET /transactions
# ---------------------------------------------------------------------------

class TransactionRecord(BaseModel):
    transaction_id: str
    anomaly_score: float
    is_flagged: bool
    alert_level: str
    transaction_amt: float | None
    timestamp_dt: float | None  # TransactionDT value if provided
    top_features: list[TopFeature] = []
    explanation: str | None = None
    counterfactuals: list[CounterfactualFeature] = []
    stability_score: float | None = None


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class MetricsResponse(BaseModel):
    model_version: str
    val_roc_auc: float | None
    val_brier_score: float | None = None
    n_transactions_scored: int
    n_flagged: int
    flag_rate: float
    drift_detected: bool


# ---------------------------------------------------------------------------
# GET /calibration
# ---------------------------------------------------------------------------

class CalibrationBin(BaseModel):
    mean_predicted: float
    actual_rate: float
    count: int
    bin_lower: float
    bin_upper: float


class CalibrationResponse(BaseModel):
    brier_score: float
    baseline_brier: float  # naive predictor: base_rate * (1 - base_rate)
    n_val_samples: int
    reliability_diagram: list[CalibrationBin]


# ---------------------------------------------------------------------------
# GET /drift-status
# ---------------------------------------------------------------------------

class DriftStatusResponse(BaseModel):
    drift_detected: bool
    drifted_features: list[str]
    psi_scores: dict[str, float]
    last_checked: str | None
    n_transactions_since_last_check: int


# ---------------------------------------------------------------------------
# POST /retrain
# ---------------------------------------------------------------------------

class RetrainResponse(BaseModel):
    status: str    # "started" | "already_running"
    message: str


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    transaction_id: str
    anomaly_score: float
    top_features: list[TopFeature]


class ExplainResponse(BaseModel):
    transaction_id: str
    explanation: str


# ---------------------------------------------------------------------------
# GET /batch-metrics and GET /drift-history
# ---------------------------------------------------------------------------

class BatchMetric(BaseModel):
    batch_id: int
    is_post_drift: bool
    f1: float
    fraud_rate: float
    n_transactions: int
    n_flagged_batch: int = 0
    drift_detected: bool
    drifted_features: list[str]
    top_feature_counts: dict[str, int] = {}
    recorded_at: str


# ---------------------------------------------------------------------------
# GET /explanation-drift
# ---------------------------------------------------------------------------

class ExplanationDriftEvent(BaseModel):
    batch_id: int
    is_post_drift: bool
    top_feature_counts: dict[str, int]
    n_flagged: int
    recorded_at: str


class DriftEvent(BaseModel):
    batch_id: int
    is_post_drift: bool
    drift_detected: bool
    drifted_features: list[str]
    psi_scores: dict[str, float]
    checked_at: str
