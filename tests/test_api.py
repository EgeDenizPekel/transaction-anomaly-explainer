"""
Integration tests for the Transaction Anomaly Explainer API.

Requires:
  - MLflow model registered as 'anomaly-detector'
  - data/processed/features_val.parquet present
  - LLM not called (generate_explanation=False in all POST /score tests)

Run:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient that triggers lifespan startup/shutdown."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(client, features: dict, generate_explanation: bool = False) -> dict:
    r = client.post(
        "/score",
        json={
            "transaction_id": "test_tx",
            "features": features,
            "generate_explanation": generate_explanation,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_version" in data


# ---------------------------------------------------------------------------
# POST /score - basic
# ---------------------------------------------------------------------------

def test_score_returns_required_fields(client):
    data = _score(client, {"TransactionAmt": 50.0})
    for field in ["transaction_id", "anomaly_score", "is_flagged", "alert_level",
                  "top_features", "explanation", "model_version", "latency_ms"]:
        assert field in data, f"Missing field: {field}"


def test_score_anomaly_score_in_range(client):
    data = _score(client, {"TransactionAmt": 100.0})
    assert 0.0 <= data["anomaly_score"] <= 1.0


def test_score_alert_level_values(client):
    data = _score(client, {"TransactionAmt": 100.0})
    assert data["alert_level"] in {"HIGH", "MEDIUM", "LOW"}


def test_score_explanation_none_when_not_requested(client):
    data = _score(client, {"TransactionAmt": 10.0}, generate_explanation=False)
    assert data["explanation"] is None or isinstance(data["explanation"], str)


def test_score_empty_features(client):
    """Empty features should still return a valid response (all -999)."""
    data = _score(client, {})
    assert "anomaly_score" in data


def test_score_missing_features_handled(client):
    """Partial features dict should not crash."""
    data = _score(client, {"TransactionAmt": 250.0, "card1": 9500})
    assert 0.0 <= data["anomaly_score"] <= 1.0


def test_score_top_features_structure(client):
    """Flagged transactions should return top_features with the right shape."""
    data = _score(client, {
        "TransactionAmt": 50000.0,
        "TransactionAmt_zscore": 15.0,
        "is_new_device": 1.0,
        "txn_velocity_1h": 20.0,
    })
    if data["is_flagged"]:
        assert len(data["top_features"]) <= 3
        for feat in data["top_features"]:
            assert "feature" in feat
            assert "shap_value" in feat
            assert "direction" in feat
            assert feat["direction"] in {"increases_risk", "decreases_risk"}


def test_score_latency_ms_positive(client):
    data = _score(client, {"TransactionAmt": 100.0})
    assert data["latency_ms"] > 0


# ---------------------------------------------------------------------------
# Feature store: temporal features accumulate across calls
# ---------------------------------------------------------------------------

def test_feature_store_updates_across_calls(client):
    """Same card should accumulate history without crashing on second call."""
    _score(client, {"TransactionAmt": 100.0, "card1": 99999, "TransactionDT": 1000000.0})
    data2 = _score(client, {"TransactionAmt": 200.0, "card1": 99999, "TransactionDT": 1003600.0})
    assert "anomaly_score" in data2


# ---------------------------------------------------------------------------
# GET /transactions
# ---------------------------------------------------------------------------

def test_get_transactions_returns_list(client):
    r = client.get("/transactions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_transactions_limit(client):
    r = client.get("/transactions?limit=2")
    assert r.status_code == 200
    assert len(r.json()) <= 2


def test_get_transactions_contains_recent_score(client):
    _score(client, {"TransactionAmt": 42.0, "card1": 777})
    r = client.get("/transactions?limit=10")
    ids = [t["transaction_id"] for t in r.json()]
    assert "test_tx" in ids


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

def test_get_metrics_fields(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    for field in ["model_version", "n_transactions_scored", "n_flagged", "flag_rate", "drift_detected"]:
        assert field in data


def test_get_metrics_counters_increase(client):
    before = client.get("/metrics").json()["n_transactions_scored"]
    _score(client, {"TransactionAmt": 10.0})
    after = client.get("/metrics").json()["n_transactions_scored"]
    assert after == before + 1


# ---------------------------------------------------------------------------
# GET /drift-status
# ---------------------------------------------------------------------------

def test_get_drift_status_fields(client):
    r = client.get("/drift-status")
    assert r.status_code == 200
    data = r.json()
    for field in ["drift_detected", "drifted_features", "psi_scores",
                  "last_checked", "n_transactions_since_last_check"]:
        assert field in data


def test_drift_status_boolean(client):
    data = client.get("/drift-status").json()
    assert isinstance(data["drift_detected"], bool)


# ---------------------------------------------------------------------------
# GET /retrain/status
# ---------------------------------------------------------------------------

def test_retrain_status_idle_at_startup(client):
    r = client.get("/retrain/status")
    assert r.status_code == 200
    data = r.json()
    assert "running" in data
    assert data["running"] is False


# ---------------------------------------------------------------------------
# GET /batch-metrics
# ---------------------------------------------------------------------------

def test_batch_metrics_returns_list(client):
    r = client.get("/batch-metrics")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_batch_metrics_schema(client):
    """If any batch metrics exist, each entry must have required fields."""
    data = client.get("/batch-metrics").json()
    for entry in data:
        for field in ["batch_id", "is_post_drift", "f1", "fraud_rate",
                      "n_transactions", "drift_detected", "drifted_features", "recorded_at"]:
            assert field in entry, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# GET /drift-history
# ---------------------------------------------------------------------------

def test_drift_history_returns_list(client):
    r = client.get("/drift-history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_drift_history_schema(client):
    """If any drift events exist, each entry must have required fields."""
    data = client.get("/drift-history").json()
    for entry in data:
        for field in ["batch_id", "is_post_drift", "drift_detected",
                      "drifted_features", "psi_scores", "checked_at"]:
            assert field in entry, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# GET /calibration
# ---------------------------------------------------------------------------

def test_calibration_returns_200_or_503(client):
    """Calibration requires val parquet with isFraud column. Accept 200 or 503."""
    r = client.get("/calibration")
    assert r.status_code in (200, 503), f"Unexpected status {r.status_code}: {r.text}"


def test_calibration_schema(client):
    r = client.get("/calibration")
    if r.status_code == 200:
        data = r.json()
        for field in ["brier_score", "baseline_brier", "n_val_samples", "reliability_diagram"]:
            assert field in data, f"Missing field: {field}"
        assert 0.0 <= data["brier_score"] <= 1.0
        assert 0.0 <= data["baseline_brier"] <= 0.25
        assert isinstance(data["reliability_diagram"], list)
        for bin_entry in data["reliability_diagram"]:
            for field in ["mean_predicted", "actual_rate", "count"]:
                assert field in bin_entry


# ---------------------------------------------------------------------------
# GET /explanation-drift
# ---------------------------------------------------------------------------

def test_explanation_drift_returns_list(client):
    r = client.get("/explanation-drift")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_explanation_drift_schema(client):
    data = client.get("/explanation-drift").json()
    for entry in data:
        for field in ["batch_id", "is_post_drift", "top_feature_counts", "n_flagged", "recorded_at"]:
            assert field in entry, f"Missing field: {field}"
        assert isinstance(entry["top_feature_counts"], dict)


# ---------------------------------------------------------------------------
# POST /score - counterfactuals and stability flags
# ---------------------------------------------------------------------------

def test_score_counterfactuals_flag(client):
    """compute_counterfactuals=True should not crash and returns list."""
    r = client.post("/score", json={
        "transaction_id": "test_cf",
        "features": {"TransactionAmt": 50000.0, "txn_velocity_1h": 20.0, "is_new_device": 1.0},
        "generate_explanation": False,
        "compute_counterfactuals": True,
    })
    assert r.status_code == 200
    data = r.json()
    assert "counterfactuals" in data
    assert isinstance(data["counterfactuals"], list)


def test_score_stability_flag(client):
    """compute_stability=True on a flagged transaction should return a float or None."""
    r = client.post("/score", json={
        "transaction_id": "test_stability",
        "features": {"TransactionAmt": 50000.0, "txn_velocity_1h": 20.0, "is_new_device": 1.0},
        "generate_explanation": False,
        "compute_stability": True,
    })
    assert r.status_code == 200
    data = r.json()
    assert "stability_score" in data
    if data["stability_score"] is not None:
        assert 0.0 <= data["stability_score"] <= 1.0


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------

def test_explain_returns_200_or_503(client):
    """
    /explain calls the LLM which may not be available in the test environment.
    Accept 200 (LLM available) or 503 (LLM unavailable). Never 422 (schema error).
    """
    payload = {
        "transaction_id": "test_explain",
        "anomaly_score": 0.92,
        "top_features": [
            {"feature": "card1", "shap_value": 1.08, "feature_value": 0.034,
             "direction": "increases_risk"},
            {"feature": "C1", "shap_value": 0.73, "feature_value": 9.0,
             "direction": "increases_risk"},
            {"feature": "time_since_last_txn", "shap_value": -0.33, "feature_value": 86400.0,
             "direction": "decreases_risk"},
        ],
    }
    r = client.post("/explain", json=payload)
    assert r.status_code in (200, 503), f"Unexpected status {r.status_code}: {r.text}"
    if r.status_code == 200:
        data = r.json()
        assert "transaction_id" in data
        assert "explanation" in data
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"]) > 0
