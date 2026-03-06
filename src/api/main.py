"""
FastAPI application for the Transaction Anomaly Explainer.

Startup loads:
  - LightGBM model from MLflow registry
  - Validation set for drift monitoring reference
  - DriftMonitor on the top PSI features

Endpoints:
  POST /score           - Score + explain a transaction
  GET  /transactions    - Recent scored transactions
  GET  /metrics         - Model metrics + scoring stats
  GET  /drift-status    - PSI drift monitoring result
  POST /retrain         - Trigger background retraining
  GET  /retrain/status  - Retrain job status
  GET  /health          - Liveness probe
"""

import logging
import sys
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.model_registry import registry
from src.api.routers import admin, metrics, transactions
from src.models.shap_utils import get_feature_cols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup / lifespan
# ---------------------------------------------------------------------------

DRIFT_CHECK_INTERVAL = 1000
DRIFT_MONITOR_FEATURES = [
    "txn_velocity_1h",
    "hour_of_day",
    "TransactionAmt_zscore",
    "card_amt_std",
    "time_since_last_txn",
    "TransactionAmt",
    "card1",
]
VAL_PATH = ROOT / "data" / "processed" / "features_val.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up Transaction Anomaly Explainer API ...")

    # Load model from MLflow
    registry.load()

    # Derive feature columns + set up drift monitor from validation set
    if VAL_PATH.exists():
        val_df = pd.read_parquet(VAL_PATH)
        registry.feature_cols = get_feature_cols(val_df)
        log.info(f"Feature columns: {len(registry.feature_cols)}")

        monitor_features = [f for f in DRIFT_MONITOR_FEATURES if f in registry.feature_cols]
        if monitor_features:
            from src.drift.monitor import DriftMonitor
            app.state.drift_monitor = DriftMonitor(
                reference_data=val_df,
                features=monitor_features,
                psi_threshold=0.2,
            )
            log.info(f"Drift monitor on {len(monitor_features)} features")
        else:
            app.state.drift_monitor = None
    else:
        log.warning(f"Val parquet not found at {VAL_PATH} - drift monitor disabled")
        app.state.drift_monitor = None

    # Initialize app state
    app.state.recent_transactions = deque(maxlen=500)
    app.state.recent_feature_rows = deque(maxlen=1000)
    app.state.n_scored = 0
    app.state.n_flagged = 0
    app.state.n_since_drift_check = 0
    app.state.drift_check_interval = DRIFT_CHECK_INTERVAL
    app.state.drift_state = {
        "drift_detected": False,
        "drifted_features": [],
        "psi_scores": {},
        "last_checked": None,
    }
    app.state.batch_metrics_history = deque(maxlen=50)
    app.state.drift_history = deque(maxlen=50)

    # Start background stream seeder (replays simulated_stream.parquet for demo)
    from src.api.stream_seeder import start_seeder
    start_seeder(app)

    log.info("Startup complete.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Transaction Anomaly Explainer",
    description=(
        "Fraud detection API with SHAP-grounded LLM explanations and "
        "Evidently-based PSI drift monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(transactions.router)
app.include_router(metrics.router)
app.include_router(admin.router)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "model_version": registry.version,
        "n_scored": getattr(app.state, "n_scored", 0),
    }
