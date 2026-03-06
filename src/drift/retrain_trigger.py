"""
Retraining pipeline using FastAPI BackgroundTasks (not Celery).

Design decision: FastAPI BackgroundTasks instead of Celery.
Rationale: LightGBM retraining on tabular data takes 2-10 minutes.
BackgroundTasks handles this without distributed task infrastructure.
Production would use Celery or a managed job runner.

Usage - standalone (notebook / script):
    from src.drift.retrain_trigger import retrain
    result = retrain(new_data_df, val_df)

Usage - FastAPI BackgroundTask (Phase 5):
    background_tasks.add_task(trigger_retrain_background, new_data, val_df)
"""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.evaluate import evaluate, find_best_threshold
from src.models.shap_utils import get_feature_cols

log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow' / 'mlruns.db'}"
REGISTERED_MODEL_NAME = "anomaly-detector"
RETRAIN_EXPERIMENT = "anomaly-detector-retrain"

# Subsample of original training data to include during retrain.
# Keeps retraining fast while preserving knowledge of original fraud patterns.
N_ORIGINAL_SAMPLE = 50_000

# Same hyperparameters as initial training
RETRAIN_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 256,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# In-memory status dict for BackgroundTask tracking (used in Phase 5 API)
_status: dict = {
    "status": "idle",        # idle | running | done | failed
    "started_at": None,
    "completed_at": None,
    "auc_before": None,
    "auc_after": None,
    "auc_delta": None,
    "promoted": False,
    "new_version": None,
    "error": None,
}


def get_status() -> dict:
    """Return a copy of the current retrain status (for API polling)."""
    return _status.copy()


def _load_current_model_info() -> tuple:
    """Return (model, threshold, val_auc) for the latest registered version."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_registered_model(REGISTERED_MODEL_NAME).latest_versions
    if not versions:
        raise RuntimeError(f"No versions registered for '{REGISTERED_MODEL_NAME}'")
    version = versions[0]
    run = client.get_run(version.run_id)
    threshold = float(run.data.tags.get("threshold", 0.5))
    auc = float(run.data.metrics.get("val/roc_auc", 0.0))
    model = mlflow.lightgbm.load_model(f"runs:/{version.run_id}/model")
    return model, threshold, auc


def retrain(
    new_data: pd.DataFrame,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
) -> dict:
    """
    Retrain LightGBM on a subsample of original training data + new stream data.

    The new data contains the post-drift fraud pattern. Combining it with a
    sample of original training preserves knowledge of historical patterns while
    letting the model learn the new signal.

    Registers a new MLflow model version if the retrained AUC does not degrade
    more than 0.005 below the current model's AUC.

    Args:
        new_data:  Post-drift stream data (features + isFraud column).
        val_df:    Validation split for early stopping and metric evaluation.
        train_df:  Original training data. If None, loaded from parquet.

    Returns:
        {
            "auc_before":  float,
            "auc_after":   float,
            "auc_delta":   float,
            "threshold":   float,
            "promoted":    bool,
            "new_version": str | None,
            "model":       LGBMClassifier,
            "run_id":      str,
        }
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(RETRAIN_EXPERIMENT)

    log.info("Loading current model metrics ...")
    _, _, auc_before = _load_current_model_info()
    log.info(f"Current model val AUC: {auc_before:.4f}")

    if train_df is None:
        log.info("Loading original training data ...")
        train_df = pd.read_parquet(ROOT / "data" / "processed" / "features_train.parquet")

    feature_cols = get_feature_cols(train_df)

    # Subsample original training to avoid drowning out the new signal
    n_sample = min(N_ORIGINAL_SAMPLE, len(train_df))
    train_sample = train_df.sample(n=n_sample, random_state=42)
    log.info(f"Original training sample: {len(train_sample):,} rows")

    # Align new data columns to the model's feature set
    stream_cols = feature_cols + ["isFraud"]
    available = [c for c in stream_cols if c in new_data.columns]
    new_clean = new_data[available].copy()
    log.info(f"Post-drift rows: {len(new_clean):,} | fraud rate: {new_clean['isFraud'].mean():.3f}")

    combined = pd.concat([train_sample, new_clean], ignore_index=True)
    log.info(f"Combined training size: {len(combined):,}")

    X_train = combined[feature_cols]
    y_train = combined["isFraud"]
    X_val = val_df[feature_cols]
    y_val = val_df["isFraud"]

    with mlflow.start_run() as run:
        log.info(f"Retrain MLflow run: {run.info.run_id}")
        t0 = time.time()

        new_model = lgb.LGBMClassifier(**RETRAIN_PARAMS)
        new_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        elapsed = time.time() - t0
        log.info(f"Training done in {elapsed:.0f}s. Best iteration: {new_model.best_iteration_}")

        val_probs = new_model.predict_proba(X_val)[:, 1]
        new_threshold = find_best_threshold(y_val.values, val_probs)
        metrics = evaluate(y_val.values, val_probs, new_threshold, split_name="val")
        auc_after = metrics["val/roc_auc"]

        mlflow.log_params({**RETRAIN_PARAMS, "best_iteration": new_model.best_iteration_})
        mlflow.log_params({
            "n_original_sample": n_sample,
            "n_new_data": len(new_clean),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metric("auc_before", auc_before)
        mlflow.log_metric("auc_delta", auc_after - auc_before)
        mlflow.set_tag("threshold", str(round(new_threshold, 6)))
        mlflow.set_tag("retrain_trigger", "drift_monitor")

        # Promote if new model doesn't degrade by more than 0.005
        promoted = auc_after >= auc_before - 0.005
        new_version = None

        if promoted:
            log.info(
                f"Promoting model: AUC {auc_before:.4f} -> {auc_after:.4f} "
                f"(delta: {auc_after - auc_before:+.4f})"
            )
            mlflow.lightgbm.log_model(
                new_model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                input_example=X_train.head(5),
            )
            client = mlflow.tracking.MlflowClient()
            new_version = client.get_registered_model(
                REGISTERED_MODEL_NAME
            ).latest_versions[0].version
        else:
            log.warning(
                f"Not promoting: AUC dropped {auc_before:.4f} -> {auc_after:.4f} "
                f"(delta: {auc_after - auc_before:+.4f})"
            )
            mlflow.lightgbm.log_model(new_model, artifact_path="model")

    result = {
        "auc_before": round(auc_before, 4),
        "auc_after": round(auc_after, 4),
        "auc_delta": round(auc_after - auc_before, 4),
        "threshold": round(new_threshold, 4),
        "promoted": promoted,
        "new_version": new_version,
        "model": new_model,
        "run_id": run.info.run_id,
    }
    log.info(f"Retrain complete: {result}")
    return result


def trigger_retrain_background(
    new_data: pd.DataFrame,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
) -> None:
    """
    Wrapper for FastAPI BackgroundTasks. Updates _status in-place.

    Call via:
        background_tasks.add_task(trigger_retrain_background, new_data, val_df)
    """
    global _status
    _status["status"] = "running"
    _status["started_at"] = datetime.now(timezone.utc).isoformat()
    _status["error"] = None

    try:
        result = retrain(new_data, val_df, train_df)
        _status.update({
            "status": "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "auc_before": result["auc_before"],
            "auc_after": result["auc_after"],
            "auc_delta": result["auc_delta"],
            "promoted": result["promoted"],
            "new_version": result["new_version"],
        })
    except Exception as e:
        log.error(f"Retrain failed: {e}", exc_info=True)
        _status.update({
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })
