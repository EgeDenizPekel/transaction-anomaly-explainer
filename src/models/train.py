"""
LightGBM training script with MLflow tracking.

Usage:
    python src/models/train.py
"""

import logging
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.evaluate import evaluate, find_best_threshold, print_metrics
from src.models.shap_utils import build_explainer, compute_shap_values, get_feature_cols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PROCESSED = ROOT / "data" / "processed"
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow' / 'mlruns.db'}"
EXPERIMENT_NAME = "anomaly-detector"
REGISTERED_MODEL_NAME = "anomaly-detector"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

PARAMS = {
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

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading processed parquet files ...")
    train = pd.read_parquet(DATA_PROCESSED / "features_train.parquet")
    val   = pd.read_parquet(DATA_PROCESSED / "features_val.parquet")
    test  = pd.read_parquet(DATA_PROCESSED / "features_test.parquet")
    log.info(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMClassifier:
    log.info("Training LightGBM ...")
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    log.info(f"Best iteration: {model.best_iteration_}")
    return model


def run_training() -> None:
    # MLflow setup
    (ROOT / "mlflow").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df, val_df, test_df = load_splits()

    feature_cols = get_feature_cols(train_df)
    log.info(f"Feature count: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"]
    X_val   = val_df[feature_cols]
    y_val   = val_df["isFraud"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["isFraud"]

    with mlflow.start_run() as run:
        log.info(f"MLflow run: {run.info.run_id}")

        # Train
        model = train_model(X_train, y_train, X_val, y_val)

        # Log params
        mlflow.log_params({**PARAMS, "best_iteration": model.best_iteration_})

        # Val metrics - find optimal threshold
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val.values, val_probs)
        log.info(f"Optimal threshold (val F1): {threshold:.4f}")

        val_metrics = evaluate(y_val.values, val_probs, threshold, split_name="val")
        mlflow.log_metrics(val_metrics)
        mlflow.log_metric("threshold", threshold)

        log.info("Val metrics:")
        print_metrics(val_metrics)

        # Test metrics
        test_probs = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test.values, test_probs, threshold, split_name="test")
        mlflow.log_metrics(test_metrics)

        log.info("Test metrics:")
        print_metrics(test_metrics)

        # SHAP validation on 500 random training samples
        log.info("Computing SHAP values on 500 train samples for validation ...")
        sample_idx = np.random.default_rng(42).integers(0, len(X_train), size=500)
        X_sample = X_train.iloc[sample_idx]
        explainer = build_explainer(model)
        shap_vals = compute_shap_values(explainer, X_sample)

        # Mean absolute SHAP per feature - log top 10 as params for reference
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
        top10_features = {
            f"top_shap_feature_{i+1}": feature_cols[idx]
            for i, idx in enumerate(top10_idx)
        }
        mlflow.log_params(top10_features)
        log.info("Top 10 features by mean |SHAP|:")
        for i, idx in enumerate(top10_idx):
            log.info(f"  {i+1:2d}. {feature_cols[idx]:35s} {mean_abs_shap[idx]:.4f}")

        # Log model artifact
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5),
        )

        # Store threshold as a tag so the serving layer can retrieve it
        mlflow.set_tag("threshold", str(round(threshold, 6)))
        mlflow.set_tag("feature_cols", ",".join(feature_cols))

        log.info(f"Model registered as '{REGISTERED_MODEL_NAME}'")
        log.info(f"Run ID: {run.info.run_id}")

    log.info("Training complete.")


if __name__ == "__main__":
    run_training()
