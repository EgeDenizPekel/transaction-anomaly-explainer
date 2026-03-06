"""
MLflow model registry with hot-swap capability.

Loads the latest registered 'anomaly-detector' model on startup.
Supports atomic hot-swap after a successful retrain without restarting the API.

Usage:
    from src.api.model_registry import registry
    model = registry.model
    threshold = registry.threshold
    registry.hot_swap(new_model, new_threshold, new_version)
"""

import logging
import sys
import threading
from pathlib import Path

import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.lightgbm

from src.models.shap_utils import build_explainer, get_feature_cols

log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow' / 'mlruns.db'}"
REGISTERED_MODEL_NAME = "anomaly-detector"


class ModelRegistry:
    """
    Singleton model registry. Thread-safe hot-swap via a lock.

    Attributes:
        model:        Loaded LGBMClassifier.
        threshold:    Operating threshold (tuned on val set).
        feature_cols: Ordered list of feature names the model expects.
        version:      Registered model version string.
        val_roc_auc:  AUC from the training run (for /metrics endpoint).
        explainer:    SHAP TreeExplainer (built lazily, reset on hot-swap).
    """

    def __init__(self):
        self._model = None
        self._threshold: float = 0.5
        self._feature_cols: list[str] = []
        self._version: str = "unknown"
        self._val_roc_auc: float | None = None
        self._explainer: shap.TreeExplainer | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the latest registered model from MLflow."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_registered_model(REGISTERED_MODEL_NAME).latest_versions
        if not versions:
            raise RuntimeError(f"No model registered as '{REGISTERED_MODEL_NAME}'")

        version = versions[0]
        run = client.get_run(version.run_id)
        threshold = float(run.data.tags.get("threshold", 0.5))
        val_auc = run.data.metrics.get("val/roc_auc")

        model = mlflow.lightgbm.load_model(f"runs:/{version.run_id}/model")
        log.info(f"Loaded model version={version.version} run={version.run_id[:8]} threshold={threshold:.4f}")

        with self._lock:
            self._model = model
            self._threshold = threshold
            self._version = str(version.version)
            self._val_roc_auc = float(val_auc) if val_auc is not None else None
            self._explainer = None  # built lazily

    def hot_swap(self, new_model, new_threshold: float, new_version: str) -> None:
        """Atomically replace model after a successful retrain."""
        log.info(f"Hot-swapping model to version {new_version} (threshold={new_threshold:.4f})")
        with self._lock:
            self._model = new_model
            self._threshold = new_threshold
            self._version = new_version
            self._explainer = None  # reset so it rebuilds for new model

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    @feature_cols.setter
    def feature_cols(self, cols: list[str]) -> None:
        self._feature_cols = cols

    @property
    def version(self) -> str:
        return self._version

    @property
    def val_roc_auc(self) -> float | None:
        return self._val_roc_auc

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            log.info("Building SHAP TreeExplainer ...")
            self._explainer = build_explainer(self.model)
        return self._explainer


# Singleton - imported by all routers
registry = ModelRegistry()
