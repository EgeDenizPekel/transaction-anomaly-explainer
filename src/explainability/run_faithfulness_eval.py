"""
Standalone faithfulness evaluation script.

Generates explanations for N flagged test-set transactions using both prompt
versions, scores faithfulness, and saves results to data/processed/faithfulness_results.json.

Run with:
    python src/explainability/run_faithfulness_eval.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.shap_utils import (
    build_explainer, compute_shap_values, get_feature_cols, top_k_features,
)
from src.explainability.llm_explainer import generate_explanation
from src.explainability.faithfulness_eval import compute_faithfulness, evaluate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_EVAL = 50
OUTPUT_PATH = ROOT / "data" / "processed" / "faithfulness_results.json"
CHECKPOINT_PATH = ROOT / "data" / "processed" / "faithfulness_checkpoint.json"
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow' / 'mlruns.db'}"


def load_model_and_data():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_registered_model("anomaly-detector").latest_versions[0]
    run_id = model_version.run_id
    run = client.get_run(run_id)
    threshold = float(run.data.tags.get("threshold", 0.5))

    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    log.info(f"Loaded model run={run_id} threshold={threshold:.4f}")

    test_df = pd.read_parquet(ROOT / "data" / "processed" / "features_test.parquet")
    feature_cols = get_feature_cols(test_df)
    return model, test_df, feature_cols, threshold


def build_eval_records(model, test_df, feature_cols, threshold):
    X_test = test_df[feature_cols]
    test_probs = model.predict_proba(X_test)[:, 1]
    flagged_idx = np.where(test_probs >= threshold)[0]
    log.info(f"Flagged in test set: {len(flagged_idx):,}")

    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(flagged_idx, size=min(N_EVAL, len(flagged_idx)), replace=False))

    X_sample = X_test.iloc[sample_idx]
    scores_sample = test_probs[sample_idx]

    log.info(f"Computing SHAP for {len(sample_idx)} transactions ...")
    explainer = build_explainer(model)
    shap_vals = compute_shap_values(explainer, X_sample)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])

    records = []
    for i in range(len(sample_idx)):
        txn_id = int(test_df.iloc[sample_idx[i]].get("TransactionID", sample_idx[i]))
        top_feats = top_k_features(shap_vals[i], feature_cols, X_sample.iloc[i].values, k=3)
        records.append({
            "transaction_id": txn_id,
            "score": float(scores_sample[i]),
            "top_features": top_feats,
        })

    log.info("SHAP done.")
    return records


def run_generation_batch(records, prompt_version):
    # Load checkpoint if exists
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)

    results = []
    errors = 0
    start = time.time()

    for i, rec in enumerate(records):
        cache_key = f"{prompt_version}_{rec['transaction_id']}"

        # Resume from checkpoint if available
        if cache_key in checkpoint:
            results.append({**rec, "explanation": checkpoint[cache_key]})
            continue

        try:
            explanation = generate_explanation(
                score=rec["score"],
                features=rec["top_features"],
                prompt_version=prompt_version,
            )
            results.append({**rec, "explanation": explanation})
            checkpoint[cache_key] = explanation
        except Exception as e:
            errors += 1
            log.warning(f"Error on record {i}: {e}")
            results.append({**rec, "explanation": ""})

        # Save checkpoint every 10 calls
        if (i + 1) % 10 == 0:
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(checkpoint, f)
            elapsed = time.time() - start
            done = i + 1 - sum(1 for r in results[:i+1] if r["explanation"] == "")
            rate = max(done, 1) / max(elapsed, 1)
            eta = (len(records) - i - 1) / rate
            log.info(f"[{prompt_version}] {i+1}/{len(records)} | {rate:.1f} req/s | ETA {eta:.0f}s")

    elapsed = time.time() - start
    log.info(f"[{prompt_version}] Done. {len(results)} explanations in {elapsed:.0f}s. Errors: {errors}")
    return results


def main():
    model, test_df, feature_cols, threshold = load_model_and_data()
    records = build_eval_records(model, test_df, feature_cols, threshold)

    log.info("Generating v1 explanations (unconstrained) ...")
    v1_results = run_generation_batch(records, "v1")

    log.info("Generating v2 explanations (constrained) ...")
    v2_results = run_generation_batch(records, "v2")

    # Score faithfulness
    v1_batch = [{"explanation": r["explanation"], "top_features": r["top_features"]} for r in v1_results]
    v2_batch = [{"explanation": r["explanation"], "top_features": r["top_features"]} for r in v2_results]

    v1_summary = evaluate_batch(v1_batch, "v1")
    v2_summary = evaluate_batch(v2_batch, "v2")

    # Print results
    log.info("=" * 55)
    log.info(f"{'Metric':<35} {'v1':>8} {'v2':>8}")
    log.info("=" * 55)
    for label, key in [
        ("mention_rate",           "mean_mention_rate"),
        ("direction_accuracy",     "mean_direction_accuracy"),
        ("value_accuracy",         "mean_value_accuracy"),
        ("hallucination_rate",     "hallucination_rate"),
        ("composite_faithfulness", "mean_composite_faithfulness"),
    ]:
        v1v = v1_summary.get(key)
        v2v = v2_summary.get(key)
        log.info(f"  {label:<33} {str(round(v1v,3) if v1v else 'N/A'):>8} {str(round(v2v,3) if v2v else 'N/A'):>8}")

    # Save full results
    output = {
        "v1": {"summary": {k: v for k, v in v1_summary.items() if k != "individual"},
               "results": v1_results},
        "v2": {"summary": {k: v for k, v in v2_summary.items() if k != "individual"},
               "results": v2_results},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
