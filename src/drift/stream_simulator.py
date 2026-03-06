"""
Stream simulator for concept drift demonstration.

Generates a simulated transaction stream in two segments:
  - Pre-drift  (batches 0-2): sampled from the test set with original labels.
    The existing model scores these well.
  - Post-drift (batches 3-5): same source data but with two synthetic changes:
      1. txn_velocity_1h scaled up 4x and hour_of_day biased toward 0-5 AM
         (creates measurable PSI on these features, triggering the monitor)
      2. Fraud pattern changed: high-velocity + odd-hour transactions are
         labelled fraud at 80% rate; high-zscore transactions have their
         fraud labels partially suppressed (40% flip rate)

NOTE: This is a SYNTHETIC simulation for demo purposes. Real concept drift
would occur gradually as fraud patterns evolve over time. The injected drift
is intentionally abrupt so that a 6-batch stream produces a clear F1
degradation and recovery chart. See README for context.

Run standalone:
    python src/drift/stream_simulator.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stream configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 1000
N_PRE_BATCHES = 3
N_POST_BATCHES = 3
N_PRE_DRIFT = N_PRE_BATCHES * BATCH_SIZE
N_POST_DRIFT = N_POST_BATCHES * BATCH_SIZE

# Drift injection parameters
VELOCITY_SCALE = 4.0          # multiply txn_velocity_1h in post-drift segment
ODD_HOUR_SHIFT_PROB = 0.70    # fraction of rows to push toward odd hours (0-5 AM)
ODD_HOUR_FRAUD_RATE = 0.80    # P(isFraud=1 | high velocity AND hour in [0,5])
OLD_PATTERN_FLIP_RATE = 0.40  # P(flip fraud->legit) for high-zscore txns in post-drift

OUTPUT_PATH = ROOT / "data" / "streaming" / "simulated_stream.parquet"


# ---------------------------------------------------------------------------
# Drift injection
# ---------------------------------------------------------------------------

def _inject_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Apply synthetic concept drift to a DataFrame:
    1. Distributional shift: scale velocity, bias hour toward odd hours.
    2. Concept shift: re-label fraud based on the new pattern.

    This is SYNTHETIC - for demo purposes only.
    """
    df = df.copy()

    # 1. Distributional shift on txn_velocity_1h (triggers PSI alert)
    if "txn_velocity_1h" in df.columns:
        df["txn_velocity_1h"] = (
            df["txn_velocity_1h"].clip(lower=0) * VELOCITY_SCALE
        ).round().astype(int)

    # 2. Distributional shift on hour_of_day (triggers PSI alert)
    if "hour_of_day" in df.columns:
        shift_mask = rng.random(len(df)) < ODD_HOUR_SHIFT_PROB
        df.loc[shift_mask, "hour_of_day"] = rng.integers(0, 6, size=int(shift_mask.sum()))

    # 3. Re-label: new fraud signal = high velocity + odd hour
    if "txn_velocity_1h" in df.columns and "hour_of_day" in df.columns:
        high_vel = df["txn_velocity_1h"] > df["txn_velocity_1h"].quantile(0.75)
        odd_hour = df["hour_of_day"] < 6
        new_pattern = high_vel & odd_hour

        # New fraud labels for new-pattern transactions
        new_fraud = rng.random(int(new_pattern.sum())) < ODD_HOUR_FRAUD_RATE
        df.loc[new_pattern, "isFraud"] = new_fraud.astype(int)

        # Weaken old fraud signal: flip some high-zscore fraud to legit
        if "TransactionAmt_zscore" in df.columns:
            high_z = df["TransactionAmt_zscore"] > df["TransactionAmt_zscore"].quantile(0.75)
            old_fraud = high_z & (df["isFraud"] == 1) & ~new_pattern
            flip = old_fraud & (rng.random(len(df)) < OLD_PATTERN_FLIP_RATE)
            df.loc[flip, "isFraud"] = 0

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_stream(
    test_parquet: Path = ROOT / "data" / "processed" / "features_test.parquet",
    output_path: Path = OUTPUT_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate and persist the simulated stream.

    Returns a DataFrame with all feature columns plus:
        batch_id      : int  - 0-indexed batch (0-2 pre-drift, 3-5 post-drift)
        is_post_drift : bool - True for post-drift batches
    """
    log.info("Loading test set for stream generation ...")
    test_df = pd.read_parquet(test_parquet)
    rng = np.random.default_rng(seed)

    total = N_PRE_DRIFT + N_POST_DRIFT
    if len(test_df) < total:
        raise ValueError(f"Test set has {len(test_df)} rows; need {total}")

    # Sample without replacement, preserve temporal order
    idx = np.sort(rng.choice(len(test_df), size=total, replace=False))
    sampled = test_df.iloc[idx].reset_index(drop=True)

    pre = sampled.iloc[:N_PRE_DRIFT].copy()
    post = _inject_drift(sampled.iloc[N_PRE_DRIFT:].copy(), rng)

    pre["batch_id"] = np.repeat(np.arange(N_PRE_BATCHES), BATCH_SIZE)
    post["batch_id"] = np.repeat(
        np.arange(N_PRE_BATCHES, N_PRE_BATCHES + N_POST_BATCHES), BATCH_SIZE
    )
    pre["is_post_drift"] = False
    post["is_post_drift"] = True

    stream = pd.concat([pre, post], ignore_index=True)

    log.info(f"Pre-drift  fraud rate: {pre['isFraud'].mean():.3f}  ({N_PRE_DRIFT:,} rows)")
    log.info(f"Post-drift fraud rate: {post['isFraud'].mean():.3f}  ({N_POST_DRIFT:,} rows)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream.to_parquet(output_path, index=False)
    log.info(f"Saved {len(stream):,} rows to {output_path}")
    return stream


if __name__ == "__main__":
    generate_stream()
