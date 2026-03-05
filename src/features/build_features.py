"""
Feature engineering pipeline for IEEE-CIS fraud detection.

Design rules:
- Full dataset must be sorted by TransactionDT before any temporal feature is computed.
- All rolling/expanding operations use shift(1) to exclude the current row (leakage prevention).
- Target encoding statistics are computed on training rows only, then applied to val/test.
- Chronological split happens AFTER temporal features are computed on the full sorted dataset.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MISSING_NUM = -999
MISSING_CAT = "MISSING"
HIGH_MISSING_THRESHOLD = 0.70  # drop features with > 70% missing

# Categorical columns by expected cardinality
LOW_CARD_CATS = ["card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
HIGH_CARD_CATS = ["card1", "card2", "card3", "card5", "addr1", "addr2",
                  "P_emaildomain", "R_emaildomain",
                  "DeviceType", "DeviceInfo",
                  "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
                  "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = remaining 0.15


# ---------------------------------------------------------------------------
# Step 1: Load and join
# ---------------------------------------------------------------------------

def load_raw(data_raw: Path) -> pd.DataFrame:
    log.info("Loading train_transaction.csv ...")
    txn = pd.read_csv(data_raw / "train_transaction.csv")
    log.info("Loading train_identity.csv ...")
    idn = pd.read_csv(data_raw / "train_identity.csv")

    log.info(f"Transactions: {len(txn):,} rows x {txn.shape[1]} cols")
    log.info(f"Identity    : {len(idn):,} rows x {idn.shape[1]} cols")

    df = txn.merge(idn, on="TransactionID", how="left")
    log.info(f"After join  : {len(df):,} rows x {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# Step 2: Drop high-missing features
# ---------------------------------------------------------------------------

def drop_high_missing(df: pd.DataFrame, threshold: float = HIGH_MISSING_THRESHOLD) -> pd.DataFrame:
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

    # Never drop the target, key columns, or derived flag placeholders
    protected = {"isFraud", "TransactionID", "TransactionDT", "has_identity", "_DeviceInfo_raw"}
    cols_to_drop = [c for c in cols_to_drop if c not in protected]

    log.info(f"Dropping {len(cols_to_drop)} features with >{threshold*100:.0f}% missing")
    return df.drop(columns=cols_to_drop)


# ---------------------------------------------------------------------------
# Step 3: Flag identity coverage
# ---------------------------------------------------------------------------

def add_has_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute identity-derived binary flags before high-missing columns are dropped.
    DeviceType and DeviceInfo are >70% missing so they get dropped in drop_high_missing;
    this function must run first to extract signal from them.
    """
    # has_identity: whether this transaction has an identity row
    if "DeviceType" in df.columns:
        df["has_identity"] = df["DeviceType"].notnull().astype(np.int8)
    else:
        id_cols = [c for c in df.columns if c.startswith("id_")]
        df["has_identity"] = df[id_cols[0]].notnull().astype(np.int8) if id_cols else np.int8(0)

    log.info(f"has_identity coverage: {df['has_identity'].mean()*100:.1f}%")

    # is_new_device: first occurrence of (card1, DeviceInfo) in dataset order.
    # Requires sorting by TransactionDT first, but we compute the raw flag here
    # and re-sort in engineer_temporal_features. The sort happens at step 4 in
    # run_pipeline, so this flag is recomputed there on the sorted dataset.
    # We set a placeholder here; engineer_temporal_features overwrites it.
    if "DeviceInfo" in df.columns:
        df["_DeviceInfo_raw"] = df["DeviceInfo"].fillna("MISSING").astype(str)
    else:
        df["_DeviceInfo_raw"] = "MISSING"

    return df


# ---------------------------------------------------------------------------
# Step 4: Temporal feature engineering  (MUST run on full sorted dataset)
# ---------------------------------------------------------------------------

def _velocity_1h(group: pd.DataFrame) -> pd.Series:
    """Count transactions on the same card in the 1-hour window BEFORE current row."""
    dts = group["TransactionDT"].values
    counts = np.empty(len(dts), dtype=np.int32)
    for i in range(len(dts)):
        window_start = dts[i] - 3600
        left = np.searchsorted(dts, window_start, side="left")
        counts[i] = i - left  # excludes current row
    return pd.Series(counts, index=group.index)


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All features here use only past information for each card.
    df must already be sorted by TransactionDT before calling this function.
    """
    assert df["TransactionDT"].is_monotonic_increasing, \
        "DataFrame must be sorted by TransactionDT before calling engineer_temporal_features"

    log.info("Engineering temporal features ...")

    # Time-of-day and day-of-week (seconds elapsed mod day/week)
    df["hour_of_day"] = ((df["TransactionDT"] % 86400) / 3600).astype(np.float32)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)

    # Log-transform amount
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"]).astype(np.float32)

    # Per-card rolling statistics (expanding, shift 1 to exclude current row)
    log.info("  Computing per-card expanding mean/std ...")
    grp = df.groupby("card1")["TransactionAmt"]
    df["card_amt_mean"] = grp.transform(lambda x: x.expanding().mean().shift(1))
    df["card_amt_std"]  = grp.transform(lambda x: x.expanding().std().shift(1))

    # Z-score of transaction amount relative to card history
    df["TransactionAmt_zscore"] = (
        (df["TransactionAmt"] - df["card_amt_mean"]) / df["card_amt_std"].replace(0, np.nan)
    ).astype(np.float32)

    # Amount relative to card mean
    df["amt_to_mean_ratio"] = (
        df["TransactionAmt"] / df["card_amt_mean"].replace(0, np.nan)
    ).astype(np.float32)

    # Time since last transaction on same card
    log.info("  Computing time_since_last_txn ...")
    df["time_since_last_txn"] = (
        df.groupby("card1")["TransactionDT"]
        .transform(lambda x: x.diff())
        .astype(np.float32)
    )

    # Transaction velocity: count of transactions on same card in last 1 hour
    log.info("  Computing txn_velocity_1h (this takes a moment) ...")
    df["txn_velocity_1h"] = (
        df.groupby("card1", group_keys=False)
        .apply(_velocity_1h, include_groups=False)
        .astype(np.int32)
    )

    # is_new_device: first occurrence of (card1, DeviceInfo) pair in sorted history.
    # Uses _DeviceInfo_raw which was saved before DeviceInfo was dropped.
    log.info("  Computing is_new_device ...")
    df["card_device"] = df["card1"].astype(str) + "_" + df["_DeviceInfo_raw"]
    df["is_new_device"] = (~df.duplicated(subset=["card_device"], keep="first")).astype(np.int8)
    df.drop(columns=["card_device", "_DeviceInfo_raw"], inplace=True)

    log.info("Temporal features done.")
    return df


# ---------------------------------------------------------------------------
# Step 5: Chronological split
# ---------------------------------------------------------------------------

def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    log.info(f"Split -> train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    log.info(f"  Train TransactionDT: {train['TransactionDT'].min():,} - {train['TransactionDT'].max():,}")
    log.info(f"  Val   TransactionDT: {val['TransactionDT'].min():,} - {val['TransactionDT'].max():,}")
    log.info(f"  Test  TransactionDT: {test['TransactionDT'].min():,} - {test['TransactionDT'].max():,}")

    # Sanity: no temporal overlap
    assert train["TransactionDT"].max() <= val["TransactionDT"].min(), "Train/val overlap!"
    assert val["TransactionDT"].max() <= test["TransactionDT"].min(), "Val/test overlap!"

    return train, val, test


# ---------------------------------------------------------------------------
# Step 6: Categorical encoding
# ---------------------------------------------------------------------------

def _label_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[col] = df[col].fillna(MISSING_CAT).astype(str)
    codes = {v: i for i, v in enumerate(sorted(df[col].unique()))}
    df[col] = df[col].map(codes).fillna(-1).astype(np.int16)
    return df


def encode_categoricals(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Low-cardinality: label encoding.
    High-cardinality: target encoding (mean isFraud per category, computed from train only).
    """
    log.info("Encoding categoricals ...")

    # Label encode low-cardinality
    for col in LOW_CARD_CATS:
        if col not in train.columns:
            continue
        # Fit on train categories
        train[col] = train[col].fillna(MISSING_CAT).astype(str)
        val[col]   = val[col].fillna(MISSING_CAT).astype(str)
        test[col]  = test[col].fillna(MISSING_CAT).astype(str)
        codes = {v: i for i, v in enumerate(sorted(train[col].unique()))}
        for split in [train, val, test]:
            split[col] = split[col].map(codes).fillna(-1).astype(np.int16)

    # Target encode high-cardinality (fit on train only)
    global_mean = train["isFraud"].mean()
    for col in HIGH_CARD_CATS:
        if col not in train.columns:
            continue
        train[col] = train[col].fillna(MISSING_CAT).astype(str)
        val[col]   = val[col].fillna(MISSING_CAT).astype(str)
        test[col]  = test[col].fillna(MISSING_CAT).astype(str)

        # Smoothed target encoding: blend category mean with global mean
        # smoothing = count / (count + k) where k=10
        stats = train.groupby(col)["isFraud"].agg(["mean", "count"])
        k = 10
        stats["smoothed"] = (
            (stats["mean"] * stats["count"] + global_mean * k) / (stats["count"] + k)
        )
        encoding_map = stats["smoothed"].to_dict()

        for split in [train, val, test]:
            split[col] = split[col].map(encoding_map).fillna(global_mean).astype(np.float32)

    log.info("Categorical encoding done.")
    return train, val, test


# ---------------------------------------------------------------------------
# Step 7: Fill missing values
# ---------------------------------------------------------------------------

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    df[num_cols] = df[num_cols].fillna(MISSING_NUM)
    df[cat_cols] = df[cat_cols].fillna(MISSING_CAT)
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_raw: Path, data_processed: Path) -> None:
    data_processed.mkdir(parents=True, exist_ok=True)

    # 1. Load and join
    df = load_raw(data_raw)

    # 2. Identity coverage flag - must run BEFORE drop_high_missing because
    #    DeviceType and DeviceInfo are >70% missing and would be dropped first
    df = add_has_identity(df)

    # 3. Drop high-missing features
    df = drop_high_missing(df)

    # 4. Sort by TransactionDT - MUST happen before any temporal feature computation
    log.info("Sorting by TransactionDT ...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # 5. Temporal features (on full sorted dataset - leakage-safe)
    df = engineer_temporal_features(df)

    # 6. Chronological split
    train, val, test = chronological_split(df)

    # 7. Categorical encoding (fit on train only)
    train, val, test = encode_categoricals(train, val, test)

    # 8. Fill remaining missing values
    for split in [train, val, test]:
        fill_missing(split)

    # 9. Save
    log.info("Saving parquet files ...")
    train.to_parquet(data_processed / "features_train.parquet", index=False)
    val.to_parquet(data_processed / "features_val.parquet", index=False)
    test.to_parquet(data_processed / "features_test.parquet", index=False)

    log.info(f"Saved to {data_processed}")
    log.info(f"  features_train.parquet : {len(train):,} rows x {train.shape[1]} cols")
    log.info(f"  features_val.parquet   : {len(val):,} rows x {val.shape[1]} cols")
    log.info(f"  features_test.parquet  : {len(test):,} rows x {test.shape[1]} cols")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    run_pipeline(data_raw, data_processed)
