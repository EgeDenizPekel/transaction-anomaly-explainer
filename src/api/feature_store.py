"""
In-memory card state store for inference-time feature computation.

Maintains per-card rolling statistics that mirror the temporal features
computed by build_features.py during training:
  - TransactionAmt_zscore    : z-score of current amount vs card history
  - card_amt_mean / card_amt_std : rolling stats
  - is_new_device            : first time this card_id used this device string
  - time_since_last_txn      : seconds since last transaction on this card
  - txn_velocity_1h          : count of transactions in last 60 minutes
  - amt_to_mean_ratio        : amount / mean of card's historical amounts

IMPORTANT: compute_features() must be called BEFORE update() to avoid
leakage (the current transaction must not be included in its own statistics).

Production note: This in-memory store is reset on API restart and does not
scale horizontally. A production deployment would replace this with Redis or
a managed feature store (e.g. Feast).
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MISSING = -999.0
ONE_HOUR_SECONDS = 3600.0


@dataclass
class _CardState:
    amounts: list[float] = field(default_factory=list)
    devices: set[str] = field(default_factory=set)
    timestamps: list[float] = field(default_factory=list)


class FeatureStore:
    """
    Per-card rolling state store. Thread-safe for single-process use
    (FastAPI runs in a single process with async handlers).
    """

    def __init__(self):
        self._store: dict[int, _CardState] = defaultdict(_CardState)

    def compute_features(
        self,
        card_id: int,
        amount: float,
        device_info: str | None,
        transaction_dt: float,
    ) -> dict[str, float]:
        """
        Compute temporal features for the current transaction using only
        prior history (no leakage). Call this BEFORE update().

        Args:
            card_id:        card1 value (integer card identifier).
            amount:         TransactionAmt.
            device_info:    DeviceInfo string (or None if unknown).
            transaction_dt: TransactionDT in seconds.

        Returns:
            Dict of computed feature name -> value.
        """
        state = self._store[card_id]
        amounts = state.amounts
        timestamps = state.timestamps

        # --- TransactionAmt_zscore ---
        if len(amounts) >= 2:
            mean = sum(amounts) / len(amounts)
            variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
            std = math.sqrt(variance)
            zscore = (amount - mean) / (std + 1e-8) if std > 0 else 0.0
        elif len(amounts) == 1:
            zscore = 0.0
        else:
            zscore = 0.0

        # --- card_amt_mean / card_amt_std ---
        if amounts:
            card_amt_mean = sum(amounts) / len(amounts)
            if len(amounts) >= 2:
                variance = sum((x - card_amt_mean) ** 2 for x in amounts) / len(amounts)
                card_amt_std = math.sqrt(variance)
            else:
                card_amt_std = 0.0
        else:
            card_amt_mean = amount  # fallback: no history
            card_amt_std = 0.0

        # --- is_new_device ---
        if device_info and device_info not in state.devices:
            is_new_device = 1.0
        else:
            is_new_device = 0.0

        # --- time_since_last_txn ---
        if timestamps:
            time_since = transaction_dt - timestamps[-1]
        else:
            time_since = MISSING

        # --- txn_velocity_1h ---
        cutoff = transaction_dt - ONE_HOUR_SECONDS
        velocity = sum(1 for t in timestamps if t >= cutoff)

        # --- amt_to_mean_ratio ---
        amt_to_mean_ratio = amount / card_amt_mean if card_amt_mean != 0 else 1.0

        return {
            "TransactionAmt_zscore": round(zscore, 6),
            "card_amt_mean": round(card_amt_mean, 6),
            "card_amt_std": round(card_amt_std, 6),
            "is_new_device": is_new_device,
            "time_since_last_txn": round(time_since, 2),
            "txn_velocity_1h": float(velocity),
            "amt_to_mean_ratio": round(amt_to_mean_ratio, 6),
        }

    def update(
        self,
        card_id: int,
        amount: float,
        device_info: str | None,
        transaction_dt: float,
    ) -> None:
        """
        Record this transaction in the card's history. Call AFTER scoring
        and returning the response to avoid leakage.
        """
        state = self._store[card_id]
        state.amounts.append(amount)
        if device_info:
            state.devices.add(device_info)
        state.timestamps.append(transaction_dt)

    def card_count(self) -> int:
        """Number of distinct card_ids in the store."""
        return len(self._store)

    def clear(self) -> None:
        """Reset all state (useful for testing)."""
        self._store.clear()


# Singleton - imported by routers
feature_store = FeatureStore()
