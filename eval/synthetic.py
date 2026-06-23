"""Deterministic synthetic ModelMatrix with a KNOWN, tunable signal and the full
reserved-column contract (cost, uniqueness, timing/availability, regime)."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["ofi_integrated", "microprice_dev", "queue_imb", "spread_tick", "cvd"]


def _concurrency_uniqueness(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """uniqueness_i = 1 / (# label spans covering t_event_i)."""
    t0s = np.sort(t0); t1s = np.sort(t1)
    started = np.searchsorted(t0s, t0, side="right")
    ended = np.searchsorted(t1s, t0, side="right")
    conc = np.maximum(started - ended, 1)
    return 1.0 / conc


def make_matrix(n: int = 8000, *, signal_strength: float, seed: int,
                horizon_ns: int = 10_000_000_000, noise_bps: float = 8.0,
                latency_ns: int = 50_000_000):
    """Returns (df, feature_cols, max_lookback_ns)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURES)))
    f = X[:, 0] * 1.0 + np.tanh(X[:, 1]) * 1.5 + (X[:, 2] > 0.5) * X[:, 3]
    y = signal_strength * f + rng.standard_normal(n) * noise_bps
    step = horizon_ns // 4                       # overlapping labels (concurrency ~4)
    t_event = (np.arange(n, dtype=np.int64) + 1) * step
    t_barrier = t_event + horizon_ns
    lookback = horizon_ns                        # feature window
    regime = np.where(X[:, 3] > 0, "tight", "wide")
    df = pd.DataFrame(X, columns=FEATURES)
    df["y_fwd_bps"] = y
    df["label"] = np.sign(y).astype(int)
    df["t_event"] = t_event
    df["t_barrier"] = t_barrier
    df["t_feature_start"] = t_event - lookback
    df["t_available"] = t_event  # synchronous baseline: latency handled upstream by lagging features
    df["cost_bps"] = np.where(regime == "wide", 4.0, 1.5)
    df["half_spread_bps"] = np.where(regime == "wide", 2.0, 0.6)
    df["uniqueness"] = _concurrency_uniqueness(t_event, t_barrier)
    df["regime"] = regime
    df["horizon"] = "10s"
    return df, list(FEATURES), int(lookback)
