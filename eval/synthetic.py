"""Deterministic synthetic ModelMatrix with a KNOWN, tunable signal and the full
reserved-column contract (cost, uniqueness, timing/availability, regime). Also builds the
synthetic staged-pilot world (matched G0 arms + partition contract) issue #52's evaluator
and its tests consume — the REAL producer is issue #37; these fixtures implement the same
partition/manifest contracts on synthetic data only."""
from __future__ import annotations
import datetime as _dt

import numpy as np
import pandas as pd

from eval.manifest import MANIFEST_VERSION
from eval.matrix import RESERVED

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


def make_manifest(feature_cols, max_lookback_ns, *, gate=None, **over):
    """A schema-valid v1 feature manifest mirroring make_matrix ("10s" horizon tag with
    duration = max_lookback_ns — the generator sets lookback == horizon_ns, so this holds
    for any horizon_ns override too; embargo = look-back). Test/exploration helper — real
    builds write their own manifest. Override horizons via **over for multi-horizon or
    custom-tag manifests."""
    man = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": "synthetic",
        "build_id": "seeded",
        "bar_clock": {"kind": "synthetic"},
        "time": {"unit": "ns", "timezone": "UTC"},
        "feature_cols": list(feature_cols),
        "target_cols": ["y_fwd_bps", "label"],
        "reserved_cols": list(RESERVED),
        "venues": [{"exchange": "SYNTHETIC", "symbol": "BTC-TEST"}],
        "horizons": {"10s": int(max_lookback_ns)},
        "sources": ["eval/synthetic.py"],
        "generated_at": "2026-07-02T00:00:00+00:00",
        "max_lookback_ns": int(max_lookback_ns),
        "embargo_ns": int(max_lookback_ns),
    }
    if gate is not None:
        man["gate"] = gate
    man.update(over)
    return man


# --------------------------------------------------------------------- G0 pilot world
# Synthetic staged-pilot fixtures for the #52 evaluator: span-safe development + holdout
# partitions, a hash-pinned partition contract with REAL per-horizon boundary-drop counts
# (the generator applies the conservative prefilter itself), and matched arm builds whose
# reserved columns are identical while feature_cols differ.

G0_CB_FEATURES = ["cb_ofi", "cb_microprice_dev", "cb_queue_imb", "cb_spread_tick", "cb_cvd"]
G0_BN_FEATURES = ["bn_ofi_lag", "bn_cvd_lag", "bn_basis"]

G0_DEV_START = "2025-11-01T00:00:00+00:00"
G0_HOLDOUT_START = "2026-04-01T00:00:00+00:00"
G0_HOLDOUT_END = "2026-05-01T00:00:00+00:00"


def _iso_ns(iso: str) -> int:
    return int(_dt.datetime.fromisoformat(iso).timestamp()) * 1_000_000_000


def _g0_partition_rows(rng, *, lo_ns: int, boundary_ns: int, n_bars: int, horizons: dict,
                       guard_ns: int, cb_signal: float, bn_signal: float,
                       noise_bps: float):
    """One partition: an even t_event grid over [lo_ns, boundary_ns) run through the
    conservative prefilter (t_event + horizon + guard < boundary), returning the surviving
    bar x horizon rows plus the per-horizon drop counts the contract records."""
    step = (boundary_ns - lo_ns) // (n_bars + 1)
    t_event = lo_ns + (np.arange(n_bars, dtype=np.int64) + 1) * step
    Xcb = rng.standard_normal((n_bars, len(G0_CB_FEATURES)))
    Xbn = rng.standard_normal((n_bars, len(G0_BN_FEATURES)))
    f_cb = Xcb[:, 0] * 1.0 + np.tanh(Xcb[:, 1]) * 1.5 + (Xcb[:, 2] > 0.5) * Xcb[:, 3]
    f_bn = Xbn[:, 0] * 1.0 + np.tanh(Xbn[:, 1]) * 1.2
    regime = np.where(Xcb[:, 3] > 0, "tight", "wide")

    frames, drops = [], {}
    for tag in sorted(horizons):
        h_ns = horizons[tag]
        keep = t_event + h_ns + guard_ns < boundary_ns
        drops[tag] = int((~keep).sum())
        y = (cb_signal * f_cb[keep] + bn_signal * f_bn[keep]
             + rng.standard_normal(int(keep.sum())) * noise_bps)
        te = t_event[keep]
        df = pd.DataFrame(Xcb[keep], columns=G0_CB_FEATURES)
        for j, c in enumerate(G0_BN_FEATURES):
            df[c] = Xbn[keep, j]
        df["y_fwd_bps"] = y
        df["label"] = np.sign(y).astype(int)
        df["t_event"] = te
        df["t_barrier"] = te + h_ns
        df["t_feature_start"] = te - max(horizons.values())
        df["t_available"] = te
        df["cost_bps"] = np.where(regime[keep] == "wide", 4.0, 1.5)
        df["half_spread_bps"] = np.where(regime[keep] == "wide", 2.0, 0.6)
        df["uniqueness"] = _concurrency_uniqueness(te, te + h_ns)
        df["regime"] = regime[keep]
        df["horizon"] = tag
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["t_event", "horizon"], kind="mergesort").reset_index(drop=True), drops


def make_g0_contract(*, horizons: dict, guard_ns: int, drop_counts: dict,
                     dev_start: str = G0_DEV_START, holdout_start: str = G0_HOLDOUT_START,
                     holdout_end: str = G0_HOLDOUT_END) -> dict:
    from eval.partition import PARTITION_CONTRACT_VERSION, PREFILTER_RULE
    return {
        "partition_contract_version": PARTITION_CONTRACT_VERSION,
        "rule_version": "span-safe-v1",
        "prefilter_rule": PREFILTER_RULE,
        "dev_start_ns": _iso_ns(dev_start),
        "holdout_start_ns": _iso_ns(holdout_start),
        "holdout_end_ns": _iso_ns(holdout_end),
        "guard_ns": int(guard_ns),
        "horizons": dict(horizons),
        "boundary_drop_counts": drop_counts,
        "generated_at": "2026-07-10T00:00:00+00:00",
    }


def g0_binding(contract: dict, partition: str) -> dict:
    """The manifest `sources` entry that pins a build to one contract partition."""
    from eval.partition import contract_hash
    return {"name": "partition_contract", "sha256": contract_hash(contract),
            "partition": partition,
            "boundary_drop_counts": dict(contract["boundary_drop_counts"][partition])}


def make_g0_manifest(arm: str, feature_cols, *, contract: dict, partition: str,
                     dataset_id: str, build_id: str, gate=None, **over) -> dict:
    lookback = max(contract["horizons"].values())
    venues = {
        "coinbase_only": [{"exchange": "COINBASE", "symbol": "BTC-USD", "role": "target"}],
        "binance_only": [{"exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP",
                          "role": "signal"},
                         {"exchange": "COINBASE", "symbol": "BTC-USD", "role": "target"}],
        "combined": [{"exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP",
                      "role": "signal"},
                     {"exchange": "COINBASE", "symbol": "BTC-USD", "role": "target"}],
    }[arm]
    man = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": dataset_id,
        "build_id": build_id,
        "bar_clock": {"kind": "synthetic"},
        "time": {"unit": "ns", "timezone": "UTC"},
        "feature_cols": list(feature_cols),
        "target_cols": ["y_fwd_bps", "label"],
        "reserved_cols": list(RESERVED),
        "venues": venues,
        "horizons": dict(contract["horizons"]),
        "sources": ["eval/synthetic.py", g0_binding(contract, partition)],
        "generated_at": "2026-07-10T00:00:00+00:00",
        "max_lookback_ns": int(lookback),
        "embargo_ns": int(lookback),
    }
    if gate is not None:
        man["gate"] = gate
    man.update(over)
    return man


def make_g0_world(*, n_dev_bars: int = 400, n_holdout_bars: int = 120,
                  cb_signal: float = 4.0, bn_signal: float = 4.0, seed: int = 0,
                  noise_bps: float = 8.0, horizons: dict | None = None,
                  guard_ns: int = 60_000_000_000, dataset_id: str = "synthetic-xv-pilot"):
    """The full synthetic staged-pilot world: partition contract + matched development and
    holdout builds for the coinbase_only / binance_only / combined arms. Reserved columns
    are IDENTICAL across arms within each partition (one generation, sliced per arm);
    only feature_cols differ. `cb_signal`/`bn_signal` tune which arms carry edge."""
    horizons = dict(horizons or {"10s": 10_000_000_000})
    rng = np.random.default_rng(seed)
    dev_rows, dev_drops = _g0_partition_rows(
        rng, lo_ns=_iso_ns(G0_DEV_START), boundary_ns=_iso_ns(G0_HOLDOUT_START),
        n_bars=n_dev_bars, horizons=horizons, guard_ns=guard_ns,
        cb_signal=cb_signal, bn_signal=bn_signal, noise_bps=noise_bps)
    hold_rows, hold_drops = _g0_partition_rows(
        rng, lo_ns=_iso_ns(G0_HOLDOUT_START), boundary_ns=_iso_ns(G0_HOLDOUT_END),
        n_bars=n_holdout_bars, horizons=horizons, guard_ns=guard_ns,
        cb_signal=cb_signal, bn_signal=bn_signal, noise_bps=noise_bps)
    contract = make_g0_contract(horizons=horizons, guard_ns=guard_ns,
                                drop_counts={"development": dev_drops,
                                             "holdout": hold_drops})
    arm_feats = {"coinbase_only": list(G0_CB_FEATURES),
                 "binance_only": list(G0_BN_FEATURES),
                 "combined": list(G0_CB_FEATURES) + list(G0_BN_FEATURES)}

    def _arms(rows: pd.DataFrame, partition: str, build_id: str) -> dict:
        arms = {}
        for arm, feats in arm_feats.items():
            man = make_g0_manifest(arm, feats, contract=contract, partition=partition,
                                   dataset_id=dataset_id,
                                   build_id=f"{build_id}-{arm}")
            arms[arm] = {"manifest": man,
                         "matrix": rows[feats + list(RESERVED)].copy()}
        return arms

    holdout_days = sorted(pd.to_datetime(hold_rows["t_event"], unit="ns", utc=True)
                          .dt.strftime("%Y-%m-%d").unique().tolist())
    return {
        "contract": contract,
        "horizons": horizons,
        "dev": {"rows": dev_rows, "arms": _arms(dev_rows, "development", "dev-seeded"),
                "drop_counts": dev_drops},
        "holdout": {"rows": hold_rows,
                    "arms": _arms(hold_rows, "holdout", "holdout-seeded"),
                    "drop_counts": hold_drops},
        "holdout_days": holdout_days,
        "arm_features": arm_feats,
    }
