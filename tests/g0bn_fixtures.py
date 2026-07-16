"""Synthetic G0-BN manifest/frame builders for the T8 writer tests (issue #87).

Everything here is synthetic: fake 64-hex pins, an arbitrary NON-production fee tier,
and seeded random rows. Real pins come from #64/#68 evidence and the #69 operator;
tests only exercise the binding/writer contracts in
docs/superpowers/specs/2026-07-13-g0bn-protocol.md sections 2 and 7."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bars.cost import DRIFT_POLICY
from eval.matrix import RESERVED
from eval.writer import (
    G0BN_DEV_DATASET_ID,
    G0BN_FEATURES,
    G0BN_HORIZONS_NS,
    G0BN_INSTRUMENT,
    G0BN_OOS_DATASET_ID,
    build_id_for,
    logical_row_sha256,
    ordered_manifest_columns,
)

# Synthetic fee tier for tests only — a real freeze requires evidenced operator values
# (spec section 12); these numbers must never be copied into a production config.
FEE_BPS = 1.7
SLIP_BPS = 0.9

MAX_LOOKBACK_NS = 120_000_000_000


def hex64(i: int) -> str:
    """Deterministic fake SHA-256 pin (64 lowercase hex chars)."""
    return f"{i:064x}"


def partition_binding(partition: str = "development") -> dict:
    return {
        "name": "partition_contract",
        "schema": "g0bn-partition-plan-v1",
        "partition": partition,
        "partition_plan_sha256": hex64(1),
    }


def protocol_binding() -> dict:
    return {
        "name": "g0bn_protocol",
        "protocol": "g0bn-v1",
        "protocol_config_sha256": hex64(2),
        "source_certification_sha256": hex64(3),
        "horizon_roles_sha256": hex64(4),
        "instrument": dict(G0BN_INSTRUMENT),
    }


def holdout_plan_binding() -> dict:
    return {
        "name": "g0bn_holdout_plan",
        "protocol": "g0bn-one-shot-v1",
        "consumption_schema": "g0bn-consumption-v1",
        "holdout_universe_id": hex64(5),
        "transaction_id": hex64(6),
        "holdout_plan_sha256": hex64(7),
        "freeze_sha256": hex64(8),
    }


def cost_assumption_source() -> dict:
    return {
        "name": "cost_assumption",
        "venue": "binance",
        "product": "BTC-USDT-PERP",
        "source": "binance_futures_l2_delta/normalized-v1",
        "version": "g0bn-test-v1",
        "taker_fee_bps": FEE_BPS,
        "base_slippage_bps": SLIP_BPS,
        "drift_policy": DRIFT_POLICY,
    }


def g0bn_sources(partition: str = "development") -> list[dict]:
    src = [
        {"name": "binance_futures_l2_snapshot", "sha256": hex64(9)},
        {"name": "binance_futures_l2_delta", "sha256": hex64(10)},
        {"name": "binance_futures_trades", "sha256": hex64(11)},
        {"name": "source_certification", "sha256": hex64(12)},
        cost_assumption_source(),
        partition_binding(partition),
        protocol_binding(),
    ]
    if partition == "holdout":
        src.append({"name": "custodian_seal", "sha256": hex64(13)})
        src.append(holdout_plan_binding())
    return src


def g0bn_manifest(*, partition: str = "development", **over) -> dict:
    dataset_id = G0BN_DEV_DATASET_ID if partition == "development" else G0BN_OOS_DATASET_ID
    man = {
        "manifest_version": 1,
        "dataset_id": dataset_id,
        "build_id": hex64(99),
        "bar_clock": {
            "kind": "dollar",
            "reference_stream": "binance_futures_trades",
            "target_bars_per_day": 500,
            "time_cap_ns": 5_000_000_000,
            "warmup_days": 2,
            "threshold_schedule": "schedules/g0bn-dev-v1.json",
            "threshold_schedule_hash": hex64(14),
            "feed_lag_tail_ns": 250_000_000,
            "coverage_policy": "g0bn-coverage-v1",
        },
        "time": {"unit": "ns", "timezone": "UTC"},
        "feature_cols": list(G0BN_FEATURES),
        "target_cols": ["y_fwd_bps", "label"],
        "reserved_cols": list(RESERVED),
        "extra_cols": ["latency_drift_bps"],
        "venues": [{"exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP"}],
        "horizons": dict(G0BN_HORIZONS_NS),
        "sources": g0bn_sources(partition),
        "generated_at": "2026-07-15T00:00:00+00:00",
        "max_lookback_ns": MAX_LOOKBACK_NS,
        "embargo_ns": MAX_LOOKBACK_NS,
        "availability_lag_ns": 0,
        "dtypes": {
            "cost_bps": "float64",
            "half_spread_bps": "float64",
            "latency_drift_bps": "float64",
        },
    }
    man.update(over)
    return man


def g0bn_frame(*, rows_per_horizon: int = 5, seed: int = 7) -> pd.DataFrame:
    """One valid ModelMatrix row block per G0-BN horizon; cost identity holds exactly:
    cost_bps = 2*FEE_BPS + SLIP_BPS + latency_drift_bps (T7 / spec section 8.2)."""
    rng = np.random.default_rng(seed)
    parts = []
    base = 1_700_000_000_000_000_000
    for tag, h_ns in G0BN_HORIZONS_NS.items():
        n = rows_per_horizon
        t_event = base + (np.arange(n, dtype=np.int64) + 1) * 3_000_000_000
        df = pd.DataFrame(
            rng.standard_normal((n, len(G0BN_FEATURES))), columns=list(G0BN_FEATURES))
        y = rng.standard_normal(n) * 5.0
        drift = rng.uniform(0.0, 0.5, n)
        df["y_fwd_bps"] = y
        df["label"] = np.sign(y).astype(np.int64)
        df["t_event"] = t_event
        df["t_barrier"] = t_event + h_ns // 2
        df["t_feature_start"] = t_event - MAX_LOOKBACK_NS // 2
        df["t_available"] = t_event
        df["cost_bps"] = 2.0 * FEE_BPS + SLIP_BPS + drift
        df["half_spread_bps"] = rng.uniform(0.2, 1.0, n)
        df["uniqueness"] = rng.uniform(0.2, 1.0, n)
        df["regime"] = np.where(df["spread_tick"] > 0, "wide", "tight")
        df["horizon"] = tag
        df["latency_drift_bps"] = drift
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_params(partition: str = "development") -> dict:
    params = {
        "producer": "tests/g0bn_fixtures.py",
        "threshold_schedule_sha256": hex64(14),
        "seed": 7,
    }
    if partition == "holdout":
        params["holdout_plan_sha256"] = hex64(7)
    return params


def built_g0bn(*, partition: str = "development", frame: pd.DataFrame | None = None,
               params: dict | None = None):
    """(frame, manifest, build_params) with the manifest's build_id correctly derived
    from the frame's canonical logical rows + the build params."""
    frame = g0bn_frame() if frame is None else frame
    params = build_params(partition) if params is None else params
    man = g0bn_manifest(partition=partition)
    lrh = logical_row_sha256(frame, ordered_manifest_columns(man))
    man["build_id"] = build_id_for(
        dataset_id=man["dataset_id"], logical_row_sha256=lrh, build_params=params)
    return frame, man, params
