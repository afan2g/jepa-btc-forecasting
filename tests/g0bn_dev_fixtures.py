"""Synthetic G0-BN development bundles for the 67-B candidate engine tests (issue #88).

Builds on the spec-literal 67-A fixtures (tests/g0bn_protocol_fixtures.py) but resolves
the environment-bearing sections to the INSTALLED runtime: candidate `package_version`
and the `software` section come from the actual venv (spec section 4.1 re-resolves and
compares them before fitting, so a hard-coded synthetic version would fail the runtime
drift gate that 67-B implements). Everything else stays synthetic: fake evidence
hashes, a reduced included-day window, seeded random rows with a planted signal. No
vendor data, no real market data, no January values.
"""
from __future__ import annotations

import os
import sys
import tempfile

import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import sklearn

from eval.g0bn_config import validate_protocol_config
from eval.g0bn_engine import DEV_SOURCE_MANIFEST_SCHEMA, g0bn_candidate_code_sha256
from eval.g0bn_ledger import G0BNLedger
from eval.g0bn_selection import g0bn_dsr_code_sha256, g0bn_pbo_code_sha256
from eval.hashing import hash_obj
from eval.manifest import manifest_sha256
from eval.matrix import RESERVED
from eval.writer import build_id_for, logical_row_sha256, ordered_manifest_columns
from tests.g0bn_protocol_fixtures import (
    DEV_DATASET_ID,
    FEATURE_REGISTRY,
    INSTRUMENT,
    dev_days,
    make_candidates,
    make_config,
    make_cv,
    make_exclusions,
    make_software,
    make_source_certification,
    sha_hex,
)

DAY_NS = 86_400_000_000_000
N_INCLUDED_DAYS = 24

# Synthetic Binance source-object evidence: the manifest's data-source entries and
# the config's development_source_manifest_sha256 pin are derived from the SAME
# constants, mirroring how a real config pins the certified #64/#68 evidence.
DEV_SOURCE_SHAS = {
    "binance_futures_l2_snapshot": sha_hex("src-l2-snapshot"),
    "binance_futures_l2_delta": sha_hex("src-l2-delta"),
    "binance_futures_trades": sha_hex("src-trades"),
}


def dev_source_manifest_sha256() -> str:
    return hash_obj({"schema": DEV_SOURCE_MANIFEST_SCHEMA,
                     "sources": {name: [sha] for name, sha in
                                 DEV_SOURCE_SHAS.items()}})


def durable_ledger() -> G0BNLedger:
    """A path-bound ledger on a fresh temp file (run_g0bn_development requires
    durability; the file lives outside the repo and is disposable test state)."""
    fd, path = tempfile.mkstemp(prefix="g0bn-test-ledger-", suffix=".json")
    os.close(fd)
    os.unlink(path)
    return G0BNLedger(path=path)


def included_days(n: int = N_INCLUDED_DAYS) -> list:
    return dev_days()[:n]


def dev_exclusions(n_included: int = N_INCLUDED_DAYS) -> dict:
    """Outcome-blind day accounting covering the FULL 61-day window: the first
    `n_included` days are in scope, every later day carries an explicit synthetic
    exclusion (keeps the bootstrap day list small for cheap tests)."""
    days = dev_days()
    excluded = {d: {"reason": "synthetic_out_of_scope",
                    "evidence_sha256": sha_hex(f"excl-{d}")}
                for d in days[n_included:]}
    return make_exclusions(included_days=days[:n_included], excluded_days=excluded)


def runtime_software(**over) -> dict:
    d = make_software(
        python_version=".".join(str(v) for v in sys.version_info[:3]),
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        scikit_learn_version=sklearn.__version__,
        lightgbm_version=lightgbm.__version__,
        pyarrow_version=pyarrow.__version__,
    )
    d.update(over)
    return d


def runtime_candidates() -> list:
    """67-A spec-literal candidate definitions with package_version AND the
    candidate implementation hash resolved to the installed runtime (the
    identity-bearing pins ARE the running environment's values)."""
    candidates = make_candidates()
    running_code = g0bn_candidate_code_sha256()
    for defn in candidates:
        defn["candidate_code_sha256"] = running_code
        if defn.get("package") == "scikit-learn":
            defn["package_version"] = sklearn.__version__
        elif defn.get("package") == "lightgbm":
            defn["package_version"] = lightgbm.__version__
    return candidates


def runtime_cv(**over) -> dict:
    cv = make_cv()
    cv["dsr"] = dict(cv["dsr"], code_sha256=g0bn_dsr_code_sha256())
    cv["pbo"] = dict(cv["pbo"], code_sha256=g0bn_pbo_code_sha256())
    cv.update(over)
    return cv


def dev_config(**over) -> dict:
    over.setdefault("candidates", runtime_candidates())
    over.setdefault("software", runtime_software())
    over.setdefault("exclusions", dev_exclusions())
    over.setdefault("cv", runtime_cv())
    over.setdefault("source_certification", make_source_certification(
        development_source_manifest_sha256=dev_source_manifest_sha256()))
    return validate_protocol_config(make_config(**over))


def _day_start_ns(day: str) -> int:
    return int(np.datetime64(day, "ns").astype(np.int64))


def dev_matrix(config: dict, *, rows_per_day: int = 10, seed: int = 11,
               signal_bps: float = 18.0, noise_bps: float = 2.0,
               days: list | None = None) -> pd.DataFrame:
    """Synthetic development ModelMatrix inside the frozen included days.

    The planted signal makes `microprice_dev` (and, attenuated, `ofi_integrated`)
    predictive of y_fwd_bps: with the default amplitude the raw microprice forecast
    clears the taker-cost band on most rows (trade-eligible scenarios); with a small
    `signal_bps` (e.g. 2.0) forecasts stay inside the band -> predictive-only
    scenarios. Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    fee = config["costs"]["cost_assumption"]["taker_fee_bps"]
    slip = config["costs"]["cost_assumption"]["base_slippage_bps"]
    days = list(config["exclusions"]["included_days"]) if days is None else list(days)
    horizons = [(h["tag"], h["ns"]) for h in config["horizons"]]
    parts = []
    for day in days:
        start = _day_start_ns(day)
        for tag, h_ns in horizons:
            n = rows_per_day
            step = DAY_NS // (n + 2)
            t_event = start + (np.arange(n, dtype=np.int64) + 1) * step
            df = pd.DataFrame(rng.standard_normal((n, len(FEATURE_REGISTRY))),
                              columns=list(FEATURE_REGISTRY))
            m = rng.standard_normal(n) * signal_bps
            y = m + rng.standard_normal(n) * noise_bps
            df["microprice_dev"] = m
            # Stationarized OFI proxy correlated with the same signal (unit scale).
            df["ofi_integrated"] = (m + rng.standard_normal(n) * noise_bps) / max(
                signal_bps, 1e-9)
            df["y_fwd_bps"] = y
            label = np.zeros(n, dtype=np.int64)
            label[y > 0.5 * signal_bps] = 1
            label[y < -0.5 * signal_bps] = -1
            df["label"] = label
            df["t_event"] = t_event
            df["t_barrier"] = t_event + h_ns
            df["t_feature_start"] = t_event - 60_000_000_000
            df["t_available"] = t_event
            drift = rng.uniform(0.0, 0.5, n)
            df["latency_drift_bps"] = drift
            df["cost_bps"] = 2.0 * fee + slip + drift
            df["half_spread_bps"] = rng.uniform(0.2, 1.0, n)
            df["uniqueness"] = rng.uniform(0.5, 1.0, n)
            df["regime"] = np.where(df["spread_tick"].to_numpy() > 0, "wide", "tight")
            df["horizon"] = tag
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def horizon_roles_sha256(config: dict) -> str:
    return hash_obj({h["tag"]: h["role"] for h in config["horizons"]})


def dev_manifest(config: dict, frame: pd.DataFrame, *,
                 build_params: dict | None = None) -> dict:
    """A T8-shaped development manifest whose bindings all reconcile with `config`
    and whose build_id derives from the frame's canonical logical rows."""
    cert = config["source_certification"]
    sources = [
        {"name": name, "sha256": sha} for name, sha in DEV_SOURCE_SHAS.items()
    ] + [
        {"name": "source_certification", "sha256": cert["certification_sha256"]},
        dict(config["costs"]["cost_assumption"], name="cost_assumption"),
        {"name": "partition_contract", "schema": "g0bn-partition-plan-v1",
         "partition": "development",
         "partition_plan_sha256": config["partition"]["sha256"]},
        {"name": "g0bn_protocol", "protocol": "g0bn-v1",
         "protocol_config_sha256": config["sha256"],
         "source_certification_sha256": cert["certification_sha256"],
         "horizon_roles_sha256": horizon_roles_sha256(config),
         "instrument": dict(INSTRUMENT)},
    ]
    man = {
        "manifest_version": 1,
        "dataset_id": DEV_DATASET_ID,
        "build_id": "0" * 64,
        "bar_clock": {
            "kind": "dollar",
            "reference_stream": "binance_futures_trades",
            "target_bars_per_day": config["clock"]["target_bars_per_day"],
            "time_cap_ns": config["clock"]["time_cap_ns"],
            "threshold_schedule_hash": config["clock"]["development_schedule_sha256"],
        },
        "time": {"unit": "ns", "timezone": "UTC"},
        "feature_cols": list(FEATURE_REGISTRY),
        "target_cols": ["y_fwd_bps", "label"],
        "reserved_cols": list(RESERVED),
        "extra_cols": ["latency_drift_bps"],
        "venues": [{"exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP"}],
        "horizons": {h["tag"]: h["ns"] for h in config["horizons"]},
        "sources": sources,
        "generated_at": "2026-07-16T00:00:00+00:00",
        "max_lookback_ns": config["features"]["max_lookback_ns"],
        "embargo_ns": config["cv"]["embargo_ns"],
        "availability_lag_ns": 0,
        "dtypes": {"cost_bps": "float64", "half_spread_bps": "float64",
                   "latency_drift_bps": "float64"},
    }
    lrh = logical_row_sha256(frame, ordered_manifest_columns(man))
    params = ({"producer": "tests/g0bn_dev_fixtures.py", "seed": 11}
              if build_params is None else dict(build_params))
    man["build_id"] = build_id_for(dataset_id=man["dataset_id"],
                                   logical_row_sha256=lrh, build_params=params)
    return man


def dev_data_identity(config: dict, manifest: dict, frame: pd.DataFrame) -> dict:
    return {
        "development_dataset_id": manifest["dataset_id"],
        "development_build_id": manifest["build_id"],
        "development_manifest_sha256": manifest_sha256(manifest),
        "development_logical_row_sha256": logical_row_sha256(
            frame, ordered_manifest_columns(manifest)),
        "partition_plan_sha256": config["partition"]["sha256"],
    }


def dev_bundle(*, rows_per_day: int = 10, seed: int = 11, signal_bps: float = 18.0,
               noise_bps: float = 2.0, config: dict | None = None):
    """(frame, manifest, config, data_identity) with every cross-artifact hash bound."""
    config = dev_config() if config is None else config
    frame = dev_matrix(config, rows_per_day=rows_per_day, seed=seed,
                       signal_bps=signal_bps, noise_bps=noise_bps)
    manifest = dev_manifest(config, frame)
    return frame, manifest, config, dev_data_identity(config, manifest, frame)
