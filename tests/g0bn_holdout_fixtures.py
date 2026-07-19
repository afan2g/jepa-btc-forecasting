"""Synthetic outcome-blind custody/holdout fixtures for the 67-C tests (issue #89).

Everything here is synthetic and outcome-blind: opaque object ids, deterministic
fake checksums, and UTC-day accounting only — no vendor data, no January payloads,
no prices/sizes/returns, and no activity proxies (byte sizes, record counts).
The custody metadata defaults are consistent with tests.g0bn_dev_fixtures.dev_config()
(the same synthetic #68 evidence hashes from tests.g0bn_protocol_fixtures), mirroring
how a real inventory must reconcile with the config's source_certification pins.

Day/product literals are spec-literal (docs/superpowers/specs/2026-07-13-g0bn-protocol.md
sections 2.1, 5.1-5.2), NOT imported from the modules under test, so the tests fail
if the implementation constants drift from the binding contract.
"""
from __future__ import annotations

import datetime as _dt

from eval.matrix import RESERVED
from eval.writer import build_id_for, logical_row_sha256, ordered_manifest_columns
from tests.g0bn_protocol_fixtures import (
    INSTRUMENT,
    L2_DELTA_PRODUCT,
    L2_SNAPSHOT_PRODUCT,
    TRADE_PRODUCT,
    sha_hex,
)

# Spec-literal layer/product taxonomy: raw objects carry the certified native vendor
# product ids; normalized objects carry the T8 writer's normalized source names.
RAW_PRODUCTS = (L2_SNAPSHOT_PRODUCT, L2_DELTA_PRODUCT, TRADE_PRODUCT)
NORMALIZED_PRODUCTS = ("binance_futures_l2_snapshot", "binance_futures_l2_delta",
                       "binance_futures_trades")

DEFAULT_EXCLUDED_DAYS = ("2026-01-14", "2026-01-25")

OOS_DATASET_ID = "binance_single_venue_g0bn_oos"


def january_days() -> list:
    """All 31 UTC holdout days, 2026-01-01 .. 2026-01-31 (spec section 2.2)."""
    start = _dt.date(2026, 1, 1)
    return [(start + _dt.timedelta(days=i)).isoformat() for i in range(31)]


def make_objects(included_days) -> list:
    """One sealed object per (day, layer, product): 3 raw + 3 normalized per day."""
    objects = []
    for day in included_days:
        for layer, products in (("raw", RAW_PRODUCTS),
                                ("normalized", NORMALIZED_PRODUCTS)):
            for product in products:
                objects.append({
                    "object_id": f"custody/{layer}/{product}/{day}",
                    "layer": layer,
                    "product": product,
                    "day": day,
                    "sha256": sha_hex(f"jan-obj-{layer}-{product}-{day}"),
                })
    return objects


def make_inventory(**over) -> dict:
    """Outcome-blind custodian inventory/seal metadata consistent with dev_config():
    29 included days, 2 explicitly excluded days, and a complete object allowlist."""
    days = january_days()
    excluded = {
        "2026-01-14": {"reason": "custody_source_gap",
                       "evidence_sha256": sha_hex("jan-gap-2026-01-14")},
        "2026-01-25": {"reason": "custody_one_sided_book",
                       "evidence_sha256": sha_hex("jan-book-2026-01-25")},
    }
    included = [d for d in days if d not in excluded]
    inv = {
        "custodian_seal_sha256": sha_hex("custodian-seal"),
        "coverage_sha256": sha_hex("coverage"),
        "permission_policy_sha256": sha_hex("permission-policy"),
        "custodian_identity": "g0bn-custodian-svc",
        "operator_identity": "g0bn-operator-dev",
        "included_days": included,
        "excluded_days": excluded,
        "objects": make_objects(included),
    }
    inv.update(over)
    return inv


def oos_manifest_and_params(config: dict, plan: dict, freeze: dict,
                            frame) -> tuple:
    """A T8-shaped BLIND-HOLDOUT manifest whose bindings all derive from the real
    plan/freeze artifacts, plus the oos build params that bind holdout_plan_sha256.
    The frame is synthetic (out-of-window timestamps); write_holdout performs only
    structural checks, so this exercises the future-build binding without any
    January semantics."""
    from eval.g0bn_freeze import holdout_plan_binding, oos_build_params

    cert = config["source_certification"]

    def _normalized_sha(product: str) -> str:
        return next(o["sha256"] for o in plan["object_allowlist"]
                    if o["layer"] == "normalized" and o["product"] == product)

    sources = [
        {"name": name, "sha256": _normalized_sha(name)}
        for name in NORMALIZED_PRODUCTS
    ] + [
        {"name": "source_certification", "sha256": cert["certification_sha256"]},
        {"name": "custodian_seal", "sha256": cert["custodian_seal_sha256"]},
        dict(config["costs"]["cost_assumption"], name="cost_assumption"),
        {"name": "partition_contract", "schema": "g0bn-partition-plan-v1",
         "partition": "holdout",
         "partition_plan_sha256": config["partition"]["sha256"]},
        {"name": "g0bn_protocol", "protocol": "g0bn-v1",
         "protocol_config_sha256": config["sha256"],
         "source_certification_sha256": cert["certification_sha256"],
         "horizon_roles_sha256": plan["output_contract"]["expected_bindings"][
             "g0bn_protocol"]["horizon_roles_sha256"],
         "instrument": dict(INSTRUMENT)},
        holdout_plan_binding(plan, freeze, config=config),
    ]
    man = {
        "manifest_version": 1,
        "dataset_id": OOS_DATASET_ID,
        "build_id": "0" * 64,
        "bar_clock": {
            "kind": "dollar",
            "reference_stream": "binance_futures_trades",
            "target_bars_per_day": config["clock"]["target_bars_per_day"],
            "time_cap_ns": config["clock"]["time_cap_ns"],
            "threshold_schedule_hash": config["clock"]["development_schedule_sha256"],
        },
        "time": {"unit": "ns", "timezone": "UTC"},
        "feature_cols": list(plan["output_contract"]["feature_cols"]),
        "target_cols": ["y_fwd_bps", "label"],
        "reserved_cols": list(RESERVED),
        "extra_cols": list(plan["output_contract"]["extra_cols"]),
        "venues": [{"exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP"}],
        "horizons": dict(plan["output_contract"]["horizons"]),
        "sources": sources,
        "generated_at": "2026-07-19T00:00:00+00:00",
        "max_lookback_ns": config["features"]["max_lookback_ns"],
        "embargo_ns": config["cv"]["embargo_ns"],
        "availability_lag_ns": 0,
        "dtypes": dict(plan["output_contract"]["dtypes"]),
    }
    params = oos_build_params(plan, {"producer": "tests/g0bn_holdout_fixtures.py",
                                     "seed": 7}, config=config)
    lrh = logical_row_sha256(frame, ordered_manifest_columns(man))
    man["build_id"] = build_id_for(dataset_id=man["dataset_id"],
                                   logical_row_sha256=lrh, build_params=params)
    return man, params
