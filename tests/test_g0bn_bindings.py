"""G0-BN manifest binding + classification contract (T8, issue #87).

Pins spec sections 2 and 7 of docs/superpowers/specs/2026-07-13-g0bn-protocol.md:
exactly one partition_contract and one g0bn_protocol source binding, holdout-only
g0bn_holdout_plan, the exact single-venue template, and the manifest-only preflight
classification 67-D consumes before any parquet loader."""
from __future__ import annotations

import pytest

from eval.synthetic import make_manifest, make_matrix
from eval.writer import (
    G0BN_DEV_DATASET_ID,
    G0BN_FEATURES,
    G0BN_OOS_DATASET_ID,
    classify_manifest,
    validate_g0bn_manifest,
)
from tests.g0bn_fixtures import (
    g0bn_manifest,
    g0bn_sources,
    holdout_plan_binding,
    partition_binding,
    protocol_binding,
)


def _drop_source(man, name):
    man["sources"] = [s for s in man["sources"] if s.get("name") != name]
    return man


def _get_source(man, name):
    return next(s for s in man["sources"] if s.get("name") == name)


# ---------------------------------------------------------------- happy paths

def test_valid_development_manifest_passes():
    man = g0bn_manifest()
    assert validate_g0bn_manifest(man) is man


def test_valid_holdout_manifest_passes():
    man = g0bn_manifest(partition="holdout")
    assert validate_g0bn_manifest(man) is man


# ---------------------------------------------------------------- binding cardinality

@pytest.mark.parametrize("name", ["partition_contract", "g0bn_protocol"])
def test_missing_binding_fails(name):
    man = _drop_source(g0bn_manifest(), name)
    with pytest.raises(ValueError, match="exactly one"):
        validate_g0bn_manifest(man)


@pytest.mark.parametrize("binding", [partition_binding(), protocol_binding()])
def test_duplicate_binding_fails(binding):
    man = g0bn_manifest()
    man["sources"].append(dict(binding))
    with pytest.raises(ValueError, match="exactly one"):
        validate_g0bn_manifest(man)


def test_holdout_plan_binding_on_development_fails():
    man = g0bn_manifest()
    man["sources"].append(holdout_plan_binding())
    with pytest.raises(ValueError, match="g0bn_holdout_plan"):
        validate_g0bn_manifest(man)


def test_holdout_without_plan_binding_fails():
    man = _drop_source(g0bn_manifest(partition="holdout"), "g0bn_holdout_plan")
    with pytest.raises(ValueError, match="g0bn_holdout_plan"):
        validate_g0bn_manifest(man)


def test_holdout_duplicate_plan_binding_fails():
    man = g0bn_manifest(partition="holdout")
    man["sources"].append(holdout_plan_binding())
    with pytest.raises(ValueError, match="exactly one"):
        validate_g0bn_manifest(man)


# ---------------------------------------------------------------- binding fields

@pytest.mark.parametrize("field,value", [
    ("schema", "g0bn-partition-plan-v2"),
    ("partition", "dev"),
    ("partition_plan_sha256", "ABC123"),
    ("partition_plan_sha256", "ff" * 31),
])
def test_partition_binding_bad_field_fails(field, value):
    man = g0bn_manifest()
    _get_source(man, "partition_contract")[field] = value
    with pytest.raises(ValueError):
        validate_g0bn_manifest(man)


def test_partition_binding_missing_field_fails():
    man = g0bn_manifest()
    del _get_source(man, "partition_contract")["partition_plan_sha256"]
    with pytest.raises(ValueError, match="partition_contract"):
        validate_g0bn_manifest(man)


def test_partition_binding_unknown_key_fails():
    man = g0bn_manifest()
    _get_source(man, "partition_contract")["extra"] = "x"
    with pytest.raises(ValueError, match="partition_contract"):
        validate_g0bn_manifest(man)


@pytest.mark.parametrize("field,value", [
    ("protocol", "g0xv-v1"),
    ("protocol_config_sha256", "nothex"),
    ("source_certification_sha256", ""),
    ("horizon_roles_sha256", "F" * 64),
])
def test_protocol_binding_bad_field_fails(field, value):
    man = g0bn_manifest()
    _get_source(man, "g0bn_protocol")[field] = value
    with pytest.raises(ValueError):
        validate_g0bn_manifest(man)


def test_protocol_binding_wrong_instrument_fails():
    man = g0bn_manifest()
    _get_source(man, "g0bn_protocol")["instrument"]["symbol"] = "ETH-USDT-PERP"
    with pytest.raises(ValueError, match="instrument"):
        validate_g0bn_manifest(man)


def test_protocol_binding_instrument_missing_key_fails():
    man = g0bn_manifest()
    del _get_source(man, "g0bn_protocol")["instrument"]["settlement_asset"]
    with pytest.raises(ValueError, match="instrument"):
        validate_g0bn_manifest(man)


def test_protocol_binding_instrument_extra_key_fails():
    man = g0bn_manifest()
    _get_source(man, "g0bn_protocol")["instrument"]["margin"] = "cross"
    with pytest.raises(ValueError, match="instrument"):
        validate_g0bn_manifest(man)


@pytest.mark.parametrize("field,value", [
    ("protocol", "g0bn-v1"),
    ("consumption_schema", "g0bn-consumption-v2"),
    ("holdout_universe_id", "zz" * 32),
    ("transaction_id", ""),
    ("freeze_sha256", "short"),
])
def test_holdout_plan_binding_bad_field_fails(field, value):
    man = g0bn_manifest(partition="holdout")
    _get_source(man, "g0bn_holdout_plan")[field] = value
    with pytest.raises(ValueError):
        validate_g0bn_manifest(man)


def test_holdout_plan_binding_missing_field_fails():
    man = g0bn_manifest(partition="holdout")
    del _get_source(man, "g0bn_holdout_plan")["transaction_id"]
    with pytest.raises(ValueError, match="g0bn_holdout_plan"):
        validate_g0bn_manifest(man)


# ---------------------------------------------------------------- exact template

def test_second_venue_fails():
    man = g0bn_manifest()
    man["venues"].append({"exchange": "COINBASE", "symbol": "BTC-USD"})
    with pytest.raises(ValueError, match="venue"):
        validate_g0bn_manifest(man)


def test_wrong_venue_fails():
    man = g0bn_manifest(venues=[{"exchange": "BINANCE", "symbol": "BTC-USDT"}])
    with pytest.raises(ValueError, match="venue"):
        validate_g0bn_manifest(man)


def test_venue_role_fails():
    man = g0bn_manifest(venues=[
        {"exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP", "role": "signal"}])
    with pytest.raises(ValueError, match="venue"):
        validate_g0bn_manifest(man)


def test_permuted_feature_order_fails():
    feats = list(G0BN_FEATURES)
    feats[0], feats[1] = feats[1], feats[0]
    with pytest.raises(ValueError, match="feature"):
        validate_g0bn_manifest(g0bn_manifest(feature_cols=feats))


def test_missing_feature_fails():
    with pytest.raises(ValueError, match="feature"):
        validate_g0bn_manifest(g0bn_manifest(feature_cols=list(G0BN_FEATURES)[:-1]))


def test_extra_feature_fails():
    with pytest.raises(ValueError, match="feature"):
        validate_g0bn_manifest(
            g0bn_manifest(feature_cols=list(G0BN_FEATURES) + ["basis_binance_coinbase"]))


def test_permuted_target_order_fails():
    with pytest.raises(ValueError, match="target"):
        validate_g0bn_manifest(g0bn_manifest(target_cols=["label", "y_fwd_bps"]))


@pytest.mark.parametrize("horizons", [
    {"2s": 2_000_000_000, "10s": 10_000_000_000},                            # missing 60s
    {"2s": 2_000_000_000, "10s": 10_000_000_000, "60s": 60_000_000_000,
     "300s": 300_000_000_000},                                               # extra rung
    {"2s": 2_000_000_000, "10s": 10_000_000_000, "60s": 61_000_000_000},     # wrong ns
])
def test_wrong_horizon_map_fails(horizons):
    with pytest.raises(ValueError, match="horizon"):
        validate_g0bn_manifest(g0bn_manifest(horizons=horizons))


def test_missing_latency_drift_extra_col_fails():
    with pytest.raises(ValueError, match="latency_drift_bps"):
        validate_g0bn_manifest(g0bn_manifest(extra_cols=[]))


def test_missing_latency_drift_dtype_pin_fails():
    with pytest.raises(ValueError, match="latency_drift_bps"):
        validate_g0bn_manifest(g0bn_manifest(
            dtypes={"cost_bps": "float64", "half_spread_bps": "float64"}))


def test_float32_cost_dtype_pin_fails():
    with pytest.raises(ValueError, match="float64"):
        validate_g0bn_manifest(g0bn_manifest(dtypes={
            "cost_bps": "float32", "half_spread_bps": "float64",
            "latency_drift_bps": "float64"}))


def test_missing_dtypes_map_fails():
    man = g0bn_manifest()
    del man["dtypes"]
    with pytest.raises(ValueError, match="dtypes"):
        validate_g0bn_manifest(man)


def test_embargo_not_equal_lookback_fails():
    man = g0bn_manifest(embargo_ns=130_000_000_000)
    with pytest.raises(ValueError, match="embargo"):
        validate_g0bn_manifest(man)


def test_nonzero_availability_lag_fails():
    with pytest.raises(ValueError, match="availability_lag_ns"):
        validate_g0bn_manifest(g0bn_manifest(availability_lag_ns=1))


def test_non_hex_build_id_fails():
    with pytest.raises(ValueError, match="build_id"):
        validate_g0bn_manifest(g0bn_manifest(build_id="2026-07-15-tag"))


# ---------------------------------------------------------------- source isolation

@pytest.mark.parametrize("bad", [
    {"name": "coinbase_l2_snapshot", "sha256": "aa" * 32},
    {"name": "binance_spot_trades", "sha256": "aa" * 32},
    {"name": "funding_rates", "sha256": "aa" * 32},
    "crypto-lake/book_delta_v2",
])
def test_extra_source_fails(bad):
    man = g0bn_manifest()
    man["sources"].append(bad)
    with pytest.raises(ValueError, match="source"):
        validate_g0bn_manifest(man)


def test_data_source_missing_sha256_fails():
    man = _drop_source(g0bn_manifest(), "binance_futures_trades")
    man["sources"].append({"name": "binance_futures_trades"})
    with pytest.raises(ValueError, match="sha256"):
        validate_g0bn_manifest(man)


@pytest.mark.parametrize("name", [
    "binance_futures_l2_snapshot", "binance_futures_l2_delta",
    "binance_futures_trades", "source_certification", "cost_assumption"])
def test_missing_required_source_fails(name):
    man = _drop_source(g0bn_manifest(), name)
    with pytest.raises(ValueError, match=name):
        validate_g0bn_manifest(man)


def test_duplicate_cost_assumption_fails():
    man = g0bn_manifest()
    man["sources"].append(_get_source(man, "cost_assumption").copy())
    with pytest.raises(ValueError, match="cost_assumption"):
        validate_g0bn_manifest(man)


@pytest.mark.parametrize("field,value", [
    ("venue", "coinbase"),
    ("product", "BTC-USD"),
    ("drift_policy", "custom_drift_v2"),
    ("taker_fee_bps", -1.0),
])
def test_bad_cost_assumption_fails(field, value):
    man = g0bn_manifest()
    _get_source(man, "cost_assumption")[field] = value
    with pytest.raises(ValueError):
        validate_g0bn_manifest(man)


def test_holdout_missing_custodian_seal_fails():
    man = _drop_source(g0bn_manifest(partition="holdout"), "custodian_seal")
    with pytest.raises(ValueError, match="custodian_seal"):
        validate_g0bn_manifest(man)


# ---------------------------------------------------------------- dataset/partition isolation

def test_dev_dataset_id_with_holdout_partition_fails():
    man = g0bn_manifest(sources=g0bn_sources("holdout"))
    with pytest.raises(ValueError, match="dataset_id|partition"):
        validate_g0bn_manifest(man)


def test_oos_dataset_id_with_development_partition_fails():
    man = g0bn_manifest(dataset_id=G0BN_OOS_DATASET_ID)
    with pytest.raises(ValueError, match="dataset_id|partition"):
        validate_g0bn_manifest(man)


def test_other_dataset_id_with_bindings_fails():
    man = g0bn_manifest(dataset_id="binance_single_venue")
    with pytest.raises(ValueError, match="dataset_id"):
        validate_g0bn_manifest(man)


# ---------------------------------------------------------------- classification (67-D preflight)

def test_classify_legacy_manifest_is_not_g0bn():
    _, feats, lb = make_matrix(n=32, signal_strength=1.0, seed=1)
    cls = classify_manifest(make_manifest(feats, lb))
    assert (cls.is_g0bn, cls.partition, cls.holdout_bound) == (False, None, False)
    assert cls.dataset_id == "synthetic"


def test_classify_development_manifest():
    cls = classify_manifest(g0bn_manifest())
    assert (cls.is_g0bn, cls.partition, cls.holdout_bound) == (True, "development", False)
    assert cls.dataset_id == G0BN_DEV_DATASET_ID


def test_classify_holdout_manifest_is_holdout_bound():
    cls = classify_manifest(g0bn_manifest(partition="holdout"))
    assert (cls.is_g0bn, cls.partition, cls.holdout_bound) == (True, "holdout", True)
    assert cls.dataset_id == G0BN_OOS_DATASET_ID


def test_classify_oos_dataset_id_without_bindings_fails():
    _, feats, lb = make_matrix(n=32, signal_strength=1.0, seed=1)
    man = make_manifest(feats, lb, dataset_id=G0BN_OOS_DATASET_ID)
    with pytest.raises(ValueError):
        classify_manifest(man)


def test_classify_partition_binding_without_protocol_binding_fails():
    _, feats, lb = make_matrix(n=32, signal_strength=1.0, seed=1)
    man = make_manifest(feats, lb)
    man["sources"] = list(man["sources"]) + [partition_binding("holdout")]
    with pytest.raises(ValueError):
        classify_manifest(man)


def test_classify_rejects_ambiguous_partition():
    man = g0bn_manifest()
    man["sources"].append(partition_binding("holdout"))
    with pytest.raises(ValueError, match="exactly one"):
        classify_manifest(man)


def test_classify_validates_schema_first():
    with pytest.raises(ValueError, match="manifest missing required fields"):
        classify_manifest({"dataset_id": G0BN_OOS_DATASET_ID})
