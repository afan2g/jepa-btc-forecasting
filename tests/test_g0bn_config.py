"""Strict g0bn-protocol-config-v1 schema, canonical hashing, and exact
instrument/horizon/candidate validation (issue #86, slice 67-A).

Synthetic configs only. Legacy G0-CB/G0-XV contracts are untouched; a dedicated
regression test pins that boundary.
"""
from __future__ import annotations

import copy

import pytest

import eval.ledger as legacy_ledger
from eval.g0bn_config import (
    CANDIDATE_IDS,
    CONFIG_SCHEMA,
    DEV_DATASET_ID,
    DEV_END_NS,
    DEV_START_NS,
    FEATURE_REGISTRY,
    HOLDOUT_END_NS,
    HOLDOUT_START_NS,
    HORIZONS,
    INSTRUMENT,
    OOS_DATASET_ID,
    PILOT_ID,
    PROTOCOL_ID,
    g0bn_artifact_sha256,
    protocol_config_sha256,
    validate_protocol_config,
)
from eval.hashing import hash_obj

import g0bn_protocol_fixtures as fx
from g0bn_protocol_fixtures import make_config, with_sha


# --- constants must match the binding spec exactly ------------------------------------

def test_module_constants_match_spec_literals():
    assert CONFIG_SCHEMA == "g0bn-protocol-config-v1"
    assert PROTOCOL_ID == "g0bn-v1"
    assert PILOT_ID == "g0bn-2025-11_2026-01-v1"
    assert INSTRUMENT == fx.INSTRUMENT
    assert tuple(h["tag"] for h in HORIZONS) == ("2s", "10s", "60s")
    assert [dict(h) for h in HORIZONS] == [dict(h) for h in fx.HORIZONS]
    assert tuple(FEATURE_REGISTRY) == fx.FEATURE_REGISTRY
    assert tuple(CANDIDATE_IDS) == fx.CANDIDATE_IDS
    assert DEV_DATASET_ID == "binance_single_venue_g0bn_dev"
    assert OOS_DATASET_ID == "binance_single_venue_g0bn_oos"
    assert (DEV_START_NS, DEV_END_NS) == (1_761_955_200_000_000_000, 1_767_225_600_000_000_000)
    assert (HOLDOUT_START_NS, HOLDOUT_END_NS) == (1_767_225_600_000_000_000,
                                                  1_769_904_000_000_000_000)


def test_legacy_ledger_protocols_are_untouched():
    # G0-BN never joins the legacy trial accounting (spec section 1).
    assert legacy_ledger.TRIAL_PROTOCOLS == ("g0cb", "g0xv")
    assert legacy_ledger.PROTOCOLS == ("g0cb", "g0xv", "g0xv-verdict")
    assert not any("g0bn" in p for p in legacy_ledger.PROTOCOLS)


# --- canonical hashing ------------------------------------------------------------------

def test_valid_config_passes_and_chains():
    cfg = make_config()
    assert validate_protocol_config(cfg) is cfg


def test_config_hash_round_trip_is_deterministic_and_key_order_free():
    cfg = make_config()
    assert cfg["sha256"] == protocol_config_sha256(cfg)
    shuffled = dict(reversed(list(copy.deepcopy(cfg).items())))
    assert protocol_config_sha256(shuffled) == cfg["sha256"]
    validate_protocol_config(shuffled)


def test_hash_excludes_exactly_self_and_generated_at():
    cfg = make_config()
    other_time = copy.deepcopy(cfg)
    other_time["generated_at"] = "2026-07-16T12:34:56Z"
    assert protocol_config_sha256(other_time) == cfg["sha256"]
    validate_protocol_config(other_time)
    # g0bn_artifact_sha256 must ignore the embedded self-field itself.
    assert g0bn_artifact_sha256(cfg) == g0bn_artifact_sha256(dict(cfg, sha256="0" * 64))


@pytest.mark.parametrize("section", [
    "schema", "protocol_id", "pilot_id", "instrument", "source_certification",
    "producer", "clock", "features", "labels", "costs", "exclusions", "partition",
    "cv", "horizons", "candidates", "selection", "verdict_thresholds", "reporting",
    "oos", "software",
])
def test_every_decision_bearing_section_is_hash_bearing(section):
    cfg = make_config()
    tampered = copy.deepcopy(cfg)
    tampered[section] = "tampered" if not isinstance(tampered[section], str) else "x"
    assert protocol_config_sha256(tampered) != cfg["sha256"]


def test_tampering_is_detected_by_the_embedded_hash():
    cfg = make_config()
    stale = copy.deepcopy(cfg)
    stale["labels"]["tp_multiplier"] = 2.0  # content changed, sha256 left stale
    with pytest.raises(ValueError, match="sha256"):
        validate_protocol_config(stale)
    forged = make_config(sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        validate_protocol_config(forged)
    with pytest.raises(ValueError, match="sha256"):
        validate_protocol_config(make_config(sha256=make_config()["sha256"].upper()))


def test_hash_matches_independent_hand_computation():
    # Independent of eval.hashing: raw hashlib over the canonical encoding.
    import hashlib
    import json

    cfg = make_config()
    stripped = {k: v for k, v in cfg.items() if k not in ("sha256", "generated_at")}
    canonical = json.dumps(stripped, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == cfg["sha256"]


def test_hash_exclusion_is_top_level_only():
    # A nested generated_at (e.g. inside resolved params) stays hash-bearing.
    a = copy.deepcopy(make_config())
    b = copy.deepcopy(a)
    a["candidates"][3]["model_params"]["generated_at"] = "1"
    b["candidates"][3]["model_params"]["generated_at"] = "2"
    for cfg in (a, b):
        cfg["candidates"][3]["model_params_sha256"] = hash_obj(
            cfg["candidates"][3]["model_params"])
    assert protocol_config_sha256(a) != protocol_config_sha256(b)


def test_nested_array_order_is_decision_bearing():
    cfg = make_config()
    reordered = copy.deepcopy(cfg)
    tv = reordered["producer"]["transform_versions"]
    tv[0], tv[1] = tv[1], tv[0]
    assert protocol_config_sha256(reordered) != cfg["sha256"]


# --- exact field sets -------------------------------------------------------------------

def test_unknown_top_level_key_rejected():
    with pytest.raises(ValueError, match="unknown"):
        validate_protocol_config(make_config(extra_section={"x": 1}))


@pytest.mark.parametrize("key", [
    "schema", "protocol_id", "pilot_id", "instrument", "source_certification",
    "producer", "clock", "features", "labels", "costs", "exclusions", "partition",
    "cv", "horizons", "candidates", "selection", "verdict_thresholds", "reporting",
    "oos", "software", "generated_at", "sha256",
])
def test_missing_top_level_key_rejected(key):
    cfg = make_config()
    cfg.pop(key)
    with pytest.raises(ValueError, match=key):
        validate_protocol_config(cfg)


@pytest.mark.parametrize("section", [
    "instrument", "source_certification", "producer", "clock", "features", "labels",
    "costs", "exclusions", "partition", "cv", "selection", "verdict_thresholds",
    "reporting", "oos", "software",
])
def test_unknown_nested_key_rejected(section):
    cfg = make_config()
    cfg[section]["bogus_extra_field"] = 1
    with pytest.raises(ValueError, match="bogus_extra_field"):
        validate_protocol_config(with_sha(cfg))


def test_exact_protocol_strings_required():
    with pytest.raises(ValueError, match="schema"):
        validate_protocol_config(make_config(schema="g0xv-protocol-config-v1"))
    with pytest.raises(ValueError, match="protocol_id"):
        validate_protocol_config(make_config(protocol_id="g0xv"))
    with pytest.raises(ValueError, match="pilot_id"):
        validate_protocol_config(make_config(pilot_id="g0bn-2026-02_2026-03-v1"))


def test_generated_at_must_be_timezone_aware_iso():
    with pytest.raises(ValueError, match="generated_at"):
        validate_protocol_config(make_config(generated_at="2026-07-15T00:00:00"))
    with pytest.raises(ValueError, match="generated_at"):
        validate_protocol_config(make_config(generated_at=1767225600))


# --- instrument and source isolation ----------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("exchange", "COINBASE"),
    ("exchange", "BINANCE"),
    ("native_symbol", "BTCUSD"),
    ("symbol", "BTC-USDT"),
    ("symbol", "ETH-USDT-PERP"),
    ("contract_type", "spot"),
    ("quote_asset", "USD"),
    ("settlement_asset", "BTC"),
])
def test_instrument_must_be_exactly_the_spec_object(field, value):
    cfg = make_config(instrument=dict(fx.INSTRUMENT, **{field: value}))
    with pytest.raises(ValueError, match="instrument"):
        validate_protocol_config(with_sha(cfg))


def test_instrument_extra_or_missing_field_rejected():
    with pytest.raises(ValueError, match="instrument"):
        validate_protocol_config(
            with_sha(make_config(instrument=dict(fx.INSTRUMENT, venue2="BINANCE"))))
    partial = dict(fx.INSTRUMENT)
    partial.pop("contract_type")
    with pytest.raises(ValueError, match="instrument"):
        validate_protocol_config(with_sha(make_config(instrument=partial)))


@pytest.mark.parametrize("field,bad", [
    ("l2_product", "coinbase/book_l2"),
    ("l2_product", "binance-spot/book_delta_v2"),
    ("trade_product", "coinbase/trades"),
    ("trade_product", "binance-futures/funding_rate_v1"),
])
def test_source_certification_pins_exact_certified_products(field, bad):
    cfg = make_config(source_certification=fx.make_source_certification(**{field: bad}))
    with pytest.raises(ValueError, match=field):
        validate_protocol_config(with_sha(cfg))


@pytest.mark.parametrize("bad", ["coinapi", "coinbase", "cryptohftdata", "binance-direct"])
def test_source_certification_pins_the_certified_provider(bad):
    # Spec 2.1 rejects a provider fallback (e.g. CoinAPI) even with valid product
    # strings; the frozen source identity names exactly the #64-certified provider.
    cfg = make_config(source_certification=fx.make_source_certification(provider=bad))
    with pytest.raises(ValueError, match="provider"):
        validate_protocol_config(with_sha(cfg))


def test_source_certification_products_must_be_distinct():
    # The two feeds cannot collapse into one (spec section 2.1 needs L2 and trades).
    same = fx.make_source_certification(trade_product=fx.L2_PRODUCT)
    with pytest.raises(ValueError, match="trade_product"):
        validate_protocol_config(with_sha(make_config(source_certification=same)))


def test_clock_reference_stream_must_be_the_certified_trade_product():
    # Spec 2.1/3.2: the bar clock is driven by the single certified Binance-perp
    # trade product; a Coinbase/spot reference stream must fail closed.
    for bad in ("coinbase-spot/trades", "binance-spot/trades_v1"):
        cfg = make_config(clock=fx.make_clock(reference_stream=bad))
        with pytest.raises(ValueError, match="reference_stream"):
            validate_protocol_config(with_sha(cfg))


def test_repository_object_ids_use_one_declared_format():
    prod = fx.make_producer(repository_tree=fx.sha_hex("repo-tree"))  # 64 vs 40 commit
    with pytest.raises(ValueError, match="repository"):
        validate_protocol_config(with_sha(make_config(producer=prod)))


def test_source_certification_requires_distinct_identities_and_hashes():
    cfg = make_config(source_certification=fx.make_source_certification(
        operator_identity="g0bn-custodian-svc"))
    with pytest.raises(ValueError, match="custodian"):
        validate_protocol_config(with_sha(cfg))
    cfg = make_config(source_certification=fx.make_source_certification(
        custodian_seal_sha256="not-a-hash"))
    with pytest.raises(ValueError, match="custodian_seal_sha256"):
        validate_protocol_config(with_sha(cfg))


@pytest.mark.parametrize("value", ["TBD", "UNRESOLVED", "tbd", " unresolved ", ""])
def test_placeholder_operator_values_rejected(value):
    cfg = make_config(source_certification=fx.make_source_certification(provider=value))
    with pytest.raises(ValueError, match="provider"):
        validate_protocol_config(with_sha(cfg))


def test_null_required_value_rejected():
    cfg = make_config(labels=fx.make_labels(unresolved_barrier_policy=None))
    with pytest.raises(ValueError, match="unresolved_barrier_policy"):
        validate_protocol_config(with_sha(cfg))


# --- horizons ---------------------------------------------------------------------------

def test_horizons_exact_order_values_roles():
    cfg = make_config(horizons=[dict(h) for h in reversed(fx.HORIZONS)])
    with pytest.raises(ValueError, match="horizons"):
        validate_protocol_config(with_sha(cfg))
    cfg = make_config(horizons=[dict(h) for h in fx.HORIZONS][:2])
    with pytest.raises(ValueError, match="horizons"):
        validate_protocol_config(with_sha(cfg))
    extra = [dict(h) for h in fx.HORIZONS] + [{"tag": "5m", "ns": 300_000_000_000,
                                               "role": "primary"}]
    with pytest.raises(ValueError, match="horizons"):
        validate_protocol_config(with_sha(make_config(horizons=extra)))
    swapped_role = [dict(h) for h in fx.HORIZONS]
    swapped_role[2]["role"] = "primary"  # 60s can never be primary
    with pytest.raises(ValueError, match="horizons"):
        validate_protocol_config(with_sha(make_config(horizons=swapped_role)))


def test_horizon_ns_type_strictness():
    as_str = [dict(h) for h in fx.HORIZONS]
    as_str[0]["ns"] = "2000000000"
    with pytest.raises(ValueError, match="horizons"):
        validate_protocol_config(with_sha(make_config(horizons=as_str)))
    as_float = [dict(h) for h in fx.HORIZONS]
    as_float[0]["ns"] = 2e9
    with pytest.raises(ValueError, match="horizons"):
        validate_protocol_config(with_sha(make_config(horizons=as_float)))


# --- features ---------------------------------------------------------------------------

def test_feature_registry_exact_order_and_names():
    feats = fx.make_features()
    feats["registry"][0], feats["registry"][1] = feats["registry"][1], feats["registry"][0]
    with pytest.raises(ValueError, match="registry"):
        validate_protocol_config(with_sha(make_config(features=feats)))
    feats = fx.make_features()
    feats["registry"] = feats["registry"][:-1]
    with pytest.raises(ValueError, match="registry"):
        validate_protocol_config(with_sha(make_config(features=feats)))
    feats = fx.make_features()
    feats["registry"][3]["name"] = "spread_ticks"
    with pytest.raises(ValueError, match="registry"):
        validate_protocol_config(with_sha(make_config(features=feats)))


# --- candidates -------------------------------------------------------------------------

def test_candidate_ladder_exact_ids_and_order():
    cands = fx.make_candidates()
    cands[0], cands[1] = cands[1], cands[0]
    with pytest.raises(ValueError, match="candidates"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    with pytest.raises(ValueError, match="candidates"):
        validate_protocol_config(with_sha(make_config(candidates=fx.make_candidates()[:4])))


@pytest.mark.parametrize("param,value", [
    ("alpha", 1),            # int is not the pinned float 1.0
    ("alpha", 2.0),
    ("fit_intercept", 1),    # bool-vs-int
    ("max_iter", 0),         # pinned as present-and-null
    ("solver", "auto"),
    ("random_state", 0),
])
def test_ridge_fixed_params_are_pinned_exactly(param, value):
    cands = fx.make_candidates()
    cands[2]["model_params"][param] = value
    cands[2]["model_params_sha256"] = hash_obj(cands[2]["model_params"])
    with pytest.raises(ValueError, match=param):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_ridge_max_iter_must_be_present_null():
    cands = fx.make_candidates()
    cands[2]["model_params"].pop("max_iter")
    cands[2]["model_params_sha256"] = hash_obj(cands[2]["model_params"])
    with pytest.raises(ValueError, match="max_iter"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


@pytest.mark.parametrize("param,value", [
    ("learning_rate", 0.1),
    ("num_leaves", 31.0),    # float is not the pinned int
    ("deterministic", 1),    # bool-vs-int
    ("n_jobs", 2),
    ("objective", "huber"),
])
def test_lgbm_fixed_params_are_pinned_exactly(param, value):
    cands = fx.make_candidates()
    cands[3]["model_params"][param] = value
    cands[3]["model_params_sha256"] = hash_obj(cands[3]["model_params"])
    with pytest.raises(ValueError, match=param):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_extra_resolved_library_default_is_allowed_and_hash_bearing():
    cfg = make_config()
    cands = fx.make_candidates()
    cands[3]["model_params"]["class_weight"] = None  # future library default: allowed
    cands[3]["model_params_sha256"] = hash_obj(cands[3]["model_params"])
    cfg2 = with_sha(make_config(candidates=cands))
    validate_protocol_config(cfg2)
    assert cfg2["sha256"] != cfg["sha256"]


def test_negative_zero_is_not_the_pinned_zero():
    # IEEE -0.0 == 0.0, but canonical JSON serializes them differently, so accepting
    # -0.0 would mint a full set of phantom trial identities over the same ladder.
    cands = fx.make_candidates()
    cands[0]["model_params"] = {"forecast_bps": -0.0}
    cands[0]["model_params_sha256"] = hash_obj(cands[0]["model_params"])
    with pytest.raises(ValueError, match="forecast_bps"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[3]["model_params"]["reg_alpha"] = -0.0
    cands[3]["model_params_sha256"] = hash_obj(cands[3]["model_params"])
    with pytest.raises(ValueError, match="reg_alpha"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_prediction_transforms_are_pinned_per_candidate():
    cands = fx.make_candidates()
    cands[0]["prediction_transform"] = "constant_one_bps_v1"
    with pytest.raises(ValueError, match="prediction_transform"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[4]["prediction_transform"] = "identity_bps_v1"
    with pytest.raises(ValueError, match="prediction_transform"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_non_fitted_params_are_exact():
    cands = fx.make_candidates()
    cands[0]["model_params"] = {"forecast_bps": 0}  # int, not the pinned float 0.0
    cands[0]["model_params_sha256"] = hash_obj(cands[0]["model_params"])
    with pytest.raises(ValueError, match="forecast_bps"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[1]["model_params"]["multiplier"] = 2.0
    cands[1]["model_params_sha256"] = hash_obj(cands[1]["model_params"])
    with pytest.raises(ValueError, match="multiplier"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[0]["model_params"]["extra"] = 1.0  # non-fitted params take no extra keys
    cands[0]["model_params_sha256"] = hash_obj(cands[0]["model_params"])
    with pytest.raises(ValueError, match="model_params"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_candidate_inputs_are_pinned():
    cands = fx.make_candidates()
    cands[2]["feature_cols"] = ["microprice_dev"]  # ofi_ridge input is ofi_integrated
    cands[2]["feature_formula_sha256s"] = {"microprice_dev": fx.sha_hex("x")}
    with pytest.raises(ValueError, match="feature_cols"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[3]["feature_cols"] = list(reversed(cands[3]["feature_cols"]))
    with pytest.raises(ValueError, match="feature_cols"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_candidate_hash_consistency_enforced():
    cands = fx.make_candidates()
    cands[3]["model_params"]["learning_rate"] = 0.05  # unchanged value, stale hash below
    cands[3]["model_params_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model_params_sha256"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[4]["preprocessing_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="preprocessing_sha256"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_classifier_scale_rule_and_class_order_pinned():
    cands = fx.make_candidates()
    cands[4]["training_y_std_rule"] = "weighted_sample_std_v1"
    with pytest.raises(ValueError, match="training_y_std_rule"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[4]["class_order"] = [1, 0, -1]
    with pytest.raises(ValueError, match="class_order"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_candidate_params_reject_nested_physical_file_hash():
    # A *_file_sha256 nested in an identity-bearing candidate subtree would flow into
    # candidate_definition_sha256 -> trial_id; reject it (physical hashes are audit-only).
    cands = fx.make_candidates()
    cands[3]["model_params"]["matrix_file_sha256"] = "0" * 64
    cands[3]["model_params_sha256"] = hash_obj(cands[3]["model_params"])
    with pytest.raises(ValueError, match="file_sha256"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[2]["preprocessing"]["matrix_file_sha256"] = "0" * 64
    cands[2]["preprocessing_sha256"] = hash_obj(cands[2]["preprocessing"])
    with pytest.raises(ValueError, match="file_sha256"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_resolved_params_reject_negative_zero_in_opaque_subtree():
    # -0.0 in a non-pinned resolved default (opaque subtree) also drifts the hash;
    # _scalar_tree must guard it like _exact/_num do for pinned/typed fields.
    cands = fx.make_candidates()
    cands[3]["model_params"]["extra_resolved_default"] = -0.0
    cands[3]["model_params_sha256"] = hash_obj(cands[3]["model_params"])
    with pytest.raises(ValueError, match="negative zero"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))
    cands = fx.make_candidates()
    cands[2]["preprocessing"]["shift"] = -0.0
    cands[2]["preprocessing_sha256"] = hash_obj(cands[2]["preprocessing"])
    with pytest.raises(ValueError, match="negative zero"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


def test_ofi_ridge_must_declare_no_candidate_local_scaling():
    cands = fx.make_candidates()
    cands[2]["preprocessing"] = {"stationarization": "producer_pinned_causal_v1",
                                 "candidate_local_scaling": True}
    cands[2]["preprocessing_sha256"] = hash_obj(cands[2]["preprocessing"])
    with pytest.raises(ValueError, match="candidate_local_scaling"):
        validate_protocol_config(with_sha(make_config(candidates=cands)))


# --- typed scalars: bool-vs-int, non-finite, coercion ----------------------------------

@pytest.mark.parametrize("mutate,match", [
    (lambda c: c["partition"].update(partition_guard_ns=True), "partition_guard_ns"),
    (lambda c: c["verdict_thresholds"]["bootstrap"].update(n_boot=True), "n_boot"),
    (lambda c: c["verdict_thresholds"]["bootstrap"].update(seed="0"), "seed"),
    (lambda c: c["clock"].update(target_bars_per_day="5000"), "target_bars_per_day"),
    (lambda c: c["clock"].update(monotone_watermark="true"), "monotone_watermark"),
    (lambda c: c["labels"].update(tp_multiplier=float("nan")), "tp_multiplier"),
    (lambda c: c["labels"].update(sl_multiplier=float("inf")), "sl_multiplier"),
    (lambda c: c["labels"].update(ewma_half_life_ns=600.0), "ewma_half_life_ns"),
    (lambda c: c["costs"]["cost_assumption"].update(taker_fee_bps="4.5"), "taker_fee_bps"),
    (lambda c: c["costs"]["cost_assumption"].update(taker_fee_bps=float("nan")),
     "taker_fee_bps"),
    (lambda c: c["costs"].update(no_trade_margin_bps=True), "no_trade_margin_bps"),
    (lambda c: c["cv"].update(expected_test_multiplicity=5.0), "expected_test_multiplicity"),
])
def test_scalar_type_strictness(mutate, match):
    cfg = make_config()
    mutate(cfg)
    # NaN/inf cannot re-hash under strict JSON; skip re-hash when mutation is non-finite.
    try:
        cfg = with_sha(cfg)
    except ValueError:
        pass
    with pytest.raises(ValueError, match=match):
        validate_protocol_config(cfg)


# --- day arrays and exclusions ----------------------------------------------------------

@pytest.mark.parametrize("days,match", [
    (["2025-11-02", "2025-11-01"], "sorted"),
    (["2025-11-01", "2025-11-01"], "sorted|unique"),
    (["2025-11-1"], "YYYY-MM-DD"),
    (["2025-11-01T00:00:00Z"], "YYYY-MM-DD"),
    (["2025-11-31"], "YYYY-MM-DD"),
    (["2025-13-01"], "YYYY-MM-DD"),
    (["2025-02-30"], "YYYY-MM-DD"),
    (["2025-10-31"], "development"),
    (["2026-01-01"], "development"),
    ([], "included_days"),
])
def test_included_day_array_rules(days, match):
    exc = fx.make_exclusions(included_days=days, excluded_days={})
    with pytest.raises(ValueError, match=match):
        validate_protocol_config(with_sha(make_config(exclusions=exc)))


def test_exclusion_accounting_must_cover_the_development_window():
    exc = fx.make_exclusions()
    exc["excluded_days"].pop("2025-12-25")  # day now in neither list
    with pytest.raises(ValueError, match="2025-12-25"):
        validate_protocol_config(with_sha(make_config(exclusions=exc)))
    exc = fx.make_exclusions()
    exc["included_days"] = exc["included_days"] + ["2025-12-25"]  # in both lists
    exc["included_days"].sort()
    with pytest.raises(ValueError, match="2025-12-25"):
        validate_protocol_config(with_sha(make_config(exclusions=exc)))


def test_excluded_day_entries_need_reason_and_evidence():
    exc = fx.make_exclusions()
    exc["excluded_days"]["2025-11-14"] = {"reason": "source_gap"}
    with pytest.raises(ValueError, match="evidence_sha256"):
        validate_protocol_config(with_sha(make_config(exclusions=exc)))
    exc = fx.make_exclusions()
    exc["excluded_days"]["2025-11-14"] = {"reason": "", "evidence_sha256": fx.sha_hex("e")}
    with pytest.raises(ValueError, match="reason"):
        validate_protocol_config(with_sha(make_config(exclusions=exc)))


# --- partition plan ---------------------------------------------------------------------

def test_partition_bounds_and_schema_are_pinned():
    with pytest.raises(ValueError, match="schema"):
        validate_protocol_config(with_sha(make_config(
            partition=fx.make_partition(schema="partition-contract-v1"))))
    with pytest.raises(ValueError, match="holdout_end_ns"):
        validate_protocol_config(with_sha(make_config(
            partition=fx.make_partition(holdout_end_ns=fx.HOLDOUT_END_NS + 1))))
    with pytest.raises(ValueError, match="prefilter_rule"):
        validate_protocol_config(with_sha(make_config(
            partition=fx.make_partition(
                prefilter_rule="t_event + horizons[horizon] + guard_ns < boundary_ns"))))


def test_partition_plan_self_hash_verified():
    part = fx.make_partition()
    part["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="partition.*sha256|sha256"):
        validate_protocol_config(with_sha(make_config(partition=part)))


def test_partition_rejects_realized_holdout_counts():
    part = fx.make_partition()
    part["holdout_drop_counts"] = {"2s": 10, "10s": 12, "60s": 30}
    part["sha256"] = hash_obj(part, exclude_keys=("sha256",))
    with pytest.raises(ValueError, match="holdout_drop_counts"):
        validate_protocol_config(with_sha(make_config(partition=part)))


def test_partition_horizons_must_match_horizon_section():
    part = fx.make_partition(horizons={"2s": 2_000_000_000, "10s": 10_000_000_000})
    with pytest.raises(ValueError, match="horizons"):
        validate_protocol_config(with_sha(make_config(partition=part)))


def test_development_drop_counts_are_ints_per_horizon():
    part = fx.make_partition(development_drop_counts={"2s": 180, "10s": 240.0, "60s": 900})
    with pytest.raises(ValueError, match="development_drop_counts"):
        validate_protocol_config(with_sha(make_config(partition=part)))


# --- cv / selection / verdict thresholds -----------------------------------------------

@pytest.mark.parametrize("field,value,match", [
    ("n_groups", 5, "n_groups"),
    ("k", 3, "k"),
    ("forecast_collapse_version", "median_v1", "forecast_collapse_version"),
    ("forecast_dtype", "float32", "forecast_dtype"),
    ("expected_test_multiplicity", 4, "expected_test_multiplicity"),
])
def test_cv_constants_pinned(field, value, match):
    cv = fx.make_cv(**{field: value})
    with pytest.raises(ValueError, match=match):
        validate_protocol_config(with_sha(make_config(cv=cv)))


def test_cv_dsr_pbo_rules_pinned():
    cv = fx.make_cv()
    cv["dsr"]["rounding_rule"] = "floor_v1"
    with pytest.raises(ValueError, match="rounding_rule"):
        validate_protocol_config(with_sha(make_config(cv=cv)))
    cv = fx.make_cv()
    cv["pbo"]["is_tie_rule"] = "average_rank_v1"
    with pytest.raises(ValueError, match="is_tie_rule"):
        validate_protocol_config(with_sha(make_config(cv=cv)))
    cv = fx.make_cv()
    cv["pbo"]["n_blocks"] = 6
    with pytest.raises(ValueError, match="n_blocks"):
        validate_protocol_config(with_sha(make_config(cv=cv)))


def test_embargo_must_equal_max_lookback():
    cv = fx.make_cv(embargo_ns=900_000_000_001)
    with pytest.raises(ValueError, match="embargo_ns"):
        validate_protocol_config(with_sha(make_config(cv=cv)))


def test_selection_rules_are_pinned():
    with pytest.raises(ValueError, match="ranking_rule"):
        validate_protocol_config(with_sha(make_config(
            selection=fx.make_selection(ranking_rule="highest_point_lift_v1"))))
    with pytest.raises(ValueError, match="tie_rule"):
        validate_protocol_config(with_sha(make_config(
            selection=fx.make_selection(tie_rule="random_v1"))))


def test_selection_eligibility_is_the_four_non_persistence_candidates():
    sel = fx.make_selection(eligible_candidate_ids=[
        "persistence_zero", "microprice_raw", "ofi_ridge", "lgbm_reg"])
    with pytest.raises(ValueError, match="eligible_candidate_ids"):
        validate_protocol_config(with_sha(make_config(selection=sel)))
    sel = fx.make_selection(eligible_candidate_ids=[
        "ofi_ridge", "microprice_raw", "lgbm_reg", "lgbm_clf"])  # wrong order
    with pytest.raises(ValueError, match="eligible_candidate_ids"):
        validate_protocol_config(with_sha(make_config(selection=sel)))


@pytest.mark.parametrize("field,value", [
    ("min_valid_days", 19),
    ("min_uniqueness_sum", 99),
    ("min_trades", 29),
    ("min_effective_trades", 9),
    ("dsr_threshold", 0.9),
    ("pbo_threshold", 0.55),
    ("alpha_dev", 0.05),
    ("alpha_oos", 0.05),
])
def test_verdict_threshold_constants_pinned(field, value):
    vt = fx.make_verdict_thresholds(**{field: value})
    with pytest.raises(ValueError, match=field):
        validate_protocol_config(with_sha(make_config(verdict_thresholds=vt)))


@pytest.mark.parametrize("field,value", [
    ("kind", "row_iid"),
    ("block_length_days", 1),
    ("n_boot", 5000),
    ("seed", 1),
    ("bit_generator", "MT19937"),
    ("percentile_method", "nearest"),
])
def test_bootstrap_constants_pinned(field, value):
    vt = fx.make_verdict_thresholds()
    vt["bootstrap"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_protocol_config(with_sha(make_config(verdict_thresholds=vt)))


# --- costs ------------------------------------------------------------------------------

@pytest.mark.parametrize("mutate,match", [
    (lambda c: c["cost_assumption"].update(taker_fee_bps=-0.0), "taker_fee_bps"),
    (lambda c: c["cost_assumption"].update(base_slippage_bps=-0.0), "base_slippage_bps"),
    (lambda c: c.update(no_trade_margin_bps=-0.0), "no_trade_margin_bps"),
])
def test_costs_reject_negative_zero(mutate, match):
    # -0.0 passes `< 0` but hashes differently from 0.0, minting a distinct cost
    # identity for the same economics.
    costs = fx.make_costs()
    mutate(costs)
    with pytest.raises(ValueError, match=match):
        validate_protocol_config(with_sha(make_config(costs=costs)))


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c["cost_assumption"].update(venue="coinbase"), "venue"),
    (lambda c: c["cost_assumption"].update(product="BTC-USDT"), "product"),
    (lambda c: c["cost_assumption"].update(drift_policy="linear_latency_v1"), "drift_policy"),
    (lambda c: c.update(fee_sides=1), "fee_sides"),
    (lambda c: c.update(spread_crossings=1), "spread_crossings"),
    (lambda c: c.update(cost_column_dtype="float32"), "cost_column_dtype"),
    (lambda c: c.update(reconciliation_rel_tol=1e-9), "reconciliation_rel_tol"),
    (lambda c: c.update(reconciliation_abs_tol=1e-9), "reconciliation_abs_tol"),
])
def test_costs_contract_pinned(mutate, match):
    costs = fx.make_costs()
    mutate(costs)
    with pytest.raises(ValueError, match=match):
        validate_protocol_config(with_sha(make_config(costs=costs)))


def test_fee_applicability_must_cover_both_partitions():
    costs = fx.make_costs()
    costs["fee_tier"]["applicability_end_ns"] = fx.DEV_END_NS  # stops before January
    with pytest.raises(ValueError, match="applicability"):
        validate_protocol_config(with_sha(make_config(costs=costs)))


# --- oos / software ---------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("dataset_id", "binance_single_venue_g0bn_dev"),
    ("holdout_universe_schema", "g0xv-holdout"),
    ("transaction_schema", "g0xv-transaction"),
    ("raw_access_claim_schema", "g0bn-raw-access-claim-v2"),
    ("matrix_access_claim_schema", "g0bn-matrix-access-claim-v2"),
    ("consumption_schema", "record-v1"),
    ("terminal_failure_policy", "retry_v1"),
])
def test_oos_pins(field, value):
    oos = fx.make_oos(**{field: value})
    with pytest.raises(ValueError, match=field):
        validate_protocol_config(with_sha(make_config(oos=oos)))


def test_oos_lock_contention_result_pinned():
    oos = fx.make_oos()
    oos["lock"]["contention_result"] = "steal_lock"
    with pytest.raises(ValueError, match="contention_result"):
        validate_protocol_config(with_sha(make_config(oos=oos)))


def test_software_commit_must_match_producer_commit():
    cfg = make_config(software=fx.make_software(repository_commit="a" * 40))
    with pytest.raises(ValueError, match="repository_commit"):
        validate_protocol_config(with_sha(cfg))


def test_costs_product_must_match_instrument_symbol():
    costs = fx.make_costs()
    costs["cost_assumption"]["product"] = "BTC-USD-PERP"
    with pytest.raises(ValueError, match="product"):
        validate_protocol_config(with_sha(make_config(costs=costs)))
