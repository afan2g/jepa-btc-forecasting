"""67-C outcome-blind holdout plan and selection freeze (issue #89; spec sections
5-6 and 10-12 of docs/superpowers/specs/2026-07-13-g0bn-protocol.md).

All fixtures are synthetic: fake custody checksums, seeded development rows, and
UTC-day accounting only. No vendor data, no January payloads, no real market data.
The module-scoped strong/weak bundles each run the full 15-trial base ladder once
(mirroring tests/test_g0bn_selection.py) so freeze construction re-derives a real
deterministic development selection.
"""
from __future__ import annotations

import builtins
import copy
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from eval.g0bn_engine import run_g0bn_development
from eval.g0bn_freeze import (
    FREEZE_SCHEMA,
    HOLDOUT_PLAN_SCHEMA,
    build_freeze,
    build_holdout_plan,
    freeze_sha256,
    holdout_plan_binding,
    holdout_plan_sha256,
    oos_build_params,
    validate_custody_inventory,
    validate_freeze,
    validate_holdout_plan,
    verify_holdout_manifest_binding,
    verify_oos_build_binding,
)
from eval.g0bn_identity import G0BN_HOLDOUT_UNIVERSE_ID, G0BN_TRANSACTION_ID
from eval.g0bn_selection import bootstrap_draws, development_selection
from eval.guard import preflight_generic_manifest
from eval.hashing import hash_obj
from eval.writer import write_holdout
from tests.g0bn_dev_fixtures import (
    dev_bundle,
    dev_config,
    dev_data_identity,
    dev_manifest,
    dev_source_manifest_sha256,
    durable_ledger,
)
from tests.g0bn_fixtures import g0bn_frame
from tests.g0bn_holdout_fixtures import (
    NORMALIZED_PRODUCTS,
    RAW_PRODUCTS,
    january_days,
    make_inventory,
    make_objects,
    oos_manifest_and_params,
    sealed_config_kwargs,
)
from tests.g0bn_protocol_fixtures import (
    make_costs,
    make_source_certification,
    sha_hex,
)

GEN_AT = "2026-07-19T00:00:00Z"

CANDIDATE_LADDER = ("persistence_zero", "microprice_raw", "ofi_ridge",
                    "lgbm_reg", "lgbm_clf")
SELECTABLE = CANDIDATE_LADDER[1:]

# Every January outcome/build field the plan and freeze must structurally refuse
# (issue #89 acceptance criteria; spec sections 5.2-5.3).
PROHIBITED_JANUARY_FIELDS = (
    "holdout_build_id",
    "holdout_manifest_sha256",
    "holdout_logical_row_sha256",
    "oos_logical_row_sha256",
    "matrix_file_sha256",
    "holdout_row_count",
    "row_counts",
    "holdout_drop_counts",
    "realized_threshold_schedule_sha256",
    "realized_normalizer_state_sha256",
    "realized_schedule",
    "oos_forecasts",
    "forecasts",
    "oos_metrics",
    "metrics",
    "results",
    "verdict",
)


def _resha(artifact: dict) -> dict:
    """Recompute the embedded self-hash after a deliberate mutation (keeps tamper
    tests targeted at cross-checks instead of the self-hash)."""
    out = copy.deepcopy(artifact)
    out.pop("sha256", None)
    out["sha256"] = hash_obj(out, exclude_keys=("sha256", "generated_at"))
    return out


@pytest.fixture(scope="module")
def strong():
    inventory = make_inventory()
    frame, manifest, config, identity = dev_bundle(
        config=dev_config(**sealed_config_kwargs(inventory)))
    run = run_g0bn_development(frame, manifest, config, identity, durable_ledger())
    plan = build_holdout_plan(config, inventory, generated_at=GEN_AT)
    freeze = build_freeze(run, plan, inventory=inventory, generated_at=GEN_AT)
    return {"config": config, "run": run, "inventory": inventory,
            "plan": plan, "freeze": freeze}


@pytest.fixture(scope="module")
def weak():
    inventory = make_inventory()
    frame, manifest, config, identity = dev_bundle(
        signal_bps=2.0, noise_bps=0.5, seed=23,
        config=dev_config(**sealed_config_kwargs(inventory)))
    run = run_g0bn_development(frame, manifest, config, identity, durable_ledger())
    plan = build_holdout_plan(config, inventory, generated_at=GEN_AT)
    freeze = build_freeze(run, plan, inventory=inventory, generated_at=GEN_AT)
    return {"config": config, "run": run, "plan": plan, "freeze": freeze}


# --------------------------------------------------------------------------- plan


def test_plan_round_trips_and_hash_excludes_generated_at_only(strong):
    plan = strong["plan"]
    assert plan["schema"] == HOLDOUT_PLAN_SCHEMA
    # canonical JSON round trip revalidates cleanly (lists everywhere, no tuples)
    round_tripped = json.loads(json.dumps(plan))
    validate_holdout_plan(round_tripped, strong["config"])
    # deterministic rebuild: same inputs -> same content hash
    rebuilt = build_holdout_plan(strong["config"], strong["inventory"],
                                 generated_at=GEN_AT)
    assert rebuilt == plan
    # generated_at is the only volatile field: a different timestamp, same hash
    other = build_holdout_plan(strong["config"], strong["inventory"],
                               generated_at="2026-07-19T12:00:00Z")
    assert other["sha256"] == plan["sha256"]
    assert holdout_plan_sha256(plan) == plan["sha256"]


def test_plan_pins_exact_31_day_scope_and_stable_ids(strong):
    plan = strong["plan"]
    days = january_days()
    assert len(days) == 31
    assert sorted(plan["included_days"] + list(plan["excluded_days"])) == days
    assert len(plan["included_days"]) == 29
    assert set(plan["excluded_days"]) == {"2026-01-14", "2026-01-25"}
    for entry in plan["excluded_days"].values():
        assert entry["reason"]
        assert len(entry["evidence_sha256"]) == 64
    assert plan["holdout_universe_id"] == G0BN_HOLDOUT_UNIVERSE_ID
    assert plan["transaction_id"] == G0BN_TRANSACTION_ID
    assert plan["holdout_start_ns"] == 1_767_225_600_000_000_000
    assert plan["holdout_end_ns"] == 1_769_904_000_000_000_000


def test_plan_allowlist_accounting_reconciles_exactly(strong):
    plan = strong["plan"]
    allow = plan["object_allowlist"]
    assert plan["n_allowlist_objects"] == len(allow)
    assert len(allow) == 29 * (len(RAW_PRODUCTS) + len(NORMALIZED_PRODUCTS))
    # canonical deterministic order
    keys = [(o["day"], o["layer"], o["product"], o["object_id"]) for o in allow]
    assert keys == sorted(keys)
    # complete coverage: every included day carries every certified product once
    seen = {(o["day"], o["layer"], o["product"]) for o in allow}
    for day in plan["included_days"]:
        for product in RAW_PRODUCTS:
            assert (day, "raw", product) in seen
        for product in NORMALIZED_PRODUCTS:
            assert (day, "normalized", product) in seen


def test_plan_binds_config_evidence_and_recipe(strong):
    plan, config = strong["plan"], strong["config"]
    cert = config["source_certification"]
    assert plan["protocol_config_sha256"] == config["sha256"]
    assert plan["source_certification_sha256"] == cert["certification_sha256"]
    assert plan["custodian_seal_sha256"] == cert["custodian_seal_sha256"]
    assert plan["coverage_sha256"] == cert["coverage_sha256"]
    assert plan["partition_plan_sha256"] == config["partition"]["sha256"]
    assert plan["producer"] == config["producer"]
    assert plan["software_sha256"] == hash_obj(config["software"])
    assert plan["oos"] == config["oos"]
    assert plan["output_contract"]["dataset_id"] == "binance_single_venue_g0bn_oos"
    assert plan["drop_count_categories"] == \
        config["partition"]["holdout_drop_count_categories"]
    assert plan["sufficiency_thresholds"] == \
        config["partition"]["sufficiency_thresholds"]


def test_plan_precomputes_the_oos_bootstrap_draw_hash(strong):
    plan = strong["plan"]
    boot = plan["oos_bootstrap"]
    assert boot["days"] == plan["included_days"]
    assert boot["alpha_oos"] == 0.025
    assert boot["seed"] == 0 and boot["n_boot"] == 10000
    _, draw_sha = bootstrap_draws(plan["included_days"])
    assert boot["draw_sha256"] == draw_sha


@pytest.mark.parametrize("mutate,match", [
    # 31-day accounting breaks
    (lambda inv: inv["excluded_days"].pop("2026-01-14"), "included_days nor"),
    (lambda inv: inv.__setitem__(
        "included_days", sorted(inv["included_days"] + ["2026-01-14"])),
     "both included_days and excluded_days"),
    (lambda inv: inv["included_days"].__setitem__(0, "2025-12-31"), "outside"),
    (lambda inv: inv["included_days"].append("2026-02-01"), "outside"),
    # allowlist breaks
    (lambda inv: inv["objects"].append(
        {"object_id": "custody/raw/book/2026-01-14", "layer": "raw",
         "product": "book", "day": "2026-01-14",
         "sha256": sha_hex("excluded-day-object")}), "not an included day"),
    (lambda inv: inv["objects"].append(
        {"object_id": "custody/raw/book/2026-02-02", "layer": "raw",
         "product": "book", "day": "2026-02-02",
         "sha256": sha_hex("outside-object")}), "not an included day"),
    (lambda inv: inv["objects"].append(
        {"object_id": "custody/raw/funding/2026-01-02", "layer": "raw",
         "product": "funding_rates", "day": "2026-01-02",
         "sha256": sha_hex("funding")}), "certified"),
    (lambda inv: inv["objects"].append(
        {"object_id": "custody/cooked/trades/2026-01-02", "layer": "cooked",
         "product": "binance_futures_trades", "day": "2026-01-02",
         "sha256": sha_hex("bad-layer")}), "must be one of"),
    (lambda inv: inv["objects"].append(
        {"object_id": "custody/normalized/spot/2026-01-02", "layer": "normalized",
         "product": "binance_spot_trades", "day": "2026-01-02",
         "sha256": sha_hex("wrong-normalized-product")}), "certified"),
    # a raw product id on the normalized layer: each layer has its OWN set
    (lambda inv: inv["objects"].append(
        {"object_id": "custody/normalized/rawname/2026-01-02",
         "layer": "normalized", "product": "book_delta_v2",
         "day": "2026-01-02", "sha256": sha_hex("raw-on-normalized")}),
     "certified"),
    (lambda inv: inv["objects"].append(dict(inv["objects"][0])), "duplicate"),
    # a SECOND object (fresh object_id) for an already-covered slot is
    # ambiguous custody and an activity-proxy channel: exactly one object per
    # (day, layer, product)
    (lambda inv: inv["objects"].append(
        {"object_id": "custody/raw/book/2026-01-01/part-2", "layer": "raw",
         "product": "book", "day": "2026-01-01",
         "sha256": sha_hex("second-object-same-slot")}), "duplicate"),
    (lambda inv: inv["objects"].__setitem__(
        3, dict(inv["objects"][3], object_id=inv["objects"][2]["object_id"])),
     "duplicate"),
    (lambda inv: inv.__setitem__(
        "objects", [o for o in inv["objects"]
                    if not (o["day"] == "2026-01-02" and o["product"] == "trades"
                            and o["layer"] == "raw")]), "missing"),
    # activity proxies stay inside custody (spec section 5.1)
    (lambda inv: inv["objects"][0].__setitem__("byte_size", 123), "unknown"),
    (lambda inv: inv["objects"][0].__setitem__("record_count", 5), "unknown"),
    # custody evidence must reconcile with the config pins
    (lambda inv: inv.__setitem__("custodian_seal_sha256", sha_hex("foreign-seal")),
     "custodian_seal_sha256"),
    (lambda inv: inv.__setitem__("operator_identity", "g0bn-custodian-svc"),
     "distinct"),
    (lambda inv: inv.__setitem__("custodian_identity", "TBD"), "placeholder"),
    (lambda inv: inv.pop("permission_policy_sha256"), "missing"),
])
def test_plan_rejects_broken_custody_inventory(strong, mutate, match):
    inventory = copy.deepcopy(strong["inventory"])
    mutate(inventory)
    with pytest.raises(ValueError, match=match):
        build_holdout_plan(strong["config"], inventory, generated_at=GEN_AT)


def test_plan_rejects_duplicate_checksums_within_a_product(strong):
    """Byte-identical sealed objects within one (layer, product) would collapse
    in any hash-set audit and leave a sealed object unaccounted for: ambiguous
    custody fails closed (Codex round 9)."""
    inventory = copy.deepcopy(strong["inventory"])
    objs = [o for o in inventory["objects"]
            if o["layer"] == "normalized"
            and o["product"] == "binance_futures_trades"]
    objs[1]["sha256"] = objs[0]["sha256"]
    with pytest.raises(ValueError, match="checksum"):
        build_holdout_plan(strong["config"], inventory, generated_at=GEN_AT)


def test_plan_rejects_insufficient_included_days(strong):
    days = january_days()
    excluded = {d: {"reason": "custody_source_gap",
                    "evidence_sha256": sha_hex(f"x-{d}")} for d in days[:25]}
    included = days[25:]
    # carry the config's sealed pin so the scalar check passes and the
    # day-scope sufficiency check (this test's target) fires first
    inventory = make_inventory(
        included_days=included, excluded_days=excluded,
        objects=make_objects(included),
        custodian_seal_sha256=strong["config"]["source_certification"][
            "custodian_seal_sha256"])
    with pytest.raises(ValueError, match="min_valid_days"):
        build_holdout_plan(strong["config"], inventory, generated_at=GEN_AT)


def test_plan_tamper_detection(strong):
    plan, config = strong["plan"], strong["config"]
    # any mutation without rehashing breaks the embedded self-hash
    for key, value in (("included_days", strong["plan"]["included_days"][:-1]),
                       ("n_allowlist_objects", 7),
                       ("build_binding_rule", "other_rule")):
        tampered = copy.deepcopy(plan)
        tampered[key] = value
        with pytest.raises(ValueError):
            validate_holdout_plan(tampered, config)
    # a rehashed foreign config pin fails the cross-check, not the self-hash
    tampered = copy.deepcopy(plan)
    tampered["protocol_config_sha256"] = sha_hex("foreign-config")
    with pytest.raises(ValueError, match="protocol_config_sha256"):
        validate_holdout_plan(_resha(tampered), config)
    tampered = copy.deepcopy(plan)
    tampered["transaction_id"] = sha_hex("foreign-transaction")
    with pytest.raises(ValueError, match="transaction_id"):
        validate_holdout_plan(_resha(tampered), config)
    # validating against a different (edited) config fails
    other = dev_config(costs=make_costs(no_trade_margin_bps=0.75))
    with pytest.raises(ValueError, match="protocol_config_sha256"):
        validate_holdout_plan(plan, other)


@pytest.mark.parametrize("field", PROHIBITED_JANUARY_FIELDS)
def test_plan_rejects_every_prohibited_january_field(strong, field):
    tampered = copy.deepcopy(strong["plan"])
    tampered[field] = sha_hex("january-outcome")
    with pytest.raises(ValueError, match="unknown"):
        validate_holdout_plan(_resha(tampered), strong["config"])


def test_plan_rejects_nested_january_and_unknown_fields(strong):
    config = strong["config"]
    tampered = copy.deepcopy(strong["plan"])
    tampered["output_contract"]["holdout_row_count"] = 12345
    with pytest.raises(ValueError, match="unknown"):
        validate_holdout_plan(_resha(tampered), config)
    tampered = copy.deepcopy(strong["plan"])
    tampered["object_allowlist"][0]["record_count"] = 9
    with pytest.raises(ValueError, match="unknown"):
        validate_holdout_plan(_resha(tampered), config)
    tampered = copy.deepcopy(strong["plan"])
    tampered["oos_bootstrap"]["realized_days"] = ["2026-01-05"]
    with pytest.raises(ValueError, match="unknown"):
        validate_holdout_plan(_resha(tampered), config)


def test_stable_ids_survive_config_source_and_freeze_edits(strong):
    plan = strong["plan"]
    # config edit (different cost margin -> different config sha) — same transaction
    inv_b = make_inventory()
    config_b = dev_config(costs=make_costs(no_trade_margin_bps=0.75),
                          **sealed_config_kwargs(inv_b))
    assert config_b["sha256"] != strong["config"]["sha256"]
    plan_b = build_holdout_plan(config_b, inv_b, generated_at=GEN_AT)
    assert plan_b["holdout_universe_id"] == plan["holdout_universe_id"]
    assert plan_b["transaction_id"] == plan["transaction_id"]
    # source-certification edit — same transaction
    inv_c = make_inventory()
    cert_c = make_source_certification(
        certification_sha256=sha_hex("alternate-64-evidence"),
        custodian_seal_sha256=inv_c["custodian_seal_sha256"],
        development_source_manifest_sha256=dev_source_manifest_sha256())
    config_c = dev_config(source_certification=cert_c)
    plan_c = build_holdout_plan(config_c, inv_c, generated_at=GEN_AT)
    assert plan_c["holdout_universe_id"] == plan["holdout_universe_id"]
    assert plan_c["transaction_id"] == plan["transaction_id"]
    # plan/freeze regeneration — same transaction
    freeze_b = build_freeze(strong["run"], plan, inventory=strong["inventory"],
                            generated_at="2026-07-19T23:00:00Z")
    assert freeze_b["transaction_id"] == plan["transaction_id"]
    assert freeze_b["sha256"] == strong["freeze"]["sha256"]


# ------------------------------------------------------------------ build binding


def test_oos_build_params_bind_the_plan_hash(strong):
    plan, config = strong["plan"], strong["config"]
    inventory = strong["inventory"]
    params = oos_build_params(plan, {"producer": "t9", "seed": 1},
                              config=config, inventory=inventory)
    assert params["holdout_plan_sha256"] == holdout_plan_sha256(plan)
    verify_oos_build_binding(params, plan, config=config, inventory=inventory)
    # the caller cannot supply or pre-empt the binding
    with pytest.raises(ValueError, match="holdout_plan_sha256"):
        oos_build_params(plan, {"holdout_plan_sha256": sha_hex("supplied")},
                         config=config, inventory=inventory)
    with pytest.raises(ValueError, match="generated_at"):
        oos_build_params(plan, {"generated_at": "2026-07-19T00:00:00Z"},
                         config=config, inventory=inventory)
    # a VALID but different plan (different extra_cols) fails the hash binding
    plan_variant = build_holdout_plan(
        config, inventory,
        extra_cols=["latency_drift_bps", "emitted_by_time_cap"],
        generated_at=GEN_AT)
    with pytest.raises(ValueError, match="holdout_plan_sha256"):
        verify_oos_build_binding(params, plan_variant, config=config,
                                 inventory=inventory)
    # the verifier mirrors the builder's invariants: a timestamp-bearing
    # recipe must fail the binding gate, not only the later writer path
    with pytest.raises(ValueError, match="generated_at"):
        verify_oos_build_binding(
            dict(params, generated_at="2026-07-19T00:00:00Z"), plan,
            config=config, inventory=inventory)
    # an internally tampered (rehashed) plan fails its own config validation
    tampered = _resha(dict(copy.deepcopy(plan),
                           included_days=plan["included_days"][:-1]))
    with pytest.raises(ValueError):
        verify_oos_build_binding(params, tampered, config=config,
                                 inventory=inventory)
    # a hand-edited plan whose embedded self-hash was not updated is rejected
    stale = copy.deepcopy(plan)
    stale["n_allowlist_objects"] += 1
    with pytest.raises(ValueError, match="sha256"):
        holdout_plan_sha256(stale)


def test_future_build_binds_plan_hash_through_the_t8_writer(strong, tmp_path):
    plan, freeze, config = strong["plan"], strong["freeze"], strong["config"]
    inventory = strong["inventory"]
    frame = g0bn_frame()
    man, params = oos_manifest_and_params(config, plan, freeze, frame, inventory)
    result = write_holdout(frame, man, build_params=params,
                           matrix_path=tmp_path / "oos.parquet",
                           manifest_path=tmp_path / "oos_manifest.json")
    # the physical schema the writer attests matches the plan's pinned expectation
    assert result.physical_schema_sha256 == \
        plan["output_contract"]["expected_physical_schema_sha256"]
    verify_holdout_manifest_binding(man, plan, freeze, config=config,
                                    inventory=inventory)
    # the generic-runner guard (67-D) still refuses the bound manifest pre-loader
    with pytest.raises(ValueError, match="holdout"):
        preflight_generic_manifest(man)
    # a build params dict missing the binding is refused by the writer itself
    bad = {k: v for k, v in params.items() if k != "holdout_plan_sha256"}
    with pytest.raises(ValueError, match="holdout_plan_sha256"):
        write_holdout(frame, man, build_params=bad,
                      matrix_path=tmp_path / "b.parquet",
                      manifest_path=tmp_path / "b.json")
    # a VALID but different plan breaks the manifest binding verification
    plan_variant = build_holdout_plan(
        config, inventory,
        extra_cols=["latency_drift_bps", "emitted_by_time_cap"],
        generated_at=GEN_AT)
    with pytest.raises(ValueError, match="holdout_plan"):
        verify_holdout_manifest_binding(man, plan_variant, freeze,
                                        config=config, inventory=inventory)


def test_manifest_binding_rejects_coordinated_plan_freeze_rehash(strong):
    # An attacker who edits the plan and rehashes BOTH the plan and the freeze
    # so they agree internally must still fail: the binding verifiers re-anchor
    # both artifacts to the immutable config AND the custody inventory.
    plan, freeze, config = strong["plan"], strong["freeze"], strong["config"]
    inventory = strong["inventory"]
    man, _ = oos_manifest_and_params(config, plan, freeze, g0bn_frame(),
                                     inventory)

    def _coordinated(mutate):
        plan_t = copy.deepcopy(plan)
        mutate(plan_t)
        plan_t = _resha(plan_t)
        freeze_t = copy.deepcopy(freeze)
        freeze_t["holdout_plan_sha256"] = hash_obj(
            plan_t, exclude_keys=("sha256", "generated_at"))
        return plan_t, _resha(freeze_t)

    # loosened output contract -> caught by the config anchor
    plan_t, freeze_t = _coordinated(
        lambda p: p["output_contract"]["dtypes"].__setitem__("cost_bps",
                                                             "float32"))
    with pytest.raises(ValueError, match="output_contract"):
        verify_holdout_manifest_binding(man, plan_t, freeze_t, config=config,
                                        inventory=inventory)
    # swapped sealed-object hash -> caught by the custody-inventory anchor
    # (the config cannot pin January object hashes; the #68 inventory does)
    plan_t, freeze_t = _coordinated(
        lambda p: p["object_allowlist"][0].__setitem__(
            "sha256", sha_hex("never-sealed-object")))
    with pytest.raises(ValueError, match="object_allowlist"):
        verify_holdout_manifest_binding(man, plan_t, freeze_t, config=config,
                                        inventory=inventory)
    with pytest.raises(ValueError, match="object_allowlist"):
        verify_oos_build_binding(
            {"holdout_plan_sha256": hash_obj(
                plan_t, exclude_keys=("sha256", "generated_at"))},
            plan_t, config=config, inventory=inventory)


def test_manifest_binding_rejects_unsealed_source_and_foreign_freeze(strong, weak):
    plan, freeze, config = strong["plan"], strong["freeze"], strong["config"]
    inventory = strong["inventory"]
    man, _ = oos_manifest_and_params(config, plan, freeze, g0bn_frame(),
                                     inventory)
    # a normalized data-source pin the custodian never sealed fails closed
    bad = copy.deepcopy(man)
    next(s for s in bad["sources"]
         if s.get("name") == "binance_futures_trades")["sha256"] = \
        sha_hex("unsealed-normalized-object")
    with pytest.raises(ValueError, match="sealed normalized allowlist"):
        verify_holdout_manifest_binding(bad, plan, freeze, config=config,
                                        inventory=inventory)
    # an incomplete per-product hash set (one sealed object dropped) fails:
    # the manifest must account for the COMPLETE sealed scope, not a subset
    incomplete = copy.deepcopy(man)
    for i, s in enumerate(incomplete["sources"]):
        if s.get("name") == "binance_futures_trades":
            del incomplete["sources"][i]
            break
    with pytest.raises(ValueError, match="complete sealed scope"):
        verify_holdout_manifest_binding(incomplete, plan, freeze, config=config,
                                        inventory=inventory)
    # duplicate source entries break one-to-one custody accounting even when
    # the unique hash set still equals the sealed allowlist
    dup = copy.deepcopy(man)
    dup["sources"].append(dict(next(
        s for s in dup["sources"] if s.get("name") == "binance_futures_trades")))
    with pytest.raises(ValueError, match="one-to-one"):
        verify_holdout_manifest_binding(dup, plan, freeze, config=config,
                                        inventory=inventory)
    # outcome-blind clock/timing metadata must match the frozen config pins
    drift = copy.deepcopy(man)
    drift["bar_clock"]["target_bars_per_day"] = 9999
    with pytest.raises(ValueError, match="bar_clock"):
        verify_holdout_manifest_binding(drift, plan, freeze, config=config,
                                        inventory=inventory)
    drift = copy.deepcopy(man)
    drift["max_lookback_ns"] = drift["embargo_ns"] = 123_000_000_000
    with pytest.raises(ValueError, match="lookback|embargo"):
        verify_holdout_manifest_binding(drift, plan, freeze, config=config,
                                        inventory=inventory)
    # a coverage evidence entry, when present, must pin the plan's coverage
    good = copy.deepcopy(man)
    good["sources"].append({"name": "coverage",
                            "sha256": plan["coverage_sha256"]})
    verify_holdout_manifest_binding(good, plan, freeze, config=config,
                                    inventory=inventory)
    bad_cov = copy.deepcopy(man)
    bad_cov["sources"].append({"name": "coverage",
                               "sha256": sha_hex("foreign-coverage")})
    with pytest.raises(ValueError, match="coverage"):
        verify_holdout_manifest_binding(bad_cov, plan, freeze, config=config,
                                        inventory=inventory)
    # a foreign freeze (different run, same plan/config) cannot verify a
    # manifest bound to the original freeze
    assert weak["freeze"]["sha256"] != freeze["sha256"]
    with pytest.raises(ValueError, match="freeze_sha256"):
        verify_holdout_manifest_binding(man, plan, weak["freeze"],
                                        config=config, inventory=inventory)


# -------------------------------------------------------------------------- freeze


def test_freeze_builds_validates_and_reproduces(strong):
    freeze, config, plan = strong["freeze"], strong["config"], strong["plan"]
    assert freeze["schema"] == FREEZE_SCHEMA
    validate_freeze(json.loads(json.dumps(freeze)), config=config, plan=plan)
    rebuilt = build_freeze(strong["run"], plan,
                           inventory=strong["inventory"], generated_at=GEN_AT)
    assert rebuilt == freeze
    assert freeze_sha256(freeze) == freeze["sha256"]
    # the re-derived development result is what the freeze pins
    derived = development_selection(strong["run"])
    assert freeze["development"]["result_sha256"] == derived["result_sha256"]
    assert freeze["ledger"]["n_effective_trials"] == 15
    assert freeze["ledger"]["ledger_sha256"] == strong["run"].ledger.ledger_sha256()
    assert freeze["holdout_plan_sha256"] == holdout_plan_sha256(plan)
    assert freeze["holdout_universe_id"] == G0BN_HOLDOUT_UNIVERSE_ID
    assert freeze["transaction_id"] == G0BN_TRANSACTION_ID


def test_freeze_pins_selected_primaries_and_60s_control_set(strong):
    freeze = strong["freeze"]
    run = strong["run"]
    assert set(freeze["selected"]) == {"2s", "10s"}
    for tag, sel in freeze["selected"].items():
        assert sel["candidate_id"] in SELECTABLE
        assert sel["mode"] == "trade"
        assert sel["ranked_candidate_ids"][0] == sel["candidate_id"]
        assert "persistence_zero" not in sel["ranked_candidate_ids"]
        # the embedded identity re-derives the pinned trial id and is in the ledger
        assert sel["identity"]["horizon"] == tag
        assert sel["identity"]["candidate_id"] == sel["candidate_id"]
        assert sel["trial_id"] in run.ledger.trial_ids()
        assert sel["dsr"]["provenance"]["n_trials"] == 15
        assert sel["dsr"]["value"] > 0.95
    controls = freeze["control_candidates"]["60s"]
    assert [c["candidate_id"] for c in controls] == list(CANDIDATE_LADDER)
    for c in controls:
        assert c["identity"]["horizon"] == "60s"
        assert c["trial_id"] in run.ledger.trial_ids()
    for tag, block in freeze["pbo"].items():
        assert block["available"] is True
        assert block["value"] < 0.5
        assert block["split_sha256"] == freeze["development"]["split_sha256s"][tag]


def test_freeze_contains_no_january_value_anywhere(strong):
    # The freeze must not mention any January DAY, build, schedule, or result:
    # its canonical JSON text carries no 2026-01-DD day string anywhere (the
    # holdout bounds appear only as pinned integers; the pilot_id names the
    # window as the fixed protocol identity, not a day).
    text = json.dumps({k: v for k, v in strong["freeze"].items()
                       if k != "generated_at"})
    assert re.search(r"2026-01-\d{2}", text) is None


def test_weak_run_freezes_in_predictive_mode(weak):
    freeze = weak["freeze"]
    validate_freeze(freeze, config=weak["config"], plan=weak["plan"])
    for tag in ("2s", "10s"):
        assert freeze["selected"][tag]["mode"] == "predictive"


def test_blocked_selection_cannot_build_a_freeze(strong):
    blocked_inventory = make_inventory()
    frame, manifest, config, identity = dev_bundle(
        seed=31, config=dev_config(**sealed_config_kwargs(blocked_inventory)))
    rng = np.random.default_rng(99)
    shuffled = frame.copy()
    for tag in ("2s", "10s", "60s"):
        idx = shuffled.index[shuffled["horizon"] == tag].to_numpy()
        y = shuffled.loc[idx, "y_fwd_bps"].to_numpy()
        y = y[rng.permutation(len(y))]
        shuffled.loc[idx, "y_fwd_bps"] = y
        label = np.zeros(len(y), dtype=np.int64)
        label[y > 9.0] = 1
        label[y < -9.0] = -1
        shuffled.loc[idx, "label"] = label
    manifest2 = dev_manifest(config, shuffled)
    identity2 = dev_data_identity(config, manifest2, shuffled)
    run = run_g0bn_development(shuffled, manifest2, config, identity2,
                               durable_ledger())
    plan = build_holdout_plan(config, blocked_inventory, generated_at=GEN_AT)
    with pytest.raises(ValueError, match="predictive-eligible"):
        build_freeze(run, plan, inventory=blocked_inventory,
                     generated_at=GEN_AT)


def test_freeze_rejects_expected_development_result_mismatch(strong):
    derived = development_selection(strong["run"])
    # an internally consistent but DIFFERENT result must not be frozen
    edited = copy.deepcopy(derived)
    edited["selection"]["2s"]["selected_candidate_id"] = "lgbm_reg"
    edited["result_sha256"] = hash_obj(edited, exclude_keys=("result_sha256",
                                                             "generated_at"))
    with pytest.raises(ValueError, match="development result"):
        build_freeze(strong["run"], strong["plan"], generated_at=GEN_AT,
                     inventory=strong["inventory"],
                     expected_development_result=edited)
    # a tampered embedded result hash fails before any comparison
    tampered = copy.deepcopy(derived)
    tampered["result_sha256"] = sha_hex("tampered-result")
    with pytest.raises(ValueError, match="result_sha256"):
        build_freeze(strong["run"], strong["plan"], generated_at=GEN_AT,
                     inventory=strong["inventory"],
                     expected_development_result=tampered)
    # the matching re-derived result freezes identically
    ok = build_freeze(strong["run"], strong["plan"], generated_at=GEN_AT,
                      inventory=strong["inventory"],
                      expected_development_result=derived)
    assert ok["sha256"] == strong["freeze"]["sha256"]


def test_freeze_rejects_tampered_forecasts(strong):
    run = strong["run"]
    tampered = copy.copy(run)
    tampered.forecasts = {tid: fc.copy() for tid, fc in run.forecasts.items()}
    tid = next(iter(tampered.forecasts))
    tampered.forecasts[tid] = tampered.forecasts[tid] + 1.0
    with pytest.raises(ValueError, match="forecast"):
        build_freeze(tampered, strong["plan"],
                     inventory=strong["inventory"], generated_at=GEN_AT)


def test_freeze_rejects_a_foreign_plan(strong):
    inv_b = make_inventory()
    config_b = dev_config(costs=make_costs(no_trade_margin_bps=0.75),
                          **sealed_config_kwargs(inv_b))
    plan_b = build_holdout_plan(config_b, inv_b, generated_at=GEN_AT)
    with pytest.raises(ValueError, match="protocol_config_sha256"):
        build_freeze(strong["run"], plan_b, inventory=inv_b,
                     generated_at=GEN_AT)


def test_freeze_tamper_detection(strong):
    freeze, config, plan = strong["freeze"], strong["config"], strong["plan"]
    # any unrehashed mutation breaks a cross-check or the self-hash
    tampered = copy.deepcopy(freeze)
    current = tampered["selected"]["2s"]["candidate_id"]
    tampered["selected"]["2s"]["candidate_id"] = \
        "ofi_ridge" if current != "ofi_ridge" else "lgbm_reg"
    with pytest.raises(ValueError):
        validate_freeze(tampered, config=config, plan=plan)
    # a rehashed foreign ledger pin fails the provenance cross-checks
    tampered = copy.deepcopy(freeze)
    tampered["ledger"]["ledger_sha256"] = sha_hex("foreign-ledger")
    with pytest.raises(ValueError, match="ledger"):
        validate_freeze(_resha(tampered), config=config, plan=plan)
    # a rehashed identity swap breaks the trial id re-derivation
    tampered = copy.deepcopy(freeze)
    tampered["selected"]["2s"]["identity"]["feature_cols"] = ["spread_tick"]
    with pytest.raises(ValueError):
        validate_freeze(_resha(tampered), config=config, plan=plan)
    # a VALID but different plan (different extra_cols -> different hash) breaks
    # the freeze's plan-hash binding; an internally torn plan already fails its
    # own validation first
    plan_variant = build_holdout_plan(
        config, strong["inventory"],
        extra_cols=["latency_drift_bps", "emitted_by_time_cap"],
        generated_at=GEN_AT)
    assert plan_variant["sha256"] != plan["sha256"]
    with pytest.raises(ValueError, match="holdout_plan_sha256"):
        validate_freeze(freeze, config=config, plan=plan_variant)


@pytest.mark.parametrize("field", PROHIBITED_JANUARY_FIELDS)
def test_freeze_rejects_every_prohibited_january_field(strong, field):
    tampered = copy.deepcopy(strong["freeze"])
    tampered[field] = sha_hex("january-outcome")
    with pytest.raises(ValueError, match="unknown"):
        validate_freeze(_resha(tampered), config=strong["config"],
                        plan=strong["plan"])


def test_freeze_rejects_nested_january_fields(strong):
    config, plan = strong["config"], strong["plan"]
    tampered = copy.deepcopy(strong["freeze"])
    tampered["development"]["holdout_logical_row_sha256"] = sha_hex("jan-rows")
    with pytest.raises(ValueError, match="unknown"):
        validate_freeze(_resha(tampered), config=config, plan=plan)
    tampered = copy.deepcopy(strong["freeze"])
    tampered["selected"]["2s"]["oos_lift"] = 0.25
    with pytest.raises(ValueError, match="unknown"):
        validate_freeze(_resha(tampered), config=config, plan=plan)
    tampered = copy.deepcopy(strong["freeze"])
    tampered["ledger"]["matrix_file_sha256"] = sha_hex("jan-file")
    with pytest.raises(ValueError, match="unknown"):
        validate_freeze(_resha(tampered), config=config, plan=plan)
    tampered = copy.deepcopy(strong["freeze"])
    tampered["pbo"]["2s"]["oos_realized_value"] = 0.11
    with pytest.raises(ValueError, match="unknown"):
        validate_freeze(_resha(tampered), config=config, plan=plan)
    tampered = copy.deepcopy(strong["freeze"])
    tampered["selected"]["2s"]["dsr"]["provenance"]["oos_metric"] = \
        sha_hex("jan-oos")
    with pytest.raises(ValueError, match="unknown"):
        validate_freeze(_resha(tampered), config=config, plan=plan)
    tampered = copy.deepcopy(strong["freeze"])
    tampered["control_candidates"]["60s"][0]["oos_forecast_sha256"] = \
        sha_hex("jan-fc")
    with pytest.raises(ValueError, match="unknown"):
        validate_freeze(_resha(tampered), config=config, plan=plan)


def test_run_reconciliation_rejects_fabricated_statistics(strong):
    """The freeze's DSR/PBO numbers are development evidence pinned by hashes an
    artifact editor can recompute. The structural (no-run) validation level
    cannot authenticate them; the run-anchored level — the one-shot runner's
    spec-6.3 step-1 self-verification — re-derives the whole freeze and refuses
    any value development_selection cannot reproduce."""
    config, plan, freeze = strong["config"], strong["plan"], strong["freeze"]
    fabricated = copy.deepcopy(freeze)
    fabricated["selected"]["2s"]["dsr"]["value"] = 0.9999
    fabricated["pbo"]["2s"]["value"] = 0.01
    fabricated = _resha(fabricated)
    # hash-consistent fabrication passes the structural level...
    validate_freeze(fabricated, config=config, plan=plan)
    # ...and fails closed at the run-anchored reproducibility level
    with pytest.raises(ValueError, match="reproduce"):
        validate_freeze(fabricated, config=config, plan=plan,
                        run=strong["run"], inventory=strong["inventory"])
    # the authentic freeze reconciles exactly
    validate_freeze(freeze, config=config, plan=plan, run=strong["run"],
                    inventory=strong["inventory"])
    # the run-anchored level refuses to run without the custody anchor
    with pytest.raises(ValueError, match="inventory"):
        validate_freeze(freeze, config=config, plan=plan, run=strong["run"])


# ------------------------------------------------------- outcome access boundaries


def test_freeze_construction_performs_zero_january_reads(strong, monkeypatch):
    """The loader/read spy required by issue #89: plan+freeze construction touches
    no parquet loader and opens nothing but Python module sources (the runtime
    code-identity hashes)."""
    opened: list = []
    real_open = builtins.open

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("parquet loader invoked during outcome-blind "
                             "plan/freeze construction")

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(pq, "read_pandas", forbidden)

    plan = build_holdout_plan(strong["config"], strong["inventory"],
                              generated_at=GEN_AT)
    freeze = build_freeze(strong["run"], plan,
                          inventory=strong["inventory"], generated_at=GEN_AT)

    assert freeze["sha256"] == strong["freeze"]["sha256"]
    assert opened, "the code-identity hashes read module sources"
    non_source = [p for p in opened if not p.endswith(".py")]
    assert non_source == [], f"unexpected file access: {non_source}"


@pytest.mark.parametrize("break_config,match", [
    # missing/placeholder operator, custody, source, cost, threshold, software
    (lambda c: c["source_certification"].pop("custodian_seal_sha256"), "missing"),
    (lambda c: c["source_certification"].__setitem__("operator_identity", "TBD"),
     "placeholder"),
    (lambda c: c["source_certification"].__setitem__(
        "certification_sha256", "TBD"), "hex"),
    (lambda c: c["costs"].pop("fee_tier"), "missing"),
    (lambda c: c["costs"]["cost_assumption"].__setitem__(
        "taker_fee_bps", float("nan")), "finite"),
    (lambda c: c["verdict_thresholds"].pop("dsr_threshold"), "missing"),
    (lambda c: c["software"].pop("lightgbm_version"), "missing"),
    (lambda c: c["labels"].__setitem__("ewma_half_life_ns", "TBD"), "integer"),
])
def test_missing_evidence_fails_before_any_burn(strong, break_config, match):
    # No rehash needed (or possible for the NaN case): the config section
    # validators run before the self-hash check, so the targeted evidence
    # failure fires first.
    config = copy.deepcopy(strong["config"])
    break_config(config)
    with pytest.raises(ValueError, match=match):
        build_holdout_plan(config, make_inventory(), generated_at=GEN_AT)


def test_validate_custody_inventory_is_exactly_the_build_input(strong):
    validated = validate_custody_inventory(strong["inventory"], strong["config"])
    assert validated is strong["inventory"]
    with pytest.raises(ValueError, match="unknown"):
        validate_custody_inventory(
            dict(strong["inventory"], holdout_row_count=3), strong["config"])


def test_none_inventory_cannot_bypass_the_custody_anchor(strong):
    """inventory= is required SEMANTICALLY, not just syntactically: an explicit
    None must fail closed at every custody-anchored entry point rather than
    silently degrading to config-only validation (Codex round 5, P1)."""
    plan, freeze = strong["plan"], strong["freeze"]
    config, run = strong["config"], strong["run"]
    with pytest.raises(ValueError, match="inventory"):
        build_freeze(run, plan, inventory=None, generated_at=GEN_AT)
    with pytest.raises(ValueError, match="inventory"):
        oos_build_params(plan, {"producer": "t9"}, config=config,
                         inventory=None)
    with pytest.raises(ValueError, match="inventory"):
        verify_oos_build_binding({"holdout_plan_sha256": plan["sha256"]}, plan,
                                 config=config, inventory=None)
    with pytest.raises(ValueError, match="inventory"):
        holdout_plan_binding(plan, freeze, config=config, inventory=None)
    with pytest.raises(ValueError, match="inventory"):
        verify_holdout_manifest_binding({}, plan, freeze, config=config,
                                        inventory=None)


def test_inventory_content_is_bound_to_the_sealed_pin(strong):
    """The custodian seal hash is a CONTENT commitment: an inventory whose
    scalar pins are copied from the config but whose objects or day accounting
    were edited must fail the recomputed seal-content binding (Codex round 11,
    P1) — a forged allowlist can never reuse a sealed pin."""
    forged = copy.deepcopy(strong["inventory"])
    forged["objects"][0]["sha256"] = sha_hex("forged-object-content")
    with pytest.raises(ValueError, match="custodian seal"):
        validate_custody_inventory(forged, strong["config"])
    forged = copy.deepcopy(strong["inventory"])
    forged["excluded_days"]["2026-01-14"]["reason"] = "edited_reason"
    with pytest.raises(ValueError, match="custodian seal"):
        validate_custody_inventory(forged, strong["config"])
    with pytest.raises(ValueError, match="custodian seal"):
        build_holdout_plan(strong["config"],
                           copy.deepcopy(dict(strong["inventory"],
                                              objects=strong["inventory"]["objects"][:-1]
                                              + [dict(strong["inventory"]["objects"][-1],
                                                      sha256=sha_hex("swap"))])),
                           generated_at=GEN_AT)


def test_supplied_inventory_is_itself_validated_against_the_config(strong):
    """A caller-supplied inventory is only a custody anchor if IT reconciles
    with the config's seal pins: a forged minimal dict mirroring the plan's own
    scope fields, or a full inventory carrying a foreign seal hash, must fail
    before any comparison lets a binding hash be derived (Codex round 2)."""
    plan, config = strong["plan"], strong["config"]
    forged = {
        "included_days": list(plan["included_days"]),
        "excluded_days": copy.deepcopy(plan["excluded_days"]),
        "objects": [dict(o) for o in plan["object_allowlist"]],
    }
    with pytest.raises(ValueError, match="missing"):
        validate_holdout_plan(plan, config, inventory=forged)
    mismatched = make_inventory(custodian_seal_sha256=sha_hex("foreign-seal"))
    with pytest.raises(ValueError, match="custodian_seal_sha256"):
        validate_holdout_plan(plan, config, inventory=mismatched)
    with pytest.raises(ValueError, match="custodian_seal_sha256"):
        verify_oos_build_binding(
            {"holdout_plan_sha256": plan["sha256"]}, plan,
            config=config, inventory=mismatched)
