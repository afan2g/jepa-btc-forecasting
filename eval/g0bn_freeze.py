"""Outcome-blind G0-BN holdout plan and selection freeze (issue #89, slice 67-C).

Implements spec sections 5-6 of docs/superpowers/specs/2026-07-13-g0bn-protocol.md:
the strict `g0bn-holdout-plan-v1` and `g0bn-freeze-v1` builders/validators.

Outcome-access boundary (spec section 5.1): every input here is already-produced
METADATA — the validated protocol config, the #68 custodian inventory/seal dict,
and the in-memory development run/ledger. This module performs NO filesystem or
vendor access of its own (the only file reads during construction are the Python
module sources hashed by the 67-B code-identity checks); it never opens, decodes,
or stats a January source payload, Parquet footer, matrix, or manifest. The plan
and freeze deliberately contain no January build_id, manifest/logical-row/
matrix-file hash, row/drop count, realized adaptive schedule/state, forecast,
metric, or result — every nested object is validated with an exact field set, so
there is nowhere to smuggle one in.

Boundaries owned elsewhere: #68 acquires/seals custody and signs the inventory
(this module only reconciles its hashes against the config's pins); #94/T9
materializes; #91 runs the one-shot transaction and its filesystem preflights;
#92 scores. The freeze is REPRODUCIBLE evidence: `build_freeze` re-derives the
deterministic development selection and ledger provenance from the run itself
(spec section 5.3) and never trusts supplied selected-candidate fields.

The stable `g0bn-holdout-universe-v1`/`g0bn-one-shot-v1` identities are pinned
constants from eval.g0bn_identity: a config, source, plan, or freeze edit changes
artifact hashes but can never mint a second transaction over the same January
outcomes (spec section 6.1).
"""
from __future__ import annotations

import copy
import datetime as _dt
import math

import pyarrow as pa

from eval.g0bn_config import (
    CANDIDATE_IDS,
    CONSUMPTION_SCHEMA,
    DEV_DATASET_ID,
    DSR_ROUNDING_RULE,
    FEATURE_REGISTRY,
    HOLDOUT_END_NS,
    HOLDOUT_START_NS,
    INSTRUMENT,
    ONE_SHOT_SCHEMA,
    OOS_DATASET_ID,
    PARTITION_PLAN_SCHEMA,
    PBO_IS_TIE_RULE,
    PBO_OOS_RANK_RULE,
    PROTOCOL_ID,
    BOOTSTRAP_KIND,
    _day,
    _day_array,
    _dict,
    _exact,
    _fail,
    _int,
    _num,
    _sha256,
    _str,
    _validate_generated_at,
    g0bn_artifact_sha256,
    validate_protocol_config,
)
from eval.g0bn_identity import (
    G0BN_HOLDOUT_UNIVERSE_ID,
    G0BN_TRANSACTION_ID,
    base_trial_identities,
    trial_id as _trial_id,
)
from eval.g0bn_selection import (
    BLOCK_LENGTH_DAYS,
    BOOTSTRAP_SEED,
    N_BOOT,
    PBO_MIN_COLUMNS,
    PBO_MIN_ROWS,
    PBO_N_BLOCKS,
    bootstrap_draws,
    development_selection,
    dsr_sample_count,
)
from eval.hashing import hash_obj
from eval.matrix import RESERVED, TIMING_COLS
from eval.writer import (
    G0BN_ALLOWED_EXTRA_COLS,
    G0BN_COST_DTYPES,
    G0BN_DATA_SOURCES,
    G0BN_REQUIRED_EXTRA_COLS,
    HOLDOUT_PLAN_BINDING,
    PARTITION_BINDING,
    PROTOCOL_BINDING,
    _physical_schema_sha256,
    validate_g0bn_manifest,
)

HOLDOUT_PLAN_SCHEMA = "g0bn-holdout-plan-v1"
FREEZE_SCHEMA = "g0bn-freeze-v1"

OBJECT_LAYERS = ("raw", "normalized")

# The one-shot matrix Parquet is hashed WHILE streaming the write (the T8
# _HashingSink); the artifact is never reopened. This names that algorithm in the
# plan's output contract.
MATRIX_FILE_HASH_ALGORITHM = "sha256_streaming_write_v1"

# Spec section 5.2 binding rule: the plan's own canonical hash enters the future
# OOS logical build parameters AND the manifest's g0bn_holdout_plan binding,
# breaking the config/freeze/build identity cycle without guessing a build hash.
BUILD_BINDING_RULE = \
    "holdout_plan_sha256_enters_oos_build_params_and_manifest_binding_v1"

_DAY_NS = 86_400_000_000_000

_INVENTORY_FIELDS = (
    "custodian_seal_sha256", "coverage_sha256", "permission_policy_sha256",
    "custodian_identity", "operator_identity",
    "included_days", "excluded_days", "objects",
)

_OBJECT_FIELDS = ("object_id", "layer", "product", "day", "sha256")

_PLAN_FIELDS = (
    "schema", "protocol_id", "pilot_id",
    "holdout_universe_id", "transaction_id", "instrument",
    "holdout_start_ns", "holdout_end_ns", "included_days", "excluded_days",
    "protocol_config_sha256", "source_certification_sha256",
    "custodian_seal_sha256", "coverage_sha256", "permission_policy_sha256",
    "partition_plan_sha256", "custodian_identity", "operator_identity",
    "object_allowlist", "n_allowlist_objects",
    "producer", "software_sha256", "materialization", "oos",
    "output_contract", "oos_bootstrap",
    "drop_count_categories", "sufficiency_thresholds",
    "build_binding_rule",
    "generated_at", "sha256",
)

_OUTPUT_CONTRACT_FIELDS = (
    "dataset_id", "manifest_version", "partition", "expected_bindings",
    "feature_cols", "target_cols", "reserved_cols", "extra_cols",
    "horizons", "dtypes", "expected_arrow_schema",
    "expected_physical_schema_sha256", "matrix_file_hash_algorithm",
)

_OOS_BOOTSTRAP_FIELDS = (
    "kind", "block_length_days", "n_boot", "seed", "bit_generator",
    "percentile_method", "alpha_oos", "days", "draw_sha256",
)

_FREEZE_FIELDS = (
    "schema", "protocol_id", "pilot_id",
    "holdout_universe_id", "transaction_id",
    "protocol_config_sha256", "holdout_plan_sha256",
    "source_certification_sha256", "custodian_seal_sha256", "coverage_sha256",
    "permission_policy_sha256", "partition_plan_sha256",
    "producer", "software_sha256",
    "development", "ledger", "selected", "control_candidates", "pbo",
    "verdict_thresholds", "costs", "exclusions", "oos",
    "generated_at", "sha256",
)

_DEVELOPMENT_FIELDS = (
    "dataset_id", "build_id", "manifest_sha256", "logical_row_sha256",
    "partition_plan_sha256", "result_sha256", "split_sha256s",
)

_LEDGER_FIELDS = ("n_effective_trials", "ledger_sha256", "history_sha256",
                  "identity_set_sha256")

_SELECTED_FIELDS = ("candidate_id", "trial_id", "mode", "ranked_candidate_ids",
                    "identity", "dsr")

_DSR_PROVENANCE_FIELDS = (
    "n_trials", "n_trials_source", "ledger_sha256", "sr_trials_std",
    "same_horizon_scored_trial_ids", "effective_trades", "T",
    "rounding_rule", "epsilon", "threshold", "code_sha256",
)

_PBO_FIELDS = (
    "available", "value", "reason", "n_rows", "n_columns", "n_combinations",
    "n_blocks", "is_tie_rule", "oos_rank_rule", "column_order_rule",
    "input_sha256", "column_trial_ids", "threshold", "ledger_sha256",
    "split_sha256", "code_sha256",
)

_CONTROL_FIELDS = ("candidate_id", "trial_id", "identity")

# Expected Arrow physical types by column, matching what the T8 writer's
# pa.Table.from_pandas emits for the pinned pandas dtypes (double = binary64;
# spec section 8.2 forbids any float32 round trip).
_RESERVED_ARROW_TYPES = {
    "y_fwd_bps": "double", "label": "int64", "cost_bps": "double",
    "half_spread_bps": "double", "uniqueness": "double",
    "regime": "string", "horizon": "string",
    **{c: "int64" for c in TIMING_COLS},
}
_EXTRA_ARROW_TYPES = {"latency_drift_bps": "double", "emitted_by_time_cap": "bool"}
_ARROW_TYPE_BY_NAME = {"double": pa.float64(), "int64": pa.int64(),
                       "string": pa.string(), "bool": pa.bool_()}


def holdout_window_days() -> list:
    """All UTC days of the fixed spec-2.2 holdout support window (exactly
    2026-01-01 .. 2026-01-31)."""
    if HOLDOUT_START_NS % _DAY_NS or HOLDOUT_END_NS % _DAY_NS:  # pragma: no cover
        _fail("holdout window", "bounds must be exact UTC midnights")
    start = _dt.datetime.fromtimestamp(HOLDOUT_START_NS // 1_000_000_000,
                                       tz=_dt.timezone.utc).date()
    n_days = (HOLDOUT_END_NS - HOLDOUT_START_NS) // _DAY_NS
    return [(start + _dt.timedelta(days=i)).isoformat() for i in range(n_days)]


HOLDOUT_DAYS = tuple(holdout_window_days())


def _raw_products(config: dict) -> tuple:
    """The certified native L2/trade product ids, consumed generically from the
    validated config's source_certification (the #64 evidence pin) rather than
    hard-coded here."""
    cert = config["source_certification"]
    return (cert["l2_snapshot_product"], cert["l2_delta_product"],
            cert["trade_product"])


def _object_key(obj: dict) -> tuple:
    return (obj["day"], obj["layer"], obj["product"], obj["object_id"])


def _validate_day_scope(path: str, included, excluded, config: dict) -> None:
    """Exact 31-day accounting: every UTC day of the holdout window is included
    or explicitly excluded with an outcome-blind reason and evidence hash."""
    window = list(HOLDOUT_DAYS)
    window_set = set(window)
    _day_array(f"{path}.included_days", included)
    for i, d in enumerate(included):
        if d not in window_set:
            _fail(f"{path}.included_days[{i}]",
                  f"day {d} is outside the January holdout window "
                  f"[{window[0]}, {window[-1]}]")
    if not isinstance(excluded, dict):
        _fail(f"{path}.excluded_days",
              "must be a map of day -> {reason, evidence_sha256}")
    for d, entry in excluded.items():
        epath = f"{path}.excluded_days[{d!r}]"
        _day(epath, d)
        if d not in window_set:
            _fail(epath, f"day {d} is outside the January holdout window "
                         f"[{window[0]}, {window[-1]}]")
        _dict(epath, entry, ("reason", "evidence_sha256"))
        _str(f"{epath}.reason", entry["reason"])
        _sha256(f"{epath}.evidence_sha256", entry["evidence_sha256"])
    both = sorted(set(included) & set(excluded))
    if both:
        _fail(path, f"days in both included_days and excluded_days: {both}")
    unaccounted = sorted(window_set - set(included) - set(excluded))
    if unaccounted:
        _fail(path, f"January days in neither included_days nor excluded_days "
                    f"(the outcome-blind custody accounting must cover all "
                    f"{len(window)} days): {unaccounted}")
    min_days = config["partition"]["sufficiency_thresholds"]["min_valid_days"]
    if len(included) < min_days:
        _fail(f"{path}.included_days",
              f"only {len(included)} outcome-blind included days; the scope can "
              f"never satisfy sufficiency_thresholds.min_valid_days={min_days}, "
              "so the one-shot transaction would be a guaranteed-INCONCLUSIVE "
              "burn (spec section 6.3 pre-burn predictable checks)")


def _validate_allowlist(path: str, objects, included, config: dict) -> None:
    """Exact sealed-object allowlist accounting: unique opaque ids, certified
    products only, included days only, and EXACTLY ONE object per
    (day, layer, product) — a second object for a covered slot is ambiguous
    custody and would make object multiplicity a January activity proxy, so the
    allowlist cardinality is a pure function of the included-day count. Byte
    sizes / record counts are likewise structurally rejected by the exact
    object field set (spec section 5.1). A custodian sharding scheme finer than
    day-per-product requires an explicit contract revision, never silent
    acceptance."""
    if not isinstance(objects, list) or not objects:
        _fail(path, "must be a non-empty array of sealed object entries")
    raw_products = _raw_products(config)
    products_by_layer = {"raw": raw_products, "normalized": G0BN_DATA_SOURCES}
    included_set = set(included)
    seen_ids = set()
    seen = set()
    for i, obj in enumerate(objects):
        opath = f"{path}[{i}]"
        _dict(opath, obj, _OBJECT_FIELDS)
        _str(f"{opath}.object_id", obj["object_id"])
        if obj["layer"] not in OBJECT_LAYERS:
            _fail(f"{opath}.layer", f"must be one of {OBJECT_LAYERS}; "
                                    f"got {obj['layer']!r}")
        allowed = products_by_layer[obj["layer"]]
        if obj["product"] not in allowed:
            _fail(f"{opath}.product",
                  f"{obj['product']!r} is not a certified {obj['layer']} "
                  f"L2/trade product (allowed: {tuple(allowed)}); no other feed "
                  "may enter the one-shot scope (spec section 2.1)")
        _day(f"{opath}.day", obj["day"])
        if obj["day"] not in included_set:
            _fail(f"{opath}.day",
                  f"{obj['day']} is not an included day of the outcome-blind "
                  "January scope; sealed objects on excluded or out-of-window "
                  "days must not enter the allowlist")
        _sha256(f"{opath}.sha256", obj["sha256"])
        if obj["object_id"] in seen_ids:
            _fail(f"{opath}.object_id",
                  f"duplicate object_id {obj['object_id']!r} in the sealed "
                  "allowlist")
        seen_ids.add(obj["object_id"])
        triple = (obj["day"], obj["layer"], obj["product"])
        if triple in seen:
            _fail(opath,
                  f"duplicate sealed object for (day, layer, product) {triple}; "
                  "the allowlist admits exactly one object per certified "
                  "product per included day (ambiguous custody and "
                  "activity-proxy multiplicity are forbidden)")
        seen.add(triple)
    missing = [(day, layer, product)
               for day in included
               for layer in OBJECT_LAYERS
               for product in products_by_layer[layer]
               if (day, layer, product) not in seen]
    if missing:
        _fail(path, f"missing sealed object(s) for (day, layer, product): "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}; every "
                    "included day needs every certified raw and normalized "
                    "L2/trade product")


def validate_custody_inventory(inventory: dict, config: dict) -> dict:
    """Strict consumption contract for the outcome-blind #68 custodian
    inventory/seal metadata. This module only reads such already-produced
    metadata; the custody evidence hashes and the distinct custodian/operator
    identities must reconcile exactly with the config's source_certification
    pins, or the holdout scope is unsealed/non-custodial and fails closed."""
    validate_protocol_config(config)
    cert = config["source_certification"]
    path = "custody inventory"
    _dict(path, inventory, _INVENTORY_FIELDS)
    for k in ("custodian_identity", "operator_identity"):
        _str(f"{path}.{k}", inventory[k])
    if inventory["custodian_identity"] == inventory["operator_identity"]:
        _fail(path, "custodian_identity and operator_identity must be distinct "
                    "(separation of custody; spec section 5.1)")
    for k in ("custodian_seal_sha256", "coverage_sha256",
              "permission_policy_sha256"):
        _sha256(f"{path}.{k}", inventory[k])
        _exact(f"{path}.{k}", inventory[k], cert[k])
    for k in ("custodian_identity", "operator_identity"):
        _exact(f"{path}.{k}", inventory[k], cert[k])
    _validate_day_scope(path, inventory["included_days"],
                        inventory["excluded_days"], config)
    _validate_allowlist(f"{path}.objects", inventory["objects"],
                        inventory["included_days"], config)
    return inventory


def _validate_extra_cols(path: str, extra) -> list:
    if not isinstance(extra, list):
        _fail(path, f"must be an ordered array of diagnostic columns; got {extra!r}")
    for i, c in enumerate(extra):
        _str(f"{path}[{i}]", c)
        if c not in G0BN_ALLOWED_EXTRA_COLS:
            _fail(f"{path}[{i}]", f"{c!r} is not an authorized G0-BN diagnostic "
                                  f"(allowed: {tuple(G0BN_ALLOWED_EXTRA_COLS)})")
    if len(set(extra)) != len(extra):
        _fail(path, "must be unique")
    missing = [c for c in G0BN_REQUIRED_EXTRA_COLS if c not in extra]
    if missing:
        _fail(path, f"missing required diagnostics: {missing}")
    return extra


def expected_arrow_schema(extra_cols) -> list:
    """The exact ordered Arrow physical schema the blind materializer must emit:
    manifest column order (features, reserved, extra), binary64 doubles, int64
    nanosecond timing, string tags — as [name, type, nullable] triples."""
    fields = [[c, "double", True] for c in FEATURE_REGISTRY]
    fields += [[c, _RESERVED_ARROW_TYPES[c], True] for c in RESERVED]
    fields += [[c, _EXTRA_ARROW_TYPES[c], True] for c in extra_cols]
    return fields


def _expected_physical_schema_sha256(fields) -> str:
    schema = pa.schema([pa.field(name, _ARROW_TYPE_BY_NAME[type_name],
                                 nullable=nullable)
                        for name, type_name, nullable in fields])
    return _physical_schema_sha256(schema)


def _horizon_roles_sha256(config: dict) -> str:
    return hash_obj({h["tag"]: h["role"] for h in config["horizons"]})


def _output_contract(config: dict, extra_cols: list) -> dict:
    cert = config["source_certification"]
    arrow_fields = expected_arrow_schema(extra_cols)
    return {
        "dataset_id": OOS_DATASET_ID,
        "manifest_version": 1,
        "partition": "holdout",
        "expected_bindings": {
            PARTITION_BINDING: {
                "schema": PARTITION_PLAN_SCHEMA,
                "partition": "holdout",
                "partition_plan_sha256": config["partition"]["sha256"],
            },
            PROTOCOL_BINDING: {
                "protocol": PROTOCOL_ID,
                "protocol_config_sha256": config["sha256"],
                "source_certification_sha256": cert["certification_sha256"],
                "horizon_roles_sha256": _horizon_roles_sha256(config),
                "instrument": dict(INSTRUMENT),
            },
            # The binding's holdout_plan_sha256/freeze_sha256 values are the
            # plan's own future hash and the not-yet-built freeze: they cannot be
            # embedded here. BUILD_BINDING_RULE pins how they enter (spec 5.2);
            # holdout_plan_binding() derives the complete entry once both exist.
            HOLDOUT_PLAN_BINDING: {
                "protocol": ONE_SHOT_SCHEMA,
                "consumption_schema": CONSUMPTION_SCHEMA,
                "holdout_universe_id": G0BN_HOLDOUT_UNIVERSE_ID,
                "transaction_id": G0BN_TRANSACTION_ID,
            },
        },
        "feature_cols": list(FEATURE_REGISTRY),
        "target_cols": ["y_fwd_bps", "label"],
        "reserved_cols": list(RESERVED),
        "extra_cols": list(extra_cols),
        "horizons": {h["tag"]: h["ns"] for h in config["horizons"]},
        "dtypes": dict(G0BN_COST_DTYPES),
        "expected_arrow_schema": arrow_fields,
        "expected_physical_schema_sha256":
            _expected_physical_schema_sha256(arrow_fields),
        "matrix_file_hash_algorithm": MATRIX_FILE_HASH_ALGORITHM,
    }


def build_holdout_plan(config: dict, inventory: dict, *,
                       extra_cols=None, generated_at: str | None = None) -> dict:
    """Build the outcome-blind `g0bn-holdout-plan-v1` from the validated protocol
    config and the #68 custodian inventory/seal metadata (spec section 5.2). The
    exact object allowlist comes from the seal — never from a glob at execution
    time — and the plan pins hashes and derivation rules only: the January
    build/manifest/row/file/count/schedule/result values do not exist yet."""
    validate_protocol_config(config)
    validate_custody_inventory(inventory, config)
    cert = config["source_certification"]
    extra = (list(G0BN_REQUIRED_EXTRA_COLS) if extra_cols is None
             else list(extra_cols))
    _validate_extra_cols("extra_cols", extra)
    included = list(inventory["included_days"])
    excluded = {d: dict(entry)
                for d, entry in sorted(inventory["excluded_days"].items())}
    allowlist = [dict(obj) for obj in sorted(inventory["objects"],
                                             key=_object_key)]
    _, draw_sha256 = bootstrap_draws(included)
    plan = {
        "schema": HOLDOUT_PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "pilot_id": config["pilot_id"],
        "holdout_universe_id": G0BN_HOLDOUT_UNIVERSE_ID,
        "transaction_id": G0BN_TRANSACTION_ID,
        "instrument": dict(INSTRUMENT),
        "holdout_start_ns": HOLDOUT_START_NS,
        "holdout_end_ns": HOLDOUT_END_NS,
        "included_days": included,
        "excluded_days": excluded,
        "protocol_config_sha256": config["sha256"],
        "source_certification_sha256": cert["certification_sha256"],
        "custodian_seal_sha256": cert["custodian_seal_sha256"],
        "coverage_sha256": cert["coverage_sha256"],
        "permission_policy_sha256": cert["permission_policy_sha256"],
        "partition_plan_sha256": config["partition"]["sha256"],
        "custodian_identity": cert["custodian_identity"],
        "operator_identity": cert["operator_identity"],
        "object_allowlist": allowlist,
        "n_allowlist_objects": len(allowlist),
        "producer": copy.deepcopy(config["producer"]),
        "software_sha256": hash_obj(config["software"]),
        "materialization": {
            "clock_sha256": hash_obj(config["clock"]),
            "features_sha256": hash_obj(config["features"]),
            "labels_sha256": hash_obj(config["labels"]),
            "costs_sha256": hash_obj(config["costs"]),
        },
        "oos": copy.deepcopy(config["oos"]),
        "output_contract": _output_contract(config, extra),
        "oos_bootstrap": {
            "kind": BOOTSTRAP_KIND,
            "block_length_days": BLOCK_LENGTH_DAYS,
            "n_boot": N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "bit_generator": "PCG64",
            "percentile_method": "linear",
            "alpha_oos": config["verdict_thresholds"]["alpha_oos"],
            "days": list(included),
            "draw_sha256": draw_sha256,
        },
        "drop_count_categories":
            list(config["partition"]["holdout_drop_count_categories"]),
        "sufficiency_thresholds":
            dict(config["partition"]["sufficiency_thresholds"]),
        "build_binding_rule": BUILD_BINDING_RULE,
        "generated_at": generated_at,
    }
    plan["sha256"] = g0bn_artifact_sha256(plan)
    return validate_holdout_plan(plan, config, inventory=inventory)


def validate_holdout_plan(plan: dict, config: dict, *,
                          inventory: dict | None = None) -> dict:
    """Strict fail-closed validation of one `g0bn-holdout-plan-v1` against the
    exact protocol config (and, when supplied, the source inventory). Exact
    nested field sets everywhere; the embedded self-hash is verified last so a
    targeted error names the offending field first."""
    validate_protocol_config(config)
    cert = config["source_certification"]
    _dict("g0bn holdout plan", plan, _PLAN_FIELDS)
    _exact("schema", plan["schema"], HOLDOUT_PLAN_SCHEMA)
    _exact("protocol_id", plan["protocol_id"], PROTOCOL_ID)
    _exact("pilot_id", plan["pilot_id"], config["pilot_id"])
    # Stable identities first (spec section 6.1): a plan naming a foreign
    # universe/transaction can never mint a second transaction — it fails here.
    _sha256("holdout_universe_id", plan["holdout_universe_id"])
    _exact("holdout_universe_id", plan["holdout_universe_id"],
           G0BN_HOLDOUT_UNIVERSE_ID)
    _sha256("transaction_id", plan["transaction_id"])
    _exact("transaction_id", plan["transaction_id"], G0BN_TRANSACTION_ID)
    _exact("instrument", plan["instrument"], INSTRUMENT)
    _exact("holdout_start_ns", plan["holdout_start_ns"], HOLDOUT_START_NS)
    _exact("holdout_end_ns", plan["holdout_end_ns"], HOLDOUT_END_NS)
    _exact("protocol_config_sha256", plan["protocol_config_sha256"],
           config["sha256"])
    _exact("source_certification_sha256", plan["source_certification_sha256"],
           cert["certification_sha256"])
    _exact("custodian_seal_sha256", plan["custodian_seal_sha256"],
           cert["custodian_seal_sha256"])
    _exact("coverage_sha256", plan["coverage_sha256"], cert["coverage_sha256"])
    _exact("permission_policy_sha256", plan["permission_policy_sha256"],
           cert["permission_policy_sha256"])
    _exact("partition_plan_sha256", plan["partition_plan_sha256"],
           config["partition"]["sha256"])
    _exact("custodian_identity", plan["custodian_identity"],
           cert["custodian_identity"])
    _exact("operator_identity", plan["operator_identity"],
           cert["operator_identity"])

    _validate_day_scope("g0bn holdout plan", plan["included_days"],
                        plan["excluded_days"], config)
    _validate_allowlist("object_allowlist", plan["object_allowlist"],
                        plan["included_days"], config)
    keys = [_object_key(o) for o in plan["object_allowlist"]]
    if keys != sorted(keys):
        _fail("object_allowlist", "must be in canonical "
                                  "(day, layer, product, object_id) order")
    _int("n_allowlist_objects", plan["n_allowlist_objects"], minimum=1)
    if plan["n_allowlist_objects"] != len(plan["object_allowlist"]):
        _fail("n_allowlist_objects",
              f"does not reconcile with the embedded allowlist "
              f"({plan['n_allowlist_objects']} != {len(plan['object_allowlist'])})")

    _exact("producer", plan["producer"], config["producer"])
    _exact("software_sha256", plan["software_sha256"],
           hash_obj(config["software"]))
    _exact("materialization", plan["materialization"], {
        "clock_sha256": hash_obj(config["clock"]),
        "features_sha256": hash_obj(config["features"]),
        "labels_sha256": hash_obj(config["labels"]),
        "costs_sha256": hash_obj(config["costs"]),
    })
    _exact("oos", plan["oos"], config["oos"])

    oc = plan["output_contract"]
    _dict("output_contract", oc, _OUTPUT_CONTRACT_FIELDS)
    extra = _validate_extra_cols("output_contract.extra_cols", oc["extra_cols"])
    _exact("output_contract", oc, _output_contract(config, extra))

    boot = plan["oos_bootstrap"]
    _dict("oos_bootstrap", boot, _OOS_BOOTSTRAP_FIELDS)
    _exact("oos_bootstrap.kind", boot["kind"], BOOTSTRAP_KIND)
    _exact("oos_bootstrap.block_length_days", boot["block_length_days"],
           BLOCK_LENGTH_DAYS)
    _exact("oos_bootstrap.n_boot", boot["n_boot"], N_BOOT)
    _exact("oos_bootstrap.seed", boot["seed"], BOOTSTRAP_SEED)
    _exact("oos_bootstrap.bit_generator", boot["bit_generator"], "PCG64")
    _exact("oos_bootstrap.percentile_method", boot["percentile_method"], "linear")
    _exact("oos_bootstrap.alpha_oos", boot["alpha_oos"],
           config["verdict_thresholds"]["alpha_oos"])
    _exact("oos_bootstrap.days", boot["days"], plan["included_days"])
    _sha256("oos_bootstrap.draw_sha256", boot["draw_sha256"])
    _, expected_draw = bootstrap_draws(plan["included_days"])
    if boot["draw_sha256"] != expected_draw:
        _fail("oos_bootstrap.draw_sha256",
              f"does not reproduce the pinned OOS draw recipe on the frozen day "
              f"list ({boot['draw_sha256']} != {expected_draw})")

    _exact("drop_count_categories", plan["drop_count_categories"],
           list(config["partition"]["holdout_drop_count_categories"]))
    _exact("sufficiency_thresholds", plan["sufficiency_thresholds"],
           config["partition"]["sufficiency_thresholds"])
    _exact("build_binding_rule", plan["build_binding_rule"], BUILD_BINDING_RULE)
    if plan["generated_at"] is not None:
        _validate_generated_at(plan["generated_at"])

    if inventory is not None:
        # The inventory is only a custody anchor if IT reconciles with the
        # config's seal/coverage/permission/identity pins: a forged minimal
        # dict mirroring the plan's own scope fields (or a full inventory
        # carrying foreign custody evidence) must fail before any comparison.
        validate_custody_inventory(inventory, config)
        _exact("included_days", plan["included_days"],
               list(inventory["included_days"]))
        _exact("excluded_days", plan["excluded_days"],
               inventory["excluded_days"])
        expected_allow = [dict(o) for o in sorted(inventory["objects"],
                                                  key=_object_key)]
        _exact("object_allowlist", plan["object_allowlist"], expected_allow)

    embedded = _sha256("sha256", plan["sha256"])
    recomputed = g0bn_artifact_sha256(plan)
    if embedded != recomputed:
        _fail("sha256", f"embedded holdout-plan sha256 does not match the "
                        f"canonical content (tampered or stale): "
                        f"{embedded} != {recomputed}")
    return plan


def _artifact_content_sha256(path: str, artifact: dict, schema: str) -> str:
    """Recompute an artifact's canonical content hash and require the embedded
    self-hash to match: a hand-edited artifact whose sha256 was not updated is
    rejected even by the light hash accessors (a coordinated rehash is instead
    caught by the config-anchored validators)."""
    if not isinstance(artifact, dict) or artifact.get("schema") != schema:
        _fail(path, f"must be a {schema!r} artifact; got schema "
                    f"{artifact.get('schema') if isinstance(artifact, dict) else artifact!r}")
    recomputed = g0bn_artifact_sha256(artifact)
    if artifact.get("sha256") != recomputed:
        _fail(f"{path}.sha256",
              f"embedded self-hash does not match the canonical content "
              f"(tampered or stale): {artifact.get('sha256')!r} != {recomputed}")
    return recomputed


def holdout_plan_sha256(plan: dict) -> str:
    """Canonical plan content hash (self-hash and generated_at excluded). This is
    the value that enters the future OOS build parameters and manifest binding."""
    return _artifact_content_sha256("holdout plan", plan, HOLDOUT_PLAN_SCHEMA)


def freeze_sha256(freeze: dict) -> str:
    return _artifact_content_sha256("freeze", freeze, FREEZE_SCHEMA)


# ------------------------------------------------------------ future-build binding


def oos_build_params(plan: dict, build_params: dict, *, config: dict,
                     inventory: dict) -> dict:
    """Derive the one-shot OOS logical build parameters: the caller's explicit
    parameters plus the binding `holdout_plan_sha256` (spec section 5.2). The
    plan is first re-anchored to BOTH the immutable config and the #68 custody
    inventory — the config cannot pin January object hashes, so without the
    inventory a coordinated plan+freeze rehash could bind objects the custodian
    never sealed. The binding value is always injected from the plan itself: a
    caller-supplied value could pin a stale/tampered plan and is refused."""
    validate_holdout_plan(plan, config, inventory=inventory)
    if not isinstance(build_params, dict):
        _fail("build_params", "must be a dict of explicit build parameters")
    if "holdout_plan_sha256" in build_params:
        _fail("build_params.holdout_plan_sha256",
              "must not be caller-supplied; it is derived from the plan so a "
              "stale or tampered pin cannot enter the build identity")
    if "generated_at" in build_params:
        _fail("build_params.generated_at",
              "must not appear in build parameters (identical rebuilds must "
              "share a build identity)")
    return dict(build_params, holdout_plan_sha256=holdout_plan_sha256(plan))


def verify_oos_build_binding(build_params: dict, plan: dict, *, config: dict,
                             inventory: dict) -> dict:
    """Verify that build parameters bind exactly this config- and
    custody-validated plan. Tampering with the plan after binding (even with a
    recomputed self-hash, and even if the allowlist edit is coordinated with a
    freeze rehash) or binding a foreign plan fails closed."""
    validate_holdout_plan(plan, config, inventory=inventory)
    if not isinstance(build_params, dict):
        _fail("build_params", "must be a dict of explicit build parameters")
    expected = holdout_plan_sha256(plan)
    if build_params.get("holdout_plan_sha256") != expected:
        _fail("build_params.holdout_plan_sha256",
              f"does not bind this holdout plan "
              f"({build_params.get('holdout_plan_sha256')!r} != {expected}); "
              "the OOS build recipe must pin the exact outcome-blind plan")
    return build_params


def holdout_plan_binding(plan: dict, freeze: dict, *, config: dict,
                         inventory: dict) -> dict:
    """The exact `g0bn_holdout_plan` manifest source entry the blind-materialized
    OOS manifest must carry (spec section 7), derived from the real plan/freeze
    pair. Both artifacts are first re-anchored to the immutable config AND the
    #68 custody inventory (a coordinated plan+freeze rehash still fails, even
    one that only swaps a sealed object hash), and the freeze must bind exactly
    this plan."""
    validate_freeze(freeze, config=config, plan=plan,
                    inventory=inventory)   # validates the plan too
    return {
        "name": HOLDOUT_PLAN_BINDING,
        "protocol": ONE_SHOT_SCHEMA,
        "consumption_schema": CONSUMPTION_SCHEMA,
        "holdout_universe_id": plan["holdout_universe_id"],
        "transaction_id": plan["transaction_id"],
        "holdout_plan_sha256": holdout_plan_sha256(plan),
        "freeze_sha256": freeze_sha256(freeze),
    }


def verify_holdout_manifest_binding(manifest: dict, plan: dict, freeze: dict, *,
                                    config: dict, inventory: dict) -> dict:
    """Verify a blind-materialized holdout manifest against the config- and
    custody-validated plan's pinned output contract and the plan/freeze hashes.
    Any tampering — with the plan, the freeze, or a manifest binding — fails
    closed, including a coordinated rehash of the plan/freeze pair (the config
    and the #68 inventory are the anchors)."""
    expected_plan_binding = holdout_plan_binding(plan, freeze, config=config,
                                                 inventory=inventory)
    validate_g0bn_manifest(manifest)
    oc = plan["output_contract"]
    named: dict = {}
    for s in manifest["sources"]:
        if isinstance(s, dict):
            named.setdefault(s["name"], []).append(s)
    _exact("manifest.dataset_id", manifest["dataset_id"], oc["dataset_id"])
    _exact(f"manifest.{PARTITION_BINDING}", named[PARTITION_BINDING][0],
           dict(oc["expected_bindings"][PARTITION_BINDING],
                name=PARTITION_BINDING))
    _exact(f"manifest.{PROTOCOL_BINDING}", named[PROTOCOL_BINDING][0],
           dict(oc["expected_bindings"][PROTOCOL_BINDING], name=PROTOCOL_BINDING))
    _exact(f"manifest.{HOLDOUT_PLAN_BINDING}", named[HOLDOUT_PLAN_BINDING][0],
           expected_plan_binding)
    _exact("manifest.feature_cols", list(manifest["feature_cols"]),
           oc["feature_cols"])
    _exact("manifest.target_cols", list(manifest["target_cols"]),
           oc["target_cols"])
    _exact("manifest.reserved_cols", list(manifest["reserved_cols"]),
           oc["reserved_cols"])
    _exact("manifest.extra_cols", list(manifest.get("extra_cols", [])),
           oc["extra_cols"])
    _exact("manifest.horizons", manifest["horizons"], oc["horizons"])
    _exact("manifest.dtypes", manifest.get("dtypes"), oc["dtypes"])
    _exact("manifest.custodian_seal", named["custodian_seal"][0]["sha256"],
           plan["custodian_seal_sha256"])
    _exact("manifest.source_certification",
           named["source_certification"][0]["sha256"],
           plan["source_certification_sha256"])
    cost_entry = {k: v for k, v in named["cost_assumption"][0].items()
                  if k != "name"}
    _exact("manifest.cost_assumption", cost_entry,
           freeze["costs"]["cost_assumption"])
    # Every normalized data-source pin must come from the sealed allowlist: a
    # manifest naming an object the custodian never sealed fails closed.
    allowed_shas: dict = {}
    for obj in plan["object_allowlist"]:
        if obj["layer"] == "normalized":
            allowed_shas.setdefault(obj["product"], set()).add(obj["sha256"])
    for name in G0BN_DATA_SOURCES:
        for entry in named.get(name, []):
            if entry["sha256"] not in allowed_shas.get(name, set()):
                _fail(f"manifest.{name}",
                      f"source object hash {entry['sha256']} is not in the "
                      "plan's sealed normalized allowlist for that product")
    return manifest


# ------------------------------------------------------------------------- freeze


def build_freeze(run, plan: dict, *, inventory: dict,
                 expected_development_result: dict | None = None,
                 generated_at: str | None = None) -> dict:
    """Build the reproducible `g0bn-freeze-v1` from development evidence and
    outcome-blind metadata only (spec section 5.3).

    The plan is re-anchored to BOTH the immutable config and the required #68
    custody inventory before its hash is frozen: without the inventory, a plan
    whose object allowlist was edited and rehashed could be frozen over objects
    the custodian never sealed. Selection and ledger provenance are RE-DERIVED
    from the run via `development_selection` — never trusted from supplied
    fields. The optional `expected_development_result` is a reconciliation
    input: it must reproduce its own embedded hash AND equal the re-derived
    deterministic result, so a stale, tampered, or hand-edited development
    result can never be frozen."""
    config = run.config
    validate_holdout_plan(plan, config, inventory=inventory)
    if expected_development_result is not None:
        exp = expected_development_result
        if not isinstance(exp, dict) or "result_sha256" not in exp:
            _fail("expected_development_result",
                  "must be a development-result dict with an embedded "
                  "result_sha256")
        recomputed = hash_obj(exp, exclude_keys=("result_sha256", "generated_at"))
        if exp["result_sha256"] != recomputed:
            _fail("expected_development_result.result_sha256",
                  f"does not match its own content (tampered or stale): "
                  f"{exp['result_sha256']} != {recomputed}")
    derived = development_selection(run)
    if expected_development_result is not None and \
            expected_development_result["result_sha256"] != derived["result_sha256"]:
        _fail("expected_development_result",
              f"the supplied development result "
              f"{expected_development_result['result_sha256']} does not match "
              f"the re-derived deterministic development result "
              f"{derived['result_sha256']}; the freeze pins re-derived evidence "
              "only (spec section 5.3)")
    if derived["freeze_blocked"]:
        _fail("selection",
              "the development selection is freeze-blocked: a primary horizon "
              "has no predictive-eligible candidate (spec section 4.3), so the "
              "freeze cannot be built and January remains unopened")

    roles = {h["tag"]: h["role"] for h in config["horizons"]}
    ladder_tags = [h["tag"] for h in config["horizons"]]
    primary_tags = [t for t in ladder_tags if roles[t] == "primary"]
    control_tags = [t for t in ladder_tags if roles[t] != "primary"]
    thresholds = config["verdict_thresholds"]

    selected = {}
    pbo = {}
    for tag in primary_tags:
        sel = derived["selection"][tag]
        cid = sel["selected_candidate_id"]
        tid = sel["selected_trial_id"]
        cand = derived["horizons"][tag]["candidates"][cid]
        block = copy.deepcopy(derived["horizons"][tag]["pbo"])
        if not block["available"] or not (block["value"] <
                                          thresholds["pbo_threshold"]):
            _fail(f"pbo[{tag!r}]",
                  "an unavailable or above-threshold PBO cannot back a freeze "
                  "(spec section 5.3)")
        selected[tag] = {
            "candidate_id": cid,
            "trial_id": tid,
            "mode": sel["mode"],
            "ranked_candidate_ids": list(sel["ranked_candidate_ids"]),
            "identity": run.ledger.identity_for(tid),
            "dsr": {
                "value": cand["dsr"],
                "provenance": copy.deepcopy(cand["dsr_provenance"]),
            },
        }
        pbo[tag] = block

    base_ids = base_trial_identities(config, run.data_identity)
    control_candidates = {
        tag: [{"candidate_id": identity["candidate_id"],
               "trial_id": _trial_id(identity),
               "identity": copy.deepcopy(identity)}
              for identity in base_ids if identity["horizon"] == tag]
        for tag in control_tags
    }

    data_identity = run.data_identity
    freeze = {
        "schema": FREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "pilot_id": config["pilot_id"],
        "holdout_universe_id": plan["holdout_universe_id"],
        "transaction_id": plan["transaction_id"],
        "protocol_config_sha256": config["sha256"],
        "holdout_plan_sha256": holdout_plan_sha256(plan),
        "source_certification_sha256": plan["source_certification_sha256"],
        "custodian_seal_sha256": plan["custodian_seal_sha256"],
        "coverage_sha256": plan["coverage_sha256"],
        "permission_policy_sha256": plan["permission_policy_sha256"],
        "partition_plan_sha256": config["partition"]["sha256"],
        "producer": copy.deepcopy(config["producer"]),
        "software_sha256": hash_obj(config["software"]),
        "development": {
            "dataset_id": data_identity["development_dataset_id"],
            "build_id": data_identity["development_build_id"],
            "manifest_sha256": data_identity["development_manifest_sha256"],
            "logical_row_sha256":
                data_identity["development_logical_row_sha256"],
            "partition_plan_sha256": data_identity["partition_plan_sha256"],
            "result_sha256": derived["result_sha256"],
            "split_sha256s": {tag: derived["horizons"][tag]["split_sha256"]
                              for tag in ladder_tags},
        },
        "ledger": dict(derived["ledger"]),
        "selected": selected,
        "control_candidates": control_candidates,
        "pbo": pbo,
        "verdict_thresholds": copy.deepcopy(config["verdict_thresholds"]),
        "costs": copy.deepcopy(config["costs"]),
        "exclusions": copy.deepcopy(config["exclusions"]),
        "oos": copy.deepcopy(config["oos"]),
        "generated_at": generated_at,
    }
    freeze["sha256"] = g0bn_artifact_sha256(freeze)
    return validate_freeze(freeze, config=config, plan=plan)


def validate_freeze(freeze: dict, *, config: dict, plan: dict,
                    run=None, inventory: dict | None = None) -> dict:
    """Strict fail-closed validation of one `g0bn-freeze-v1` against the exact
    protocol config and holdout plan: exact nested field sets (no January
    outcome field can exist anywhere), stable identities, config-drift and
    plan-tamper detection, base-ladder identity re-derivation, and DSR/PBO
    provenance reconciliation.

    Two validation levels. Without `run`, this is the STRUCTURAL level: it
    checks everything checkable without development data, but the freeze's
    development-derived statistics (DSR/PBO values, result hash) are pinned
    only by hashes an artifact editor could recompute. With `run` (the
    development run the one-shot runner reloads in its spec-6.3 step-1
    self-verification), the freeze is additionally REBUILT via
    `build_freeze` — re-running `development_selection` — and must reproduce
    exactly, so a fabricated or stale statistic fails closed. `inventory`
    (the #68 seal metadata, likewise loaded by the runner) re-anchors the
    plan's day scope and object allowlist to custody; the binding entry
    points require it because the config alone cannot pin January object
    hashes."""
    validate_holdout_plan(plan, config, inventory=inventory)
    cert = config["source_certification"]
    thresholds = config["verdict_thresholds"]
    _dict("g0bn freeze", freeze, _FREEZE_FIELDS)
    _exact("schema", freeze["schema"], FREEZE_SCHEMA)
    _exact("protocol_id", freeze["protocol_id"], PROTOCOL_ID)
    _exact("pilot_id", freeze["pilot_id"], config["pilot_id"])
    _exact("holdout_universe_id", freeze["holdout_universe_id"],
           G0BN_HOLDOUT_UNIVERSE_ID)
    _exact("transaction_id", freeze["transaction_id"], G0BN_TRANSACTION_ID)
    _exact("protocol_config_sha256", freeze["protocol_config_sha256"],
           config["sha256"])
    expected_plan_sha = g0bn_artifact_sha256(plan)
    if freeze["holdout_plan_sha256"] != expected_plan_sha:
        _fail("holdout_plan_sha256",
              f"the freeze binds holdout plan {freeze['holdout_plan_sha256']}, "
              f"not this plan's canonical content {expected_plan_sha} (tampered "
              "or foreign plan)")
    _exact("source_certification_sha256", freeze["source_certification_sha256"],
           cert["certification_sha256"])
    _exact("custodian_seal_sha256", freeze["custodian_seal_sha256"],
           cert["custodian_seal_sha256"])
    _exact("coverage_sha256", freeze["coverage_sha256"], cert["coverage_sha256"])
    _exact("permission_policy_sha256", freeze["permission_policy_sha256"],
           cert["permission_policy_sha256"])
    _exact("partition_plan_sha256", freeze["partition_plan_sha256"],
           config["partition"]["sha256"])
    _exact("producer", freeze["producer"], config["producer"])
    _exact("software_sha256", freeze["software_sha256"],
           hash_obj(config["software"]))

    roles = {h["tag"]: h["role"] for h in config["horizons"]}
    ladder_tags = [h["tag"] for h in config["horizons"]]
    primary_tags = [t for t in ladder_tags if roles[t] == "primary"]
    control_tags = [t for t in ladder_tags if roles[t] != "primary"]

    dev = freeze["development"]
    _dict("development", dev, _DEVELOPMENT_FIELDS)
    _exact("development.dataset_id", dev["dataset_id"], DEV_DATASET_ID)
    for k in ("build_id", "manifest_sha256", "logical_row_sha256",
              "result_sha256"):
        _sha256(f"development.{k}", dev[k])
    _exact("development.partition_plan_sha256", dev["partition_plan_sha256"],
           config["partition"]["sha256"])
    _dict("development.split_sha256s", dev["split_sha256s"], tuple(ladder_tags))
    for tag in ladder_tags:
        _sha256(f"development.split_sha256s[{tag!r}]", dev["split_sha256s"][tag])

    ledger = freeze["ledger"]
    _dict("ledger", ledger, _LEDGER_FIELDS)
    n_base = len(CANDIDATE_IDS) * len(ladder_tags)
    _int("ledger.n_effective_trials", ledger["n_effective_trials"],
         minimum=n_base)
    for k in ("ledger_sha256", "history_sha256", "identity_set_sha256"):
        _sha256(f"ledger.{k}", ledger[k])

    data_identity = {
        "development_dataset_id": dev["dataset_id"],
        "development_build_id": dev["build_id"],
        "development_manifest_sha256": dev["manifest_sha256"],
        "development_logical_row_sha256": dev["logical_row_sha256"],
        "partition_plan_sha256": dev["partition_plan_sha256"],
    }
    base_by_key = {(identity["horizon"], identity["candidate_id"]): identity
                   for identity in base_trial_identities(config, data_identity)}
    base_tids_by_tag = {
        tag: [_trial_id(base_by_key[(tag, cid)]) for cid in CANDIDATE_IDS]
        for tag in ladder_tags
    }

    eligible = config["selection"]["eligible_candidate_ids"]
    _dict("selected", freeze["selected"], tuple(primary_tags))
    for tag in primary_tags:
        spath = f"selected[{tag!r}]"
        sel = freeze["selected"][tag]
        _dict(spath, sel, _SELECTED_FIELDS)
        if sel["candidate_id"] not in eligible:
            _fail(f"{spath}.candidate_id",
                  f"{sel['candidate_id']!r} is not an eligible selectable "
                  f"candidate ({eligible})")
        if sel["mode"] not in ("trade", "predictive"):
            _fail(f"{spath}.mode", f"must be 'trade' or 'predictive'; "
                                   f"got {sel['mode']!r}")
        ranked = sel["ranked_candidate_ids"]
        if (not isinstance(ranked, list) or not ranked
                or len(set(ranked)) != len(ranked)
                or any(c not in eligible for c in ranked)
                or ranked[0] != sel["candidate_id"]):
            _fail(f"{spath}.ranked_candidate_ids",
                  f"must be a unique eligible ranking whose first entry is the "
                  f"selected candidate; got {ranked!r}")
        expected_identity = base_by_key[(tag, sel["candidate_id"])]
        _exact(f"{spath}.identity", sel["identity"], expected_identity)
        _sha256(f"{spath}.trial_id", sel["trial_id"])
        if sel["trial_id"] != _trial_id(sel["identity"]):
            _fail(f"{spath}.trial_id",
                  "does not re-derive from the embedded canonical identity "
                  "(tampered identity or id)")
        dsr = sel["dsr"]
        _dict(f"{spath}.dsr", dsr, ("value", "provenance"))
        _num(f"{spath}.dsr.value", dsr["value"])
        if sel["mode"] == "trade" and not (dsr["value"] >
                                           thresholds["dsr_threshold"]):
            _fail(f"{spath}.dsr.value",
                  f"a trade-mode selection requires DSR > "
                  f"{thresholds['dsr_threshold']}; got {dsr['value']}")
        prov = dsr["provenance"]
        ppath = f"{spath}.dsr.provenance"
        _dict(ppath, prov, _DSR_PROVENANCE_FIELDS)
        _exact(f"{ppath}.n_trials", prov["n_trials"],
               ledger["n_effective_trials"])
        _exact(f"{ppath}.n_trials_source", prov["n_trials_source"],
               "g0bn_ledger_unique_identity_count_v1")
        if prov["ledger_sha256"] != ledger["ledger_sha256"]:
            _fail(f"{ppath}.ledger_sha256",
                  "DSR provenance does not pin the freeze's ledger hash "
                  f"({prov['ledger_sha256']} != {ledger['ledger_sha256']})")
        _num(f"{ppath}.sr_trials_std", prov["sr_trials_std"], positive=True)
        scored = prov["same_horizon_scored_trial_ids"]
        if (not isinstance(scored, list)
                or len(set(scored)) != len(scored)
                or scored[:len(CANDIDATE_IDS)] != base_tids_by_tag[tag]):
            _fail(f"{ppath}.same_horizon_scored_trial_ids",
                  "must be the unique scored identities with the complete base "
                  "ladder first, in ladder order")
        for i, tid in enumerate(scored):
            _sha256(f"{ppath}.same_horizon_scored_trial_ids[{i}]", tid)
        _num(f"{ppath}.effective_trades", prov["effective_trades"], minimum=0)
        _int(f"{ppath}.T", prov["T"], minimum=2)
        if prov["T"] != dsr_sample_count(prov["effective_trades"])[0]:
            _fail(f"{ppath}.T",
                  "does not reproduce nearest_ties_to_even_int64_v1 over the "
                  "recorded effective trades")
        _exact(f"{ppath}.rounding_rule", prov["rounding_rule"],
               DSR_ROUNDING_RULE)
        _exact(f"{ppath}.epsilon", prov["epsilon"], 1e-9)
        _exact(f"{ppath}.threshold", prov["threshold"],
               thresholds["dsr_threshold"])
        _exact(f"{ppath}.code_sha256", prov["code_sha256"],
               config["cv"]["dsr"]["code_sha256"])

    _dict("control_candidates", freeze["control_candidates"],
          tuple(control_tags))
    for tag in control_tags:
        controls = freeze["control_candidates"][tag]
        cpath = f"control_candidates[{tag!r}]"
        if not isinstance(controls, list) or len(controls) != len(CANDIDATE_IDS):
            _fail(cpath, f"must pin the complete {len(CANDIDATE_IDS)}-candidate "
                         f"control set in ladder order")
        for i, entry in enumerate(controls):
            epath = f"{cpath}[{i}]"
            _dict(epath, entry, _CONTROL_FIELDS)
            _exact(f"{epath}.candidate_id", entry["candidate_id"],
                   CANDIDATE_IDS[i])
            expected_identity = base_by_key[(tag, CANDIDATE_IDS[i])]
            _exact(f"{epath}.identity", entry["identity"], expected_identity)
            _sha256(f"{epath}.trial_id", entry["trial_id"])
            if entry["trial_id"] != base_tids_by_tag[tag][i]:
                _fail(f"{epath}.trial_id",
                      "does not re-derive from the embedded canonical identity")

    _dict("pbo", freeze["pbo"], tuple(primary_tags))
    for tag in primary_tags:
        bpath = f"pbo[{tag!r}]"
        block = freeze["pbo"][tag]
        _dict(bpath, block, _PBO_FIELDS)
        _exact(f"{bpath}.available", block["available"], True)
        if block["reason"] is not None:
            _fail(f"{bpath}.reason", f"must be JSON null for an available PBO; "
                                     f"got {block['reason']!r}")
        _num(f"{bpath}.value", block["value"], minimum=0)
        if not (block["value"] < thresholds["pbo_threshold"]):
            _fail(f"{bpath}.value",
                  f"a freeze requires PBO < {thresholds['pbo_threshold']}; "
                  f"got {block['value']} (spec section 5.3)")
        _exact(f"{bpath}.threshold", block["threshold"],
               thresholds["pbo_threshold"])
        _exact(f"{bpath}.n_blocks", block["n_blocks"], PBO_N_BLOCKS)
        _exact(f"{bpath}.is_tie_rule", block["is_tie_rule"], PBO_IS_TIE_RULE)
        _exact(f"{bpath}.oos_rank_rule", block["oos_rank_rule"],
               PBO_OOS_RANK_RULE)
        _exact(f"{bpath}.column_order_rule", block["column_order_rule"],
               "base_ladder_then_ascending_trial_id_v1")
        _int(f"{bpath}.n_rows", block["n_rows"], minimum=PBO_MIN_ROWS)
        columns = block["column_trial_ids"]
        if (not isinstance(columns, list)
                or len(set(columns)) != len(columns)
                or columns[:len(CANDIDATE_IDS)] != base_tids_by_tag[tag]):
            _fail(f"{bpath}.column_trial_ids",
                  "must be the unique scored identities with the complete base "
                  "ladder first, in ladder order")
        for i, tid in enumerate(columns):
            _sha256(f"{bpath}.column_trial_ids[{i}]", tid)
        _exact(f"{bpath}.n_columns", block["n_columns"], len(columns))
        if block["n_columns"] < PBO_MIN_COLUMNS:
            _fail(f"{bpath}.n_columns", f"must be >= {PBO_MIN_COLUMNS}")
        _exact(f"{bpath}.n_combinations", block["n_combinations"],
               math.comb(PBO_N_BLOCKS, PBO_N_BLOCKS // 2))
        _sha256(f"{bpath}.input_sha256", block["input_sha256"])
        if block["ledger_sha256"] != freeze["ledger"]["ledger_sha256"]:
            _fail(f"{bpath}.ledger_sha256",
                  "PBO provenance does not pin the freeze's ledger hash "
                  f"({block['ledger_sha256']} != "
                  f"{freeze['ledger']['ledger_sha256']})")
        _exact(f"{bpath}.split_sha256", block["split_sha256"],
               dev["split_sha256s"][tag])
        _exact(f"{bpath}.code_sha256", block["code_sha256"],
               config["cv"]["pbo"]["code_sha256"])

    _exact("verdict_thresholds", freeze["verdict_thresholds"],
           config["verdict_thresholds"])
    _exact("costs", freeze["costs"], config["costs"])
    _exact("exclusions", freeze["exclusions"], config["exclusions"])
    _exact("oos", freeze["oos"], config["oos"])
    if freeze["generated_at"] is not None:
        _validate_generated_at(freeze["generated_at"])

    embedded = _sha256("sha256", freeze["sha256"])
    recomputed = g0bn_artifact_sha256(freeze)
    if embedded != recomputed:
        _fail("sha256", f"embedded freeze sha256 does not match the canonical "
                        f"content (tampered or stale): {embedded} != {recomputed}")
    if run is not None:
        if inventory is None:
            _fail("inventory",
                  "run-anchored freeze validation requires the #68 custody "
                  "inventory: the rebuild must re-anchor the plan to custody, "
                  "not merely to the config (the one-shot runner loads the seal "
                  "metadata anyway; spec section 6.3 step 1)")
        # build_freeze validates the plan against run.config, whose sha the plan
        # already pins, so the rebuild is anchored to the same immutable config
        # AND the custody inventory.
        rebuilt = build_freeze(run, plan, inventory=inventory,
                               generated_at=freeze["generated_at"])
        if rebuilt != freeze:
            diff = sorted(k for k in _FREEZE_FIELDS
                          if rebuilt.get(k) != freeze.get(k))
            _fail("freeze",
                  f"does not reproduce from the development run and the "
                  f"outcome-blind plan (differing fields: {diff}); the freeze "
                  "is re-derived development evidence (spec section 5.3), so a "
                  "value development_selection cannot reproduce is fabricated "
                  "or stale")
    return freeze
