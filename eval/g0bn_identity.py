"""Stable G0-BN identities (issue #86, slice 67-A of issue #67).

Implements spec section 6.1 (stable `g0bn-holdout-universe-v1` and
`g0bn-one-shot-v1` identities) and the identity-bearing inputs of section 4.2
(`g0bn-trial-v1`, development logical build identity).

The universe/transaction identities hash the exact spec objects and nothing
else: pilot, config, candidate, freeze, source, seal, plan, build, manifest,
row, attempt, and result values are not inputs, so changing them cannot mint a
second transaction over the same January outcomes.

The development data identity binds trials to the producer's canonical
logical-row/content hash and logical build id only. Physical Parquet file
hashes (`matrix_file_sha256`) are audit/custody evidence: the exact field sets
here reject them, so Parquet metadata, row-group layout, compression, or writer
version can never reidentify a trial (spec section 3.1).
"""
from __future__ import annotations

import copy

from eval.g0bn_config import (
    CANDIDATE_IDS,
    DEV_DATASET_ID,
    HOLDOUT_END_NS,
    HOLDOUT_START_NS,
    FEATURE_REGISTRY,
    HORIZON_ROLES,
    INSTRUMENT,
    ONE_SHOT_SCHEMA,
    PROTOCOL_ID,
    TRIAL_SCHEMA,
    UNIVERSE_SCHEMA,
    _dict,
    _exact,
    _fail,
    _int,
    _scalar_tree,
    _sha256,
    _str,
    validate_protocol_config,
)
from eval.hashing import hash_obj

_UNIVERSE_FIELDS = ("schema", "protocol_id", "instrument", "holdout_start_ns",
                    "holdout_end_ns")


def holdout_universe(instrument: dict | None = None, *,
                     holdout_start_ns: int | None = None,
                     holdout_end_ns: int | None = None) -> dict:
    """Build the validated stable outcome-universe object. Defaults are exactly the
    pinned spec 6.1 values; overrides exist only so tests can prove identity
    sensitivity — the v1 protocol universe is the default object."""
    instrument = dict(INSTRUMENT) if instrument is None else instrument
    start = HOLDOUT_START_NS if holdout_start_ns is None else holdout_start_ns
    end = HOLDOUT_END_NS if holdout_end_ns is None else holdout_end_ns
    _dict("instrument", instrument, tuple(INSTRUMENT))
    for k in INSTRUMENT:
        _str(f"instrument.{k}", instrument[k])
    _int("holdout_start_ns", start, minimum=1)
    _int("holdout_end_ns", end, minimum=1)
    if end <= start:
        _fail("holdout_end_ns", f"must be greater than holdout_start_ns; "
                                f"got [{start}, {end})")
    return {
        "schema": UNIVERSE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "instrument": dict(instrument),
        "holdout_start_ns": start,
        "holdout_end_ns": end,
    }


def holdout_universe_id(universe: dict | None = None) -> str:
    """SHA-256 of the canonical universe object (spec 6.1). No pilot, config, freeze,
    source, seal, plan, build, manifest, row, attempt, or result value is an input."""
    if universe is None:
        universe = holdout_universe()
    _dict("holdout universe", universe, _UNIVERSE_FIELDS)
    if universe["schema"] != UNIVERSE_SCHEMA:
        _fail("holdout universe schema", f"must equal {UNIVERSE_SCHEMA!r}; "
                                         f"got {universe['schema']!r}")
    if universe["protocol_id"] != PROTOCOL_ID:
        _fail("holdout universe protocol_id", f"must equal {PROTOCOL_ID!r}; "
                                              f"got {universe['protocol_id']!r}")
    # The stable universe is EXACTLY the G0-BN BTC-USDT perpetual + fixed bounds
    # (spec 6.1): an out-of-protocol instrument or window must fail before it derives a
    # separate transaction/lock/consumption path, not silently mint a second identity.
    _exact("holdout universe.instrument", universe["instrument"], INSTRUMENT)
    _exact("holdout universe.holdout_start_ns", universe["holdout_start_ns"],
           HOLDOUT_START_NS)
    _exact("holdout universe.holdout_end_ns", universe["holdout_end_ns"], HOLDOUT_END_NS)
    return hash_obj(universe)


def one_shot_transaction_id(universe_id: str | None = None) -> str:
    """SHA-256 of exactly {"schema": "g0bn-one-shot-v1", "holdout_universe_id": ...}."""
    if universe_id is None:
        universe_id = G0BN_HOLDOUT_UNIVERSE_ID
    _sha256("holdout_universe_id", universe_id)
    return hash_obj({"schema": ONE_SHOT_SCHEMA, "holdout_universe_id": universe_id})


G0BN_HOLDOUT_UNIVERSE_ID = holdout_universe_id()
G0BN_TRANSACTION_ID = one_shot_transaction_id(G0BN_HOLDOUT_UNIVERSE_ID)


# --- development logical build identity (no physical Parquet metadata) -----------------

_DATA_IDENTITY_FIELDS = (
    "development_dataset_id", "development_build_id", "development_manifest_sha256",
    "development_logical_row_sha256", "partition_plan_sha256",
)


def development_data_identity(identity: dict) -> dict:
    """Validate the logical development-data identity a trial binds to. The exact
    field set rejects any physical file hash (e.g. matrix_file_sha256): those are
    audit-only and never identity-bearing."""
    _dict("development data identity", identity, _DATA_IDENTITY_FIELDS)
    if identity["development_dataset_id"] != DEV_DATASET_ID:
        _fail("development_dataset_id",
              f"must equal {DEV_DATASET_ID!r} (G0-BN trials are development-only; "
              f"got {identity['development_dataset_id']!r})")
    _str("development_build_id", identity["development_build_id"])
    _sha256("development_manifest_sha256", identity["development_manifest_sha256"])
    _sha256("development_logical_row_sha256", identity["development_logical_row_sha256"])
    _sha256("partition_plan_sha256", identity["partition_plan_sha256"])
    return identity


# --- g0bn-trial-v1 identity (spec section 4.2) ------------------------------------------

TRIAL_IDENTITY_FIELDS = (
    "schema",
    "protocol_config_sha256",
    "source_certification_sha256",
    "development_dataset_id",
    "development_build_id",
    "development_manifest_sha256",
    "development_logical_row_sha256",
    "partition_plan_sha256",
    "candidate_id",
    "candidate_definition_sha256",
    "candidate_code_sha256",
    "preprocessing",
    "preprocessing_sha256",
    "model_params",
    "model_params_sha256",
    "feature_cols",
    "seed_and_thread_settings",
    "software_versions_sha256",
    "horizon",
    "horizon_role",
    "cv_sha256",
    "label_sha256",
    "cost_sha256",
    "thresholds_sha256",
    "variant",
    "variant_params",
)

_TRIAL_SHA_FIELDS = (
    "protocol_config_sha256", "source_certification_sha256",
    "development_manifest_sha256", "development_logical_row_sha256",
    "partition_plan_sha256", "candidate_definition_sha256", "candidate_code_sha256",
    "preprocessing_sha256", "model_params_sha256", "software_versions_sha256",
    "cv_sha256", "label_sha256", "cost_sha256", "thresholds_sha256",
)


def validate_trial_identity(identity: dict) -> dict:
    """Exact-field-set validation of one g0bn-trial-v1 identity object. The complete
    resolved preprocessing/model_params objects are identity-bearing, and their
    embedded convenience hashes must match them (no silent drift). Execution
    ordinals/timestamps are ledger-event fields and are rejected here."""
    _dict("g0bn trial identity", identity, TRIAL_IDENTITY_FIELDS)
    if identity["schema"] != TRIAL_SCHEMA:
        _fail("schema", f"must equal {TRIAL_SCHEMA!r}; got {identity['schema']!r}")
    for k in _TRIAL_SHA_FIELDS:
        _sha256(k, identity[k])
    if identity["development_dataset_id"] != DEV_DATASET_ID:
        _fail("development_dataset_id",
              f"must equal {DEV_DATASET_ID!r} (trials never bind OOS/holdout data); "
              f"got {identity['development_dataset_id']!r}")
    _str("development_build_id", identity["development_build_id"])
    if identity["candidate_id"] not in CANDIDATE_IDS:
        _fail("candidate_id", f"must be one of {CANDIDATE_IDS}; "
                              f"got {identity['candidate_id']!r}")
    # An off-ladder horizon (e.g. an accidental data-derived tau rung attempted after
    # v1 trials start) is still a valid, RECORDABLE trial identity that counts toward
    # effective N / DSR provenance (spec §4.2); it is simply not eligible for v1
    # selection (base_trial_identities never emits it). So accept a non-empty
    # off-ladder horizon/role, but keep exact role consistency for the three base
    # horizons. Eligibility is enforced at enumeration/selection, not here.
    _str("horizon", identity["horizon"])
    _str("horizon_role", identity["horizon_role"])
    if (identity["horizon"] in HORIZON_ROLES
            and identity["horizon_role"] != HORIZON_ROLES[identity["horizon"]]):
        _fail("horizon_role", f"must equal {HORIZON_ROLES[identity['horizon']]!r} for "
                              f"base horizon {identity['horizon']!r}; "
                              f"got {identity['horizon_role']!r}")
    for k in ("preprocessing", "model_params", "seed_and_thread_settings",
              "variant_params"):
        if not isinstance(identity[k], dict):
            _fail(k, f"must be a dict; got {type(identity[k]).__name__}")
        _scalar_tree(k, identity[k])
    cols = identity["feature_cols"]
    if not isinstance(cols, list):
        _fail("feature_cols", f"must be an ordered array; got {cols!r}")
    for i, c in enumerate(cols):
        if c not in FEATURE_REGISTRY:
            _fail("feature_cols", f"entry {c!r} is not in the G0-BN feature registry")
    if len(set(cols)) != len(cols):
        _fail("feature_cols", "must be unique (ordered, no duplicates)")
    _str("variant", identity["variant"])
    if hash_obj(identity["preprocessing"]) != identity["preprocessing_sha256"]:
        _fail("preprocessing_sha256", "does not match the preprocessing object")
    if hash_obj(identity["model_params"]) != identity["model_params_sha256"]:
        _fail("model_params_sha256", "does not match the model_params object")
    return identity


def trial_id(identity: dict) -> str:
    """Canonical trial identity hash: SHA-256 of the exact validated identity object
    (never of a container carrying the resulting id)."""
    return hash_obj(validate_trial_identity(identity))


def base_trial_identities(config: dict, data_identity: dict) -> list:
    """Enumerate the complete preregistered v1 ladder: 5 candidates x 3 horizons = 15
    unique trial identities, in horizon-major, ladder-order sequence. Fails closed on
    an invalid config/data identity or a partition-plan mismatch between them."""
    validate_protocol_config(config)
    development_data_identity(data_identity)
    if data_identity["partition_plan_sha256"] != config["partition"]["sha256"]:
        _fail("partition_plan_sha256",
              "development data identity does not bind the config's partition plan "
              f"({data_identity['partition_plan_sha256']} != "
              f"{config['partition']['sha256']})")
    # Bind the trial to the explicit #64 source-certification artifact hash (spec §2.1
    # "source-certification hash"), not a hash of the whole config section, so trial /
    # manifest-binding / freeze reconciliation all reference the same evidence hash.
    source_certification_sha256 = config["source_certification"]["certification_sha256"]
    software_versions_sha256 = hash_obj(config["software"])
    cv_sha256 = hash_obj(config["cv"])
    label_sha256 = hash_obj(config["labels"])
    cost_sha256 = hash_obj(config["costs"])
    thresholds_sha256 = hash_obj(config["verdict_thresholds"])
    identities = []
    for horizon in config["horizons"]:
        for defn in config["candidates"]:
            identity = {
                "schema": TRIAL_SCHEMA,
                "protocol_config_sha256": config["sha256"],
                "source_certification_sha256": source_certification_sha256,
                "development_dataset_id": data_identity["development_dataset_id"],
                "development_build_id": data_identity["development_build_id"],
                "development_manifest_sha256":
                    data_identity["development_manifest_sha256"],
                "development_logical_row_sha256":
                    data_identity["development_logical_row_sha256"],
                "partition_plan_sha256": data_identity["partition_plan_sha256"],
                "candidate_id": defn["candidate_id"],
                "candidate_definition_sha256": hash_obj(defn),
                "candidate_code_sha256": defn["candidate_code_sha256"],
                "preprocessing": copy.deepcopy(defn["preprocessing"]),
                "preprocessing_sha256": defn["preprocessing_sha256"],
                "model_params": copy.deepcopy(defn["model_params"]),
                "model_params_sha256": defn["model_params_sha256"],
                "feature_cols": list(defn["feature_cols"]),
                "seed_and_thread_settings":
                    copy.deepcopy(defn.get("seed_and_thread_settings", {})),
                "software_versions_sha256": software_versions_sha256,
                "horizon": horizon["tag"],
                "horizon_role": horizon["role"],
                "cv_sha256": cv_sha256,
                "label_sha256": label_sha256,
                "cost_sha256": cost_sha256,
                "thresholds_sha256": thresholds_sha256,
                "variant": "base",
                "variant_params": {},
            }
            identities.append(validate_trial_identity(identity))
    return identities
