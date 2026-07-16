"""G0-BN stable identities (issue #86, slice 67-A): holdout-universe / one-shot
transaction IDs, development logical build identity, and g0bn-trial-v1 identities.

All fixtures are synthetic; expected digests are hand-computed with hashlib/json
straight from the spec literals so the implementation cannot self-confirm.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from eval.g0bn_identity import (
    G0BN_HOLDOUT_UNIVERSE_ID,
    G0BN_TRANSACTION_ID,
    base_trial_identities,
    development_data_identity,
    holdout_universe,
    holdout_universe_id,
    one_shot_transaction_id,
    trial_id,
    validate_trial_identity,
)
from eval.hashing import matrix_content_hash

from g0bn_protocol_fixtures import (
    CANDIDATE_IDS,
    DEV_DATASET_ID,
    HOLDOUT_END_NS,
    HOLDOUT_START_NS,
    INSTRUMENT,
    make_config,
    make_data_identity,
    make_trial_identity,
    spec_transaction_id,
    spec_universe_id,
    spec_universe_object,
)


# --- holdout universe / one-shot transaction IDs --------------------------------------

def test_universe_id_matches_hand_computed_spec_literal():
    # The exact canonical encoding of the spec 6.1 object, written out by hand.
    canonical = (
        '{"holdout_end_ns":1769904000000000000,'
        '"holdout_start_ns":1767225600000000000,'
        '"instrument":{"base_asset":"BTC","contract_type":"linear_perpetual",'
        '"exchange":"BINANCE_FUTURES","native_symbol":"BTCUSDT","quote_asset":"USDT",'
        '"settlement_asset":"USDT","symbol":"BTC-USDT-PERP"},'
        '"protocol_id":"g0bn-v1","schema":"g0bn-holdout-universe-v1"}'
    )
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    assert holdout_universe_id() == expected
    assert G0BN_HOLDOUT_UNIVERSE_ID == expected
    assert spec_universe_id() == expected


def test_transaction_id_matches_hand_computed_derivation():
    canonical = json.dumps(
        {"schema": "g0bn-one-shot-v1", "holdout_universe_id": holdout_universe_id()},
        sort_keys=True, separators=(",", ":"),
    )
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    assert one_shot_transaction_id() == expected
    assert G0BN_TRANSACTION_ID == expected
    assert spec_transaction_id() == expected


def test_default_universe_object_is_exactly_the_spec_object():
    assert holdout_universe() == spec_universe_object()


def test_universe_id_rejects_out_of_protocol_instrument_or_window():
    # The stable universe is EXACTLY the G0-BN BTC-USDT perpetual + fixed bounds
    # (spec 6.1); an out-of-protocol instrument/window must fail rather than mint a
    # separate transaction/lock/consumption path.
    other_instrument = dict(INSTRUMENT, symbol="ETH-USDT-PERP", native_symbol="ETHUSDT",
                            base_asset="ETH")
    with pytest.raises(ValueError, match="instrument"):
        holdout_universe_id(holdout_universe(instrument=other_instrument))
    with pytest.raises(ValueError, match="holdout_start_ns"):
        holdout_universe_id(
            holdout_universe(holdout_start_ns=HOLDOUT_START_NS + 86_400_000_000_000))
    with pytest.raises(ValueError, match="holdout_end_ns"):
        holdout_universe_id(
            holdout_universe(holdout_end_ns=HOLDOUT_END_NS + 86_400_000_000_000))
    # The identity object has exactly these five fields, so pilot/config/freeze/source/
    # plan/build/result values cannot mint a second transaction over the same outcomes.
    assert set(holdout_universe()) == {
        "schema", "protocol_id", "instrument", "holdout_start_ns", "holdout_end_ns",
    }


def test_universe_object_validation_fails_closed():
    with pytest.raises(ValueError, match="instrument"):
        holdout_universe(instrument={**INSTRUMENT, "extra_venue": "x"})
    missing = dict(INSTRUMENT)
    missing.pop("settlement_asset")
    with pytest.raises(ValueError, match="instrument"):
        holdout_universe(instrument=missing)
    with pytest.raises(ValueError, match="holdout_start_ns"):
        holdout_universe(holdout_start_ns=True)
    with pytest.raises(ValueError, match="holdout_start_ns"):
        holdout_universe(holdout_start_ns="1767225600000000000")
    with pytest.raises(ValueError, match="holdout_end_ns"):
        holdout_universe(holdout_end_ns=HOLDOUT_START_NS)  # empty window


def test_transaction_id_rejects_malformed_universe_id():
    for bad in ("", "xyz", "A" * 64, spec_universe_id().upper(), spec_universe_id()[:-1]):
        with pytest.raises(ValueError, match="holdout_universe_id"):
            one_shot_transaction_id(bad)


# --- development data identity (logical build inputs; no physical file hashes) --------

def test_data_identity_valid_and_chains():
    ident = make_data_identity()
    assert development_data_identity(ident) == ident


def test_data_identity_requires_exact_field_set():
    ident = make_data_identity()
    ident["matrix_file_sha256"] = "0" * 64  # physical hash is audit-only, never identity
    with pytest.raises(ValueError, match="matrix_file_sha256"):
        development_data_identity(ident)
    missing = make_data_identity()
    missing.pop("development_logical_row_sha256")
    with pytest.raises(ValueError, match="development_logical_row_sha256"):
        development_data_identity(missing)


def test_data_identity_field_validation():
    with pytest.raises(ValueError, match="development_dataset_id"):
        development_data_identity(make_data_identity(development_dataset_id="g0xv_dev"))
    with pytest.raises(ValueError, match="development_build_id"):
        development_data_identity(make_data_identity(development_build_id=""))
    with pytest.raises(ValueError, match="development_manifest_sha256"):
        development_data_identity(
            make_data_identity(development_manifest_sha256="0" * 63))
    with pytest.raises(ValueError, match="development_logical_row_sha256"):
        development_data_identity(
            make_data_identity(development_logical_row_sha256=("0" * 63) + "G"))


# --- g0bn-trial-v1 identity ------------------------------------------------------------

def test_trial_identity_valid_and_hash_deterministic():
    ident = make_trial_identity()
    assert validate_trial_identity(ident) == ident
    a = trial_id(ident)
    b = trial_id(copy.deepcopy(ident))
    assert a == b and len(a) == 64 and a == a.lower()


def test_trial_identity_requires_exact_field_set():
    ident = make_trial_identity()
    ident["execution_ordinal"] = 3  # ledger-event field, never identity-bearing
    with pytest.raises(ValueError, match="execution_ordinal"):
        validate_trial_identity(ident)
    ident = make_trial_identity()
    ident["matrix_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="matrix_file_sha256"):
        validate_trial_identity(ident)
    for field in ("schema", "cv_sha256", "variant_params", "seed_and_thread_settings"):
        broken = make_trial_identity()
        broken.pop(field)
        with pytest.raises(ValueError, match=field):
            validate_trial_identity(broken)


@pytest.mark.parametrize("field", [
    "preprocessing", "model_params", "variant_params", "seed_and_thread_settings",
])
def test_trial_identity_rejects_nested_physical_file_hash(field):
    # Physical Parquet file hashes are audit-only; nesting one inside an opaque
    # identity subtree must not let it reidentify a logically-identical trial.
    from eval.hashing import hash_obj

    ident = make_trial_identity()
    poisoned = dict(ident[field], matrix_file_sha256="0" * 64)
    over = {field: poisoned}
    if field == "preprocessing":
        over["preprocessing_sha256"] = hash_obj(poisoned)
    if field == "model_params":
        over["model_params_sha256"] = hash_obj(poisoned)
    with pytest.raises(ValueError, match="file_sha256"):
        validate_trial_identity(make_trial_identity(**over))


def test_trial_identity_rejects_negative_zero_in_subtree():
    poisoned = dict(make_trial_identity()["variant_params"], drift=-0.0)
    with pytest.raises(ValueError, match="negative zero"):
        validate_trial_identity(make_trial_identity(variant_params=poisoned))


def test_trial_identity_field_validation():
    with pytest.raises(ValueError, match="schema"):
        validate_trial_identity(make_trial_identity(schema="g0xv-trial-v1"))
    with pytest.raises(ValueError, match="candidate_id"):
        validate_trial_identity(make_trial_identity(candidate_id="magic_extra_model"))
    with pytest.raises(ValueError, match="horizon"):
        validate_trial_identity(make_trial_identity(horizon="30s"))
    with pytest.raises(ValueError, match="horizon_role"):
        validate_trial_identity(make_trial_identity(horizon="60s"))  # role stays primary
    with pytest.raises(ValueError, match="development_dataset_id"):
        validate_trial_identity(
            make_trial_identity(development_dataset_id="binance_single_venue_g0bn_oos"))
    with pytest.raises(ValueError, match="variant"):
        validate_trial_identity(make_trial_identity(variant=""))
    with pytest.raises(ValueError, match="feature_cols"):
        validate_trial_identity(make_trial_identity(feature_cols=["not_in_registry"]))
    with pytest.raises(ValueError, match="feature_cols"):
        validate_trial_identity(
            make_trial_identity(feature_cols=["ofi_integrated", "ofi_integrated"]))


def test_trial_identity_embedded_hashes_must_match_objects():
    ident = make_trial_identity()
    ident["preprocessing"] = dict(ident["preprocessing"], tweak=True)
    with pytest.raises(ValueError, match="preprocessing_sha256"):
        validate_trial_identity(ident)
    ident = make_trial_identity()
    ident["model_params"] = dict(ident["model_params"], alpha=2.0)
    with pytest.raises(ValueError, match="model_params_sha256"):
        validate_trial_identity(ident)


def test_trial_id_sensitivity_and_order_sensitivity():
    base = trial_id(make_trial_identity())
    assert trial_id(make_trial_identity(horizon="10s")) != base
    assert trial_id(make_trial_identity(variant="tweaked",
                                        variant_params={"margin_bps": 1.0})) != base
    reordered = make_trial_identity()
    reordered["feature_cols"] = ["microprice_dev", "ofi_integrated"]
    ordered = make_trial_identity(feature_cols=["ofi_integrated", "microprice_dev"])
    assert trial_id(reordered) != trial_id(ordered)  # array order is decision-bearing


def test_trial_id_is_sensitive_to_every_identity_field():
    from eval.hashing import hash_obj

    base = trial_id(make_trial_identity())
    # Seed/thread, code, version, data, and section-hash fields (spec 11 item 3).
    for field, value in [
        ("seed_and_thread_settings", {"random_state": 1}),
        ("candidate_code_sha256", "1" * 64),
        ("candidate_definition_sha256", "1" * 64),
        ("software_versions_sha256", "1" * 64),
        ("protocol_config_sha256", "1" * 64),
        ("source_certification_sha256", "1" * 64),
        ("development_build_id", "g0bn-dev-build-0002"),
        ("development_manifest_sha256", "1" * 64),
        ("development_logical_row_sha256", "1" * 64),
        ("partition_plan_sha256", "1" * 64),
        ("cv_sha256", "1" * 64),
        ("label_sha256", "1" * 64),
        ("cost_sha256", "1" * 64),
        ("thresholds_sha256", "1" * 64),
    ]:
        assert trial_id(make_trial_identity(**{field: value})) != base, field
    # Full resolved objects are identity-bearing (with consistent convenience hashes).
    preprocessing = {"stationarization": "other_v1", "candidate_local_scaling": False}
    changed = make_trial_identity(preprocessing=preprocessing,
                                  preprocessing_sha256=hash_obj(preprocessing))
    assert trial_id(changed) != base
    params = dict(make_trial_identity()["model_params"], alpha=2.0)
    changed = make_trial_identity(model_params=params,
                                  model_params_sha256=hash_obj(params))
    assert trial_id(changed) != base


def test_base_trial_enumeration_is_exactly_the_15_spec_trials():
    config = make_config()
    identities = base_trial_identities(config, make_data_identity(config))
    assert len(identities) == 15
    ids = [trial_id(i) for i in identities]
    assert len(set(ids)) == 15
    combos = {(i["candidate_id"], i["horizon"]) for i in identities}
    assert combos == {(c, h) for c in CANDIDATE_IDS for h in ("2s", "10s", "60s")}
    for ident in identities:
        assert ident["schema"] == "g0bn-trial-v1"
        assert ident["variant"] == "base"
        assert ident["variant_params"] == {}
        assert ident["protocol_config_sha256"] == config["sha256"]
        assert ident["development_dataset_id"] == DEV_DATASET_ID
    zero = [i for i in identities if i["candidate_id"] == "persistence_zero"]
    assert all(i["feature_cols"] == [] for i in zero)  # empty-feature persistence identity
    # Section hashes must bind the RIGHT config sections (not swapped/stale); the
    # source certification binds the explicit #64 artifact hash, not the section hash.
    from eval.hashing import hash_obj
    for ident in identities:
        assert ident["source_certification_sha256"] == \
            config["source_certification"]["certification_sha256"]
        assert ident["source_certification_sha256"] != hash_obj(
            config["source_certification"])
        assert ident["software_versions_sha256"] == hash_obj(config["software"])
        assert ident["cv_sha256"] == hash_obj(config["cv"])
        assert ident["label_sha256"] == hash_obj(config["labels"])
        assert ident["cost_sha256"] == hash_obj(config["costs"])
        assert ident["thresholds_sha256"] == hash_obj(config["verdict_thresholds"])
        assert ident["partition_plan_sha256"] == config["partition"]["sha256"]
    # Deterministic: re-enumeration reproduces the same identity hashes.
    again = base_trial_identities(make_config(), make_data_identity(make_config()))
    assert [trial_id(i) for i in again] == ids


def test_base_trial_enumeration_cross_checks_partition_plan():
    config = make_config()
    ident = make_data_identity(config, partition_plan_sha256="1" * 64)
    with pytest.raises(ValueError, match="partition_plan_sha256"):
        base_trial_identities(config, ident)


# --- logical identity is independent of Parquet physical metadata ---------------------

def test_logical_row_hash_survives_different_parquet_encodings(tmp_path):
    df = pd.DataFrame({
        "t_event": pd.array([1_000, 2_000, 3_000, 4_000], dtype="int64"),
        "horizon": ["2s", "2s", "10s", "10s"],
        "y_fwd_bps": [0.5, -1.25, 2.0, 0.0],
        "uniqueness": [1.0, 0.5, 1.0, 0.25],
    })
    cols = list(df.columns)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    df.to_parquet(a, engine="pyarrow", compression="snappy", row_group_size=2)
    df.to_parquet(b, engine="pyarrow", compression="zstd", row_group_size=1)
    file_a, file_b = a.read_bytes(), b.read_bytes()
    assert hashlib.sha256(file_a).hexdigest() != hashlib.sha256(file_b).hexdigest()

    logical_a = matrix_content_hash(pd.read_parquet(a), cols)
    logical_b = matrix_content_hash(pd.read_parquet(b), cols)
    assert logical_a == logical_b  # physical metadata cannot change the logical identity

    config = make_config()
    ident_a = make_data_identity(config, development_logical_row_sha256=logical_a)
    ident_b = make_data_identity(config, development_logical_row_sha256=logical_b)
    trials_a = [trial_id(t) for t in base_trial_identities(config, ident_a)]
    trials_b = [trial_id(t) for t in base_trial_identities(config, ident_b)]
    assert trials_a == trials_b  # same trial IDs and effective N across encodings
