"""Append-only G0-BN attempt ledger tests (issue #88, slice 67-B; spec section 4.2).

Covers spec section 11 item 4: append-only behavior, exact-rerun idempotency,
conflicting-result rejection, unique aborted/additional-variant counting with no
replacement, and structural rejection of legacy G0-CB/G0-XV ledger content.
All identities are synthetic (tests/g0bn_protocol_fixtures.py); no vendor data.
"""
from __future__ import annotations

import json
import os

import pytest

from eval.g0bn_identity import trial_id
from eval.g0bn_ledger import LEDGER_SCHEMA, G0BNLedger
from eval.hashing import hash_obj
from eval.ledger import TrialLedger, trial_identity as legacy_trial_identity
from tests.g0bn_protocol_fixtures import make_trial_identity, sha_hex


def _identity(**over) -> dict:
    return make_trial_identity(**over)


def _result(tag: str = "r0", **over) -> dict:
    d = {
        "schema": "g0bn-trial-result-v1",
        "n_rows": 240,
        "forecasts_sha256": sha_hex(f"forecasts-{tag}"),
        "collapse_version": "mean_repeated_test_forecasts_v1",
        "split_scales": None,
    }
    d.update(over)
    return d


# --------------------------------------------------------------- registration / counting

def test_schema_constant_is_spec_literal():
    assert LEDGER_SCHEMA == "g0bn-ledger-v1"


def test_completion_registers_unique_identity_and_result():
    led = G0BNLedger()
    ident = _identity()
    tid = led.record_start(ident)
    assert tid == trial_id(ident)
    assert led.record_completion(ident, _result()) == tid
    assert led.n_effective_trials() == 1
    assert led.scored_trial_ids() == [tid]
    assert led.result_for(tid) == _result()
    assert led.identity_for(tid) == ident


def test_effective_n_counts_unique_identities_not_executions():
    led = G0BNLedger()
    ident = _identity()
    led.record_start(ident)
    led.record_completion(ident, _result())
    # A second full execution of the SAME identity is an idempotent execution event.
    led.record_start(ident)
    led.record_completion(ident, _result())
    assert led.n_effective_trials() == 1
    assert len(led.events()) == 4  # every execution event is recorded, append-only


def test_exact_rerun_must_reproduce_existing_result_hash():
    led = G0BNLedger()
    ident = _identity()
    led.record_completion(ident, _result())
    events_before = len(led.events())
    with pytest.raises(ValueError, match="DIFFERENT result"):
        led.record_completion(ident, _result(n_rows=241))
    # Conflict appends nothing and replaces nothing (no silent overwrite).
    assert len(led.events()) == events_before
    assert led.result_for(trial_id(ident)) == _result()
    assert led.n_effective_trials() == 1


def test_aborted_only_identity_counts_once_in_effective_n():
    led = G0BNLedger()
    ident = _identity(horizon="10s")
    led.record_start(ident)
    tid = led.record_abort(ident, error="lightgbm segfault (infrastructure)")
    assert led.n_effective_trials() == 1
    assert led.result_for(tid) is None
    assert led.scored_trial_ids() == []
    # A second abort of the same identity still counts once.
    led.record_abort(ident, error="oom")
    assert led.n_effective_trials() == 1


def test_retry_after_abort_may_complete_under_same_identity():
    led = G0BNLedger()
    ident = _identity()
    led.record_start(ident)
    led.record_abort(ident, error="transient")
    led.record_start(ident)
    tid = led.record_completion(ident, _result())
    assert led.n_effective_trials() == 1
    assert led.result_for(tid) == _result()


def test_changed_variant_is_a_new_immutable_trial():
    led = G0BNLedger()
    base = _identity()
    led.record_completion(base, _result())
    variant = _identity(variant="alpha_sweep", variant_params={"alpha": 2.0})
    vid = led.record_completion(variant, _result("variant"))
    assert vid != trial_id(base)
    assert led.n_effective_trials() == 2
    # The base result is untouched (no replacement of prior history).
    assert led.result_for(trial_id(base)) == _result()


def test_off_ladder_horizon_is_recordable_and_counted():
    led = G0BNLedger()
    off = _identity(horizon="5s", horizon_role="exploratory")
    led.record_abort(off, error="not run")
    assert led.n_effective_trials() == 1


def test_distinct_identities_across_horizons_count_separately():
    led = G0BNLedger()
    for horizon, role in (("2s", "primary"), ("10s", "primary"), ("60s", "control-only")):
        led.record_completion(_identity(horizon=horizon, horizon_role=role),
                              _result(horizon))
    assert led.n_effective_trials() == 3


# --------------------------------------------------------------------------- validation

def test_invalid_identity_is_rejected():
    led = G0BNLedger()
    bad = _identity()
    bad.pop("cv_sha256")
    with pytest.raises(ValueError):
        led.record_start(bad)
    assert led.n_effective_trials() == 0


def test_non_finite_result_values_are_rejected():
    led = G0BNLedger()
    with pytest.raises(ValueError, match="finite"):
        led.record_completion(_identity(), _result(extra=float("nan")))
    assert led.n_effective_trials() == 0


def test_abort_requires_a_non_empty_error():
    led = G0BNLedger()
    with pytest.raises(ValueError, match="error"):
        led.record_abort(_identity(), error="   ")


def test_result_must_be_a_dict():
    led = G0BNLedger()
    with pytest.raises(ValueError, match="dict"):
        led.record_completion(_identity(), [1, 2, 3])


# ------------------------------------------------------------------------- event chain

def test_events_are_hash_chained_in_order():
    led = G0BNLedger()
    ident = _identity()
    led.record_start(ident)
    led.record_abort(ident, error="transient")
    led.record_completion(ident, _result())
    events = led.events()
    assert [e["event"] for e in events] == ["started", "aborted", "completed"]
    assert [e["ordinal"] for e in events] == [0, 1, 2]
    prev = led.genesis_sha256()
    for e in events:
        assert e["prev_sha256"] == prev
        assert e["sha256"] == hash_obj({k: v for k, v in e.items() if k != "sha256"})
        prev = e["sha256"]
    assert led.history_sha256() == events[-1]["sha256"]


def test_ledger_hash_covers_history_and_identity_result_set():
    led = G0BNLedger()
    ident = _identity()
    led.record_completion(ident, _result())
    h1 = led.ledger_sha256()
    ids1 = led.identity_set_sha256()
    # An idempotent rerun changes the ordered event history but not the identity set.
    led.record_completion(ident, _result())
    assert led.identity_set_sha256() == ids1
    assert led.ledger_sha256() != h1


# -------------------------------------------------------------------------- persistence

def test_save_load_round_trip(tmp_path):
    led = G0BNLedger()
    a = _identity()
    b = _identity(horizon="10s")
    led.record_start(a)
    led.record_completion(a, _result("a"))
    led.record_start(b)
    led.record_abort(b, error="died")
    path = tmp_path / "g0bn_ledger.json"
    led.save(path)
    loaded = G0BNLedger.load(path)
    assert loaded.ledger_sha256() == led.ledger_sha256()
    assert loaded.history_sha256() == led.history_sha256()
    assert loaded.n_effective_trials() == 2
    assert loaded.events() == led.events()
    assert loaded.result_for(trial_id(a)) == _result("a")
    assert loaded.result_for(trial_id(b)) is None


def test_load_rejects_tampered_result(tmp_path):
    led = G0BNLedger()
    led.record_completion(_identity(), _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    payload["identities"][0]["result"]["n_rows"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        G0BNLedger.load(path)


def test_load_rejects_tampered_event_chain(tmp_path):
    led = G0BNLedger()
    ident = _identity()
    led.record_start(ident)
    led.record_completion(ident, _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    payload["events"][0]["event"] = "completed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        G0BNLedger.load(path)


def test_load_rejects_tampered_identity(tmp_path):
    led = G0BNLedger()
    led.record_completion(_identity(), _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    payload["identities"][0]["identity"]["horizon"] = "10s"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        G0BNLedger.load(path)


def test_load_rejects_result_without_completed_event(tmp_path):
    # A crafted identity record carrying a result no execution event ever produced
    # is a fabricated outcome; the load-path invariant must catch it even before
    # the summary hashes are compared.
    led = G0BNLedger()
    ident = _identity()
    led.record_start(ident)                      # started, never completed
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    fabricated = _result("fabricated")
    payload["identities"][0]["result"] = fabricated
    payload["identities"][0]["result_sha256"] = hash_obj(fabricated)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no completed event"):
        G0BNLedger.load(path)


def test_load_rejects_duplicate_identity_records(tmp_path):
    led = G0BNLedger()
    led.record_completion(_identity(), _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    payload["identities"].append(json.loads(json.dumps(payload["identities"][0])))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate identity"):
        G0BNLedger.load(path)


def test_load_rejects_completed_event_with_stripped_result(tmp_path):
    led = G0BNLedger()
    ident = _identity()
    led.record_start(ident)
    led.record_completion(ident, _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    payload["identities"][0]["result"] = None
    payload["identities"][0]["result_sha256"] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="immutable non-null result"):
        G0BNLedger.load(path)


def test_load_rejects_identity_with_no_execution_event(tmp_path):
    # Every real registration appends an event, so an identity record with zero
    # started/aborted/completed events is fabricated — even when the crafted file
    # recomputes every summary hash, it must not inflate effective N.
    from eval.g0bn_ledger import LEDGER_SCHEMA
    led = G0BNLedger()
    led.record_completion(_identity(), _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    extra = _identity(horizon="60s", horizon_role="control-only")
    payload["identities"].append({"trial_id": trial_id(extra), "identity": extra,
                                  "result": None, "result_sha256": None})
    pairs = sorted((r["trial_id"], r["result_sha256"])
                   for r in payload["identities"])
    payload["identity_set_sha256"] = hash_obj(
        {"schema": LEDGER_SCHEMA, "trials": [list(p) for p in pairs]})
    payload["n_effective_trials"] += 1
    payload["ledger_sha256"] = hash_obj({
        "schema": LEDGER_SCHEMA,
        "n_effective_trials": payload["n_effective_trials"],
        "identity_set_sha256": payload["identity_set_sha256"],
        "history_sha256": payload["history_sha256"]})
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no execution event"):
        G0BNLedger.load(path)


def test_load_rejects_wrong_schema(tmp_path):
    led = G0BNLedger()
    led.record_completion(_identity(), _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    payload["schema"] = "g0xv-ledger-v1"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        G0BNLedger.load(path)


# -------------------------------------------------------- config immutability (3.1)

def test_register_rejects_protocol_config_drift():
    # Once any g0bn-trial-v1 identity is registered, the v1 protocol config is
    # immutable: a later attempt under an edited config fails closed instead of
    # becoming an ordinary new trial (spec section 3.1).
    led = G0BNLedger()
    led.record_start(_identity())
    drifted = _identity(horizon="10s", protocol_config_sha256="a" * 64)
    with pytest.raises(ValueError, match="config drift"):
        led.record_start(drifted)
    with pytest.raises(ValueError, match="config drift"):
        led.record_abort(drifted, error="x")
    with pytest.raises(ValueError, match="config drift"):
        led.record_completion(drifted, _result())
    assert led.n_effective_trials() == 1
    assert led.protocol_config_sha256() == _identity()["protocol_config_sha256"]


def test_load_rejects_mixed_config_ledger_file(tmp_path):
    led = G0BNLedger()
    led.record_completion(_identity(), _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    foreign = _identity(horizon="10s", protocol_config_sha256="b" * 64)
    payload["identities"].append({"trial_id": trial_id(foreign), "identity": foreign,
                                  "result": None, "result_sha256": None})
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="config drift"):
        G0BNLedger.load(path)


def test_register_rejects_development_data_identity_drift():
    # One protocol instance has ONE canonical development build (immutable config
    # + deterministic producer): an identity binding a different build must fail
    # closed at every registration path — an aborted-only foreign attempt would
    # otherwise silently inflate effective N and every DSR benchmark.
    led = G0BNLedger()
    led.record_start(_identity())
    for field in ("development_build_id", "development_manifest_sha256",
                  "development_logical_row_sha256", "partition_plan_sha256"):
        foreign = _identity(horizon="10s", **{field: "c" * 64})
        with pytest.raises(ValueError, match="data-identity drift"):
            led.record_start(foreign)
        with pytest.raises(ValueError, match="data-identity drift"):
            led.record_abort(foreign, error="x")
        with pytest.raises(ValueError, match="data-identity drift"):
            led.record_completion(foreign, _result())
    assert led.n_effective_trials() == 1


def test_load_rejects_mixed_data_identity_ledger_file(tmp_path):
    led = G0BNLedger()
    led.record_completion(_identity(), _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    foreign = _identity(horizon="10s", development_build_id="d" * 64)
    payload["identities"].append({"trial_id": trial_id(foreign), "identity": foreign,
                                  "result": None, "result_sha256": None})
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="data-identity drift"):
        G0BNLedger.load(path)


def test_symlinked_paths_share_one_writer_lock(tmp_path):
    # abspath alone would leave symlink aliases distinct, deriving different
    # .lock files for the same underlying ledger; canonicalization through
    # symlinks must make every alias contend on ONE lock.
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    alias = tmp_path / "alias"
    os.symlink(real_dir, alias)
    led = G0BNLedger(path=real_dir / "ledger.json")
    led.record_start(_identity())
    with pytest.raises(ValueError, match="lock"):
        G0BNLedger.load(alias / "ledger.json")       # alias resolves to same lock
    snapshot = G0BNLedger.load(alias / "ledger.json", bind=False)
    assert snapshot.n_effective_trials() == 1
    led.close()
    resumed = G0BNLedger.load(alias / "ledger.json")  # released -> alias may bind
    resumed.record_abort(_identity(), error="via alias")
    resumed.close()
    assert len(G0BNLedger.load(real_dir / "ledger.json", bind=False).events()) == 2


# ------------------------------------------------------------ durable persistence

def test_path_bound_ledger_persists_every_event_immediately(tmp_path):
    path = tmp_path / "durable.json"
    led = G0BNLedger(path=path)
    assert led.is_durable()
    ident = _identity()
    led.record_start(ident)
    # No explicit save(): a crash after this point must not lose the start.
    # (bind=False: read-only inspection that neither locks nor resumes.)
    recovered = G0BNLedger.load(path, bind=False)
    assert not recovered.is_durable()
    assert recovered.n_effective_trials() == 1
    assert [e["event"] for e in recovered.events()] == ["started"]
    led.record_abort(ident, error="synthetic crash evidence")
    recovered = G0BNLedger.load(path, bind=False)
    assert [e["event"] for e in recovered.events()] == ["started", "aborted"]


def test_rejected_mutation_leaves_the_durable_file_unchanged(tmp_path):
    path = tmp_path / "durable.json"
    led = G0BNLedger(path=path)
    ident = _identity()
    led.record_completion(ident, _result())
    before = path.read_text()
    with pytest.raises(ValueError, match="DIFFERENT result"):
        led.record_completion(ident, _result(n_rows=999))
    assert path.read_text() == before


def test_bound_constructor_refuses_an_existing_file(tmp_path):
    path = tmp_path / "durable.json"
    led = G0BNLedger(path=path)
    led.record_start(_identity())
    # While the writer is alive the lock (acquired BEFORE the existence check)
    # fires first; after close, the under-lock existence check refuses the fork.
    with pytest.raises(ValueError, match="lock"):
        G0BNLedger(path=path)
    led.close()
    with pytest.raises(ValueError, match="exists"):
        G0BNLedger(path=path)
    resumed = G0BNLedger.load(path)          # load is the resume path and stays bound
    resumed.record_abort(_identity(), error="resumed abort")
    resumed.close()
    assert len(G0BNLedger.load(path, bind=False).events()) == 2


def test_concurrent_writers_are_locked_out(tmp_path):
    # Two live writers on one path would let the last os.replace() silently
    # discard the other's append-only events; the exclusive per-path lock makes
    # the second binder fail closed instead.
    path = tmp_path / "durable.json"
    led = G0BNLedger(path=path)
    led.record_start(_identity())
    with pytest.raises(ValueError, match="lock"):
        G0BNLedger.load(path)                    # second WRITER refused
    inspect = G0BNLedger.load(path, bind=False)  # read-only inspection still fine
    assert inspect.n_effective_trials() == 1
    led.close()
    resumed = G0BNLedger.load(path)              # lock released -> resume works
    assert resumed.is_durable()
    resumed.close()


def test_inspection_snapshot_cannot_overwrite_a_live_ledger(tmp_path):
    # save() must hold the target path's writer lock: a stale bind=False snapshot
    # replacing the live file would silently delete append-only events.
    path = tmp_path / "durable.json"
    led = G0BNLedger(path=path)
    ident = _identity()
    led.record_start(ident)
    snapshot = G0BNLedger.load(path, bind=False)
    led.record_abort(ident, error="event the snapshot does not know about")
    with pytest.raises(ValueError, match="lock"):
        snapshot.save(path)
    # the live history is intact and exporting to a FRESH path still works
    assert len(G0BNLedger.load(path, bind=False).events()) == 2
    snapshot.save(tmp_path / "export.json")
    assert len(G0BNLedger.load(tmp_path / "export.json", bind=False).events()) == 1
    led.close()


def test_stale_snapshot_cannot_replace_a_closed_ledger(tmp_path):
    # The transient lock alone only stops SIMULTANEOUS clobbering: once the live
    # writer closes, a stale snapshot's save() must still be refused unless the
    # on-disk history is a hash-chain prefix of the snapshot's own history.
    path = tmp_path / "durable.json"
    led = G0BNLedger(path=path)
    ident = _identity()
    led.record_start(ident)
    snapshot = G0BNLedger.load(path, bind=False)
    led.record_abort(ident, error="event recorded after the snapshot")
    led.close()                                  # lock released
    with pytest.raises(ValueError, match="append-only"):
        snapshot.save(path)
    assert len(G0BNLedger.load(path, bind=False).events()) == 2
    # an up-to-date snapshot may rewrite idempotently (equal history = prefix)
    current = G0BNLedger.load(path, bind=False)
    current.save(path)
    assert len(G0BNLedger.load(path, bind=False).events()) == 2


def test_fresh_bind_acquires_the_lock_before_the_existence_check(tmp_path):
    # Closes the create/create race: with the path lock held by someone else, a
    # fresh bind must fail on CONTENTION even though no ledger file exists yet —
    # proving the existence check happens under the lock, not before it.
    import fcntl as _fcntl
    path = tmp_path / "durable.json"
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    _fcntl.flock(fd, _fcntl.LOCK_EX)
    try:
        with pytest.raises(ValueError, match="lock"):
            G0BNLedger(path=path)
    finally:
        _fcntl.flock(fd, _fcntl.LOCK_UN)
        os.close(fd)
    led = G0BNLedger(path=path)                  # free again -> binds normally
    led.record_start(_identity())
    led.close()


def test_bound_path_is_normalized_against_chdir(tmp_path, monkeypatch):
    # A relative bound path would re-resolve after chdir: persistence would write
    # a DIFFERENT unlocked file while the locked original went stale.
    monkeypatch.chdir(tmp_path)
    led = G0BNLedger(path="relative.json")
    ident = _identity()
    led.record_start(ident)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    led.record_abort(ident, error="recorded after chdir")
    led.close()
    recovered = G0BNLedger.load(tmp_path / "relative.json", bind=False)
    assert [e["event"] for e in recovered.events()] == ["started", "aborted"]
    assert not (elsewhere / "relative.json").exists()


def test_load_rejects_completed_event_with_null_result_hash(tmp_path):
    # A crafted 'completed' event with result_sha256: null against a nulled
    # identity record must not pass as None == None: a completion without a
    # pinned result is impossible evidence that a later real completion could
    # silently supersede.
    from eval.g0bn_ledger import LEDGER_SCHEMA
    led = G0BNLedger()
    ident = _identity()
    led.record_start(ident)
    led.record_completion(ident, _result())
    path = tmp_path / "ledger.json"
    led.save(path)
    payload = json.loads(path.read_text())
    payload["identities"][0]["result"] = None
    payload["identities"][0]["result_sha256"] = None
    prev = None
    for event in payload["events"]:              # rebuild a VALID chain
        if event["event"] == "completed":
            event["payload"]["result_sha256"] = None
        if prev is not None:
            event["prev_sha256"] = prev
        event["sha256"] = hash_obj({k: v for k, v in event.items()
                                    if k != "sha256"})
        prev = event["sha256"]
    payload["history_sha256"] = prev
    pairs = sorted((r["trial_id"], r["result_sha256"])
                   for r in payload["identities"])
    payload["identity_set_sha256"] = hash_obj(
        {"schema": LEDGER_SCHEMA, "trials": [list(p) for p in pairs]})
    payload["ledger_sha256"] = hash_obj({
        "schema": LEDGER_SCHEMA,
        "n_effective_trials": payload["n_effective_trials"],
        "identity_set_sha256": payload["identity_set_sha256"],
        "history_sha256": payload["history_sha256"]})
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="completed event"):
        G0BNLedger.load(path)


# ----------------------------------------------------------------- legacy isolation

def test_never_imports_legacy_ledgers(tmp_path):
    # No import path exists at all (spec section 4.2: never imports G0-CB/G0-XV entries).
    assert not hasattr(G0BNLedger, "import_history")
    legacy = TrialLedger()
    legacy.register(
        legacy_trial_identity(protocol="g0cb", arm="baseline", dataset_id="ds",
                              build_id="b1", feature_cols=["f"], config="ridge",
                              horizon="2s"),
        {"net_pnl": 1.0},
    )
    path = tmp_path / "legacy.json"
    legacy.save(path)
    with pytest.raises(ValueError):
        G0BNLedger.load(path)


def test_legacy_ledger_identity_shape_is_rejected():
    led = G0BNLedger()
    legacy_shape = legacy_trial_identity(
        protocol="g0xv", arm="cross", dataset_id="ds", build_id="b1",
        feature_cols=["f"], config="lgbm_reg", horizon="2s")
    with pytest.raises(ValueError):
        led.record_start(legacy_shape)


def test_entries_are_deep_copies_not_aliases():
    led = G0BNLedger()
    ident = _identity()
    result = _result()
    led.record_completion(ident, result)
    result["n_rows"] = 999          # caller-side mutation must not reach the ledger
    stored = led.result_for(trial_id(ident))
    assert stored["n_rows"] == 240
    stored["n_rows"] = 1            # reader-side mutation must not reach the ledger
    assert led.result_for(trial_id(ident))["n_rows"] == 240
    got = led.identity_for(trial_id(ident))
    got["horizon"] = "60s"
    assert led.identity_for(trial_id(ident))["horizon"] == "2s"


def test_registration_order_is_preserved_and_hash_is_order_sensitive():
    a = _identity()
    b = _identity(horizon="10s")
    led_ab = G0BNLedger()
    led_ab.record_completion(a, _result("a"))
    led_ab.record_completion(b, _result("b"))
    led_ba = G0BNLedger()
    led_ba.record_completion(b, _result("b"))
    led_ba.record_completion(a, _result("a"))
    assert led_ab.trial_ids() == list(reversed(led_ba.trial_ids()))
    # The canonical identity/result SET hash is order-independent...
    assert led_ab.identity_set_sha256() == led_ba.identity_set_sha256()
    # ...while the ordered event history hash is order-sensitive (spec section 4.2
    # hashes both).
    assert led_ab.history_sha256() != led_ba.history_sha256()
