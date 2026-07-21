"""T9 blind holdout materializer tests (issue #94; spec §6.3 steps 4-5, §11.9).

Everything is synthetic and tiny: the sealed January "objects" are the
tests/produce_fixtures.py world written to tmp files, the custody inventory pins
their REAL content hashes, and the plan/freeze chain is built through the merged
67-B/67-C machinery from a synthetic development run. No vendor I/O and no real
January access occur anywhere here.
"""
from __future__ import annotations

import builtins
import copy
import json

import pytest
import pyarrow.parquet as pq

from bars.clock import ThresholdConfig
from bars.produce import (
    CLOCK_STATE_SCHEMA,
    DROP_COUNT_CATEGORIES,
    RuntimeParams,
    clock_state_sha256,
    materialize_holdout,
)
from eval.g0bn_config import RAW_ACCESS_CLAIM_SCHEMA, g0bn_artifact_sha256
from eval.g0bn_engine import run_g0bn_development
from eval.g0bn_freeze import (
    build_freeze,
    build_holdout_plan,
    verify_holdout_manifest_binding,
)
from eval.guard import preflight_generic_manifest
from eval.hashing import hash_obj
from eval.writer import classify_manifest
from tests.g0bn_dev_fixtures import dev_bundle, dev_config, durable_ledger
from tests.g0bn_holdout_fixtures import make_inventory, sealed_config_kwargs
from tests.g0bn_protocol_fixtures import (
    dev_days,
    make_clock,
    make_exclusions,
    make_partition,
    sha_hex,
)
from tests.produce_fixtures import (
    SEED_THRESHOLD,
    TARGET_BARS_PER_DAY,
    TIME_CAP_NS,
    SyntheticWorld,
    write_day_objects,
)

GEN_AT = "2026-07-19T00:00:00Z"

RAW_PRODUCTS = ("book", "book_delta_v2", "trades")
NORMALIZED_PRODUCTS = ("binance_futures_l2_snapshot", "binance_futures_l2_delta",
                       "binance_futures_trades")

RUNTIME = RuntimeParams(
    threshold=ThresholdConfig(
        target_bars_per_day=TARGET_BARS_PER_DAY, window_days=3, warmup_days=1,
        seed_threshold=SEED_THRESHOLD, min_covered_fraction=0.0),
    top_k=3, tick_size=0.01, min_returns=2, vol_floor_bps=0.25)

# frozen development-end clock state: three recorded December days at the synthetic
# world's daily notional, giving January 1 a live (non-warm-up) trailing threshold.
# They must be INCLUDED development days (the validator rejects out-of-scope
# history), so the custody config below includes them explicitly.
CLOCK_STATE = {
    "schema": CLOCK_STATE_SCHEMA,
    "threshold_config": {
        "target_bars_per_day": TARGET_BARS_PER_DAY, "window_days": 3,
        "warmup_days": 1, "seed_threshold": float(SEED_THRESHOLD),
        "min_covered_fraction": 0.0,
    },
    "history": [
        {"day": d, "completed_notional": 36_000.0, "covered_fraction": 1.0}
        for d in ("2025-12-29", "2025-12-30", "2025-12-31")
    ],
}


def _custody_exclusions() -> dict:
    """Outcome-blind day accounting whose included set ends at 2025-12-31: the
    frozen clock state must embed included days inside the trailing window
    before January 1 (the plain dev fixture includes the first 24 November
    days only, which would make every January day warm-up)."""
    days = dev_days()
    included = days[:21] + days[-3:]
    excluded = {d: {"reason": "synthetic_out_of_scope",
                    "evidence_sha256": sha_hex(f"excl-{d}")}
                for d in days if d not in included}
    return make_exclusions(included_days=included, excluded_days=excluded)

# January day quirks exercising the drop taxonomy end to end:
# - Jan 1 delays its first delta 5s: bar 1 is no_prior_read (feature_rejection),
#   the next bars read the prior-day snapshot only (before_start), and the last
#   snapshot-only bar ages past the 5s staleness cap;
# - Jan 13 (before the excluded Jan 14) is late-active: coverage_gap drops;
# - Jan 31 (partition end) is late-active: prefilter drops.
DAY_KWARGS = {
    "2026-01-01": {"first_delta_offset_ns": 5_000_000_000},
    "2026-01-13": {"late_active": True},
    "2026-01-31": {"late_active": True},
}


def _write_claim(path, *, config, plan, freeze, **over) -> dict:
    claim = {
        "schema": RAW_ACCESS_CLAIM_SCHEMA,
        "holdout_universe_id": plan["holdout_universe_id"],
        "transaction_id": plan["transaction_id"],
        "protocol_config_sha256": config["sha256"],
        "holdout_plan_sha256": plan["sha256"],
        "freeze_sha256": freeze["sha256"],
        "generated_at": GEN_AT,
    }
    claim.update(over)
    claim["sha256"] = over.get("sha256", g0bn_artifact_sha256(claim))
    path.write_text(json.dumps(claim, sort_keys=True) + "\n", encoding="utf-8")
    return claim


@pytest.fixture(scope="module")
def custody(tmp_path_factory):
    """The full synthetic custody chain: sealed object files with REAL content
    hashes, the paired inventory/config/plan/freeze, the frozen clock state, and
    a durable raw-access claim."""
    root = tmp_path_factory.mktemp("jan-objects")
    world = SyntheticWorld()
    inventory_probe = make_inventory()  # for the included-day list only
    included = list(inventory_probe["included_days"])

    object_paths, objects = {}, []
    for day in included:
        paths, shas = write_day_objects(world, day, root,
                                        **DAY_KWARGS.get(day, {}))
        for product in RAW_PRODUCTS:
            objects.append({"object_id": f"custody/raw/{product}/{day}",
                            "layer": "raw", "product": product, "day": day,
                            "sha256": sha_hex(f"jan-raw-{product}-{day}")})
        for product in NORMALIZED_PRODUCTS:
            oid = f"custody/normalized/{product}/{day}"
            objects.append({"object_id": oid, "layer": "normalized",
                            "product": product, "day": day,
                            "sha256": shas[product]})
            object_paths[oid] = object_paths.get(oid, paths[product])
    inventory = make_inventory(objects=objects)

    state_sha = clock_state_sha256(CLOCK_STATE)
    config = dev_config(
        clock=make_clock(target_bars_per_day=TARGET_BARS_PER_DAY,
                         time_cap_ns=TIME_CAP_NS,
                         development_end_state_sha256=state_sha),
        exclusions=_custody_exclusions(),
        **sealed_config_kwargs(inventory))
    frame, manifest, config, identity = dev_bundle(config=config)
    run = run_g0bn_development(frame, manifest, config, identity, durable_ledger())
    plan = build_holdout_plan(config, inventory, generated_at=GEN_AT)
    freeze = build_freeze(run, plan, inventory=inventory, generated_at=GEN_AT)

    claim_path = root / "g0bn-raw-access-claim-v1.json"
    claim = _write_claim(claim_path, config=config, plan=plan, freeze=freeze)
    return {"root": root, "config": config, "inventory": inventory,
            "plan": plan, "freeze": freeze, "run": run, "claim": claim,
            "claim_path": claim_path, "object_paths": object_paths}


def _materialize(custody_bundle, out_dir, *, generated_at=GEN_AT, **over):
    kwargs = dict(
        config=custody_bundle["config"], plan=custody_bundle["plan"],
        freeze=custody_bundle["freeze"], inventory=custody_bundle["inventory"],
        runtime=RUNTIME, clock_state=CLOCK_STATE,
        raw_access_claim_path=custody_bundle["claim_path"],
        object_paths=custody_bundle["object_paths"],
        matrix_path=out_dir / "oos_matrix.parquet",
        manifest_path=out_dir / "oos_manifest.json",
        attestation_path=out_dir / "g0bn-materialization-attestation-v1.json",
        generated_at=generated_at)
    kwargs.update(over)
    return materialize_holdout(**kwargs)


@pytest.fixture(scope="module")
def materialized(custody, tmp_path_factory):
    out = tmp_path_factory.mktemp("oos-out")
    return {"out": out, "result": _materialize(custody, out)}


# ------------------------------------------------------------------ happy path


def test_materialization_publishes_attested_blind_artifacts(custody, materialized):
    result = materialized["result"]
    out = materialized["out"]
    for name in ("oos_matrix.parquet", "oos_manifest.json",
                 "g0bn-materialization-attestation-v1.json"):
        assert (out / name).exists()
    assert not (out / "g0bn-materialization-attestation-v1.json.tmp").exists()
    manifest = json.loads((out / "oos_manifest.json").read_text())
    cls = classify_manifest(manifest)
    assert cls.holdout_bound and cls.partition == "holdout"
    # the published manifest reconciles with the frozen plan/freeze and accounts
    # for the COMPLETE sealed normalized scope
    verify_holdout_manifest_binding(manifest, custody["plan"], custody["freeze"],
                                    config=custody["config"],
                                    inventory=custody["inventory"])
    # the generic-runner guard refuses it before any loader
    with pytest.raises(ValueError, match="holdout guard"):
        preflight_generic_manifest(manifest)
    # all three horizons survive with real support
    assert all(n > 0 for n in result.row_counts.values())
    assert set(result.row_counts) == {"2s", "10s", "60s"}
    # the written physical schema matches the plan's frozen Arrow pin
    oc = custody["plan"]["output_contract"]
    assert result.write.physical_schema_sha256 == \
        oc["expected_physical_schema_sha256"]


def test_attestation_binds_claim_plan_freeze_and_write(custody, materialized):
    att = materialized["result"].attestation
    on_disk = json.loads(
        (materialized["out"] / "g0bn-materialization-attestation-v1.json")
        .read_text())
    assert on_disk == att
    assert att["schema"] == "g0bn-materialization-attestation-v1"
    assert g0bn_artifact_sha256(att) == att["sha256"]
    assert att["holdout_plan_sha256"] == custody["plan"]["sha256"]
    assert att["freeze_sha256"] == custody["freeze"]["sha256"]
    assert att["raw_access_claim_sha256"] == custody["claim"]["sha256"]
    write = materialized["result"].write
    assert att["build_id"] == write.build_id
    assert att["logical_row_sha256"] == write.logical_row_sha256
    assert att["manifest_sha256"] == write.manifest_sha256
    assert att["matrix_file_sha256"] == write.matrix_file_sha256
    assert att["physical_schema_sha256"] == write.physical_schema_sha256
    assert att["row_count"] == write.row_count == sum(att["row_counts"].values())
    assert att["drop_count_categories"] == list(DROP_COUNT_CATEGORIES)
    assert att["counts_sha256"] == hash_obj({
        "schema": "g0bn-materialization-counts-v1",
        "row_count": att["row_count"], "row_counts": att["row_counts"],
        "drop_counts": att["drop_counts"]})
    assert att["days_built"] == list(custody["plan"]["included_days"])
    assert [d["day"] for d in att["realized_threshold_schedule"]] == \
        att["days_built"]


def test_january_exercises_the_full_drop_taxonomy(materialized):
    drops = materialized["result"].drop_counts
    assert tuple(drops) == DROP_COUNT_CATEGORIES
    # the frozen development-end state makes January 1 a live threshold day
    assert all(n == 0 for n in drops["warmup"].values())
    # Jan 1 bar 1: no prior observable read
    assert all(n >= 1 for n in drops["feature_rejection"].values())
    # Jan 1 early bars read only the prior-day snapshot state
    assert all(n >= 1 for n in drops["before_start"].values())
    # quiet-tail cap bars age past the staleness cap
    assert all(n > 0 for n in drops["staleness"].values())
    # late-active Jan 13 runs into the excluded Jan 14 coverage gap
    assert drops["coverage_gap"]["60s"] > 0
    # late-active Jan 31 runs into the February partition boundary
    assert drops["prefilter"]["60s"] > 0
    # the schedule is live all month: no warm-up entries in the realized schedule
    sched = materialized["result"].realized_threshold_schedule
    assert not any(s["is_warmup"] for s in sched)


def test_opt_in_time_cap_diagnostic_materializes_cleanly(custody, tmp_path):
    # Codex round 4: a plan that opts into the emitted_by_time_cap diagnostic is
    # a SUPPORTED frozen contract — the manifest must reproduce the plan's
    # output contract verbatim (including its dtypes map), never extend it into
    # a post-burn binding rejection
    plan2 = build_holdout_plan(
        custody["config"], custody["inventory"],
        extra_cols=["latency_drift_bps", "emitted_by_time_cap"],
        generated_at=GEN_AT)
    freeze2 = build_freeze(custody["run"], plan2,
                           inventory=custody["inventory"], generated_at=GEN_AT)
    claim2 = tmp_path / "claim2.json"
    _write_claim(claim2, config=custody["config"], plan=plan2, freeze=freeze2)
    result = _materialize(custody, tmp_path, plan=plan2, freeze=freeze2,
                          raw_access_claim_path=claim2)
    assert result.write.row_count > 0
    schema = pq.read_schema(tmp_path / "oos_matrix.parquet")
    assert schema.field("emitted_by_time_cap").type == "bool"
    assert result.write.physical_schema_sha256 == \
        plan2["output_contract"]["expected_physical_schema_sha256"]


def test_rematerialization_is_logically_identical(custody, materialized, tmp_path):
    again = _materialize(custody, tmp_path, generated_at="2026-07-20T00:00:00Z")
    result = materialized["result"]
    assert again.write.build_id == result.write.build_id
    assert again.write.logical_row_sha256 == result.write.logical_row_sha256
    assert again.write.manifest_sha256 == result.write.manifest_sha256
    assert again.attestation["counts_sha256"] == \
        result.attestation["counts_sha256"]
    assert again.realized_threshold_schedule_sha256 == \
        result.realized_threshold_schedule_sha256
    assert again.clock_state_sha256 == result.clock_state_sha256


def test_existing_outputs_refuse_before_any_source_open(custody, materialized,
                                                        monkeypatch):
    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    with pytest.raises(FileExistsError, match="fresh"):
        _materialize(custody, materialized["out"])
    object_files = {str(p) for p in custody["object_paths"].values()}
    assert not (set(opened) & object_files)


def test_aliasing_output_paths_reject_before_any_source_open(custody, tmp_path,
                                                             monkeypatch):
    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path)
    cases = [
        {"manifest_path": tmp_path / "oos_matrix.parquet"},
        {"attestation_path": tmp_path / "oos_matrix.parquet"},
        {"matrix_path":
         tmp_path / "g0bn-materialization-attestation-v1.json.tmp"},
        # different literal strings resolving to the same file via a symlinked
        # parent must also reject (resolved aliasing, not string equality)
        {"manifest_path": alias / "oos_matrix.parquet"},
    ]
    for over in cases:
        with pytest.raises(ValueError, match="pairwise distinct"):
            _materialize(custody, tmp_path, **over)
    object_files = {str(p) for p in custody["object_paths"].values()}
    assert not (set(opened) & object_files)
    # the output preflight is pure path metadata and precedes the claim read
    assert str(custody["claim_path"]) not in opened
    for name in ("oos_matrix.parquet", "oos_manifest.json",
                 "g0bn-materialization-attestation-v1.json",
                 "g0bn-materialization-attestation-v1.json.tmp"):
        assert not (tmp_path / name).exists()


def test_missing_output_parent_rejects_before_any_source_open(custody, tmp_path,
                                                              monkeypatch):
    """A mistyped output location (missing parent directory, or a parent that is
    a file) must fail before the raw claim/source reads — not mid-write after
    the burn, which would strand a matrix without a manifest/attestation."""
    parent_file = tmp_path / "not_a_dir"
    parent_file.write_text("occupied", encoding="utf-8")

    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    cases = [
        {"manifest_path": tmp_path / "missing" / "oos_manifest.json"},
        {"attestation_path": tmp_path / "absent" / "attestation.json"},
        {"matrix_path": parent_file / "oos_matrix.parquet"},
    ]
    for over in cases:
        with pytest.raises(ValueError, match="parent directory"):
            _materialize(custody, tmp_path, **over)
    object_files = {str(p) for p in custody["object_paths"].values()}
    assert not (set(opened) & object_files)
    # the output preflight is pure path metadata and precedes the claim read
    assert str(custody["claim_path"]) not in opened
    for name in ("oos_matrix.parquet", "oos_manifest.json",
                 "g0bn-materialization-attestation-v1.json",
                 "g0bn-materialization-attestation-v1.json.tmp"):
        assert not (tmp_path / name).exists()


def test_dangling_output_symlink_rejects_before_any_source_open(custody, tmp_path,
                                                                monkeypatch):
    """A dangling symlink at an output path passes an existence check (exists()
    follows the link) but makes the later O_EXCL create fail with FileExistsError
    — after the burn. Output paths must not be symlinks at all."""
    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    for name in ("dangling_manifest.json", "dangling_attestation.json"):
        link = tmp_path / name
        link.symlink_to(tmp_path / f"missing-target-{name}")
    cases = [
        {"manifest_path": tmp_path / "dangling_manifest.json"},
        {"attestation_path": tmp_path / "dangling_attestation.json"},
    ]
    for over in cases:
        with pytest.raises(ValueError, match="symlink"):
            _materialize(custody, tmp_path, **over)
    object_files = {str(p) for p in custody["object_paths"].values()}
    assert not (set(opened) & object_files)
    assert str(custody["claim_path"]) not in opened
    for name in ("oos_matrix.parquet", "oos_manifest.json",
                 "g0bn-materialization-attestation-v1.json",
                 "g0bn-materialization-attestation-v1.json.tmp"):
        assert not (tmp_path / name).exists()


# ------------------------------------------------------------------- read spies


def test_read_spy_claim_precedes_every_source_open(custody, tmp_path, monkeypatch):
    import bars.produce as produce_mod

    order = []
    real_open = builtins.open
    real_pin = produce_mod._open_pinned_fd
    real_pf = pq.ParquetFile
    real_meta = pq.read_metadata
    real_schema = pq.read_schema

    def spy_open(file, mode="r", *args, **kwargs):
        order.append((str(file), mode))
        return real_open(file, mode, *args, **kwargs)

    def spy_pin(path):
        order.append((str(path), "pinned-open"))
        return real_pin(path)

    derived = {str(tmp_path / "oos_matrix.parquet"),
               str(tmp_path / "oos_manifest.json"),
               str(tmp_path / "g0bn-materialization-attestation-v1.json")}

    def _reject_derived(path):
        if str(path) in derived:
            raise AssertionError(f"derived artifact reopened via parquet: {path}")

    # every pyarrow entry that can touch a parquet payload OR footer records into
    # the same order log, so the claim-precedes-source assertion is closed over
    # footer/metadata reads too, not only builtins.open events
    def spy_parquet(path, *args, **kwargs):
        _reject_derived(path)
        order.append((str(path), "parquet"))
        return real_pf(path, *args, **kwargs)

    def spy_metadata(path, *args, **kwargs):
        _reject_derived(path)
        order.append((str(path), "parquet-metadata"))
        return real_meta(path, *args, **kwargs)

    def spy_schema(path, *args, **kwargs):
        _reject_derived(path)
        order.append((str(path), "parquet-schema"))
        return real_schema(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("generic pandas/pyarrow table loaders must not run "
                             "inside the blind materializer")

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(produce_mod, "_open_pinned_fd", spy_pin)
    monkeypatch.setattr(pq, "ParquetFile", spy_parquet)
    monkeypatch.setattr(pq, "read_metadata", spy_metadata)
    monkeypatch.setattr(pq, "read_schema", spy_schema)
    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(pq, "read_pandas", forbidden)
    import pandas as pd
    monkeypatch.setattr(pd, "read_parquet", forbidden)

    result = _materialize(custody, tmp_path)
    assert result.write.row_count > 0

    object_files = {str(p) for p in custody["object_paths"].values()}
    claim_file = str(custody["claim_path"])
    first_source = next(i for i, (f, _) in enumerate(order)
                        if f in object_files)
    claim_read = next(i for i, (f, _) in enumerate(order) if f == claim_file)
    assert claim_read < first_source, \
        "a January source was opened before the raw-access claim was read"
    # descriptor binding (Codex P2): each sealed object is opened exactly once,
    # via the pinned-fd primitive; no by-name reopen path exists for sources
    for path in object_files:
        assert [m for f, m in order if f == path] == ["pinned-open"], path
    # derived artifacts are opened exactly once each, write-only/exclusive, and
    # never reread; the attestation FINAL path is never opened at all — it is
    # published by atomic rename from the exclusive temp file
    assert [m for f, m in order
            if f == str(tmp_path / "oos_matrix.parquet")] == ["xb"]
    assert [m for f, m in order
            if f == str(tmp_path / "oos_manifest.json")] == ["x"]
    att = str(tmp_path / "g0bn-materialization-attestation-v1.json")
    assert [m for f, m in order if f == att] == []
    assert [m for f, m in order if f == att + ".tmp"] == ["x"]


def _spy_all_source_opens(monkeypatch, opened):
    """Record every source-open route: builtins.open AND the pinned-fd primitive."""
    import bars.produce as produce_mod

    real_open = builtins.open
    real_pin = produce_mod._open_pinned_fd

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def spy_pin(path):
        opened.append(str(path))
        return real_pin(path)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(produce_mod, "_open_pinned_fd", spy_pin)


def test_tampered_claim_blocks_every_source_open(custody, tmp_path, monkeypatch):
    bad_claim = tmp_path / "claim.json"
    _write_claim(bad_claim, config=custody["config"], plan=custody["plan"],
                 freeze=custody["freeze"],
                 holdout_plan_sha256=sha_hex("foreign-plan"))
    opened = []
    _spy_all_source_opens(monkeypatch, opened)
    with pytest.raises(ValueError, match="holdout_plan_sha256"):
        _materialize(custody, tmp_path, raw_access_claim_path=bad_claim)
    assert not (set(opened) & {str(p) for p in custody["object_paths"].values()})


def test_degenerate_runtime_rejects_before_claim_and_sources(custody, tmp_path,
                                                             monkeypatch):
    # Codex P1: a deterministic bad operator parameter (min_returns=0) must fail
    # the boundary validation before the claim is consumed or any sealed object
    # is opened — never surface mid-build after the raw burn
    opened = []
    _spy_all_source_opens(monkeypatch, opened)
    with pytest.raises(ValueError, match="min_returns"):
        _materialize(custody, tmp_path,
                     runtime=RUNTIME._replace(min_returns=0))
    touched = set(opened)
    assert str(custody["claim_path"]) not in touched
    assert not (touched & {str(p) for p in custody["object_paths"].values()})


def test_symlinked_object_path_is_rejected(custody, tmp_path):
    oid = "custody/normalized/binance_futures_trades/2026-01-07"
    link = tmp_path / "linked.parquet"
    link.symlink_to(custody["object_paths"][oid])
    paths = dict(custody["object_paths"], **{oid: link})
    with pytest.raises(ValueError, match="symlink"):
        _materialize(custody, tmp_path, object_paths=paths)


def test_missing_claim_means_unclaimed_invocation(custody, tmp_path):
    with pytest.raises(ValueError, match="unclaimed"):
        _materialize(custody, tmp_path,
                     raw_access_claim_path=tmp_path / "absent-claim.json")


def test_claim_self_hash_tamper_is_rejected(custody, tmp_path):
    stale = tmp_path / "stale-claim.json"
    _write_claim(stale, config=custody["config"], plan=custody["plan"],
                 freeze=custody["freeze"], sha256=sha_hex("stale"))
    with pytest.raises(ValueError, match="sha256"):
        _materialize(custody, tmp_path, raw_access_claim_path=stale)


def test_foreign_object_hash_rejects_before_any_decode(custody, tmp_path,
                                                       monkeypatch):
    oid = "custody/normalized/binance_futures_trades/2026-01-05"
    original = custody["object_paths"][oid]
    corrupted = tmp_path / "corrupted.parquet"
    corrupted.write_bytes(original.read_bytes() + b"\x00")
    paths = dict(custody["object_paths"], **{oid: corrupted})

    def no_decode(*args, **kwargs):
        raise AssertionError("no payload may be decoded once any sealed object "
                             "hash fails to verify")

    monkeypatch.setattr(pq, "ParquetFile", no_decode)
    with pytest.raises(ValueError, match="foreign or corrupted"):
        _materialize(custody, tmp_path, object_paths=paths)


# --------------------------------------------------------- allowlist discipline


def test_missing_extra_duplicate_and_glob_objects_reject(custody, tmp_path,
                                                         monkeypatch):
    opened = []
    _spy_all_source_opens(monkeypatch, opened)
    object_files = {str(p) for p in custody["object_paths"].values()}

    paths = dict(custody["object_paths"])
    oid = "custody/normalized/binance_futures_l2_delta/2026-01-03"
    removed = paths.pop(oid)
    with pytest.raises(ValueError, match="missing sealed normalized object"):
        _materialize(custody, tmp_path, object_paths=paths)
    # allowlist-discipline rejections happen before ANY sealed payload byte is
    # opened (issue #94: reject before payload access where possible)
    assert not (set(opened) & object_files)

    paths = dict(custody["object_paths"])
    paths["custody/normalized/binance_futures_trades/2026-01-14"] = removed
    with pytest.raises(ValueError, match="unknown object id"):
        _materialize(custody, tmp_path, object_paths=paths)

    paths = dict(custody["object_paths"])
    other = "custody/normalized/binance_futures_l2_delta/2026-01-04"
    paths[oid] = paths[other]
    with pytest.raises(ValueError, match="same file"):
        _materialize(custody, tmp_path, object_paths=paths)

    paths = dict(custody["object_paths"])
    paths[oid] = str(tmp_path / "*.parquet")
    with pytest.raises(ValueError, match="glob"):
        _materialize(custody, tmp_path, object_paths=paths)
    # none of the four rejections opened a sealed payload
    assert not (set(opened) & object_files)


def test_incompatible_drop_count_schema_fails_closed(custody, tmp_path):
    config2 = dev_config(
        clock=make_clock(target_bars_per_day=TARGET_BARS_PER_DAY,
                         time_cap_ns=TIME_CAP_NS,
                         development_end_state_sha256=clock_state_sha256(
                             CLOCK_STATE)),
        partition=make_partition(holdout_drop_count_categories=["a", "b"]),
        **sealed_config_kwargs(custody["inventory"]))
    plan2 = build_holdout_plan(config2, custody["inventory"], generated_at=GEN_AT)
    with pytest.raises(ValueError, match="drop_count_categories"):
        _materialize(custody, tmp_path, config=config2, plan=plan2)


def test_clock_state_must_reproduce_the_frozen_pin(custody, tmp_path):
    drifted = copy.deepcopy(CLOCK_STATE)
    drifted["history"][0]["completed_notional"] = 40_000.0
    with pytest.raises(ValueError, match="development-end clock state"):
        _materialize(custody, tmp_path, clock_state=drifted)

    in_partition = copy.deepcopy(CLOCK_STATE)
    in_partition["history"].append(
        {"day": "2026-01-02", "completed_notional": 1.0, "covered_fraction": 1.0})
    with pytest.raises(ValueError, match="prior day"):
        _materialize(custody, tmp_path, clock_state=in_partition)

    wrong_runtime = RUNTIME._replace(threshold=RUNTIME.threshold._replace(
        window_days=5))
    with pytest.raises(ValueError, match="threshold_config"):
        _materialize(custody, tmp_path, runtime=wrong_runtime)


def test_clock_state_history_must_stay_in_included_development_days(custody):
    """Even a hash-matching frozen pin is rejected when its history embeds a day
    outside the config's outcome-blind included development scope (a pin minted
    over an out-of-window or excluded day would otherwise smuggle out-of-scope
    volume into the holdout schedule seed)."""
    import bars.produce as produce_mod

    for bad_day in ("2025-10-31",   # before the development window
                    "2025-12-25"):  # an explicitly excluded development day
        state = copy.deepcopy(CLOCK_STATE)
        state["history"].insert(0, {"day": bad_day,
                                    "completed_notional": 36_000.0,
                                    "covered_fraction": 1.0})
        config2 = copy.deepcopy(custody["config"])
        config2["clock"]["development_end_state_sha256"] = clock_state_sha256(
            state)
        with pytest.raises(ValueError, match="included development days"):
            produce_mod._validate_clock_state(
                state, runtime=RUNTIME, config=config2,
                before_ns=int(
                    custody["config"]["partition"]["holdout_start_ns"]))
