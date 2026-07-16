"""67-D generic-runner holdout guard (issue #90; protocol spec section 7).

Loader spies prove rejection happens BEFORE any matrix access (pandas, pyarrow,
footer, or caller-supplied loader); every holdout marker rejects independently;
renaming/removing bindings cannot downgrade a holdout build into a generic one;
in-memory preloaded frames are rejected on the manifest alone; development G0-BN,
generic synthetic, and legacy G0 development manifests keep flowing through their
existing paths. Everything here is synthetic — no vendor I/O, no real data, no
January values, no scoring."""
from __future__ import annotations

import copy
import itertools
import json

import pytest

import scripts.run_baseline as run_baseline
from eval.guard import guarded_read_matrix, preflight_generic_manifest
from eval.runner import run_from_manifest
from eval.synthetic import make_manifest, make_matrix
from eval.writer import G0BN_DEV_DATASET_ID, G0BN_OOS_DATASET_ID
from tests.g0bn_fixtures import g0bn_frame, g0bn_manifest, hex64, holdout_plan_binding

# The stable refusal prefix every generic entry point surfaces; the message must never
# advertise an escape hatch (asserted in test_rejection_message_stable_and_offers_no_override).
GUARD_MSG = "holdout guard"

GATE = {"n_groups": 4, "k": 2, "min_trades": 1, "min_eff_trades": 1.0}


def dev_man(**over) -> dict:
    return g0bn_manifest(partition="development", **over)


def holdout_man() -> dict:
    return g0bn_manifest(partition="holdout")


def _set_partition(man: dict, value) -> dict:
    for s in man["sources"]:
        if isinstance(s, dict) and s.get("name") == "partition_contract":
            s["partition"] = value
    return man


def _drop_source(man: dict, name: str) -> dict:
    man["sources"] = [s for s in man["sources"]
                      if not (isinstance(s, dict) and s.get("name") == name)]
    return man


def _binding(man: dict, name: str) -> dict:
    return next(s for s in man["sources"]
                if isinstance(s, dict) and s.get("name") == name)


def _with_plan_binding(man: dict) -> dict:
    man["sources"].append(holdout_plan_binding())
    return man


# Each marker ALONE on an otherwise-development manifest: the guard must reject every
# one independently, so scrubbing the others can never authorize a generic load.
MARKERS = {
    "partition_holdout": lambda: _set_partition(dev_man(), "holdout"),
    "holdout_plan_binding": lambda: _with_plan_binding(dev_man()),
    "oos_dataset_id": lambda: dev_man(dataset_id=G0BN_OOS_DATASET_ID),
}

# Scrub one holdout marker at a time off a full holdout manifest; any surviving marker
# must still reject (renaming/removing bindings is not a downgrade path).
SCRUBS = {
    "partition": lambda man: _set_partition(man, "development"),
    "plan": lambda man: _drop_source(man, "g0bn_holdout_plan"),
    "dataset": lambda man: man.update(dataset_id=G0BN_DEV_DATASET_ID),
}
SCRUB_SUBSETS = [c for r in (1, 2) for c in itertools.combinations(sorted(SCRUBS), r)]


class PoisonFrame:
    """A 'preloaded matrix' that fails on ANY use: the in-memory entry point must
    reject on the manifest alone, before the frame is touched at all."""

    def __getattr__(self, name):
        raise AssertionError(f"matrix accessed (.{name}) before the guard rejected")

    def __getitem__(self, key):
        raise AssertionError(f"matrix accessed ([{key!r}]) before the guard rejected")

    def __len__(self):
        raise AssertionError("matrix accessed (len) before the guard rejected")


def _forbid_loaders(monkeypatch):
    """Poison every parquet loader run_baseline/eval.guard could reach: pandas,
    pyarrow's footer-reading ParquetFile, and pyarrow's table reader."""
    import pyarrow.parquet as pq

    def boom(*a, **k):
        raise AssertionError("a parquet loader ran before the holdout guard rejected")

    monkeypatch.setattr(run_baseline.pd, "read_parquet", boom)
    monkeypatch.setattr(pq, "ParquetFile", boom)
    monkeypatch.setattr(pq, "read_table", boom)


def _manifest_path(tmp_path, man: dict) -> str:
    p = tmp_path / "feature_manifest.json"
    p.write_text(json.dumps(man))
    return str(p)


# ------------------------------------------------------------ preflight: rejections

@pytest.mark.parametrize("marker", sorted(MARKERS))
def test_each_marker_rejects_independently(marker):
    with pytest.raises(ValueError, match=GUARD_MSG):
        preflight_generic_manifest(MARKERS[marker]())


def test_full_holdout_manifest_rejected():
    with pytest.raises(ValueError, match=GUARD_MSG):
        preflight_generic_manifest(holdout_man())


@pytest.mark.parametrize("scrubs", SCRUB_SUBSETS)
def test_scrubbing_markers_cannot_downgrade_a_holdout_build(scrubs):
    man = holdout_man()
    for name in scrubs:
        SCRUBS[name](man)
    with pytest.raises(ValueError, match=GUARD_MSG):
        preflight_generic_manifest(man)


def test_renamed_plan_binding_cannot_downgrade():
    # Scrub every raw marker AND rename the plan binding: the g0bn_protocol marker still
    # routes the manifest through the full validator, which refuses the unknown source.
    man = holdout_man()
    SCRUBS["partition"](man)
    SCRUBS["dataset"](man)
    _binding(man, "g0bn_holdout_plan")["name"] = "g0bn_holdout_plan_v2"
    with pytest.raises(ValueError, match="forbidden/unknown"):
        preflight_generic_manifest(man)


def test_removed_partition_binding_on_holdout_manifest_rejected():
    man = _drop_source(holdout_man(), "partition_contract")
    with pytest.raises(ValueError, match=GUARD_MSG):  # plan binding + dataset still trip
        preflight_generic_manifest(man)


def test_missing_partition_binding_on_marked_dev_manifest_rejected():
    # A g0bn_protocol-marked manifest with NO partition binding is ambiguous: reject,
    # never fall back to generic handling.
    man = _drop_source(dev_man(), "partition_contract")
    with pytest.raises(ValueError, match="exactly one"):
        preflight_generic_manifest(man)


def test_duplicate_partition_bindings_rejected():
    man = dev_man()
    man["sources"].append(copy.deepcopy(_binding(man, "partition_contract")))
    with pytest.raises(ValueError, match="exactly one"):
        preflight_generic_manifest(man)


def test_duplicate_partition_binding_smuggling_holdout_rejected_as_marker():
    # Two bindings, development first and holdout second: the raw scan checks EVERY
    # entry, so ordering cannot hide the holdout one behind a development twin.
    man = dev_man()
    dup = copy.deepcopy(_binding(man, "partition_contract"))
    dup["partition"] = "holdout"
    man["sources"].append(dup)
    with pytest.raises(ValueError, match=GUARD_MSG):
        preflight_generic_manifest(man)


@pytest.mark.parametrize("bad", ["Holdout", "both", "development-holdout", "", None, 7])
def test_malformed_or_ambiguous_partition_value_rejected(bad):
    man = _set_partition(dev_man(), bad)
    with pytest.raises(ValueError, match="partition"):
        preflight_generic_manifest(man)


def test_partition_binding_missing_partition_field_rejected():
    man = dev_man()
    del _binding(man, "partition_contract")["partition"]
    with pytest.raises(ValueError, match="exactly the fields"):
        preflight_generic_manifest(man)


def test_string_source_entry_on_marked_manifest_rejected():
    man = dev_man()
    man["sources"].append("crypto-lake/raw")
    with pytest.raises(ValueError, match="structured dicts"):
        preflight_generic_manifest(man)


def test_string_source_entry_does_not_hide_holdout_markers():
    man = holdout_man()
    man["sources"].insert(0, "stray-string-source")
    with pytest.raises(ValueError, match=GUARD_MSG):  # raw scan skips strings, still trips
        preflight_generic_manifest(man)


def test_oos_dataset_id_rejected_even_on_schema_invalid_dict():
    # Raw marker scans run BEFORE schema validation: even a torn manifest naming the
    # OOS dataset gets the stable guard refusal, not an incidental schema error.
    with pytest.raises(ValueError, match=GUARD_MSG):
        preflight_generic_manifest({"dataset_id": G0BN_OOS_DATASET_ID})


def test_plan_binding_rejected_even_on_schema_invalid_dict():
    with pytest.raises(ValueError, match=GUARD_MSG):
        preflight_generic_manifest({"sources": [holdout_plan_binding()]})


@pytest.mark.parametrize("garbage", [None, [], "feature_manifest.json", 42])
def test_non_dict_manifest_rejected_with_valueerror(garbage):
    # The guard's fail-closed contract is ValueError for ANY input, even garbage a
    # broken caller hands it — never an incidental AttributeError.
    with pytest.raises(ValueError, match="manifest must be a dict"):
        preflight_generic_manifest(garbage)


def test_classifier_holdout_verdict_refused_even_if_raw_scans_pass(monkeypatch):
    # Defense in depth: if a future classifier learns a holdout marker the raw scans do
    # not know, the guard must still refuse on the classification verdict.
    from eval import guard
    from eval.writer import ManifestClass
    _, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    man = make_manifest(feats, lb)
    monkeypatch.setattr(guard, "classify_manifest", lambda m: ManifestClass(
        dataset_id=m["dataset_id"], is_g0bn=True, partition="holdout",
        holdout_bound=True))
    with pytest.raises(ValueError, match=GUARD_MSG):
        guard.preflight_generic_manifest(man)


def test_rejection_message_stable_and_offers_no_override():
    with pytest.raises(ValueError) as ei:
        preflight_generic_manifest(holdout_man())
    msg = str(ei.value)
    assert msg.startswith("holdout guard:")
    for token in ("override", "force", "bypass", "--", "validation-only"):
        assert token not in msg.lower()


# ---------------------------------------------------------- preflight: pass-through

def test_development_g0bn_manifest_passes():
    cls = preflight_generic_manifest(dev_man())
    assert cls.is_g0bn and cls.partition == "development" and not cls.holdout_bound


def test_generic_manifest_passes():
    _, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    cls = preflight_generic_manifest(make_manifest(feats, lb))
    assert not cls.is_g0bn and cls.partition is None and not cls.holdout_bound


def test_legacy_development_partition_binding_passes():
    # Legacy G0/G0-XV manifests share the partition_contract source NAME (eval/partition
    # BINDING_SOURCE_NAME); a development binding keeps flowing through generic paths.
    _, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    legacy = {"name": "partition_contract", "sha256": hex64(21),
              "partition": "development", "boundary_drop_counts": {"10s": 0}}
    cls = preflight_generic_manifest(
        make_manifest(feats, lb, sources=["eval/synthetic.py", legacy]))
    assert not cls.is_g0bn and not cls.holdout_bound


def test_legacy_holdout_partition_binding_rejected():
    # The generic runner is not the legacy one-shot consumer either: a holdout-partition
    # binding refuses regardless of G0-BN markers. scripts/run_g0.py holdout-score stays
    # the authorized legacy path (its own fail-before-load machinery, untouched here).
    _, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    legacy = {"name": "partition_contract", "sha256": hex64(21),
              "partition": "holdout", "boundary_drop_counts": {"10s": 0}}
    with pytest.raises(ValueError, match=GUARD_MSG):
        preflight_generic_manifest(
            make_manifest(feats, lb, sources=["eval/synthetic.py", legacy]))


# ------------------------------------------------- in-memory runner (run_from_manifest)

@pytest.mark.parametrize("marker", sorted(MARKERS))
def test_run_from_manifest_rejects_each_marker_before_any_frame_access(marker):
    with pytest.raises(ValueError, match=GUARD_MSG):
        run_from_manifest(PoisonFrame(), MARKERS[marker]())


def test_run_from_manifest_rejects_preloaded_holdout_frame_immediately():
    with pytest.raises(ValueError, match=GUARD_MSG):
        run_from_manifest(PoisonFrame(), holdout_man())


def test_run_from_manifest_versionless_holdout_dict_still_rejected_pre_frame():
    # The v1 migration gate fires first (existing contract, pinned by test_runner.py);
    # a version-less holdout dict is still refused before any frame use.
    man = holdout_man()
    del man["manifest_version"]
    with pytest.raises(ValueError, match="manifest_version"):
        run_from_manifest(PoisonFrame(), man)


def test_run_from_manifest_dev_g0bn_passes_guard_into_existing_validation():
    # Development G0-BN flows PAST the guard into the unchanged baseline checks: the
    # fixture manifest carries no 'gate' block, so the pre-registration error surfaces.
    with pytest.raises(ValueError, match="pre-registered 'gate' block"):
        run_from_manifest(g0bn_frame(), dev_man())


# ------------------------------------------------------------------ CLI (run_baseline)

@pytest.mark.parametrize("marker", sorted(MARKERS))
def test_cli_rejects_each_marker_before_any_loader(tmp_path, monkeypatch, marker):
    _forbid_loaders(monkeypatch)
    manifest_path = _manifest_path(tmp_path, MARKERS[marker]())
    # The matrix path deliberately does not exist: any loader reaching the filesystem
    # would surface a FileNotFoundError instead of the guard's ValueError.
    with pytest.raises(ValueError, match=GUARD_MSG):
        run_baseline.main(str(tmp_path / "model_matrix.parquet"), manifest_path)


def test_cli_rejects_full_holdout_manifest_with_guard_error_not_gate_error(
        tmp_path, monkeypatch):
    # The holdout fixture has no 'gate' block: the guard refusal must win over the gate
    # pre-registration error, so the CLI surfaces one stable holdout message.
    _forbid_loaders(monkeypatch)
    manifest_path = _manifest_path(tmp_path, holdout_man())
    with pytest.raises(ValueError, match=GUARD_MSG):
        run_baseline.main(str(tmp_path / "model_matrix.parquet"), manifest_path)


def test_cli_dev_g0bn_fails_on_gate_before_any_loader(tmp_path, monkeypatch):
    # Development G0-BN passes the guard; the existing pre-read gate check still fires
    # before any parquet loader (loaders are poisoned).
    _forbid_loaders(monkeypatch)
    manifest_path = _manifest_path(tmp_path, dev_man())
    with pytest.raises(ValueError, match="pre-registered 'gate' block"):
        run_baseline.main(str(tmp_path / "model_matrix.parquet"), manifest_path)


def test_cli_generic_manifest_still_loads_and_runs(tmp_path, monkeypatch, capsys):
    # Legacy/generic regression: an ordinary v1 manifest flows through the guard, the
    # loader runs exactly once, and the study completes end to end.
    matrix, feats, lb = make_matrix(n=900, signal_strength=4.0, seed=8)
    manifest_path = _manifest_path(tmp_path, make_manifest(feats, lb, gate=dict(GATE)))
    calls = []

    def fake_read_parquet(path, *a, **k):
        calls.append(str(path))
        return matrix

    monkeypatch.setattr(run_baseline.pd, "read_parquet", fake_read_parquet)
    run_baseline.main("model_matrix.parquet", manifest_path)
    assert calls == ["model_matrix.parquet"]
    out = capsys.readouterr().out
    assert "manifest: synthetic" in out and "resolved gate" in out


# ------------------------------------------------------- guarded_read_matrix (loaders)

def test_guarded_read_matrix_rejects_before_caller_supplied_loader():
    calls = []

    def loader(path):
        calls.append(path)
        return "FRAME"

    with pytest.raises(ValueError, match=GUARD_MSG):
        guarded_read_matrix("model_matrix.parquet", holdout_man(), loader=loader)
    assert calls == []


def test_guarded_read_matrix_authorized_manifest_reaches_loader():
    _, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    calls = []

    def loader(path):
        calls.append(path)
        return "FRAME"

    out = guarded_read_matrix("model_matrix.parquet", make_manifest(feats, lb),
                              loader=loader)
    assert out == "FRAME" and calls == ["model_matrix.parquet"]


def test_guarded_read_matrix_default_loader_never_reached_for_holdout(monkeypatch):
    _forbid_loaders(monkeypatch)
    with pytest.raises(ValueError, match=GUARD_MSG):
        guarded_read_matrix("model_matrix.parquet", holdout_man())
