"""T8 writer contract (issue #87): deterministic ModelMatrix + manifest publication.

Pins: logical-row/build identity vs matrix-file/physical-schema hash separation,
development validate-before-publication, holdout blind write with no derived-output
reopen, development/holdout isolation, and deterministic canonicalized output
(producer plan sections H and I)."""
from __future__ import annotations

import builtins
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from eval.manifest import build_manifest, load_manifest, manifest_sha256, validate_frame, write_manifest
from eval.matrix import RESERVED, validate_matrix
from eval.writer import (
    G0BN_DEV_DATASET_ID,
    WriteResult,
    build_id_for,
    classify_manifest,
    logical_row_sha256,
    ordered_manifest_columns,
    write_development,
    write_holdout,
)
from tests.g0bn_fixtures import FEE_BPS, SLIP_BPS, built_g0bn, g0bn_frame, g0bn_manifest, hex64


def _paths(tmp_path, tag=""):
    return tmp_path / f"matrix{tag}.parquet", tmp_path / f"manifest{tag}.json"


# ---------------------------------------------------------------- identity primitives

def test_ordered_manifest_columns_is_explicit_manifest_order():
    man = g0bn_manifest()
    assert ordered_manifest_columns(man) == (
        man["feature_cols"] + man["reserved_cols"] + man["extra_cols"])


def test_logical_row_hash_ignores_row_and_column_order():
    frame = g0bn_frame()
    cols = ordered_manifest_columns(g0bn_manifest())
    scrambled = frame.sample(frac=1.0, random_state=3)[list(reversed(frame.columns))]
    assert logical_row_sha256(frame, cols) == logical_row_sha256(scrambled, cols)


def test_logical_row_hash_changes_with_values():
    frame = g0bn_frame()
    cols = ordered_manifest_columns(g0bn_manifest())
    bumped = frame.copy()
    bumped.loc[0, "cvd"] += 1e-9
    assert logical_row_sha256(frame, cols) != logical_row_sha256(bumped, cols)


def test_logical_row_hash_rejects_duplicate_event_horizon():
    frame = g0bn_frame()
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="t_event"):
        logical_row_sha256(dup, ordered_manifest_columns(g0bn_manifest()))


def test_logical_row_hash_rejects_missing_column():
    frame = g0bn_frame().drop(columns=["cvd"])
    with pytest.raises(ValueError, match="cvd"):
        logical_row_sha256(frame, ordered_manifest_columns(g0bn_manifest()))


def test_build_id_binds_dataset_rows_and_params():
    frame = g0bn_frame()
    lrh = logical_row_sha256(frame, ordered_manifest_columns(g0bn_manifest()))
    base = build_id_for(dataset_id="a", logical_row_sha256=lrh, build_params={"seed": 7})
    assert base == build_id_for(dataset_id="a", logical_row_sha256=lrh,
                                build_params={"seed": 7})
    assert base != build_id_for(dataset_id="b", logical_row_sha256=lrh,
                                build_params={"seed": 7})
    assert base != build_id_for(dataset_id="a", logical_row_sha256=lrh,
                                build_params={"seed": 8})
    assert base != build_id_for(dataset_id="a", logical_row_sha256="0" * 64,
                                build_params={"seed": 7})


def test_build_id_rejects_generated_at_param():
    with pytest.raises(ValueError, match="generated_at"):
        build_id_for(dataset_id="a", logical_row_sha256="0" * 64,
                     build_params={"generated_at": "2026-07-15T00:00:00+00:00"})


def test_build_id_rejects_non_canonical_params():
    with pytest.raises(ValueError, match="build_params"):
        build_id_for(dataset_id="a", logical_row_sha256="0" * 64,
                     build_params={"x": float("nan")})


# ---------------------------------------------------------------- manifest helpers

def test_build_manifest_round_trip(tmp_path):
    man = g0bn_manifest()
    rebuilt = build_manifest(
        dataset_id=man["dataset_id"], build_id=man["build_id"],
        bar_clock=man["bar_clock"], feature_cols=man["feature_cols"],
        target_cols=man["target_cols"], venues=man["venues"], horizons=man["horizons"],
        sources=man["sources"], generated_at=man["generated_at"],
        max_lookback_ns=man["max_lookback_ns"], embargo_ns=man["embargo_ns"],
        extra_cols=man["extra_cols"], dtypes=man["dtypes"],
        availability_lag_ns=man["availability_lag_ns"])
    assert rebuilt == man
    path = tmp_path / "man.json"
    sha = write_manifest(rebuilt, path)
    assert load_manifest(path) == man
    assert sha == manifest_sha256(man)
    assert json.loads(path.read_text()) == man


def test_build_manifest_omits_unset_optionals():
    man = g0bn_manifest()
    rebuilt = build_manifest(
        dataset_id="d", build_id="b", bar_clock={"kind": "dollar"},
        feature_cols=["f1"], target_cols=["y_fwd_bps", "label"],
        venues=man["venues"], horizons={"10s": 10_000_000_000}, sources=["src"],
        generated_at=man["generated_at"], max_lookback_ns=1, embargo_ns=1)
    assert set(rebuilt) == {
        "manifest_version", "dataset_id", "build_id", "bar_clock", "time",
        "feature_cols", "target_cols", "reserved_cols", "venues", "horizons",
        "sources", "generated_at", "max_lookback_ns", "embargo_ns"}
    assert rebuilt["reserved_cols"] == list(RESERVED)


def test_build_manifest_validates():
    with pytest.raises(ValueError, match="leaky"):
        build_manifest(
            dataset_id="d", build_id="b", bar_clock={"kind": "dollar"},
            feature_cols=["fwd_ret"], target_cols=["y_fwd_bps", "label"],
            venues=[{"exchange": "E", "symbol": "S"}],
            horizons={"10s": 10_000_000_000}, sources=["src"],
            generated_at="2026-07-15T00:00:00+00:00", max_lookback_ns=1, embargo_ns=1)


def test_manifest_sha256_excludes_generated_at_only():
    man = g0bn_manifest()
    assert manifest_sha256(man) == manifest_sha256(
        g0bn_manifest(generated_at="2026-07-16T12:00:00+00:00"))
    assert manifest_sha256(man) != manifest_sha256(g0bn_manifest(build_id=hex64(42)))


# ---------------------------------------------------------------- development writes

def test_development_write_round_trip(tmp_path):
    frame, man, params = built_g0bn()
    mpath, jpath = _paths(tmp_path)
    res = write_development(frame, man, build_params=params,
                            matrix_path=mpath, manifest_path=jpath)
    assert isinstance(res, WriteResult)
    assert res.build_id == man["build_id"]
    assert res.row_count == len(frame)
    assert res.matrix_file_sha256 == hashlib.sha256(mpath.read_bytes()).hexdigest()
    loaded = load_manifest(jpath)
    assert loaded == man
    assert res.manifest_sha256 == manifest_sha256(loaded)
    back = pd.read_parquet(mpath)
    # published artifact is canonical: manifest column order, (t_event, horizon) row order
    assert list(back.columns) == ordered_manifest_columns(man)
    key = list(zip(back["t_event"], back["horizon"]))
    assert key == sorted(key)
    validate_frame(back, loaded)
    validate_matrix(back, loaded["feature_cols"])
    assert logical_row_sha256(back, ordered_manifest_columns(loaded)) == res.logical_row_sha256
    assert classify_manifest(loaded).holdout_bound is False


def test_development_write_is_deterministic_and_order_insensitive(tmp_path):
    frame, man, params = built_g0bn()
    scrambled = frame.sample(frac=1.0, random_state=5)[list(reversed(frame.columns))]
    m1, j1 = _paths(tmp_path, "1")
    m2, j2 = _paths(tmp_path, "2")
    r1 = write_development(frame, man, build_params=params, matrix_path=m1, manifest_path=j1)
    r2 = write_development(scrambled, man, build_params=params, matrix_path=m2, manifest_path=j2)
    assert r1.logical_row_sha256 == r2.logical_row_sha256
    assert r1.build_id == r2.build_id
    assert r1.manifest_sha256 == r2.manifest_sha256
    assert r1.physical_schema_sha256 == r2.physical_schema_sha256
    assert r1.matrix_file_sha256 == r2.matrix_file_sha256
    assert m1.read_bytes() == m2.read_bytes()


def test_two_physical_encodings_share_logical_identity(tmp_path):
    frame, man, params = built_g0bn()
    m1, j1 = _paths(tmp_path, "1")
    m2, j2 = _paths(tmp_path, "2")
    r1 = write_development(frame, man, build_params=params, matrix_path=m1,
                           manifest_path=j1, parquet_options={"compression": "snappy"})
    r2 = write_development(frame, man, build_params=params, matrix_path=m2,
                           manifest_path=j2, parquet_options={"compression": "gzip"})
    assert r1.matrix_file_sha256 != r2.matrix_file_sha256
    assert r1.logical_row_sha256 == r2.logical_row_sha256
    assert r1.build_id == r2.build_id
    assert r1.physical_schema_sha256 == r2.physical_schema_sha256
    # and the logical identity survives both physical round trips
    for path, res in ((m1, r1), (m2, r2)):
        back = pd.read_parquet(path)
        assert logical_row_sha256(back, ordered_manifest_columns(man)) == res.logical_row_sha256


def test_development_write_rejects_build_id_drift(tmp_path):
    frame, man, params = built_g0bn()
    man["build_id"] = hex64(500)
    mpath, jpath = _paths(tmp_path)
    with pytest.raises(ValueError, match="build_id"):
        write_development(frame, man, build_params=params,
                          matrix_path=mpath, manifest_path=jpath)
    assert not mpath.exists() and not jpath.exists()


@pytest.mark.parametrize("corrupt", [
    lambda f: f.assign(cvd=np.nan),                                    # NaN feature
    lambda f: f.assign(queue_imb=np.inf),                              # inf feature
    lambda f: f.assign(t_available=f["t_available"] - 1),              # timing violation
    lambda f: f.assign(label=2),                                       # bad label
    lambda f: f.assign(uniqueness=0.0),                                # bad uniqueness
    lambda f: f.assign(t_barrier=f["t_event"] + 120_000_000_000),      # horizon overrun
])
def test_development_write_validates_before_publication(tmp_path, corrupt):
    frame, man, params = built_g0bn()
    bad = corrupt(frame)
    mpath, jpath = _paths(tmp_path)
    with pytest.raises(ValueError):
        write_development(bad, man, build_params=params,
                          matrix_path=mpath, manifest_path=jpath)
    assert not mpath.exists() and not jpath.exists()


def test_development_write_rejects_undeclared_column(tmp_path):
    frame, man, params = built_g0bn()
    frame["mystery"] = 1.0
    with pytest.raises(ValueError, match="mystery"):
        write_development(frame, man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


def test_development_write_requires_declared_extra_col_in_frame(tmp_path):
    frame, man, params = built_g0bn()
    frame = frame.drop(columns=["latency_drift_bps"])
    with pytest.raises(ValueError, match="latency_drift_bps"):
        write_development(frame, man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


def test_development_write_rejects_float32_cost_diagnostic(tmp_path):
    frame, man, params = built_g0bn()
    frame["latency_drift_bps"] = frame["latency_drift_bps"].astype(np.float32)
    with pytest.raises(ValueError, match="latency_drift_bps"):
        write_development(frame, man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


def test_development_write_rejects_float32_feature(tmp_path):
    frame, man, params = built_g0bn()
    frame["cvd"] = frame["cvd"].astype(np.float32)
    with pytest.raises(ValueError, match="cvd"):
        write_development(frame, man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


def test_development_write_rejects_dtype_drift(tmp_path):
    frame, man, params = built_g0bn()
    frame["latency_drift_bps"] = np.zeros(len(frame), dtype=np.int64)
    with pytest.raises(ValueError, match="latency_drift_bps"):
        write_development(frame, man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


@pytest.mark.parametrize("mutate", [
    lambda f: f.assign(latency_drift_bps=-0.1),
    lambda f: f.assign(latency_drift_bps=np.inf),
])
def test_development_write_rejects_bad_drift_diagnostic(tmp_path, mutate):
    frame, man, params = built_g0bn()
    with pytest.raises(ValueError, match="latency_drift_bps"):
        write_development(mutate(frame), man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


def test_development_write_rejects_cost_reconciliation_break(tmp_path):
    frame, man, params = built_g0bn()
    frame.loc[0, "cost_bps"] = 2.0 * FEE_BPS + SLIP_BPS + frame.loc[0, "latency_drift_bps"] + 0.5
    with pytest.raises(ValueError, match="cost_bps"):
        write_development(frame, man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


def test_development_write_allows_rebuild_overwrite(tmp_path):
    frame, man, params = built_g0bn()
    mpath, jpath = _paths(tmp_path)
    r1 = write_development(frame, man, build_params=params, matrix_path=mpath, manifest_path=jpath)
    r2 = write_development(frame, man, build_params=params, matrix_path=mpath, manifest_path=jpath)
    assert r1 == r2


def test_development_write_accepts_non_g0bn_manifest(tmp_path):
    """The writer is the generic v1 publication path; G0-BN rules bind only G0-BN builds."""
    from eval.synthetic import make_manifest, make_matrix
    frame, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    params = {"seed": 1}
    man = make_manifest(feats, lb)
    lrh = logical_row_sha256(frame, ordered_manifest_columns(man))
    man["build_id"] = build_id_for(dataset_id=man["dataset_id"],
                                   logical_row_sha256=lrh, build_params=params)
    mpath, jpath = _paths(tmp_path)
    res = write_development(frame, man, build_params=params,
                            matrix_path=mpath, manifest_path=jpath)
    assert res.logical_row_sha256 == lrh
    validate_frame(pd.read_parquet(mpath), load_manifest(jpath))


# ---------------------------------------------------------------- dev/holdout isolation

def test_development_writer_refuses_holdout_manifest(tmp_path):
    frame, man, params = built_g0bn(partition="holdout")
    with pytest.raises(ValueError, match="holdout"):
        write_development(frame, man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")
    assert not (tmp_path / "m.parquet").exists()


def test_holdout_writer_refuses_development_manifest(tmp_path):
    frame, man, params = built_g0bn()
    with pytest.raises(ValueError, match="development"):
        write_holdout(frame, man, build_params=params,
                      matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")
    assert not (tmp_path / "m.parquet").exists()


def test_development_params_must_not_bind_holdout_plan(tmp_path):
    frame, man, params = built_g0bn()
    params = {**params, "holdout_plan_sha256": hex64(7)}
    with pytest.raises(ValueError, match="holdout_plan_sha256"):
        write_development(frame, man, build_params=params,
                          matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


def test_holdout_params_must_bind_plan_hash(tmp_path):
    frame, man, params = built_g0bn(partition="holdout")
    missing = {k: v for k, v in params.items() if k != "holdout_plan_sha256"}
    with pytest.raises(ValueError, match="holdout_plan_sha256"):
        write_holdout(frame, man, build_params=missing,
                      matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")
    wrong = {**params, "holdout_plan_sha256": hex64(1000)}
    with pytest.raises(ValueError, match="holdout_plan_sha256"):
        write_holdout(frame, man, build_params=wrong,
                      matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")


def test_dev_and_holdout_identities_differ_on_identical_rows():
    frame, dev_man, dev_params = built_g0bn()
    _, oos_man, oos_params = built_g0bn(partition="holdout", frame=frame)
    assert dev_man["dataset_id"] != oos_man["dataset_id"]
    assert dev_man["build_id"] != oos_man["build_id"]
    # same canonical logical rows: the difference is the staged identity, not content
    cols = ordered_manifest_columns(dev_man)
    assert logical_row_sha256(frame, cols) == logical_row_sha256(frame, ordered_manifest_columns(oos_man))


# ---------------------------------------------------------------- holdout blind write

def test_holdout_write_never_reopens_output(tmp_path, monkeypatch):
    frame, man, params = built_g0bn(partition="holdout")
    mpath, jpath = _paths(tmp_path)
    outputs = {str(mpath), str(jpath)}

    def boom(*a, **k):
        raise AssertionError("parquet reader invoked during blind holdout write")

    real_open = builtins.open

    def spy_open(file, mode="r", *a, **k):
        if not isinstance(file, int) and str(file) in outputs and (
                "r" in mode or "+" in mode):
            raise AssertionError(f"derived output reopened: {file} mode={mode}")
        return real_open(file, mode, *a, **k)

    real_os_open = os.open

    def spy_os_open(path, flags, *a, **k):
        writing = flags & (os.O_WRONLY | os.O_RDWR)
        if not isinstance(path, int) and str(path) in outputs and not writing:
            raise AssertionError(f"derived output reopened via os.open: {path}")
        return real_os_open(path, flags, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", boom)
    monkeypatch.setattr(pq, "read_table", boom)
    monkeypatch.setattr(pq, "ParquetFile", boom)
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    res = write_holdout(frame, man, build_params=params,
                        matrix_path=mpath, manifest_path=jpath)
    monkeypatch.undo()

    # every attestation input was computed while producing the artifacts
    assert res.matrix_file_sha256 == hashlib.sha256(mpath.read_bytes()).hexdigest()
    assert res.manifest_sha256 == manifest_sha256(json.loads(jpath.read_text()))
    assert res.build_id == man["build_id"]
    assert res.row_count == len(frame)
    back = pd.read_parquet(mpath)
    assert logical_row_sha256(back, ordered_manifest_columns(man)) == res.logical_row_sha256
    assert classify_manifest(load_manifest(jpath)).holdout_bound is True


def test_holdout_write_requires_fresh_paths(tmp_path):
    frame, man, params = built_g0bn(partition="holdout")
    mpath, jpath = _paths(tmp_path)
    write_holdout(frame, man, build_params=params, matrix_path=mpath, manifest_path=jpath)
    frame2, man2, params2 = built_g0bn(partition="holdout")
    with pytest.raises(FileExistsError):
        write_holdout(frame2, man2, build_params=params2,
                      matrix_path=mpath, manifest_path=jpath)


def test_holdout_write_defers_value_validation(tmp_path):
    """Blind materialization must not inspect outcome values: a frame that would fail
    validate_frame/validate_matrix still writes; the post-burn scorer owns that judgment."""
    frame = g0bn_frame().copy()
    frame.loc[0, "cvd"] = np.nan                       # would fail validate_matrix
    frame.loc[1, "t_available"] -= 1                   # would fail validate_frame
    frame, man, params = built_g0bn(partition="holdout", frame=frame)
    mpath, jpath = _paths(tmp_path)
    res = write_holdout(frame, man, build_params=params,
                        matrix_path=mpath, manifest_path=jpath)
    assert res.row_count == len(frame)
    assert mpath.exists() and jpath.exists()


def test_holdout_write_still_refuses_schema_drift(tmp_path):
    frame, man, params = built_g0bn(partition="holdout")
    f32 = frame.assign(cost_bps=frame["cost_bps"].astype(np.float32))
    with pytest.raises(ValueError, match="cost_bps"):
        write_holdout(f32, man, build_params=params,
                      matrix_path=tmp_path / "m.parquet", manifest_path=tmp_path / "m.json")
    undeclared = frame.assign(mystery=1.0)
    with pytest.raises(ValueError, match="mystery"):
        write_holdout(undeclared, man, build_params=params,
                      matrix_path=tmp_path / "m2.parquet", manifest_path=tmp_path / "m2.json")
