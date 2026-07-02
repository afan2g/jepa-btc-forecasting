import json

import pandas as pd
import pytest

from eval.manifest import (
    MANIFEST_VERSION,
    feature_list,
    leaky_feature_names,
    load_manifest,
    target_list,
    unsafe_infer_feature_cols,
    validate_frame,
    validate_manifest,
)
from eval.matrix import RESERVED
from eval.runner import run_from_manifest
from eval.synthetic import FEATURES, make_matrix

H_NS = 10_000_000_000  # matches make_matrix's default horizon_ns / max_lookback_ns


def _manifest(**over):
    man = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": "bars_dollar_v1",
        "build_id": "2026-07-01-abc123",
        "bar_clock": {"kind": "dollar", "threshold_usd": 500_000.0, "time_cap_s": 5.0},
        "time": {"unit": "ns", "timezone": "UTC"},
        "feature_cols": list(FEATURES),
        "target_cols": ["y_fwd_bps", "label"],
        "reserved_cols": list(RESERVED),
        "venues": [
            {"exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP", "role": "signal"},
            {"exchange": "COINBASE", "symbol": "BTC-USD", "role": "target"},
        ],
        "horizons": {"10s": H_NS},
        "sources": [{"name": "crypto-lake/book_delta_v2", "exchange": "BINANCE_FUTURES"}],
        "generated_at": "2026-07-01T00:00:00+00:00",
        "max_lookback_ns": H_NS,
        "embargo_ns": H_NS,
    }
    man.update(over)
    return man


def _frame(n=64):
    df, _, _ = make_matrix(n=n, signal_strength=1.0, seed=1)
    return df


# ---------- schema-level: validate_manifest ----------

def test_valid_manifest_passes():
    man = _manifest()
    assert validate_manifest(man) is man  # no raise; returns the manifest for chaining


def test_missing_required_fields_fail():
    man = _manifest(); del man["dataset_id"]
    with pytest.raises(ValueError, match="dataset_id"):
        validate_manifest(man)
    man2 = _manifest(); del man2["feature_cols"]
    with pytest.raises(ValueError, match="feature_cols"):
        validate_manifest(man2)


def test_unknown_manifest_key_rejected():
    # Mirrors resolve_gate: a misspelled key must fail, not silently do nothing.
    with pytest.raises(ValueError, match="unknown manifest keys"):
        validate_manifest(_manifest(feature_colz=["ofi_integrated"]))


def test_unsupported_manifest_version_rejected():
    with pytest.raises(ValueError, match="manifest_version"):
        validate_manifest(_manifest(manifest_version=MANIFEST_VERSION + 1))


def test_duplicate_feature_cols_fail():
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(_manifest(feature_cols=list(FEATURES) + [FEATURES[0]]))


def test_feature_target_overlap_fails():
    # Targets must never be selectable as features.
    with pytest.raises(ValueError, match="target"):
        validate_manifest(_manifest(feature_cols=list(FEATURES) + ["label"]))


def test_feature_reserved_overlap_fails():
    with pytest.raises(ValueError, match="reserved"):
        validate_manifest(_manifest(feature_cols=list(FEATURES) + ["cost_bps"]))


def test_leaky_named_features_rejected():
    # Future-return / forward-mid / barrier-outcome style names are label-derived by
    # construction and can never be valid features, whatever the manifest claims.
    with pytest.raises(ValueError, match="leak"):
        validate_manifest(_manifest(feature_cols=list(FEATURES) + ["mid_fwd_10s"]))
    with pytest.raises(ValueError, match="leak"):
        validate_manifest(_manifest(feature_cols=list(FEATURES) + ["barrier_hit"]))
    with pytest.raises(ValueError, match="leak"):
        validate_manifest(_manifest(feature_cols=list(FEATURES) + ["y_future_ret"]))


def test_target_cols_must_be_reserved():
    with pytest.raises(ValueError, match="target_cols"):
        validate_manifest(_manifest(target_cols=["y_fwd_bps", "my_custom_target"]))


def test_core_non_label_columns_rejected_as_targets():
    # Codex P2: declaring cost/weight/tag/timing columns as "targets" would make them
    # optional on the require_targets=False path, letting frames validate while missing
    # required non-label reserved data.
    for bad in ("cost_bps", "regime", "uniqueness", "t_event"):
        with pytest.raises(ValueError, match="cannot be target_cols"):
            validate_manifest(_manifest(target_cols=["y_fwd_bps", "label", bad]))


def test_reserved_cols_must_include_core_registry():
    trimmed = [c for c in RESERVED if c != "t_available"]
    with pytest.raises(ValueError, match="t_available"):
        validate_manifest(_manifest(reserved_cols=trimmed))


def test_time_metadata_is_strict():
    # v1 supports exactly int64 ns UTC; anything else must fail, not be reinterpreted.
    with pytest.raises(ValueError, match="time.unit"):
        validate_manifest(_manifest(time={"unit": "us", "timezone": "UTC"}))
    with pytest.raises(ValueError, match="time.timezone"):
        validate_manifest(_manifest(time={"unit": "ns", "timezone": "America/New_York"}))


def test_bar_clock_requires_kind():
    with pytest.raises(ValueError, match="bar_clock"):
        validate_manifest(_manifest(bar_clock={"threshold_usd": 500_000.0}))


def test_embargo_must_cover_lookback():
    with pytest.raises(ValueError, match="embargo"):
        validate_manifest(_manifest(embargo_ns=H_NS - 1))


def test_horizons_validated():
    with pytest.raises(ValueError, match="horizon"):
        validate_manifest(_manifest(horizons={}))
    with pytest.raises(ValueError, match="horizon"):
        validate_manifest(_manifest(horizons={"10s": -1}))


def test_venues_validated():
    with pytest.raises(ValueError, match="venue"):
        validate_manifest(_manifest(venues=[]))
    with pytest.raises(ValueError, match="venue"):
        validate_manifest(_manifest(venues=[{"exchange": "COINBASE"}]))  # no symbol
    with pytest.raises(ValueError, match="role"):
        validate_manifest(_manifest(venues=[{"exchange": "COINBASE", "symbol": "BTC-USD",
                                             "role": "speculative"}]))


def test_generated_at_must_be_tz_aware_iso():
    with pytest.raises(ValueError, match="generated_at"):
        validate_manifest(_manifest(generated_at="not-a-date"))
    with pytest.raises(ValueError, match="generated_at"):
        validate_manifest(_manifest(generated_at="2026-07-01T00:00:00"))  # naive


def test_invalid_dtype_expectations_fail():
    with pytest.raises(ValueError, match="dtype"):
        validate_manifest(_manifest(dtypes={"ofi_integrated": "floaty64"}))
    with pytest.raises(ValueError, match="undeclared"):
        validate_manifest(_manifest(dtypes={"no_such_col": "float64"}))


def test_gate_block_is_allowed_passthrough():
    # The study gate stays owned by eval.runner.resolve_gate; the contract only carries it.
    validate_manifest(_manifest(gate={"k": 3}))  # no raise
    with pytest.raises(ValueError, match="gate"):
        validate_manifest(_manifest(gate="loose"))


# ---------- data-level: validate_frame ----------

def test_valid_frame_passes():
    validate_frame(_frame(), _manifest())  # no raise


def test_missing_required_feature_fails():
    with pytest.raises(ValueError, match="cvd"):
        validate_frame(_frame().drop(columns=["cvd"]), _manifest())


def test_missing_target_fails_only_when_targets_required():
    df = _frame().drop(columns=["y_fwd_bps"])
    with pytest.raises(ValueError, match="target"):
        validate_frame(df, _manifest())
    validate_frame(df, _manifest(), require_targets=False)  # unsupervised/JEPA path: no raise


def test_unknown_frame_column_fails_unless_declared_extra():
    df = _frame(); df["debug_flag"] = 1
    with pytest.raises(ValueError, match="extra_cols"):
        validate_frame(df, _manifest())
    validate_frame(df, _manifest(extra_cols=["debug_flag"]))  # explicit opt-in: no raise


def test_duplicate_frame_columns_fail():
    df = _frame()
    dup = pd.concat([df, df[["cvd"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_frame(dup, _manifest())


def test_undeclared_horizon_tag_fails():
    df = _frame(); df.loc[0, "horizon"] = "2s"
    with pytest.raises(ValueError, match="not declared"):
        validate_frame(df, _manifest())


def test_barrier_beyond_declared_horizon_fails():
    # t_barrier may come early (barrier touch) but can never pass the vertical barrier.
    df = _frame(); df.loc[0, "t_barrier"] = df.loc[0, "t_event"] + H_NS + 1
    with pytest.raises(ValueError, match="horizon"):
        validate_frame(df, _manifest())


def test_lookback_beyond_declared_max_fails():
    df = _frame(); df.loc[0, "t_feature_start"] = df.loc[0, "t_event"] - (H_NS + 1)
    with pytest.raises(ValueError, match="look-back"):
        validate_frame(df, _manifest())


def test_availability_violations_fail():
    df = _frame(); df.loc[0, "t_available"] = df.loc[0, "t_event"] - 1
    with pytest.raises(ValueError, match="t_available"):
        validate_frame(df, _manifest())
    df2 = _frame(); df2.loc[0, "t_available"] = df2.loc[0, "t_event"] + 1
    with pytest.raises(ValueError, match="availability"):
        validate_frame(df2, _manifest())          # default lag is 0: must be synchronous
    validate_frame(df2, _manifest(availability_lag_ns=5))  # declared lag covers it: no raise


def test_as_of_violation_fails():
    df = _frame()
    latest = int(df["t_available"].max())
    with pytest.raises(ValueError, match="as_of"):
        validate_frame(df, _manifest(as_of_ns=latest - 1))
    validate_frame(df, _manifest(as_of_ns=latest))  # no raise


def test_dtype_mismatch_fails():
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_frame(_frame(), _manifest(dtypes={"cvd": "int64"}))  # actual is float64
    validate_frame(_frame(), _manifest(dtypes={"cvd": "float64", "t_event": "int64"}))


# ---------- extraction and inference ----------

def test_feature_list_returns_exactly_manifest_features_in_order():
    rev = list(reversed(FEATURES))
    man = _manifest(feature_cols=rev)
    feats = feature_list(man)
    assert feats == rev                       # exact content AND order, LightGBM-ready
    feats.append("mutant")
    assert man["feature_cols"] == rev         # a copy: callers cannot mutate the manifest


def test_target_list_returns_manifest_targets():
    assert target_list(_manifest()) == ["y_fwd_bps", "label"]


def test_unsafe_infer_rejects_leaky_columns_by_default():
    df = _frame(); df["mid_fwd_5s"] = 0.0
    with pytest.raises(ValueError, match="leak"):
        unsafe_infer_feature_cols(df)


def test_unsafe_infer_returns_non_reserved_in_frame_order():
    inferred = unsafe_infer_feature_cols(_frame())
    assert inferred == list(FEATURES)  # everything else in make_matrix output is reserved


# ---------- persistence and runner integration ----------

def test_load_manifest_roundtrip_and_validation(tmp_path):
    p = tmp_path / "feature_manifest.json"
    p.write_text(json.dumps(_manifest()))
    man = load_manifest(p)
    assert man["feature_cols"] == list(FEATURES) and man["horizons"]["10s"] == H_NS
    bad = _manifest(); del bad["horizons"]
    p2 = tmp_path / "bad.json"
    p2.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="horizons"):
        load_manifest(p2)


def test_run_from_manifest_rejects_leaky_versioned_manifest():
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    man = _manifest(feature_cols=feats + ["label"], gate={"n_groups": 4, "k": 2})
    with pytest.raises(ValueError, match="target"):
        run_from_manifest(m, man)


def test_run_from_manifest_accepts_valid_versioned_manifest():
    m, feats, lb = make_matrix(n=900, signal_strength=4.0, seed=8)
    man = _manifest(feature_cols=feats, max_lookback_ns=lb, embargo_ns=lb,
                    gate={"n_groups": 4, "k": 2, "min_trades": 1, "min_eff_trades": 1.0})
    res = run_from_manifest(m, man)
    assert res["gate"]["k"] == 2 and "10s" in res["horizons"]


def test_run_from_manifest_refuses_v1_fields_without_version():
    # A typo in 'manifest_version' itself must not silently downgrade a rich manifest
    # to the unvalidated legacy path.
    m, feats, _ = make_matrix(n=64, signal_strength=1.0, seed=1)
    man = _manifest(feature_cols=feats, gate={"n_groups": 4, "k": 2})
    del man["manifest_version"]
    with pytest.raises(ValueError, match="manifest_version"):
        run_from_manifest(m, man)


def test_run_from_manifest_refuses_optional_v1_fields_without_version():
    # Codex P2: optional v1 fields (dtypes/as_of_ns/availability_lag_ns/extra_cols) on a
    # legacy-shaped manifest must also fail closed, not be silently ignored.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    man = {"feature_cols": feats, "embargo_ns": lb, "max_lookback_ns": lb,
           "gate": {"n_groups": 4, "k": 2}, "dtypes": {"cvd": "float64"}}
    with pytest.raises(ValueError, match="manifest_version"):
        run_from_manifest(m, man)


def test_run_from_manifest_categorical_horizon_with_unused_categories():
    # Codex P2: validate_frame accepts unused horizon categories (observed=True), so the
    # runner's own groupby must not hand run_study an empty subframe for "2s".
    m, feats, lb = make_matrix(n=900, signal_strength=4.0, seed=8)
    m["horizon"] = pd.Categorical(m["horizon"], categories=["10s", "2s"])
    man = _manifest(feature_cols=feats, max_lookback_ns=lb, embargo_ns=lb,
                    gate={"n_groups": 4, "k": 2, "min_trades": 1, "min_eff_trades": 1.0})
    res = run_from_manifest(m, man)
    assert set(res["horizons"]) == {"10s"}


# ---------- schema guard pins (review findings: deletable checks) ----------

def test_scalar_schema_guards_fail_closed():
    with pytest.raises(ValueError, match="manifest must be a dict"):
        validate_manifest(["not", "a", "dict"])
    with pytest.raises(ValueError, match="manifest_version"):
        validate_manifest(_manifest(manifest_version=True))  # bool is not a version
    with pytest.raises(ValueError, match="dataset_id"):
        validate_manifest(_manifest(dataset_id=""))
    with pytest.raises(ValueError, match="build_id"):
        validate_manifest(_manifest(build_id=1))
    with pytest.raises(ValueError, match="time"):
        validate_manifest(_manifest(time=None))
    with pytest.raises(ValueError, match="unknown time keys"):
        validate_manifest(_manifest(time={"unit": "ns", "timezone": "UTC", "cal": "utc"}))
    with pytest.raises(ValueError, match="sources"):
        validate_manifest(_manifest(sources=[]))
    with pytest.raises(ValueError, match="sources"):
        validate_manifest(_manifest(sources=[{"table": "book_delta_v2"}]))  # no 'name'
    with pytest.raises(ValueError, match="dtypes"):
        validate_manifest(_manifest(dtypes="float64"))
    with pytest.raises(ValueError, match="as_of_ns"):
        validate_manifest(_manifest(as_of_ns="latest"))


def test_ns_fields_must_be_nonnegative_ints():
    with pytest.raises(ValueError, match="max_lookback_ns"):
        validate_manifest(_manifest(max_lookback_ns=-1))
    with pytest.raises(ValueError, match="max_lookback_ns"):
        validate_manifest(_manifest(max_lookback_ns=True))
    with pytest.raises(ValueError, match="embargo_ns"):
        validate_manifest(_manifest(embargo_ns="big"))
    with pytest.raises(ValueError, match="availability_lag_ns"):
        validate_manifest(_manifest(availability_lag_ns=-5))


def test_column_list_shapes_validated():
    with pytest.raises(ValueError, match="feature_cols"):
        validate_manifest(_manifest(feature_cols="ofi_integrated"))  # str, not list
    with pytest.raises(ValueError, match="feature_cols"):
        validate_manifest(_manifest(feature_cols=[]))
    with pytest.raises(ValueError, match="target_cols"):
        validate_manifest(_manifest(target_cols=[1]))


def test_every_leaky_pattern_is_enforced():
    # Pins the full deny-list: deleting any one pattern (or the y/y_* prefix rule)
    # must break this test.
    for bad in ["x_future", "forward_ret", "px_fwd", "hit_barrier", "label_side",
                "target_up", "outcome_z", "y_hat", "y"]:
        with pytest.raises(ValueError, match="leak"):
            validate_manifest(_manifest(feature_cols=list(FEATURES) + [bad]))


def test_extra_cols_overlap_fails():
    with pytest.raises(ValueError, match="extra_cols"):
        validate_manifest(_manifest(extra_cols=["cvd"]))     # already a feature
    with pytest.raises(ValueError, match="extra_cols"):
        validate_manifest(_manifest(extra_cols=["regime"]))  # already reserved


def test_feature_and_target_list_validate_first():
    man = _manifest(); del man["horizons"]
    with pytest.raises(ValueError, match="horizons"):
        feature_list(man)
    with pytest.raises(ValueError, match="horizons"):
        target_list(man)


# ---------- frame guard pins and hardening (review findings) ----------

def test_frame_missing_reserved_column_fails():
    with pytest.raises(ValueError, match="regime"):
        validate_frame(_frame().drop(columns=["regime"]), _manifest())


def test_frame_timing_order_violations_fail():
    df = _frame(); df.loc[0, "t_barrier"] = df.loc[0, "t_event"] - 1
    with pytest.raises(ValueError, match="t_barrier >= t_event"):
        validate_frame(df, _manifest())
    df2 = _frame(); df2.loc[0, "t_feature_start"] = df2.loc[0, "t_event"] + 1
    with pytest.raises(ValueError, match="t_feature_start <= t_event"):
        validate_frame(df2, _manifest())


def test_null_timing_values_fail_closed():
    # Nullable Int64 + pd.NA must not vacuously pass the timing comparisons
    # (Series.all() skips NA), silently waving through rows with unknown timing.
    df = _frame()
    df["t_barrier"] = df["t_barrier"].astype("Int64")
    df.loc[0, "t_barrier"] = pd.NA
    with pytest.raises(ValueError, match="null"):
        validate_frame(df, _manifest())


def test_non_integer_timing_dtype_fails_clearly():
    df = _frame()
    df["t_event"] = pd.to_datetime(df["t_event"])  # datetime64, not int ns
    with pytest.raises(ValueError, match="integer"):
        validate_frame(df, _manifest())


def test_empty_frame_fails_clearly():
    with pytest.raises(ValueError, match="empty"):
        validate_frame(_frame().iloc[0:0], _manifest())


def test_categorical_horizon_with_unused_categories_validates():
    # pandas groupby(observed=False) yields empty groups for unused categories; the
    # validator must not KeyError on a tag that never occurs in the frame.
    df = _frame()
    df["horizon"] = pd.Categorical(df["horizon"], categories=["10s", "2s"])
    validate_frame(df, _manifest())  # no raise


def test_non_string_horizon_tags_fail():
    df = _frame(); df["horizon"] = 10
    with pytest.raises(ValueError, match="strings"):
        validate_frame(df, _manifest())


def test_structural_columns_required_even_without_targets():
    # Declaring a timing column as a target must not let it vanish from the frame on the
    # require_targets=False path. Today the schema check inside validate_frame refuses the
    # manifest outright (core non-label columns cannot be targets); the _STRUCTURAL_COLS
    # guard in the presence loop stays as defense-in-depth behind it.
    man = _manifest(target_cols=["y_fwd_bps", "label", "t_event"])
    df = _frame().drop(columns=["t_event"])
    with pytest.raises(ValueError, match="t_event"):
        validate_frame(df, man, require_targets=False)


def test_unsafe_infer_param_hygiene():
    df = _frame()
    with pytest.raises(ValueError, match="string"):
        unsafe_infer_feature_cols(df, extra_cols="cvd")  # a str would be split to chars
    assert unsafe_infer_feature_cols(df, extra_cols=("cvd",)) == \
        [c for c in FEATURES if c != "cvd"]
    df2 = _frame(); df2["mid_fwd_5s"] = 0.0
    assert "mid_fwd_5s" not in unsafe_infer_feature_cols(df2, extra_cols=("mid_fwd_5s",))
    dup = pd.concat([df, df[["cvd"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        unsafe_infer_feature_cols(dup)
    bad = df.copy(); bad[42] = 1.0
    with pytest.raises(ValueError, match="strings"):
        unsafe_infer_feature_cols(bad)


def test_leaky_feature_names_public_helper():
    # Public wrapper for the runner's legacy-branch screen (and manifest-authoring tools).
    assert leaky_feature_names(["ofi_integrated", "mid_fwd_5s", "y", "spread_tick"]) == \
        ["mid_fwd_5s", "y"]
