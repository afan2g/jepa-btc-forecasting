import numpy as np
import pytest

from data.uniqueness import (apply_lookback_cap, concurrency_uniqueness, max_lookback_ns,
                             uniqueness_by_horizon)


def _reference_uniqueness(t0, t1):
    # The eval.synthetic._concurrency_uniqueness reference this module ports; kept inline
    # as the parity oracle so the pinned semantics survive the private copy's removal.
    t0s = np.sort(t0)
    t1s = np.sort(t1)
    started = np.searchsorted(t0s, t0, side="right")
    ended = np.searchsorted(t1s, t0, side="right")
    return 1.0 / np.maximum(started - ended, 1)


def test_known_overlap_fixture():
    # Spans [0,10) [5,15) [10,20) [30,40): coverage at each row's t_event is 1,2,2,1.
    t0 = np.array([0, 5, 10, 30], dtype=np.int64)
    t1 = np.array([10, 15, 20, 40], dtype=np.int64)
    np.testing.assert_array_equal(concurrency_uniqueness(t0, t1),
                                  np.array([1.0, 0.5, 0.5, 1.0]))


def test_boundary_span_ending_at_t_event_does_not_cover():
    # Half-open [t0, t1): a span ending EXACTLY at another row's t_event does not count
    # against it (pinned to the eval.synthetic reference behavior).
    t0 = np.array([0, 10], dtype=np.int64)
    t1 = np.array([10, 20], dtype=np.int64)
    np.testing.assert_array_equal(concurrency_uniqueness(t0, t1), np.array([1.0, 1.0]))
    # ...whereas ending one tick later does cover it.
    t1_covering = np.array([11, 20], dtype=np.int64)
    np.testing.assert_array_equal(concurrency_uniqueness(t0, t1_covering),
                                  np.array([1.0, 0.5]))


def test_matches_reference_on_seeded_valid_inputs():
    rng = np.random.default_rng(7)
    for trial in range(20):
        n = int(rng.integers(1, 200))
        t0 = np.sort(rng.choice(10_000, size=n, replace=False)).astype(np.int64)
        t1 = t0 + rng.integers(1, 500, size=n)
        got = concurrency_uniqueness(t0, t1)
        np.testing.assert_array_equal(got, _reference_uniqueness(t0, t1))


def test_values_finite_in_unit_interval():
    rng = np.random.default_rng(11)
    t0 = np.sort(rng.choice(100_000, size=500, replace=False)).astype(np.int64)
    t1 = t0 + rng.integers(1, 5_000, size=500)
    u = concurrency_uniqueness(t0, t1)
    assert np.isfinite(u).all()
    assert ((u > 0) & (u <= 1)).all()


def test_isolated_row_has_unit_uniqueness():
    t0 = np.array([100], dtype=np.int64)
    t1 = np.array([200], dtype=np.int64)
    np.testing.assert_array_equal(concurrency_uniqueness(t0, t1), np.array([1.0]))


def test_empty_input_yields_empty_output():
    empty = np.array([], dtype=np.int64)
    out = concurrency_uniqueness(empty, empty)
    assert out.shape == (0,)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        concurrency_uniqueness(np.array([0, 5], dtype=np.int64),
                               np.array([10], dtype=np.int64))


def test_rejects_non_1d_input():
    sq = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError):
        concurrency_uniqueness(sq, sq)


def test_rejects_non_finite_times():
    good = np.array([0.0, 5.0])
    for bad in (np.array([0.0, np.nan]), np.array([0.0, np.inf])):
        with pytest.raises(ValueError):
            concurrency_uniqueness(bad, good + 10.0)
        with pytest.raises(ValueError):
            concurrency_uniqueness(good, bad)


def test_rejects_reversed_and_zero_length_spans():
    t0 = np.array([0, 5], dtype=np.int64)
    with pytest.raises(ValueError):                      # reversed: t_barrier < t_event
        concurrency_uniqueness(t0, np.array([10, 4], dtype=np.int64))
    with pytest.raises(ValueError):                      # degenerate: t_barrier == t_event
        concurrency_uniqueness(t0, np.array([10, 5], dtype=np.int64))


def test_rejects_duplicate_t_event():
    # One row per (t_event, horizon) is the matrix contract; duplicates make row
    # identity/weighting ambiguous and must fail closed, not produce plausible weights.
    t0 = np.array([0, 5, 5], dtype=np.int64)
    t1 = np.array([10, 15, 20], dtype=np.int64)
    with pytest.raises(ValueError):
        concurrency_uniqueness(t0, t1)


# ------------------------------------------------------------------ per-horizon API

def _multi_horizon_fixture(n_bars=40, horizons=(("2s", 2_000), ("10s", 10_000),
                                                ("60s", 60_000))):
    """bar x horizon rows on a shared t_event grid, interleaved by bar (NOT grouped by
    horizon) so alignment is actually exercised."""
    step = 1_000
    te_bar = (np.arange(n_bars, dtype=np.int64) + 1) * step
    t0, t1, tag = [], [], []
    for te in te_bar:
        for h_tag, h_ns in horizons:
            t0.append(te)
            t1.append(te + h_ns)
            tag.append(h_tag)
    return (np.array(t0, dtype=np.int64), np.array(t1, dtype=np.int64),
            np.array(tag, dtype=object))


def test_by_horizon_matches_per_subset_core():
    t0, t1, tag = _multi_horizon_fixture()
    u = uniqueness_by_horizon(t0, t1, tag)
    for h in ("2s", "10s", "60s"):
        m = tag == h
        np.testing.assert_array_equal(u[m], concurrency_uniqueness(t0[m], t1[m]))


def test_horizons_never_contribute_to_each_other():
    # Adding another horizon's rows must not change an existing horizon's weights, even
    # though the added spans overlap the existing rows' t_events in time.
    t0, t1, tag = _multi_horizon_fixture(horizons=(("10s", 10_000), ("60s", 60_000)))
    base = uniqueness_by_horizon(t0, t1, tag)
    t0x, t1x, tagx = _multi_horizon_fixture(horizons=(("2s", 2_000), ("10s", 10_000),
                                                      ("60s", 60_000)))
    withx = uniqueness_by_horizon(t0x, t1x, tagx)
    for h in ("10s", "60s"):
        np.testing.assert_array_equal(withx[tagx == h], base[tag == h])


def test_by_horizon_output_alignment_is_positional():
    t0, t1, tag = _multi_horizon_fixture(n_bars=12)
    u = uniqueness_by_horizon(t0, t1, tag)
    for i in range(len(t0)):
        m = tag == tag[i]
        sub = concurrency_uniqueness(t0[m], t1[m])
        assert u[i] == sub[np.where(np.where(m)[0] == i)[0][0]]


def test_permutation_invariance_after_realignment():
    t0, t1, tag = _multi_horizon_fixture()
    base = uniqueness_by_horizon(t0, t1, tag)
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(t0))
    permuted = uniqueness_by_horizon(t0[perm], t1[perm], tag[perm])
    np.testing.assert_array_equal(permuted, base[perm])


def test_same_t_event_across_different_horizons_is_valid():
    # The multi-horizon matrix duplicates every t_event across horizon arms; only a
    # duplicate WITHIN one horizon is an ambiguous key.
    t0 = np.array([100, 100], dtype=np.int64)
    t1 = np.array([200, 700], dtype=np.int64)
    tag = np.array(["10s", "60s"], dtype=object)
    np.testing.assert_array_equal(uniqueness_by_horizon(t0, t1, tag),
                                  np.array([1.0, 1.0]))


def test_by_horizon_rejects_duplicate_key_within_horizon():
    t0 = np.array([100, 100], dtype=np.int64)
    t1 = np.array([200, 300], dtype=np.int64)
    tag = np.array(["10s", "10s"], dtype=object)
    with pytest.raises(ValueError):
        uniqueness_by_horizon(t0, t1, tag)


def test_by_horizon_rejects_malformed_horizons():
    t0 = np.array([0, 5], dtype=np.int64)
    t1 = np.array([10, 15], dtype=np.int64)
    for bad in (np.array(["10s", None], dtype=object),
                np.array(["10s", np.nan], dtype=object),
                np.array(["10s", ""], dtype=object),
                np.array(["10s", 10], dtype=object)):
        with pytest.raises(ValueError):
            uniqueness_by_horizon(t0, t1, bad)


def test_by_horizon_rejects_length_mismatch():
    t0 = np.array([0, 5], dtype=np.int64)
    t1 = np.array([10, 15], dtype=np.int64)
    with pytest.raises(ValueError):
        uniqueness_by_horizon(t0, t1, np.array(["10s"], dtype=object))


def test_by_horizon_values_finite_in_unit_interval():
    t0, t1, tag = _multi_horizon_fixture()
    u = uniqueness_by_horizon(t0, t1, tag)
    assert np.isfinite(u).all()
    assert ((u > 0) & (u <= 1)).all()


# --------------------------------------------------------- embargo / look-back sizing

def test_max_lookback_is_exact_max_over_rows():
    te = np.array([100, 200, 300], dtype=np.int64)
    tfs = np.array([40, 195, 250], dtype=np.int64)     # look-backs 60, 5, 50
    got = max_lookback_ns(te, tfs)
    assert got == 60
    assert isinstance(got, int)                         # manifest/embargo scalar


def test_embargo_never_adds_the_label_horizon():
    # cpcv_splits embargoes from the merged test interval's UPPER bound (max t_barrier),
    # which already includes the label horizon; embargo_ns must equal the feature
    # look-back exactly. With look-back L and a much larger horizon H, the helper must
    # return L — not L + H.
    L, H = 5_000, 60_000_000_000
    te = (np.arange(50, dtype=np.int64) + 1) * 1_000_000
    got = max_lookback_ns(te, te - L)
    assert got == L
    assert got != L + H


def test_max_lookback_never_understates_any_retained_row():
    rng = np.random.default_rng(5)
    te = np.sort(rng.choice(10**9, size=300, replace=False)).astype(np.int64)
    tfs = te - rng.integers(1, 10**6, size=300)
    got = max_lookback_ns(te, tfs)
    assert (te - tfs <= got).all()
    assert got == int((te - tfs).max())


def test_max_lookback_accepts_zero_lookback_rows():
    # t_feature_start == t_event (no look-back observed) is a valid degenerate row.
    te = np.array([10, 20], dtype=np.int64)
    assert max_lookback_ns(te, np.array([10, 5], dtype=np.int64)) == 15


def test_max_lookback_rejects_feature_start_after_event():
    te = np.array([10, 20], dtype=np.int64)
    with pytest.raises(ValueError):
        max_lookback_ns(te, np.array([5, 21], dtype=np.int64))


def test_max_lookback_never_understates_fractional_float_lookback():
    # Float inputs with a fractional-ns look-back must round UP, never truncate: a
    # truncated max (60 for a true 60.9) would under-embargo by the fraction and slip
    # past eval/study.py's identically-truncating cross-check.
    te = np.array([1000.0, 2000.0])
    tfs = np.array([939.1, 1995.0])                     # true look-backs 60.9, 5.0
    got = max_lookback_ns(te, tfs)
    assert got >= (te - tfs).max()
    assert got == 61


def test_max_lookback_rejects_int64_overflow():
    # A pre-epoch t_feature_start near -INT64_MAX with a t_event near +INT64_MAX wraps
    # te - tfs negative; that garbage must fail closed, not become the embargo.
    big = np.iinfo(np.int64).max
    with pytest.raises(ValueError):
        max_lookback_ns(np.array([big], dtype=np.int64),
                        np.array([-big], dtype=np.int64))


def test_rejects_timedelta_and_datetime_dtypes():
    # np.issubdtype(timedelta64, np.number) is True; times must still be plain int/float
    # nanoseconds so malformed inputs (e.g. NaT) fail with an honest dtype error.
    td = np.array([0, 5], dtype="timedelta64[ns]")
    dt = np.array([0, 5], dtype="datetime64[ns]")
    for bad in (td, dt):
        with pytest.raises(ValueError):
            concurrency_uniqueness(bad, bad)
        with pytest.raises(ValueError):
            max_lookback_ns(bad, bad)


def test_max_lookback_rejects_unsigned_wraparound():
    # uint64 inputs must not let t_feature_start > t_event wrap to a huge positive
    # look-back that sails through the negative-look-back check.
    te = np.array([10, 20], dtype=np.uint64)
    tfs = np.array([5, 21], dtype=np.uint64)
    with pytest.raises(ValueError):
        max_lookback_ns(te, tfs)


def test_max_lookback_rejects_empty_and_mismatch_and_nonfinite():
    empty = np.array([], dtype=np.int64)
    with pytest.raises(ValueError):                     # empty: no defined maximum
        max_lookback_ns(empty, empty)
    with pytest.raises(ValueError):
        max_lookback_ns(np.array([10], dtype=np.int64), empty)
    with pytest.raises(ValueError):
        max_lookback_ns(np.array([10.0, np.nan]), np.array([5.0, 5.0]))


# --------------------------------------------------------------- robust look-back cap

def _capped_fixture():
    """Look-backs [60, 5, 50, 600]: one late-received old-origin straggler (600) that a
    raw max would let inflate the embargo across every fold."""
    te = np.array([1_000, 2_000, 3_000, 4_000], dtype=np.int64)
    tfs = te - np.array([60, 5, 50, 600], dtype=np.int64)
    return te, tfs


def test_cap_drops_straggler_and_reports_retained_max():
    te, tfs = _capped_fixture()
    res = apply_lookback_cap(te, tfs, cap_ns=100)
    np.testing.assert_array_equal(res.keep, np.array([True, True, True, False]))
    assert res.n_dropped == 1
    assert res.cap_ns == 100
    # The reported value is the exact retained maximum (60) — not the cap (100), which
    # would overstate, and never anything below 60, which would understate the embargo.
    assert res.retained_max_lookback_ns == 60


def test_cap_never_understates_fractional_float_lookback():
    te = np.array([1000.0, 2000.0])
    tfs = np.array([939.1, 1995.0])                     # true look-backs 60.9, 5.0
    res = apply_lookback_cap(te, tfs, cap_ns=100)
    assert res.retained_max_lookback_ns >= (te - tfs).max()
    assert res.retained_max_lookback_ns == 61


def test_cap_rejects_fractional_cap():
    # Nanosecond caps are integers; a fractional cap cannot be echoed exactly in the
    # integer report and must fail closed rather than truncate.
    te, tfs = _capped_fixture()
    with pytest.raises(ValueError):
        apply_lookback_cap(te, tfs, cap_ns=59.5)
    assert apply_lookback_cap(te, tfs, cap_ns=100.0).cap_ns == 100  # integral float OK


def test_cap_boundary_row_at_cap_is_kept():
    te, tfs = _capped_fixture()
    res = apply_lookback_cap(te, tfs, cap_ns=60)        # row 0's look-back == cap
    np.testing.assert_array_equal(res.keep, np.array([True, True, True, False]))
    assert res.retained_max_lookback_ns == 60


def test_cap_never_clips_t_feature_start():
    te, tfs = _capped_fixture()
    te_before, tfs_before = te.copy(), tfs.copy()
    apply_lookback_cap(te, tfs, cap_ns=100)
    np.testing.assert_array_equal(te, te_before)        # inputs untouched: rows are
    np.testing.assert_array_equal(tfs, tfs_before)      # dropped, never clipped


def test_cap_chains_with_embargo_helper():
    te, tfs = _capped_fixture()
    res = apply_lookback_cap(te, tfs, cap_ns=100)
    assert max_lookback_ns(te[res.keep], tfs[res.keep]) == res.retained_max_lookback_ns


def test_cap_is_deterministic_and_parameterized():
    te, tfs = _capped_fixture()
    a = apply_lookback_cap(te, tfs, cap_ns=100)
    b = apply_lookback_cap(te, tfs, cap_ns=100)
    np.testing.assert_array_equal(a.keep, b.keep)
    assert a[1:] == b[1:]                               # n_dropped, cap, retained max
    wider = apply_lookback_cap(te, tfs, cap_ns=1_000)   # cap is an explicit parameter
    assert wider.n_dropped == 0
    assert wider.retained_max_lookback_ns == 600


def test_cap_rejects_all_rows_dropped():
    te, tfs = _capped_fixture()
    with pytest.raises(ValueError):
        apply_lookback_cap(te, tfs, cap_ns=1)


def test_cap_rejects_invalid_cap():
    te, tfs = _capped_fixture()
    for bad in (0, -5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            apply_lookback_cap(te, tfs, cap_ns=bad)


def test_cap_rejects_malformed_spans():
    te = np.array([10, 20], dtype=np.int64)
    with pytest.raises(ValueError):                     # t_feature_start > t_event
        apply_lookback_cap(te, np.array([5, 21], dtype=np.int64), cap_ns=100)


# ------------------------------------------------------- E0.4 leakage-control gate

def test_leakage_control_random_unpurged_cv_is_inflated():
    # Deterministic E0.4 control (plan §F/§J): on an overlapping-label series, an
    # intentionally leaky evaluation (random folds, no purge, no embargo) must score
    # HIGHER than the purged+embargoed CPCV path. The world is a seeded random walk with
    # y_i = the forward return over [t_event_i, t_barrier_i): adjacent rows share K-1 of
    # K increments, so a 1-nearest-train-neighbor oracle (no model training) recovers y
    # almost exactly WHEN overlapping neighbors leak into train — and collapses to noise
    # once cpcv_splits purges the overlap and embargoes the feature look-back.
    from itertools import combinations
    from data.cv import cpcv_splits

    rng = np.random.default_rng(0)
    n, K, step = 720, 8, 1_000
    H = K * step
    level = np.cumsum(rng.standard_normal(n + K))
    y = level[K:] - level[:-K]
    t_event = (np.arange(n, dtype=np.int64) + 1) * step
    t_barrier = t_event + H
    t_feature_start = t_event - H

    u = concurrency_uniqueness(t_event, t_barrier)      # the overlap the leak rides on
    np.testing.assert_array_equal(u[K - 1:], np.full(n - K + 1, 1.0 / K))

    emb = max_lookback_ns(t_event, t_feature_start)
    assert emb == H                                     # look-back only, no horizon added

    def nn_oos_corr(splits):
        preds, actuals = [], []
        for tr, te_idx in splits:
            tr = np.sort(tr)
            pos = np.searchsorted(t_event[tr], t_event[te_idx])
            left = np.clip(pos - 1, 0, len(tr) - 1)
            right = np.clip(pos, 0, len(tr) - 1)
            nearer_right = (np.abs(t_event[tr][right] - t_event[te_idx])
                            < np.abs(t_event[te_idx] - t_event[tr][left]))
            nn = np.where(nearer_right, right, left)
            preds.append(y[tr[nn]])
            actuals.append(y[te_idx])
        p, a = np.concatenate(preds), np.concatenate(actuals)
        return float(np.corrcoef(p, a)[0, 1])

    purged = nn_oos_corr(cpcv_splits(t_event, t_event, t_barrier,
                                     n_groups=6, k=2, embargo_ns=emb))

    # Matched leaky control: same 6-fold / k=2 combo structure, but folds are random and
    # nothing is purged or embargoed — overlapping neighbors stay in train.
    fold = rng.permutation(n) % 6
    leaky_splits = [(np.where(~np.isin(fold, combo))[0], np.where(np.isin(fold, combo))[0])
                    for combo in combinations(range(6), 2)]
    leaky = nn_oos_corr(leaky_splits)

    assert leaky > 0.6                                  # leak recovers the shared path
    assert abs(purged) < 0.2                            # controls reduce it to noise
    assert leaky - purged > 0.4                         # the inflation the gate detects
