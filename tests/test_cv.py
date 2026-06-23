import numpy as np
from math import comb
from data.cv import make_time_groups, cpcv_splits


def _spans(n, span=10):
    t0 = (np.arange(n) * 5).astype(np.int64)
    return t0, (t0 + span).astype(np.int64)


def test_groups_balanced():
    t0, _ = _spans(120)
    g = make_time_groups(t0, n_groups=6)
    assert set(g.tolist()) == set(range(6))


def test_path_count_equals_n_choose_k():
    t0, t1 = _spans(120)
    assert len(list(cpcv_splits(t0, t0, t1, n_groups=6, k=2, embargo_ns=0))) == comb(6, 2)


def test_no_train_span_overlaps_any_test_span():
    t0, t1 = _spans(120)
    for tr, te in cpcv_splits(t0, t0, t1, n_groups=6, k=2, embargo_ns=0):
        for j in tr:
            assert not ((t0[j] <= t1[te]) & (t1[j] >= t0[te])).any()


def test_noncontiguous_combo_keeps_substantial_train():
    # THE rev-2 guard: union-span purge would empty this; per-span purge must not.
    t0, t1 = _spans(120)
    splits = list(cpcv_splits(t0, t0, t1, n_groups=6, k=2, embargo_ns=0))
    for tr, te in splits:
        assert len(tr) >= 40          # ~ (6-2)/6 of 120 minus purge halo, never empty


def test_embargo_drops_post_test_window():
    t0, t1 = _spans(120)
    emb = 50
    for tr, te in cpcv_splits(t0, t0, t1, n_groups=6, k=1, embargo_ns=emb):
        hi = t1[te].max()
        assert not ((t0[tr] > hi) & (t0[tr] <= hi + emb)).any()
