"""Offline tests: the Stage-2 seed-source crossed-rate gate (plan Requirement 5 / Task 7 Step 5).

Drives the PURE per-day core of `scripts/run_binance_recon.py` with synthetic Binance-shaped
`book_delta_v2` deltas + `book` seed frames — no vendor I/O, no parquet. The contract mirrors the
Coinbase quality-map (`run_coinbase_quality_map.classify_day`): a day whose thinned `book` seed
candidates are >5% crossed is `inconclusive` and emits NO certified top-K frame, EVEN when an
individually-valid snapshot is accepted — an accepted seed off an unreliable source stayed crossed
for hours on Coinbase (docs §5a-QualityMap). `seed_source_crossed_frac` is recorded so a flaky
source can never silently certify an output, and the gate runs BEFORE the expensive replay
(fail-fast) with its candidate classifications pinned equal to the replay's own reason codes.
"""
import datetime as dt
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from recon.reseed import ReseedPolicy, reconstruct_lake_l2_at_samples_seeded

# scripts/ is not a package — load the runner by path (same pattern as test_quality_map).
_SPEC = importlib.util.spec_from_file_location(
    "run_binance_recon",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_binance_recon.py")
rbr = importlib.util.module_from_spec(_SPEC)
sys.modules["run_binance_recon"] = rbr
_SPEC.loader.exec_module(rbr)

NS = 1_000_000_000
DAY = dt.date(2026, 4, 1)
DAY_OPEN = int(pd.Timestamp("2026-04-01").value)
GRID_MS = 3_600_000        # hourly grid -> 24 samples (fast synthetic replay)
POLICY = ReseedPolicy(min_levels_per_side=1)


def _delta_df(rows):
    """Binance-shaped book_delta_v2 frame from (sec_offset, seq, is_bid, price, size)."""
    return pd.DataFrame(
        [(DAY_OPEN + int(s * NS), q, b, p, z) for s, q, b, p, z in rows],
        columns=["origin_time", "sequence_number", "side_is_bid", "price", "size"])


def _book_df(candidates):
    """2-level `book` seed frame from (sec_offset, bid0, ask0) — prices on the $0.10 perp grid."""
    rows = []
    for s, bid0, ask0 in candidates:
        rows.append({"origin_time": DAY_OPEN + int(s * NS),
                     "received_time": DAY_OPEN + int(s * NS),
                     "bid_0_price": bid0, "bid_0_size": 1.0,
                     "bid_1_price": bid0 - 0.1, "bid_1_size": 2.0,
                     "ask_0_price": ask0, "ask_0_size": 1.0,
                     "ask_1_price": ask0 + 0.1, "ask_1_size": 2.0})
    return pd.DataFrame(rows)


def _clean_deltas():
    """Deltas that keep the seeded book two-sided and uncrossed all day."""
    return _delta_df([(1, 1, True, 100.0, 2.0), (2, 2, False, 100.1, 2.0),
                      (3, 3, True, 99.9, 1.0), (4, 4, False, 100.2, 1.0)])


def _candidates(n, n_crossed):
    """`n` seed candidates (one per second from the day open) with the FIRST one valid — so a seed
    IS accepted — and `n_crossed` crossed ones after it (bid 100.2 > ask 100.1)."""
    out = [(0, 100.0, 100.1)]
    for i in range(1, n):
        out.append((i, 100.2, 100.1) if i <= n_crossed else (i, 100.0, 100.1))
    return out


def _run(delta_df, book_df, **over):
    kw = dict(day=DAY, k=2, grid_ms=GRID_MS, engine="python", price_scale=None,
              policy=POLICY, book_stride_ms=1000, thresholds=rbr.Thresholds())
    kw.update(over)
    return rbr.recon_topk_day(delta_df, book_df, **kw)


# --------------------------------------------------------------------------- the gate
def test_over_5pct_crossed_source_is_inconclusive_with_no_certified_output():
    # 3/40 = 7.5% crossed candidates, yet the FIRST candidate is valid and would seed the day.
    frame, info = _run(_clean_deltas(), _book_df(_candidates(40, 3)))
    assert frame is None                                        # no certified top-K, fail closed
    assert info["classification"] == rbr.INCONCLUSIVE
    assert rbr.SEED_SOURCE_UNRELIABLE in info["reasons"]
    assert info["seed_source_crossed_frac"] == pytest.approx(3 / 40)
    assert info["seed"]["seed_accepted"] is True                # accepted seed does NOT rescue it
    assert info["gated_before_replay"] is True                  # fail-fast: replay never ran
    assert info["quality"] is None                              # no replay metrics to report


def test_clean_source_certifies_and_emits_exact_topk_frame():
    frame, info = _run(_clean_deltas(), _book_df(_candidates(40, 1)))   # 2.5% crossed
    assert info["classification"] == rbr.CERTIFIED
    assert info["seed_source_crossed_frac"] == pytest.approx(1 / 40)
    assert info["gated_before_replay"] is False
    assert info["quality"]["crossed_rate"] == 0.0
    # Exact top-K contract: mid, microprice, per-level bid/ask price/size, sample_ts LAST (k=2).
    assert list(frame.columns) == [
        "mid", "microprice",
        "bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size",
        "bid_1_price", "bid_1_size", "ask_1_price", "ask_1_size",
        "sample_ts"]
    assert len(frame) == 24                                     # hourly grid


def test_certified_frame_equals_python_oracle():
    # The runner must REUSE the engine, not reimplement reconstruction semantics: its certified
    # frame is byte-identical to a direct oracle call on the same inputs.
    book = _book_df(_candidates(10, 0))
    frame, info = _run(_clean_deltas(), book)
    from recon.reseed import snapshots_from_lake_book_df
    snaps = snapshots_from_lake_book_df(book, engine_time_col=info["engine_time_col"],
                                        max_levels=20, stride_ns=1000 * 1_000_000)
    grid = rbr.build_grid(DAY, GRID_MS)
    expect, _ = reconstruct_lake_l2_at_samples_seeded(
        _clean_deltas(), grid, k=2, engine_time_col=info["engine_time_col"],
        snapshots=snaps, policy=POLICY, frame_out=True)
    pd.testing.assert_frame_equal(frame, expect, check_dtype=True)


def test_exactly_5pct_crossed_is_not_gated():
    frame, info = _run(_clean_deltas(), _book_df(_candidates(40, 2)))   # 2/40 = 5.0%, bar is >5%
    assert info["classification"] == rbr.CERTIFIED
    assert frame is not None


# --------------------------------------------------------------------------- no-seed / cold start
def test_missing_seed_product_is_inconclusive_without_replay():
    frame, info = _run(_clean_deltas(), None)
    assert frame is None
    assert info["classification"] == rbr.INCONCLUSIVE
    assert "no_seed_snapshots" in info["reasons"]
    assert info["seed"]["seed_accepted"] is False
    assert info["gated_before_replay"] is True


def test_all_rejected_seed_candidates_are_inconclusive():
    # Every candidate crossed -> no valid seed; the day must never emit a silently-bad book.
    frame, info = _run(_clean_deltas(), _book_df([(i, 100.2, 100.1) for i in range(5)]))
    assert frame is None
    assert info["classification"] == rbr.INCONCLUSIVE
    assert any(r.startswith("seed_rejected:") for r in info["reasons"])
    assert info["seed"]["seed_accepted"] is False


# --------------------------------------------------------------------------- degraded fail-closed
def test_degraded_day_emits_no_certified_output_but_reports_metrics():
    # Clean seed source, but a delta strands a crossing bid all day (no snapshot to reseed from)
    # -> crossed_rate >> 1% usable bar -> degraded: metrics recorded, NO output frame published.
    deltas = _delta_df([(1, 1, True, 100.5, 1.0)])              # bid 100.5 > seeded ask 100.1
    frame, info = _run(deltas, _book_df([(0, 100.0, 100.1)]))
    assert frame is None
    assert info["classification"] == rbr.DEGRADED
    assert info["quality"]["crossed_rate"] > rbr.Thresholds().crossed_usable_max
    assert info["gated_before_replay"] is False
    assert any("crossed_rate" in r for r in info["reasons"])


# --------------------------------------------------------------------------- gate/replay conformance
def test_pre_gate_reason_codes_match_replay_reason_codes():
    # The fail-fast gate classifies candidates BEFORE replay; its counts must equal the replay's
    # own meta["snapshot_reason_codes"] on the same candidates + policy (same classify_snapshot).
    from recon.reseed import snapshots_from_lake_book_df
    book = _book_df(_candidates(20, 4))
    snaps = snapshots_from_lake_book_df(book, engine_time_col="origin_time",
                                        max_levels=20, stride_ns=1000 * 1_000_000)
    pre = rbr.preclassify_snapshots(snaps, POLICY)
    grid = rbr.build_grid(DAY, GRID_MS)
    _, meta = reconstruct_lake_l2_at_samples_seeded(
        _clean_deltas(), grid, k=2, engine_time_col="origin_time",
        snapshots=snaps, policy=POLICY, frame_out=False)
    assert pre["reason_codes"] == meta["snapshot_reason_codes"]
    assert pre["seed_accepted"] == meta["seed_accepted"]
    assert pre["seed_reason"] == meta["seed_reason"]
    assert pre["seed_ts"] == meta["seed_ts"]


# --------------------------------------------------------------------------- joint engine time
def test_joint_engine_time_recorded_and_shared_with_seed_frame():
    # A partial-origin delta frame forces BOTH frames onto received_time (one shared clock,
    # plan Requirement 4); the choice and the fallback flag are recorded for the manifest.
    deltas = _clean_deltas()
    deltas["received_time"] = deltas["origin_time"]
    deltas.loc[0, "origin_time"] = 0                            # 1 unpopulated origin row
    n = 400                                                     # keep origin_time >99% populated
    deltas = pd.concat([deltas] + [
        _delta_df([(10 + i, 100 + i, True, 100.0, 2.0)]).assign(
            received_time=lambda d: d["origin_time"]) for i in range(n)],
        ignore_index=True)
    frame, info = _run(deltas, _book_df(_candidates(10, 0)))
    assert info["engine_time_col"] == "received_time"
    assert info["engine_time_fallback"] is True
    assert info["classification"] == rbr.CERTIFIED
