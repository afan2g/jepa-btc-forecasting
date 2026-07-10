"""CoinAPI snapshot-only seeding experiment harness (issue #54).

EXPERIMENT CODE — deliberately separate from production policy. The partial-day fill
policy (docs/superpowers/plans/2026-07-02-partial-day-fill-policy.md) PROHIBITS
cross-vendor seeding ("The CoinAPI book is never injected into the Lake replay as a
seed"); this module exists to test whether that prohibition can be relaxed by a separate
reviewed policy change. Nothing here is imported by recon/, scripts/run_coinbase_*.py,
or ingest/; a GO verdict authorizes a follow-up PR, never a silent semantic change.

The harness converts a *trusted CoinAPI bootstrap* into the same validated-`BookSnapshot`
currency the §5a-Recon seed/reseed machinery already consumes, so the seeded Lake replay
is byte-for-byte the production replay (`recon.reseed` / `recon.native`) with only the
snapshot SOURCE swapped. Snapshot sources are emulated offline from full-day CoinAPI
`limitbook_full` files we already own — no live vendor calls anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from recon.coinapi import L3Book, _iter_actions
from recon.reseed import BookSnapshot, book_snapshot, classify_snapshot


def _chunks(frame_or_chunks):
    if isinstance(frame_or_chunks, pd.DataFrame):
        return [frame_or_chunks]
    return frame_or_chunks


def snapshots_from_topk_frame(frame: pd.DataFrame, *, max_levels: int,
                              stride_ns: int | None = None
                              ) -> tuple[list[BookSnapshot], dict]:
    """Emulate a Flat Files `limitbook_snapshot_X` day from a reconstructed top-K frame.

    The real product records the top-X levels once per second, but only for seconds
    where the top-X book changed ("recorded every second ... if the order book changed
    in at least one level in the first X best levels at the end of the interval").
    A `reconstruct_coinapi_l2_at_samples` frame on the 1 s grid IS that product's state
    stream (as-of-end-of-interval), so each row becomes a candidate `BookSnapshot` at
    its own `sample_ts` (NaN level pads dropped, never poisoning the candidate).

    Returns `(snapshots, stats)`. `stats["n_changed"]` counts the rows whose top-X
    levels differ from the previous row (first row always counts) — the row count the
    REAL product would store, which is what its file size, and hence its per-GB cost,
    scales with. `stride_ns` optionally thins the emitted candidates (not the stats) to
    at most one per window, mirroring `snapshots_from_lake_book_df`.
    """
    f = frame.sort_values("sample_ts")
    ts = f["sample_ts"].astype("int64").to_numpy()
    cols = []
    for i in range(max_levels):
        cols += [f"bid_{i}_price", f"bid_{i}_size", f"ask_{i}_price", f"ask_{i}_size"]
    missing = [c for c in cols if c not in f.columns]
    if missing:
        raise ValueError(f"top-K frame lacks level columns {missing}; "
                         f"was it built with k >= {max_levels}?")
    arr = f[cols].to_numpy(dtype="float64")
    # changed-vs-previous on the top-X block; NaN pads compare equal to NaN pads.
    if len(arr):
        prev = arr[:-1]
        cur = arr[1:]
        same = np.all((prev == cur) | (np.isnan(prev) & np.isnan(cur)), axis=1)
        n_changed = 1 + int((~same).sum())
    else:
        n_changed = 0
    stats = {"n_samples": int(len(arr)), "n_changed": n_changed,
             "changed_fraction": (float(n_changed / len(arr)) if len(arr) else 0.0),
             "max_levels": int(max_levels)}

    out: list[BookSnapshot] = []
    last_kept: int | None = None
    for r in range(len(arr)):
        t = int(ts[r])
        if stride_ns is not None and last_kept is not None and t - last_kept < stride_ns:
            continue
        row = arr[r]
        bids = [(row[4 * i], row[4 * i + 1]) for i in range(max_levels)
                if isfinite(row[4 * i]) and isfinite(row[4 * i + 1])]
        asks = [(row[4 * i + 2], row[4 * i + 3]) for i in range(max_levels)
                if isfinite(row[4 * i + 2]) and isfinite(row[4 * i + 3])]
        out.append(book_snapshot(t, bids, asks))
        last_kept = t
    return out, stats


@dataclass(frozen=True)
class SnapshotAcceptance:
    """Trust policy for a vendor snapshot candidate before it may seed a Lake replay.

    Extends the production seed gate (`recon.reseed.classify_snapshot`: two-sided,
    finite/positive, deep enough, sorted, uncrossed, sane spread) with the checks a
    CROSS-VENDOR bootstrap additionally needs:

      * causality — a snapshot stamped after the time it was requested for can never be
        used (it would leak future state into the replay);
      * staleness — a snapshot older than `max_age_s` at the requested time is not the
        state we asked for (e.g. a daily-00:00-only product answering an intraday
        request) and is rejected, never silently substituted;
      * tick alignment — every price must be an exact multiple of the venue tick
        (`tick_scale` ticks per $1, e.g. 100 for COINBASE BTC-USD); an off-tick price
        signals unit/venue drift in the snapshot source. `tick_scale=None` skips the
        check (symbol without a verified tick scale).
    """
    min_levels_per_side: int = 5
    max_age_s: float = 60.0
    max_spread_frac: float | None = None
    tick_scale: int | None = 100

    @property
    def max_age_ns(self) -> int:
        return int(self.max_age_s * 1e9)

    def as_dict(self) -> dict:
        return {"min_levels_per_side": int(self.min_levels_per_side),
                "max_age_s": float(self.max_age_s),
                "max_spread_frac": (None if self.max_spread_frac is None
                                    else float(self.max_spread_frac)),
                "tick_scale": (None if self.tick_scale is None else int(self.tick_scale))}


def classify_candidate(snap: BookSnapshot, *, requested_ts: int,
                       policy: SnapshotAcceptance) -> str:
    """Validate a snapshot candidate for seeding; return `"ok"` or a rejection reason.

    Precedence: causality (`"future"`) first — a future-stamped snapshot is a harness
    bug or a lookahead leak and must dominate any structural verdict — then staleness,
    then the production structural checks (`classify_snapshot` reason codes), then tick
    alignment. A non-`"ok"` candidate must NEVER be injected into a replay.
    """
    requested_ts = int(requested_ts)
    if snap.ts > requested_ts:
        return "future"
    if requested_ts - snap.ts > policy.max_age_ns:
        return "stale"
    reason = classify_snapshot(snap, min_levels_per_side=policy.min_levels_per_side,
                               max_spread_frac=policy.max_spread_frac)
    if reason != "ok":
        return reason
    if policy.tick_scale is not None:
        scale = float(policy.tick_scale)
        for p, _ in (*snap.bids, *snap.asks):
            # exact tick multiple: float prices at cent ticks are exactly representable
            # after round(); mirror recon.native's `round(price * scale)` tick mapping.
            if abs(p * scale - round(p * scale)) > 1e-6:
                return "off_tick"
    return "ok"


def coinapi_snapshot_at(chunks, *, day, at_ts: int, max_levels: int | None = None,
                        size_policy: str = "decrement",
                        source: dict | None = None) -> tuple[BookSnapshot, dict]:
    """Extract the CoinAPI L2 book state AS OF `at_ts` from a `limitbook_full` L3 stream.

    This is the offline emulation of "a trusted CoinAPI snapshot at time T": replay the
    L3 events whose label time is <= `at_ts` (the `sample_topk_as_of` as-of convention,
    with the opening SNAPSHOT block label-clamped to the day open exactly as
    `recon.coinapi._iter_actions` does), aggregate to L2 price levels, and return a
    `BookSnapshot` stamped at `at_ts` plus a provenance dict. `max_levels` truncates each
    side to its best-N price levels — the emulation of a depth-capped vendor snapshot
    product (e.g. the REST L2 book's 20-level cap, or Flat Files `limitbook_snapshot_X`).

    The returned snapshot is a CANDIDATE: callers must pass it through
    `classify_candidate` before seeding anything with it.
    """
    book = L3Book(size_policy=size_policy, on_unknown="count")
    day_open_ns = int(pd.Timestamp(day).value)
    at_ts = int(at_ts)
    events_applied = 0
    last_label = None
    for ev in _iter_actions(_chunks(chunks), book, day_open_ns):
        if ev[0] > at_ts:
            break
        book.apply(ev[1], ev[2], ev[3], ev[4], ev[5])
        events_applied += 1
        last_label = ev[0]
    # Full-depth aggregated levels (experiment-scoped read of L3Book internals: the
    # public snapshot(k) is top-K only, and a seed wants the whole side pre-truncation).
    bids = sorted(book._l2.bids.items(), key=lambda x: x[0], reverse=True)
    asks = sorted(book._l2.asks.items(), key=lambda x: x[0])
    levels_available = {"bids": len(bids), "asks": len(asks)}
    if max_levels is not None:
        bids, asks = bids[:max_levels], asks[:max_levels]
    snap = book_snapshot(at_ts, bids, asks)
    prov = {
        "vendor": "coinapi",
        "product": "limitbook_full",
        "method": "l3_replay_as_of",
        "day": str(day),
        "at_ts": at_ts,
        "size_policy": size_policy,
        "max_levels": max_levels,
        "levels_available": levels_available,
        "levels_used": {"bids": len(snap.bids), "asks": len(snap.asks)},
        "events_applied": events_applied,
        "last_event_label_ts": last_label,
        "quality_counters": dict(book.q),
        "source": dict(source or {}),
    }
    return snap, prov
