"""Crypto Lake `book_delta_v2` seed/reseed policy (docs/data.md §5a-Recon).

WHY THIS EXISTS — the live 2025-06-01 failure (docs/data.md §5a "Measured results" #2). Lake
`book_delta_v2` is a *mid-stream* incremental L2 feed: absolute size per level, `size==0` removes
it, and there is **no per-day snapshot block**. Cold-starting from an empty book is therefore
invalid for production parity — and worse, it strands levels: when a level's `size=0` clearing
update is missing from the slice we hold, that stale level never clears, best bid/ask freeze and
the book crosses (measured: ~67% of the day crossed, median spread −$60..−$695). A single day-open
seed does not fix intraday stranding; we need *reseed-on-crossing*.

THE POLICY (this module):
  * **Seed** the book from Crypto Lake's `book` (20-level snapshot) product — but only from a
    snapshot we have VALIDATED (two-sided, uncrossed, finite/positive, enough depth). The `book`
    product is itself intermittently crossed on some days (2026-04-01: 31.75% crossed), so a seed
    is never trusted blindly; an invalid candidate is skipped and we fall back to the next valid
    one (or cold-start if none).
  * **Reseed** whenever the reconstructed book stays crossed beyond a tolerance window — replace the
    whole state with the nearest VALID snapshot at/just-after the episode, which drops the stranded
    levels. Reseeds are injected as events at the snapshot's OWN timestamp, so a sample at grid `g`
    only ever reflects a reseed with `ts <= g` — no look-ahead (samples inside the crossed window,
    before the fixing snapshot, stay crossed and are reported as excluded).

WHAT WE DELIBERATELY DO NOT DO — `sequence_number` is NOT a row-gap detector. Coinbase
`book_delta_v2` duplicates `sequence_number` across ~91% of consecutive rows (it is per-event, and
the channel sequence also counts trades), so a naive row-to-row `seq` diff is meaningless as a
"dropped data" signal. The reseed trigger is the *observable* symptom — a crossed reconstructed
book — not a seq discontinuity. `OrderBook.apply()`'s monotonicity return value is informational
only and is never consumed here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np
import pandas as pd

from recon.events import Delta, order_key
from recon.ingest import _pick, _require_populated
from recon.orderbook import OrderBook
from recon.reconstruct import _decode_sides

# Canonical seed/reseed quality reason codes (stable strings for the JSON report).
OK = "ok"


@dataclass(frozen=True)
class BookSnapshot:
    """A full-book L2 snapshot candidate (one Crypto Lake `book` row). `bids`/`asks` are
    `(price, size)` tuples kept sorted (bids descending, asks ascending) so `bids[0]`/`asks[0]`
    are the touch. Build via `book_snapshot(...)`, which normalizes ordering."""
    ts: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


def book_snapshot(ts, bids: Iterable[tuple[float, float]],
                  asks: Iterable[tuple[float, float]]) -> BookSnapshot:
    """Construct a `BookSnapshot`, sorting bids descending / asks ascending by price."""
    b = tuple(sorted(((float(p), float(s)) for p, s in bids), key=lambda x: x[0], reverse=True))
    a = tuple(sorted(((float(p), float(s)) for p, s in asks), key=lambda x: x[0]))
    return BookSnapshot(int(ts), b, a)


def classify_snapshot(snap: BookSnapshot, *, min_levels_per_side: int = 1,
                      max_spread_frac: float | None = None) -> str:
    """Validate a seed candidate; return `"ok"` or a rejection reason code.

    Checks (in precedence order): both sides present, finite & strictly-positive prices/sizes,
    enough depth, strictly-monotonic non-duplicate prices, uncrossed touch, and (optional) a
    sane spread. A non-`"ok"` snapshot must NOT be used as a seed/reseed source."""
    bids, asks = snap.bids, snap.asks
    if not bids or not asks:
        return "one_sided"
    for p, s in (*bids, *asks):
        if not (isfinite(p) and isfinite(s)) or p <= 0.0 or s <= 0.0:
            return "bad_values"
    if len(bids) < min_levels_per_side or len(asks) < min_levels_per_side:
        return "thin_depth"
    if any(bids[i][0] <= bids[i + 1][0] for i in range(len(bids) - 1)) or \
       any(asks[i][0] >= asks[i + 1][0] for i in range(len(asks) - 1)):
        return "unsorted"
    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_bid >= best_ask:
        return "crossed"
    if max_spread_frac is not None:
        mid = (best_bid + best_ask) / 2.0
        if mid > 0 and (best_ask - best_bid) / mid > max_spread_frac:
            return "wide_spread"
    return OK


def snapshots_from_lake_book_df(df: pd.DataFrame, *, engine_time_col: str,
                                max_levels: int | None = None,
                                stride_ns: int | None = None) -> list[BookSnapshot]:
    """Parse a Crypto Lake `book` (snapshot) DataFrame into time-sorted `BookSnapshot`s.

    Reads `bid_i_price/size`, `ask_i_price/size` (NaN-padded levels dropped) and `engine_time_col`
    (origin_time). `stride_ns` thins the stream to at most one candidate per `stride_ns` window —
    essential because the Coinbase `book` product is large; reseeds only need a sparse set of
    validated candidates, not every row. Levels beyond `max_levels` are ignored."""
    _require_populated(df, engine_time_col)
    ts = df[engine_time_col].astype("int64").to_numpy()
    nlev = 0
    while f"bid_{nlev}_price" in df.columns and f"ask_{nlev}_price" in df.columns:
        nlev += 1
    if max_levels is not None:
        nlev = min(nlev, max_levels)
    if nlev == 0:
        raise ValueError(
            f"no bid_i_price/ask_i_price level columns in Lake book frame; cols={list(df.columns)}")
    bp = [df[f"bid_{i}_price"].to_numpy(dtype="float64") for i in range(nlev)]
    bs = [df[f"bid_{i}_size"].to_numpy(dtype="float64") for i in range(nlev)]
    ap = [df[f"ask_{i}_price"].to_numpy(dtype="float64") for i in range(nlev)]
    az = [df[f"ask_{i}_size"].to_numpy(dtype="float64") for i in range(nlev)]
    order = np.argsort(ts, kind="stable")
    ts_sorted = ts[order]

    # Pick the sorted positions to KEEP. Thinning uses np.searchsorted (O(kept·log N)) instead of a
    # per-row Python scan, so a multi-million-row `book` day never costs a 34M-iteration loop — only
    # the ≤(day/stride) kept candidates are materialized into BookSnapshot objects.
    if stride_ns is not None and stride_ns > 0:
        positions: list[int] = []
        i, N = 0, len(ts_sorted)
        while i < N:
            positions.append(i)
            nxt = int(np.searchsorted(ts_sorted, ts_sorted[i] + stride_ns, side="left"))
            i = nxt if nxt > i else i + 1
    else:
        positions = range(len(ts_sorted))  # type: ignore[assignment]

    out: list[BookSnapshot] = []
    for p in positions:
        o = int(order[p])
        # A level is padding (dropped) when EITHER price or size is non-finite — not price alone —
        # so a `(finite_price, NaN_size)` pad does not poison the whole snapshot into `bad_values`
        # and silently disable seeding. Genuine malformed levels (≤0) are still caught by
        # classify_snapshot. Lake pads thin books with NaN; this is robust if it ever pads otherwise.
        bids = [(float(bp[i][o]), float(bs[i][o])) for i in range(nlev)
                if isfinite(bp[i][o]) and isfinite(bs[i][o])]
        asks = [(float(ap[i][o]), float(az[i][o])) for i in range(nlev)
                if isfinite(ap[i][o]) and isfinite(az[i][o])]
        out.append(book_snapshot(int(ts[o]), bids, asks))
    return out


@dataclass(frozen=True)
class ReseedPolicy:
    """Seed/reseed configuration (docs/data.md §5a-Recon).

      * `enabled`       — apply INTRADAY reseed on crossing. The initial day-open seed is applied
                          whenever snapshots are present even if `enabled=False` (so `enabled=False`
                          is the seed-once-only A/B arm: seed, then no crossing repair).
      * `min_levels_per_side` — seed-validity depth floor (rejects thin/broken snapshots).
      * `reseed_after_crossed_s` — reseed only once the book has been crossed CONTINUOUSLY for at
                          least this long, so a transient one-tick cross that self-heals on the next
                          delta does not force a reseed (which would over-rely on the snapshot
                          product and stop testing the delta reconstruction).
      * `max_spread_frac` — optional sane-spread guard for seed validation (off by default)."""
    enabled: bool = True
    min_levels_per_side: int = 1
    reseed_after_crossed_s: float = 2.0
    max_spread_frac: float | None = None

    @property
    def reseed_after_crossed_ns(self) -> int:
        return int(self.reseed_after_crossed_s * 1_000_000_000)

    def as_dict(self) -> dict:
        return {"enabled": bool(self.enabled),
                "min_levels_per_side": int(self.min_levels_per_side),
                "reseed_after_crossed_s": float(self.reseed_after_crossed_s),
                "max_spread_frac": (None if self.max_spread_frac is None
                                    else float(self.max_spread_frac))}


def _merge_time_ordered(deltas: Iterable[Delta], snapshots: list[BookSnapshot]):
    """Merge a (ts,seq)-ordered `Delta` iterable with a ts-sorted snapshot list into one
    non-decreasing event stream of `("delta", Delta)` / `("snap", BookSnapshot)`. At equal ts the
    delta is yielded first and the snapshot second, so a same-ts snapshot (authoritative book
    state) overwrites rather than being overwritten."""
    it = iter(snapshots)
    nxt = next(it, None)
    for d in deltas:
        while nxt is not None and nxt.ts < d.ts_engine:
            yield ("snap", nxt)
            nxt = next(it, None)
        yield ("delta", d)
    while nxt is not None:
        yield ("snap", nxt)
        nxt = next(it, None)


def _replay_seeded(sample_ts, events, *, k: int, policy: ReseedPolicy,
                   collect_frame: bool = True) -> tuple[pd.DataFrame | None, dict]:
    """Replay a merged delta/snapshot stream, applying the seed/reseed policy, and emit the top-K
    book AS OF each `sample_ts` (apply-before-read; identical sampling convention to
    `recon.reconstruct.sample_topk_as_of`). Returns `(frame, metrics)`.

    `collect_frame=False` (metrics-only) skips the per-sample top-K `snapshot()` and the output
    DataFrame build — used by the A/B 'before' arm, which only needs the cold-start crossed rate, so
    a discarded 86,400-row frame is never materialized on a 34M-row day; crossed/missing/thin
    counters still come from the O(1)/best-bid-ask state."""
    ob = OrderBook()
    sample_ts = [int(t) for t in sample_ts]
    n = len(sample_ts)
    rows: list[dict] = []
    si = 0

    crossed_sample_ts: list[int] = []
    crossed_samples = 0
    missing_book_samples = 0
    thin_depth_samples = 0

    seeded = False
    seed_ts: int | None = None
    seed_accepted = False
    seed_reason = "no_snapshots"
    reseed_count = 0
    reseed_ts: list[int] = []
    reseed_blocked = 0
    reason_codes: Counter = Counter()

    crossed_since: int | None = None
    crossed_duration_ns = 0
    last_t: int | None = None

    def emit(g: int) -> None:
        nonlocal crossed_samples, missing_book_samples, thin_depth_samples
        if collect_frame:
            snap = ob.snapshot(k)
            snap["sample_ts"] = int(g)
            rows.append(snap)
        bb, ba = ob.best_bid(), ob.best_ask()
        if bb is None or ba is None:
            missing_book_samples += 1
        elif bb >= ba:
            crossed_samples += 1
            crossed_sample_ts.append(int(g))
        elif len(ob.bids) < k or len(ob.asks) < k:
            thin_depth_samples += 1

    def update_crossed(t: int) -> None:
        # Only account crossing of the ESTABLISHED book — pre-seed cold-start deltas may briefly
        # cross the empty book, but that is warm-up (excluded at the gate), not "established-book
        # crossed time", so crossed_duration_s starts only once a valid seed has landed. This also
        # keeps crossed_since (the reseed trigger) consistent with the seed-first policy.
        nonlocal crossed_since, crossed_duration_ns
        if not seeded:
            return
        bb, ba = ob.best_bid(), ob.best_ask()
        is_crossed = bb is not None and ba is not None and bb >= ba
        if is_crossed:
            if crossed_since is None:
                crossed_since = t
        elif crossed_since is not None:
            crossed_duration_ns += t - crossed_since
            crossed_since = None

    for kind, ev in events:
        t = ev.ts_engine if kind == "delta" else ev.ts
        while si < n and sample_ts[si] < t:
            emit(sample_ts[si])
            si += 1
        if kind == "delta":
            ob.apply(ev)
        else:
            reason = classify_snapshot(ev, min_levels_per_side=policy.min_levels_per_side,
                                       max_spread_frac=policy.max_spread_frac)
            reason_codes[reason] += 1
            usable = reason == OK
            if not seeded:
                if usable:
                    ob.reseed(ev.bids, ev.asks)
                    seeded, seed_ts, seed_accepted, seed_reason = True, t, True, OK
                elif seed_reason == "no_snapshots":
                    seed_reason = reason  # remember the first rejection cause
            elif (policy.enabled and crossed_since is not None
                  and t - crossed_since >= policy.reseed_after_crossed_ns):
                if usable:
                    ob.reseed(ev.bids, ev.asks)
                    reseed_count += 1
                    reseed_ts.append(int(t))
                else:
                    reseed_blocked += 1
        update_crossed(t)
        last_t = t

    while si < n:
        emit(sample_ts[si])
        si += 1
    if crossed_since is not None and last_t is not None and last_t > crossed_since:
        crossed_duration_ns += last_t - crossed_since

    frame = pd.DataFrame(rows) if collect_frame else None
    metrics = {
        "seed_accepted": bool(seed_accepted),
        "seed_ts": (int(seed_ts) if seed_ts is not None else None),
        "seed_reason": seed_reason,
        "reseed_count": int(reseed_count),
        "reseed_ts": reseed_ts,
        "reseed_blocked_invalid_snapshot": int(reseed_blocked),
        "snapshot_reason_codes": dict(reason_codes),
        "n_samples": int(n),
        "crossed_samples": int(crossed_samples),
        "crossed_rate": (float(crossed_samples / n) if n else 0.0),
        "crossed_sample_ts": crossed_sample_ts,
        "excluded_samples": int(crossed_samples),
        "crossed_duration_ns": int(crossed_duration_ns),
        "crossed_duration_s": float(crossed_duration_ns / 1e9),
        "missing_book_samples": int(missing_book_samples),
        "missing_book_fraction": (float(missing_book_samples / n) if n else 0.0),
        "thin_depth_samples": int(thin_depth_samples),
        "thin_depth_fraction": (float(thin_depth_samples / n) if n else 0.0),
        "policy": policy.as_dict(),
    }
    return frame, metrics


def reconstruct_seeded(deltas: Iterable[Delta], sample_ts, *, k: int,
                       snapshots: list[BookSnapshot] | None = None,
                       policy: ReseedPolicy | None = None) -> tuple[pd.DataFrame, dict]:
    """`Delta`-list seed/reseed reconstruction (the hand-checkable analogue of
    `recon.reconstruct.reconstruct_book_at_samples`). Sorts deltas by the canonical `order_key`,
    merges validated snapshots, and replays under `policy`."""
    policy = policy or ReseedPolicy()
    snaps = sorted(snapshots or [], key=lambda s: s.ts)
    ordered = sorted(deltas, key=order_key)
    return _replay_seeded(sample_ts, _merge_time_ordered(iter(ordered), snaps), k=k, policy=policy)


def reconstruct_lake_l2_at_samples_seeded(
        df: pd.DataFrame, sample_ts, *, k: int, engine_time_col: str,
        snapshots: list[BookSnapshot] | None = None,
        policy: ReseedPolicy | None = None,
        frame_out: bool = True) -> tuple[pd.DataFrame | None, dict]:
    """Memory-safe array seed/reseed reconstruction — the seeded counterpart of
    `recon.reconstruct.reconstruct_lake_l2_at_samples`. Resolves the same `book_delta_v2` column
    aliases, lexsorts by `(engine_time, sequence_number)`, feeds a LAZY `Delta` generator (no 34M
    NamedTuple list), and replays it merged with the validated `book` snapshots under `policy`.

    With `snapshots=None` (and no seed) the output frame is byte-identical to the cold-start
    `reconstruct_lake_l2_at_samples` — the A/B "before" arm — so the seed/reseed effect is measured
    against the exact same reconstruction, not a different code path."""
    policy = policy or ReseedPolicy()
    _require_populated(df, engine_time_col)
    seq_col = _pick(df, ("sequence_number", "seq"), field="delta sequence")
    side_col = _pick(df, ("side_is_bid", "side", "is_bid"), field="delta side")
    size_col = _pick(df, ("size", "amount"), field="delta size")
    ts = df[engine_time_col].astype("int64").to_numpy()
    seq = df[seq_col].astype("int64").to_numpy()
    price = df["price"].astype("float64").to_numpy()
    size = df[size_col].astype("float64").to_numpy()
    sides = _decode_sides(df[side_col].to_numpy())
    order = np.lexsort((seq, ts))  # primary key = ts, secondary = seq (matches order_key)
    snaps = sorted(snapshots or [], key=lambda s: s.ts)

    def gen():
        for o in order:
            yield Delta(int(ts[o]), int(seq[o]), sides[o], float(price[o]), float(size[o]))

    return _replay_seeded(sample_ts, _merge_time_ordered(gen(), snaps), k=k, policy=policy,
                          collect_frame=frame_out)
