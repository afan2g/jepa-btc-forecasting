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

import hashlib
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from eval.hashing import hash_obj
from recon.coinapi import L3Book, _iter_actions
from recon.ingest import shared_engine_time_col
from recon.reseed import (BookSnapshot, ReseedPolicy, book_snapshot, classify_snapshot,
                          reconstruct_lake_l2_at_samples_seeded)


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


def frame_replay_hash(frame: pd.DataFrame | None) -> str | None:
    """Deterministic content hash of a reconstructed top-K frame (the replay hash).

    Rows in `sample_ts` order, columns in a fixed canonical order (`sample_ts` first,
    the rest sorted by name); numeric buffers hashed as int64/float64 bytes so the hash
    is a function of logical content, not file bytes or column insertion order. Two
    replays of the same inputs must produce the same hash — the determinism invariant
    every arm report pins.
    """
    if frame is None:
        return None
    f = frame.sort_values("sample_ts").reset_index(drop=True)
    cols = ["sample_ts"] + sorted(c for c in f.columns if c != "sample_ts")
    h = hashlib.sha256()
    for c in cols:
        h.update(c.encode())
        h.update(b"\x00")
        if c == "sample_ts":
            h.update(np.ascontiguousarray(f[c].to_numpy(np.int64)).tobytes())
        else:
            h.update(np.ascontiguousarray(f[c].to_numpy(np.float64)).tobytes())
        h.update(b"\x00")
    return h.hexdigest()


def seed_lake_replay(lake_df: pd.DataFrame, candidates, *, grid, k: int,
                     acceptance: SnapshotAcceptance, reseed: bool = True,
                     reseed_after_crossed_s: float = 2.0,
                     engine: str = "python", price_scale: int | None = None,
                     engine_time_col: str | None = None,
                     frame_out: bool = True) -> tuple[pd.DataFrame | None, dict]:
    """Seed/reseed the PRODUCTION Lake `book_delta_v2` replay from vendor snapshot
    candidates, with the cross-vendor acceptance gate applied up front.

    `candidates` is a list of `(BookSnapshot, provenance_dict)`; the requested time of
    each candidate is `provenance["at_ts"]` when present (an extracted/requested
    snapshot), else the snapshot's own stamp (a streamed candidate). Candidates failing
    `classify_candidate` are recorded in the rejection ledger and NEVER injected;
    accepted ones are handed unmodified to the production seeded replay
    (`recon.reseed.reconstruct_lake_l2_at_samples_seeded`, or its native twin), which
    re-validates them structurally — the experiment swaps only the snapshot SOURCE,
    never the replay semantics. With zero accepted candidates the result is
    byte-identical to the production cold start.

    Returns `(frame, meta)`: the production replay meta plus the acceptance ledger,
    `frame_hash` (replay hash) and `report_hash` (canonical-JSON meta hash).
    """
    ledger: dict = {"n_total": len(candidates), "n_accepted": 0,
                    "accepted": [], "rejected": []}
    accepted: list[BookSnapshot] = []
    for snap, prov in candidates:
        requested_ts = int(prov.get("at_ts", snap.ts))
        reason = classify_candidate(snap, requested_ts=requested_ts, policy=acceptance)
        entry = {"ts": int(snap.ts), "requested_ts": requested_ts,
                 "levels": {"bids": len(snap.bids), "asks": len(snap.asks)},
                 "provenance": dict(prov)}
        if reason == "ok":
            accepted.append(snap)
            ledger["accepted"].append(entry)
        else:
            ledger["rejected"].append({**entry, "reason": reason})
    ledger["n_accepted"] = len(accepted)

    policy = ReseedPolicy(enabled=reseed,
                          min_levels_per_side=acceptance.min_levels_per_side,
                          reseed_after_crossed_s=reseed_after_crossed_s,
                          max_spread_frac=acceptance.max_spread_frac)
    etc = engine_time_col or shared_engine_time_col(lake_df)
    if engine == "native":
        from recon import native as _native
        frame, meta = _native.reconstruct_lake_l2_at_samples_seeded_native(
            lake_df, grid, k=k, engine_time_col=etc, snapshots=accepted or None,
            policy=policy, frame_out=frame_out, price_scale=price_scale)
    else:
        frame, meta = reconstruct_lake_l2_at_samples_seeded(
            lake_df, grid, k=k, engine_time_col=etc, snapshots=accepted or None,
            policy=policy, frame_out=frame_out)
    meta = dict(meta)
    meta["engine"] = engine
    meta["engine_time_col"] = etc
    meta["acceptance"] = acceptance.as_dict()
    meta["candidates"] = ledger
    meta["frame_hash"] = frame_replay_hash(frame)
    meta["report_hash"] = hash_obj(meta, exclude_keys=("report_hash",))
    return frame, meta


def _sustained_cross_trigger(frame: pd.DataFrame, *, trigger_ns: int,
                             after_ts: int | None) -> int | None:
    """First causally-observable reseed trigger in a reconstructed frame.

    A trigger is `first_crossed_sample_ts + trigger_ns` for a run of CONSECUTIVE
    crossed grid samples that is still crossed at the trigger time (a transient cross
    that self-heals inside the window never triggers — the production
    `reseed_after_crossed_s` semantics at grid resolution). Only triggers strictly
    after `after_ts` qualify. Uses nothing later than the trigger time itself except
    run persistence, which a live requester would observe by simply waiting.
    """
    f = frame.sort_values("sample_ts")
    ts = f["sample_ts"].astype("int64").to_numpy()
    bid = f["bid_0_price"].to_numpy(dtype="float64")
    ask = f["ask_0_price"].to_numpy(dtype="float64")
    crossed = np.isfinite(bid) & np.isfinite(ask) & (bid >= ask)
    i, n = 0, len(ts)
    while i < n:
        if not crossed[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and crossed[j + 1]:
            j += 1
        trig = int(ts[i]) + int(trigger_ns)
        if (after_ts is None or trig > after_ts) and int(ts[j]) >= trig:
            return trig
        i = j + 1
    return None


def on_demand_reseed_arm(lake_df: pd.DataFrame, provider, *, grid, k: int,
                         acceptance: SnapshotAcceptance,
                         trigger_after_crossed_s: float = 2.0, max_requests: int = 24,
                         engine: str = "python", price_scale: int | None = None,
                         engine_time_col: str | None = None
                         ) -> tuple[pd.DataFrame | None, dict]:
    """The ON-DEMAND strategy: request a vendor snapshot only when the Lake replay is
    observably broken (book crossed continuously past the trigger window), exactly when
    a live operator could have requested one.

    `provider(requested_ts) -> (BookSnapshot, provenance)` emulates the vendor (offline:
    an L3 as-of extraction from a full-day file we already own). Iterative fixed point:
    replay with the snapshots injected so far, find the first sustained-crossing trigger
    after the last injection, request a snapshot at that trigger, inject it if accepted,
    repeat. Each iteration's trigger uses only state observable at the trigger time, so
    the request sequence is exactly what a causal live system would have produced; the
    request count is the arm's per-day vendor request cost.

    Terminates on: no remaining trigger (`no_trigger`), the request budget
    (`max_requests`), or a rejected/ineffective snapshot at a recurring trigger
    (`no_progress` — never loops on a vendor that cannot help).
    """
    trigger_ns = int(trigger_after_crossed_s * 1e9)
    injected: list[tuple[BookSnapshot, dict]] = []
    request_log: list[dict] = []
    requested_seen: set[int] = set()
    terminated = None
    frame = meta = None
    while True:
        frame, meta = seed_lake_replay(
            lake_df, injected, grid=grid, k=k, acceptance=acceptance, reseed=True,
            reseed_after_crossed_s=trigger_after_crossed_s, engine=engine,
            price_scale=price_scale, engine_time_col=engine_time_col, frame_out=True)
        last_injected_ts = max((sn.ts for sn, _ in injected), default=None)
        trig = _sustained_cross_trigger(frame, trigger_ns=trigger_ns,
                                        after_ts=last_injected_ts)
        if trig is None:
            terminated = "no_trigger"
            break
        if trig in requested_seen:
            terminated = "no_progress"
            break
        if len(request_log) >= max_requests:
            terminated = "max_requests"
            break
        snap, prov = provider(trig)
        prov = {**prov, "at_ts": int(prov.get("at_ts", trig))}
        reason = classify_candidate(snap, requested_ts=trig, policy=acceptance)
        requested_seen.add(trig)
        request_log.append({"requested_ts": int(trig), "snap_ts": int(snap.ts),
                            "reason": reason, "injected": reason == "ok"})
        if reason == "ok":
            injected.append((snap, prov))
    meta = dict(meta)
    meta["on_demand"] = {"request_log": request_log, "terminated": terminated,
                         "n_requests": len(request_log),
                         "n_injected": sum(1 for r in request_log if r["injected"]),
                         "trigger_after_crossed_s": float(trigger_after_crossed_s),
                         "max_requests": int(max_requests)}
    meta["report_hash"] = hash_obj(meta, exclude_keys=("report_hash",))
    return frame, meta


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
