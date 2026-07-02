"""Partial-day / vendor-seam fill policy: segment planning and seam masks (docs/data.md §5a;
docs/superpowers/plans/2026-07-02-partial-day-fill-policy.md).

Pure planning helpers — no vendor I/O, no replay. `plan_day_stitch` partitions a day's grid into
`lake` / `coinapi` / `excluded` fill segments from a per-sample validity mask plus the seed/reseed
metadata the quality map already computes; the mask helpers derive the label/feature exclusions a
bar/label builder must apply so nothing trains across a vendor seam.

The vendor-switch boundary is the FIRST POST-SEED WARMUP-QUALIFIED SAMPLE — the ts of the
`warmup_consecutive`-th consecutive valid grid sample at/after the accepted seed
(`warmup_qualified_ts`). This is a strictly CONSERVATIVE REFINEMENT of the parity gate's
`cutoff = max(lake_warmup_cutoff, seed_ts)` clamp (`scripts/run_coinbase_parity.py`): the run
counter restarts at the seed, so the boundary is >= the parity clamp, equal whenever no valid run
straddles the seed. Pre-seed valid samples never count — "first valid sample" can be a cold-start
coincidence (stranded levels look two-sided), and the seed alone proves one good snapshot, not a
healthy replay. The qualification ts depends only on samples at/before itself, so later data can
never move it (no look-ahead); the plan-level SEGMENT layout may still change with later data
(e.g. a qualified island dropped for being under `min_lake_segment_s`).

Crossed/untrusted seed-source days route to FULL-DAY CoinAPI fill even when also partial
(2024-08-05: partial AND 28.78% crossed source — crossed dominates), per the provisional PR #13
cross-validation policy. All JSON-facing codes are stable strings, like the quality-map classes.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

NS_PER_S = 1_000_000_000

# ------------------------------------------------------------- stable strings (JSON contract)
LAKE = "lake"
COINAPI = "coinapi"
EXCLUDED = "excluded"  # covered by neither vendor for training (e.g. a tiny warmup window)
SOURCES = (LAKE, COINAPI, EXCLUDED)
# Not a segment source: returned by window_vendor_sources for the part of a window that reaches
# OUTSIDE the day's segment partition (e.g. a day-edge label whose target lands past day end).
UNCOVERED = "uncovered"

LAKE_ONLY = "lake_only"
FULL_DAY_FILL = "full_day_fill"
LEADING_PARTIAL_FILL = "leading_partial_fill"
TRAILING_PARTIAL_FILL = "trailing_partial_fill"
INTERNAL_GAP_FILL = "internal_gap_fill"
MIXED_PARTIAL_FILL = "mixed_partial_fill"
FILL_PROFILES = (LAKE_ONLY, FULL_DAY_FILL, LEADING_PARTIAL_FILL, TRAILING_PARTIAL_FILL,
                 INTERNAL_GAP_FILL, MIXED_PARTIAL_FILL)
# The four profiles that mix vendors within one day (summary bucketing: partial_fill ⊆ needs_fill).
PARTIAL_FILL_PROFILES = (LEADING_PARTIAL_FILL, TRAILING_PARTIAL_FILL, INTERNAL_GAP_FILL,
                         MIXED_PARTIAL_FILL)

# Per-segment reasons. `excluded` segments can only be the day-open warmup prefix: every mid-day
# non-Lake window contains an invalid run >= fill_min_s (that is what ended the Lake segment), so
# it always routes `coinapi` — a single excluded-reason code suffices.
REASON_LAKE_TRUSTED = "trusted_seeded_lake_reconstruction"
REASON_MISSING_LEADING = "lake_missing_leading_segment"
REASON_MISSING_INTERNAL = "lake_missing_internal_segment"
REASON_MISSING_TRAILING = "lake_missing_trailing_segment"
REASON_WARMUP_EXCLUDED = "leading_warmup_excluded"

# Full-day routing reasons (Q2 decision table, first match wins).
REASON_NO_SEED = "no_accepted_seed"
REASON_CROSSED_SOURCE = "crossed_seed_source"
REASON_NEVER_QUALIFIED = "lake_never_warmup_qualified"
REASON_SPAN_TOO_SHORT = "lake_trusted_span_too_short"
REASON_SPAN_QUALITY = "quality_over_trusted_span"


@dataclass(frozen=True)
class SeamPolicy:
    """Seam-policy knobs, emitted verbatim into reports (the `Thresholds.as_dict` pattern).

    `seam_guard_s` equals the longest label horizon (60 s ladder) so a guard-clean label window
    never leans on seam-adjacent book settling. `fill_min_s`: an invalid run shorter than this is
    not worth a fill segment's two seams — it stays masked samples inside the Lake segment.
    `min_lake_segment_s`: a smaller trusted Lake island saves nothing (the day's CoinAPI file is
    downloaded whole regardless) and adds seam risk. `span_invalid_max` mirrors
    `Thresholds.crossed_usable_max` — the trusted span must meet the usable-day bar."""
    seam_guard_s: float = 60.0
    warmup_consecutive: int = 3          # matches the parity gate's --warmup-consecutive
    fill_min_s: float = 300.0
    min_lake_segment_s: float = 3600.0
    span_invalid_max: float = 0.01
    exclude_labels_crossing_seam: bool = True
    exclude_features_crossing_seam: bool = True

    @property
    def seam_guard_ns(self) -> int:
        return int(round(self.seam_guard_s * NS_PER_S))

    @property
    def fill_min_ns(self) -> int:
        return int(round(self.fill_min_s * NS_PER_S))

    @property
    def min_lake_segment_ns(self) -> int:
        return int(round(self.min_lake_segment_s * NS_PER_S))

    def as_dict(self) -> dict:
        return {"seam_guard_s": self.seam_guard_s, "warmup_consecutive": self.warmup_consecutive,
                "fill_min_s": self.fill_min_s, "min_lake_segment_s": self.min_lake_segment_s,
                "span_invalid_max": self.span_invalid_max,
                "exclude_labels_crossing_seam": self.exclude_labels_crossing_seam,
                "exclude_features_crossing_seam": self.exclude_features_crossing_seam}


DEFAULT_SEAM_POLICY = SeamPolicy()


def _iso_utc(ts_ns: int) -> str:
    secs, rem = divmod(int(ts_ns), NS_PER_S)
    base = dt.datetime.fromtimestamp(secs, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{rem:09d}Z" if rem else f"{base}Z"


@dataclass(frozen=True)
class Segment:
    """One half-open [start_ts, end_ts) span of the day served by a single source."""
    source: str
    start_ts: int
    end_ts: int
    reason: str

    def as_dict(self) -> dict:
        return {"source": self.source, "start_ts": int(self.start_ts),
                "start_iso": _iso_utc(self.start_ts), "end_ts": int(self.end_ts),
                "end_iso": _iso_utc(self.end_ts), "reason": self.reason}


@dataclass(frozen=True)
class DayStitchPlan:
    """The per-day stitch plan: an exhaustive segment partition of [day_open_ts, day_end_ts),
    the seam timestamps (every boundary between different-source segments; the boundary sample
    belongs to the RIGHT segment), and the policy that produced it.

    `trusted_lake_start_ts`/`trusted_lake_end_ts` are the SURVIVING Lake coverage bounds — the
    first Lake segment's start / last Lake segment's end AFTER the `min_lake_segment_s` island
    drop — and are None on every full-day route (including crossed-seed-source days where the
    book may well sustain). They can differ from `warmup_qualified_ts` (the raw boundary
    primitive) whenever a qualified island was dropped."""
    day: str | None
    day_open_ts: int
    day_end_ts: int
    grid_ns: int
    fill_profile: str
    full_day_reason: str | None
    segments: tuple[Segment, ...]
    seams: tuple[int, ...]
    policy: SeamPolicy
    trusted_lake_start_ts: int | None
    trusted_lake_end_ts: int | None
    lake_present_start_ts: int | None
    lake_present_end_ts: int | None

    def as_dict(self) -> dict:
        return {"day": self.day, "day_open_ts": int(self.day_open_ts),
                "day_end_ts": int(self.day_end_ts), "grid_ns": int(self.grid_ns),
                "fill_profile": self.fill_profile, "full_day_reason": self.full_day_reason,
                "fill_segments": [s.as_dict() for s in self.segments],
                "seams": [int(s) for s in self.seams],
                "seam_policy": self.policy.as_dict(),
                "trusted_lake_start_ts": _opt_int(self.trusted_lake_start_ts),
                "trusted_lake_end_ts": _opt_int(self.trusted_lake_end_ts),
                "lake_present_start_ts": _opt_int(self.lake_present_start_ts),
                "lake_present_end_ts": _opt_int(self.lake_present_end_ts)}


def _opt_int(v):
    return None if v is None else int(v)


def valid_mask_from_frame(frame: pd.DataFrame, *, min_levels_per_side: int = 1) -> np.ndarray:
    """Per-sample validity of a top-K frame under the SAME predicate as
    `recon.parity.lake_warmup_cutoff`: best bid & ask present, uncrossed, ≥ `min_levels_per_side`
    price levels per side. Shared predicate — a boundary derived from this mask must agree with
    the parity gate's warmup cutoff (pinned by test)."""
    f = (frame.set_index("sample_ts").sort_index()
         if "sample_ts" in frame.columns else frame.sort_index())
    bid_cols = [c for c in f.columns if c.startswith("bid_") and c.endswith("_price")]
    ask_cols = [c for c in f.columns if c.startswith("ask_") and c.endswith("_price")]
    bid_depth = f[bid_cols].notna().sum(axis=1)
    ask_depth = f[ask_cols].notna().sum(axis=1)
    good = ((bid_depth >= min_levels_per_side) & (ask_depth >= min_levels_per_side)
            & f["bid_0_price"].notna() & f["ask_0_price"].notna()
            & (f["bid_0_price"] < f["ask_0_price"]))
    return good.to_numpy(dtype=bool)


def warmup_qualified_ts(sample_ts, valid, *, seed_ts, warmup_consecutive: int = 3) -> int | None:
    """The vendor-switch boundary primitive: ts of the `warmup_consecutive`-th consecutive valid
    sample, counting only samples at/after the accepted seed. Returns None if the seeded book
    never sustains. Pre-seed valid samples are cold-started state and never count — stricter than
    the parity gate's `max(lake_warmup_cutoff, seed_ts)` clamp (whose warmup run may straddle the
    seed), so the result is >= that clamp, equal when no valid run straddles the seed.
    The result depends only on samples at/before the returned ts — no look-ahead."""
    if seed_ts is None:
        return None
    ts = np.asarray(sample_ts, dtype=np.int64)
    ok = np.asarray(valid, dtype=bool)
    run = 0
    for i in range(len(ts)):
        run = run + 1 if (ok[i] and ts[i] >= seed_ts) else 0
        if run >= warmup_consecutive:
            return int(ts[i])
    return None


def _runs(mask) -> list[tuple[int, int]]:
    """Maximal [i0, i1) index runs where `mask` is True."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def plan_day_stitch(sample_ts, valid, *, grid_ns: int, seed_accepted: bool, seed_ts,
                    seed_source_trusted: bool, policy: SeamPolicy = DEFAULT_SEAM_POLICY,
                    present=None, day: str | None = None) -> DayStitchPlan:
    """Derive the day's fill segments from a per-sample validity mask + seed metadata.

    `sample_ts` must be the REGULAR full-day grid (never compact a grid — the same invariant that
    protects label horizons in `recon.parity.compare_topk`). `valid` is the
    `valid_mask_from_frame` predicate; `present` (optional, defaults to `valid`) marks raw
    top-of-book presence for the `lake_present_*` coverage fields.

    Normative algorithm (plan doc "Segment derivation algorithm"): full-day routing first
    (no seed / crossed source), then invalid runs ≥ `fill_min_s` become fill windows, each
    remaining span requalifies with `warmup_consecutive` valid samples at/after the seed (the
    requalification prefix joins the preceding fill window), Lake islands < `min_lake_segment_s`
    are dropped into the fill, and the surviving Lake span must beat `span_invalid_max`."""
    ts = np.asarray(sample_ts, dtype=np.int64)
    ok = np.asarray(valid, dtype=bool)
    grid_ns = int(grid_ns)
    if ts.ndim != 1 or len(ts) == 0 or len(ok) != len(ts):
        raise ValueError("sample_ts and valid must be equal-length non-empty 1-D arrays")
    if grid_ns <= 0:
        raise ValueError("grid_ns must be a positive step (ns)")
    if len(ts) > 1 and not np.all(np.diff(ts) == grid_ns):
        raise ValueError("sample_ts must be the regular full-day grid (spacing == grid_ns)")
    n = len(ts)
    day_open, day_end = int(ts[0]), int(ts[-1]) + grid_ns

    pres = ok if present is None else np.asarray(present, dtype=bool)
    if len(pres) != n:
        raise ValueError("present mask must have the same length as sample_ts")
    pres_idx = np.flatnonzero(pres)
    present_start = int(ts[pres_idx[0]]) if len(pres_idx) else None
    present_end = int(ts[pres_idx[-1]]) + grid_ns if len(pres_idx) else None

    def _full_day(reason: str) -> DayStitchPlan:
        seg = Segment(COINAPI, day_open, day_end, reason)
        return DayStitchPlan(day=day, day_open_ts=day_open, day_end_ts=day_end, grid_ns=grid_ns,
                             fill_profile=FULL_DAY_FILL, full_day_reason=reason, segments=(seg,),
                             seams=(), policy=policy, trusted_lake_start_ts=None,
                             trusted_lake_end_ts=None, lake_present_start_ts=present_start,
                             lake_present_end_ts=present_end)

    if not seed_accepted or seed_ts is None:
        return _full_day(REASON_NO_SEED)
    if not seed_source_trusted:
        return _full_day(REASON_CROSSED_SOURCE)

    # Invalid runs long enough to be worth a vendor fill window.
    long_fill = np.zeros(n, dtype=bool)
    for i0, i1 in _runs(~ok):
        if (i1 - i0) * grid_ns >= policy.fill_min_ns:
            long_fill[i0:i1] = True

    # Each remaining span requalifies independently (the boundary rule, per span). The
    # requalification prefix stays outside the Lake segment and merges into the fill window.
    lake = np.zeros(n, dtype=bool)
    any_qualified = False
    for i0, i1 in _runs(~long_fill):
        run, q_idx = 0, None
        for i in range(i0, i1):
            run = run + 1 if (ok[i] and ts[i] >= seed_ts) else 0
            if run >= policy.warmup_consecutive:
                q_idx = i
                break
        if q_idx is None:
            continue
        any_qualified = True
        if (i1 - q_idx) * grid_ns >= policy.min_lake_segment_ns:
            lake[q_idx:i1] = True

    lake_n = int(lake.sum())
    if lake_n == 0:
        return _full_day(REASON_SPAN_TOO_SHORT if any_qualified else REASON_NEVER_QUALIFIED)
    if int((lake & ~ok).sum()) / lake_n > policy.span_invalid_max:
        return _full_day(REASON_SPAN_QUALITY)

    segments: list[Segment] = []
    flip = np.flatnonzero(np.diff(lake.astype(np.int8))) + 1
    bounds = [0, *flip.tolist(), n]
    for b0, b1 in zip(bounds, bounds[1:]):
        start, end = int(ts[b0]), int(ts[b1 - 1]) + grid_ns
        if lake[b0]:
            segments.append(Segment(LAKE, start, end, REASON_LAKE_TRUSTED))
        elif (b1 - b0) * grid_ns >= policy.fill_min_ns:
            reason = (REASON_MISSING_LEADING if start == day_open
                      else REASON_MISSING_TRAILING if end == day_end
                      else REASON_MISSING_INTERNAL)
            segments.append(Segment(COINAPI, start, end, reason))
        else:
            # Structurally day-open only: a mid-day non-Lake window always contains the
            # >= fill_min_ns invalid run that ended the previous Lake segment.
            segments.append(Segment(EXCLUDED, start, end, REASON_WARMUP_EXCLUDED))

    seams = tuple(int(b.start_ts) for a, b in zip(segments, segments[1:]) if a.source != b.source)

    capi = [s for s in segments if s.source == COINAPI]
    if not capi:
        profile = LAKE_ONLY
    else:
        kinds = {("leading" if s.start_ts == day_open else
                  "trailing" if s.end_ts == day_end else "internal") for s in capi}
        profile = ({"leading": LEADING_PARTIAL_FILL, "trailing": TRAILING_PARTIAL_FILL,
                    "internal": INTERNAL_GAP_FILL}[next(iter(kinds))]
                   if len(kinds) == 1 else MIXED_PARTIAL_FILL)

    lake_segs = [s for s in segments if s.source == LAKE]
    return DayStitchPlan(day=day, day_open_ts=day_open, day_end_ts=day_end, grid_ns=grid_ns,
                         fill_profile=profile, full_day_reason=None, segments=tuple(segments),
                         seams=seams, policy=policy,
                         trusted_lake_start_ts=int(lake_segs[0].start_ts),
                         trusted_lake_end_ts=int(lake_segs[-1].end_ts),
                         lake_present_start_ts=present_start, lake_present_end_ts=present_end)


def full_day_plan(*, day_open_ts: int, day_end_ts: int, grid_ns: int, reason: str,
                  policy: SeamPolicy = DEFAULT_SEAM_POLICY, day: str | None = None) -> DayStitchPlan:
    """A full-day CoinAPI plan constructed WITHOUT per-sample data — for callers that must route a
    day to full-day fill from day-level evidence alone: the Lake partition is absent, the engine is
    metrics-only (no frame to mask), or a day-level quality bar failed with no mask-supported
    narrower fill. Same segment/JSON contract as a `plan_day_stitch` full-day route; `reason`
    becomes both `full_day_reason` and the single segment's reason (the caller owns its stability —
    the quality map reuses its day-level codes). Coverage bounds are unknown here, so
    `trusted_lake_*`/`lake_present_*` are None."""
    day_open_ts, day_end_ts, grid_ns = int(day_open_ts), int(day_end_ts), int(grid_ns)
    if grid_ns <= 0:
        raise ValueError("grid_ns must be a positive step (ns)")
    if day_end_ts <= day_open_ts:
        raise ValueError("day_end_ts must be after day_open_ts")
    seg = Segment(COINAPI, day_open_ts, day_end_ts, reason)
    return DayStitchPlan(day=day, day_open_ts=day_open_ts, day_end_ts=day_end_ts, grid_ns=grid_ns,
                         fill_profile=FULL_DAY_FILL, full_day_reason=reason, segments=(seg,),
                         seams=(), policy=policy, trusted_lake_start_ts=None,
                         trusted_lake_end_ts=None, lake_present_start_ts=None,
                         lake_present_end_ts=None)


def invalid_runs(sample_ts, valid, *, grid_ns: int) -> list[tuple[int, int]]:
    """Maximal half-open [start_ts, end_ts) spans where `valid` is False — the plan-doc Q7
    per-day report metric (`quality.invalid_runs`). Same array contract as `plan_day_stitch`."""
    ts = np.asarray(sample_ts, dtype=np.int64)
    ok = np.asarray(valid, dtype=bool)
    grid_ns = int(grid_ns)
    if grid_ns <= 0:
        raise ValueError("grid_ns must be a positive step (ns)")
    if len(ok) != len(ts):
        raise ValueError("sample_ts and valid must be equal-length")
    return [(int(ts[i0]), int(ts[i1 - 1]) + grid_ns) for i0, i1 in _runs(~ok)]


# --------------------------------------------------------------- seam masks (regular grid only)
def seam_guard_mask(sample_ts, seams, *, guard_ns: int) -> np.ndarray:
    """True where a sample falls inside a seam guard band [seam - guard, seam + guard)."""
    ts = np.asarray(sample_ts, dtype=np.int64)
    out = np.zeros(len(ts), dtype=bool)
    for s in seams:
        out |= (ts >= s - guard_ns) & (ts < s + guard_ns)
    return out


def window_crosses_seam(start_ts: int, end_ts: int, seams) -> bool:
    """True when the closed window [start_ts, end_ts] spans a seam. The boundary sample belongs
    to the right segment, so a window STARTING at a seam does not cross it."""
    return any(start_ts < s <= end_ts for s in seams)


def label_valid_mask(sample_ts, seams, *, horizon_ns: int, guard_ns: int) -> np.ndarray:
    """True where the label window [t, t + horizon] neither crosses a seam nor touches its guard
    band. Apply on the REGULAR grid and mask failures to NaN — never compact (the
    `compare_topk` label rule: compaction horizon-stretches positional shifts).

    Seam/guard geometry only — a training row additionally needs vendor coverage: intersect with
    `window_vendor_sources(...) in ({LAKE}, {COINAPI})`, else a window inside an `excluded`
    segment (no seam in it) passes this mask with no vendor behind it."""
    ts = np.asarray(sample_ts, dtype=np.int64)
    okm = np.ones(len(ts), dtype=bool)
    for s in seams:
        okm &= (ts + horizon_ns < s - guard_ns) | (ts >= s + guard_ns)
    return okm


def feature_valid_mask(sample_ts, seams, *, lookback_ns: int, guard_ns: int) -> np.ndarray:
    """True where the feature window [t - lookback, t] neither crosses a seam nor touches its
    guard band. Same regular-grid masking rule and vendor-coverage composition note as
    `label_valid_mask`."""
    ts = np.asarray(sample_ts, dtype=np.int64)
    okm = np.ones(len(ts), dtype=bool)
    for s in seams:
        okm &= (ts < s - guard_ns) | (ts - lookback_ns >= s + guard_ns)
    return okm


def vendor_source_at(sample_ts, segments) -> list[str]:
    """Per-sample `vendor_source`: the source of the segment containing each ts (segments are the
    half-open partition from `plan_day_stitch`). Raises on timestamps outside the day."""
    ts = np.asarray(sample_ts, dtype=np.int64)
    starts = np.array([s.start_ts for s in segments], dtype=np.int64)
    out = []
    for t in ts.tolist():
        if t < segments[0].start_ts or t >= segments[-1].end_ts:
            raise ValueError(f"sample_ts {t} is outside the day's segments")
        out.append(segments[int(np.searchsorted(starts, t, side='right')) - 1].source)
    return out


def window_vendor_sources(start_ts: int, end_ts: int, segments) -> set[str]:
    """The `feature_vendor_source`/`label_vendor_source` set: sources of every segment the closed
    window [start_ts, end_ts] touches. A training row requires a singleton {lake} or {coinapi}
    for both of its windows (plan doc Q6); anything else — mixed vendors, any `excluded`
    coverage, or UNCOVERED — is masked.

    A window reaching outside the partition `[segments[0].start_ts, segments[-1].end_ts)`
    additionally carries UNCOVERED: its overhang has no vendor at all, so a day-edge label whose
    target lands past day end must never read as a clean single-vendor window. Cross-midnight
    windows are resolvable only against the adjacent day's plan (bar-builder follow-up)."""
    if end_ts < start_ts:
        raise ValueError("window end_ts must be >= start_ts")
    out = {s.source for s in segments if s.start_ts <= end_ts and start_ts < s.end_ts}
    if start_ts < segments[0].start_ts or end_ts >= segments[-1].end_ts:
        out.add(UNCOVERED)
    return out
