"""G0-BN producer orchestration and blind materializer (T9, issue #94; plan T9 row,
spec docs/superpowers/specs/2026-07-13-g0bn-protocol.md sections 2, 5-7).

One deterministic, day-partitioned `binance_single_venue` orchestration over the
merged T1-T8 stages, with two entry points and no generic third path:

- `produce_development` — the rebuildable development build: explicit day/object
  allowlist, full validate-then-write via `eval.writer.write_development`, and the
  logical development data identity (`eval.g0bn_identity.development_data_identity`)
  the 67-B engine binds trials to.
- `materialize_holdout` — THE sole blind-materializer boundary for the 67-E one-shot
  runner (#91). It consumes only the frozen custody-validated `g0bn-holdout-plan-v1`
  object allowlist plus a matching durable `g0bn-raw-access-claim-v1`; rejects
  ranges, globs, discovery, fallbacks, symlinks, missing/extra/duplicate objects,
  wrong days/products, and foreign object hashes before any payload decode (each
  sealed object is pinned to one held O_NOFOLLOW descriptor, hash-verified over
  that descriptor, and decoded only from that descriptor — a path swapped after
  verification can never substitute unverified bytes); streams the
  frozen recipe exactly once through `eval.writer.write_holdout` (fresh O_EXCL
  outputs, hashes computed WHILE writing); then atomically writes and fsyncs
  `g0bn-materialization-attestation-v1`. It never reopens a derived matrix,
  manifest, parquet footer, or the attestation, and it does not create claims, own
  the consumption journal, score, or verdict — those are 67-E/67-F planes. Every
  inconsistency raises: after the raw burn the runner maps any raise to terminal
  INCONCLUSIVE (spec section 6.3 step 5 — an error is never a new exclusion).

Pipeline (per contiguous included-day segment, bounded memory: one day of trades/
bars/reads at a time, one streaming book fold, slim per-candidate records, and the
assembled matrix rows — never a full day of book events and never a segment of
member-trade tuples in memory):

  normalized trades -> `bars.clock.bars_for_day` (trailing threshold from PRIOR
  days only; watermark/index chained across days) -> `coalesce_decision_bars`
  within the day plus a one-bar cross-day carry (an equal-watermark first bar of
  the next day supersedes the held decision — the same last-closing rule, without
  retaining any day's bar list) -> per-day `bars.snapshot.dual_book_reads`
  (day book seeded from the day's own snapshot object; a bar whose watermark
  crosses its day's midnight is masked as day_end_truncation, because its true
  origin cut would need the next day's events) -> `bars.features`
  (one builder per build; the prior read carries across days and gaps — the
  lookback cap owns the drop) -> per-horizon partition/coverage prefilter ->
  `data.labels.triple_barrier_labels` per horizon over the segment's true-mid
  path -> `bars.cost.cost_row` -> per-horizon `data.uniqueness` -> ModelMatrix
  frame -> T8 writer.

Drop accounting is the produce-owned `DROP_COUNT_CATEGORIES` taxonomy, counted per
(category, horizon) with first-failure-wins ordering; a holdout plan whose pinned
`drop_count_categories` differ fails closed. Realized threshold schedule/state are
recorded per build (`g0bn-realized-threshold-schedule-v1` / `g0bn-clock-state-v1`)
and hashed into the result/attestation; the holdout schedule is seeded from the
frozen development-end clock state, verified against the config's
`clock.development_end_state_sha256` pin, and derives January thresholds causally
(spec section 3.2 — January may execute the frozen rule, never select or reset it).

Values the 67-A protocol config carries only as forward pins (top-K ladder depth,
tick size, barrier min-returns/vol-floor, and the trailing-threshold window/warm-up/
seed) enter through the explicit `RuntimeParams` bundle and are hashed into the
build parameters, so they are identity-bearing and attested; #69 seals the real
values and #93 owns the remaining source-name reconciliation.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import stat as _stat
from itertools import tee
from typing import Iterator, Mapping, NamedTuple

import pandas as pd
import pyarrow.parquet as pq

from bars.clock import (
    CLOSE_DAY_END,
    Bar,
    ThresholdConfig,
    ThresholdSchedule,
    bars_for_day,
    coalesce_decision_bars,
)
from bars.cost import CostAssumption, CostRow, cost_row, require_assumption_identity
from bars.events import MIN_ABSOLUTE_NS, clock_trades_from_df
from bars.features import (
    FEATURE_COLS,
    BarFeatureBuilder,
    FeatureConfig,
    FeatureRejection,
    FeatureRow,
)
from bars.modes import BINANCE_SINGLE_VENUE, VENUE_BINANCE, require_venue_allowed
from bars.snapshot import (
    REJECT_STALE,
    BookDelta,
    SnapshotRejection,
    _validate_event as _validate_book_event,
    dual_book_reads,
    validate_book_top,
)
from data.labels import (
    BarrierParams,
    LabelRejection,
    LabelRow,
    triple_barrier_labels,
    validate_barrier_params,
)
from data.uniqueness import uniqueness_by_horizon
from eval.g0bn_config import (
    ATTESTATION_SCHEMA,
    PROTOCOL_ID,
    RAW_ACCESS_CLAIM_SCHEMA,
    _dict,
    _exact,
    _fail,
    _sha256,
    _validate_generated_at,
    g0bn_artifact_sha256,
    validate_protocol_config,
)
from eval.g0bn_freeze import (
    _horizon_roles_sha256,
    holdout_plan_binding,
    oos_build_params,
    verify_holdout_manifest_binding,
)
from eval.g0bn_identity import development_data_identity
from eval.hashing import hash_obj
from eval.writer import (
    G0BN_CLOCK_KIND,
    G0BN_CLOCK_REFERENCE_STREAM,
    G0BN_COST_DTYPES,
    G0BN_DATA_SOURCES,
    G0BN_DEV_DATASET_ID,
    G0BN_INSTRUMENT,
    G0BN_OOS_DATASET_ID,
    G0BN_TARGETS,
    G0BN_VENUE,
    WriteResult,
    _fsync_dir,
    build_id_for,
    logical_row_sha256,
    ordered_manifest_columns,
    write_development,
    write_holdout,
)
from eval.matrix import RESERVED
from recon.events import Delta
from recon.orderbook import OrderBook

# Producer/build identity recorded in every build_params dict this module emits.
PRODUCER_VERSION = "bars.produce:g0bn_producer_v1"

# The complete produce-owned drop taxonomy, in pipeline (first-failure-wins) order.
# Bar-level categories (through `lookback_cap`) kill every horizon of the bar and
# count once per declared horizon; the remaining categories are per-(bar, horizon).
# A holdout plan must pin exactly this list as its drop-count schema (spec 5.2).
DROP_COUNT_CATEGORIES = (
    "warmup",             # trailing threshold schedule still in seed warm-up (T1 flag)
    "day_end_truncation", # CLOSE_DAY_END partition-truncation artifact bars (plan §C.3)
    "book_rejection",     # T2 missing/one-sided/invalid/crossed observable or label book
    "staleness",          # T2 observable book older than the certified staleness cap
    "feature_rejection",  # T3 no_prior_read / insufficient_depth
    "before_start",       # feature support (t_feature_start) precedes the segment's
                          # trusted span (partition start, or a post-excluded-day
                          # segment's first day open — plan §C.3 coverage)
    "lookback_cap",       # observed look-back exceeds the pinned cap (drop, never clip)
    "prefilter",          # t_event + horizon + guard >= partition end (spec §2.2 rule)
    "coverage_gap",       # horizon window lacks contiguous covered true-mid
                          # support: it overruns the day segment or crosses an
                          # invalid (one-sided/crossed) true-book stretch
    "label_rejection",    # T5 insufficient_vol_history / degenerate_barrier_width
    "actual_span",        # realized guarded span (t_barrier + guard) leaves the partition
)

CLOCK_STATE_SCHEMA = "g0bn-clock-state-v1"
REALIZED_SCHEDULE_SCHEMA = "g0bn-realized-threshold-schedule-v1"
COUNTS_SCHEMA = "g0bn-materialization-counts-v1"

# Normalized single-venue object products (the T8 writer's source taxonomy).
L2_SNAPSHOT, L2_DELTA, TRADES = G0BN_DATA_SOURCES

_L2_COLUMNS = ("origin_time", "received_time", "seq", "side", "price", "size")
_SPREAD_REGIME_RULE = "tight_le_boundary_wide_gt_boundary_v1"

# The exactly-implemented frozen clock rules (spec §3.2: "Code must compare
# runtime-resolved values to the config before the raw-access burn"). The producer
# executes T1's trailing windowed arithmetic mean over coverage-normalized prior-day
# notional, and records every certified included day at full coverage (a gappy day
# is excluded at the day level by the certified gap policy, never partially
# weighted). A config pinning any other rule identity must fail closed here rather
# than silently attesting a schedule "derived under" a rule this code never ran.
# clock.warmup_bars stays a #69/#93 forward pin: T1's schedule warms up in DAYS
# (RuntimeParams.threshold.warmup_days), so a bar-count pin is not reconcilable
# by this producer and is deliberately not consumed.
_ADAPTIVE_THRESHOLD_RULE = "trailing_window_mean_threshold_v1"
_COVERAGE_NORMALIZATION_RULE = "full_day_coverage_v1"

_RAW_CLAIM_FIELDS = (
    "schema", "holdout_universe_id", "transaction_id", "protocol_config_sha256",
    "holdout_plan_sha256", "freeze_sha256", "generated_at", "sha256",
)
_CLOCK_STATE_FIELDS = ("schema", "threshold_config", "history")
_HISTORY_FIELDS = ("day", "completed_notional", "covered_fraction")
_GLOB_CHARS = ("*", "?", "[", "]")

_DAY_NS = 86_400 * 10**9


class RuntimeParams(NamedTuple):
    """Producer runtime values the 67-A config carries only as forward hash pins
    (spec section 12 freeze blockers). They are identity-bearing: every field is
    hashed into the build parameters, and the holdout threshold config must also
    reproduce the frozen development-end clock state hash."""
    threshold: ThresholdConfig  # trailing schedule; target_bars_per_day must match config
    top_k: int                  # observable ladder depth (sampler AND feature depth)
    tick_size: float            # exchange price increment for spread_tick
    min_returns: int            # trailing returns required before a barrier width exists
    vol_floor_bps: float        # lower bound on the EWMA vol entering the width

    def as_dict(self) -> dict:
        return {
            "threshold": dict(self.threshold._asdict()),
            "top_k": int(self.top_k),
            "tick_size": float(self.tick_size),
            "min_returns": int(self.min_returns),
            "vol_floor_bps": float(self.vol_floor_bps),
        }


class DevelopmentBuild(NamedTuple):
    """One published development build: the T8 write identities, the logical data
    identity 67-B binds trials to, and the realized schedule/count evidence."""
    write: WriteResult
    data_identity: dict
    row_counts: dict
    drop_counts: dict
    realized_threshold_schedule: list
    realized_threshold_schedule_sha256: str
    clock_state: dict
    clock_state_sha256: str


class HoldoutMaterialization(NamedTuple):
    """The blind materializer's return to 67-E: everything is already durable on
    disk (matrix, manifest, attestation) — nothing here requires reopening it."""
    write: WriteResult
    attestation: dict
    attestation_path: str
    attestation_sha256: str
    row_counts: dict
    drop_counts: dict
    realized_threshold_schedule: list
    realized_threshold_schedule_sha256: str
    clock_state_sha256: str


# ------------------------------------------------------------ normalized readers


def _day_open_ns(day: str) -> int:
    d = _dt.date.fromisoformat(day)
    dt = _dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc)
    return int(dt.timestamp()) * 10**9


def read_normalized_trades(source) -> list:
    """One day of the certified normalized trade contract -> `ClockTrade` list in
    file order (defensive sorting is the clock's job). One day of trades is the
    documented bounded materialization (bars.clock.bars_for_day). `source` is a
    path (development) or a custody-pinned `_PinnedObject` (holdout)."""
    stream = source.stream() if isinstance(source, _PinnedObject) else None
    pf = pq.ParquetFile(stream if stream is not None else source)
    try:
        df = pf.read().to_pandas()
    finally:
        pf.close()
        if stream is not None:
            stream.close()
    trades = clock_trades_from_df(df)
    for t in trades:
        # the two-axis contract at the reader boundary (mirrors the L2 check):
        # a trade captured "before" it happened would understate the monotone
        # decision watermark — a lookahead the clock cannot detect itself
        if t.received_time < t.origin_time:
            raise ValueError(
                f"trade at (origin_time, seq)=({t.origin_time}, {t.seq}) has "
                f"received_time {t.received_time} < origin_time — the source's "
                "timestamp contract is broken (certify/normalize at ingestion)")
    return trades


def iter_normalized_book_events(source) -> Iterator[BookDelta]:
    """Stream one normalized L2 object (snapshot seed or delta day) as `BookDelta`
    events, validating non-decreasing `(origin_time, seq)` while yielding — this
    closes the lazy-tail hole bars.snapshot documents (an ordering violation past
    the last decision's lookahead barrier is undetectable there, so the T9 driver
    validates the day's stream order itself). Memory is one record batch.
    `source` is a path (development) or a custody-pinned `_PinnedObject`
    (holdout: every decode reads the verified descriptor's bytes)."""
    label = source.path if isinstance(source, _PinnedObject) else source
    stream = source.stream() if isinstance(source, _PinnedObject) else None
    pf = pq.ParquetFile(stream if stream is not None else source)
    try:
        names = pf.schema_arrow.names
        missing = [c for c in _L2_COLUMNS if c not in names]
        if missing:
            raise ValueError(f"normalized L2 object {label} lacks required "
                             f"column(s) {missing}; got {list(names)}")
        last_key = None
        for batch in pf.iter_batches(columns=list(_L2_COLUMNS)):
            cols = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
            for origin, received, seq, side, price, size in zip(*cols):
                key = (int(origin), int(seq))
                if last_key is not None and key < last_key:
                    raise ValueError(
                        f"normalized L2 object {label} is out of (origin_time, seq) "
                        f"order: {key} after {last_key} — the certified normalized "
                        "contract requires a sorted stream")
                last_key = key
                event = BookDelta(int(origin), int(received), int(seq),
                                  str(side), float(price), float(size))
                # T2's per-event contract, enforced AT THE READER: the lazily
                # consumed label-path fold can reach rows dual_book_reads never
                # validates (its checks cover only the consumed decision
                # prefix), so a malformed side/price/size/receipt must fail
                # closed before ANY fold sees the event. The absolute-UTC floor
                # mirrors the trades path (bars.events.MIN_ABSOLUTE_NS): a
                # time-of-day or wrong-unit stamp would otherwise pass the
                # seed's at-or-before-open check and seed books/labels.
                if event.origin_time < MIN_ABSOLUTE_NS:
                    raise ValueError(
                        f"L2 event origin_time {event.origin_time} at "
                        f"(origin_time, seq)={key} is not an absolute UTC epoch "
                        "timestamp (normalize vendor offsets at ingestion)")
                _validate_book_event(event)
                yield event
    finally:
        pf.close()
        if stream is not None:
            stream.close()


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_pinned_fd(path: str) -> int:
    """The single holdout source-open primitive (and the read-spy seam):
    O_NOFOLLOW pins a regular file's descriptor so hashing and every later
    decode read the same inode's bytes. O_NONBLOCK keeps a swapped-in FIFO from
    blocking the open, and the fstat gate rejects anything that is not a
    regular file AFTER the open — the path-level isfile/islink checks are
    advisory only (they can be invalidated before this open)."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                 | getattr(os, "O_CLOEXEC", 0))
    try:
        if not _stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(
                f"sealed object path {path!r} is not a regular file at open "
                "time (FIFO/device/directory swap); custody objects must be "
                "plain regular files")
    except BaseException:
        os.close(fd)
        raise
    return fd


class _PinnedObject:
    """A sealed normalized object bound to ONE held file descriptor: the custody
    hash is computed over this descriptor and every payload decode streams from a
    dup of the SAME descriptor, so a path swapped between verification and
    parsing can never substitute unverified bytes (the by-name reopen TOCTOU).
    Decodes are strictly sequential, so the dup's shared offset is reset per use."""

    def __init__(self, path) -> None:
        self.path = os.fspath(path)
        self._fd = _open_pinned_fd(self.path)

    def stream(self):
        f = os.fdopen(os.dup(self._fd), "rb")
        f.seek(0)
        return f

    def sha256(self) -> str:
        h = hashlib.sha256()
        with self.stream() as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def close(self) -> None:
        os.close(self._fd)


# ------------------------------------------------------------------ clock state


def clock_state_sha256(state: dict) -> str:
    """Canonical hash of a `g0bn-clock-state-v1` dict (no self-hash field: the
    config's `clock.development_end_state_sha256` is the external pin)."""
    return hash_obj(state)


def _clock_state(threshold: ThresholdConfig, history: list) -> dict:
    return {
        "schema": CLOCK_STATE_SCHEMA,
        "threshold_config": {
            "target_bars_per_day": int(threshold.target_bars_per_day),
            "window_days": int(threshold.window_days),
            "warmup_days": int(threshold.warmup_days),
            "seed_threshold": float(threshold.seed_threshold),
            "min_covered_fraction": float(threshold.min_covered_fraction),
        },
        "history": sorted((dict(h) for h in history), key=lambda h: h["day"]),
    }


def _validate_clock_state(state, *, runtime: RuntimeParams, config: dict,
                          before_ns: int) -> dict:
    path = "clock state"
    _dict(path, state, _CLOCK_STATE_FIELDS)
    _exact(f"{path}.schema", state["schema"], CLOCK_STATE_SCHEMA)
    expected_tc = _clock_state(runtime.threshold, [])["threshold_config"]
    _exact(f"{path}.threshold_config", state["threshold_config"], expected_tc)
    if not isinstance(state["history"], list) or not state["history"]:
        _fail(f"{path}.history", "must be a non-empty array of recorded prior days "
                                 "(the frozen development-end schedule input)")
    seen = set()
    included_dev_days = set(config["exclusions"]["included_days"])
    for i, entry in enumerate(state["history"]):
        epath = f"{path}.history[{i}]"
        _dict(epath, entry, _HISTORY_FIELDS)
        day = entry["day"]
        if not isinstance(day, str) or _day_open_ns(day) >= before_ns:
            _fail(f"{epath}.day", f"{day!r} is not a prior day before the build "
                                  "partition (the frozen state may not embed "
                                  "in-partition volume)")
        if day not in included_dev_days:
            _fail(f"{epath}.day",
                  f"{day!r} is not one of the config's outcome-blind included "
                  "development days; even a hash-matching frozen state may only "
                  "embed in-scope development volume (out-of-window or excluded "
                  "days must not seed the holdout schedule)")
        if day in seen:
            _fail(f"{epath}.day", f"duplicate recorded day {day}")
        seen.add(day)
        notional = entry["completed_notional"]
        if not (isinstance(notional, (int, float)) and not isinstance(notional, bool)
                and math.isfinite(float(notional)) and float(notional) >= 0.0):
            _fail(f"{epath}.completed_notional", f"must be finite >= 0; got {notional!r}")
        coverage = entry["covered_fraction"]
        if not (isinstance(coverage, (int, float)) and not isinstance(coverage, bool)
                and 0.0 < float(coverage) <= 1.0):
            _fail(f"{epath}.covered_fraction", f"must be in (0, 1]; got {coverage!r}")
    days = [h["day"] for h in state["history"]]
    if days != sorted(days):
        _fail(f"{path}.history", "must be sorted by day")
    pinned = config["clock"]["development_end_state_sha256"]
    recomputed = clock_state_sha256(state)
    if recomputed != pinned:
        _fail(f"{path}", f"does not hash to the config's frozen development-end "
                         f"clock state pin ({recomputed} != {pinned}); the one-shot "
                         "schedule must start from exactly the frozen state")
    return state


# ------------------------------------------------------------- runtime validation


def _validate_runtime(runtime: RuntimeParams, config: dict) -> None:
    if not isinstance(runtime, RuntimeParams):
        _fail("runtime", f"must be a bars.produce.RuntimeParams; got "
                         f"{type(runtime).__name__}")
    # eager sub-contract validation (each stage re-validates at construction;
    # failing here keeps every deterministic operator-parameter error at the
    # boundary — in holdout, BEFORE the raw claim is consumed or any sealed
    # source is opened, where a late raise would burn the one-shot)
    # strict TYPES first: the delegated stage validators bound VALUES but let a
    # JSON/YAML typo class through — bools compare as 1/0 (tick_size=True would
    # materialize a 1.0-tick build) and fractional day/count fields survive
    # numeric comparisons; float top_k would first explode inside heapq.nlargest
    # mid-build — after the raw burn in holdout
    def _plain_int(field: str, v) -> None:
        if isinstance(v, bool) or not isinstance(v, int):
            _fail(f"runtime.{field}",
                  f"must be a plain integer; got {v!r}")

    def _plain_real(field: str, v) -> None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _fail(f"runtime.{field}",
                  f"must be a plain real number; got {v!r}")

    _plain_int("top_k", runtime.top_k)
    _plain_real("tick_size", runtime.tick_size)
    _plain_int("threshold.target_bars_per_day",
               runtime.threshold.target_bars_per_day)
    _plain_int("threshold.window_days", runtime.threshold.window_days)
    _plain_int("threshold.warmup_days", runtime.threshold.warmup_days)
    _plain_real("threshold.seed_threshold", runtime.threshold.seed_threshold)
    _plain_real("threshold.min_covered_fraction",
                runtime.threshold.min_covered_fraction)
    ThresholdSchedule(runtime.threshold)
    BarFeatureBuilder(FeatureConfig(top_k=runtime.top_k, tick_size=runtime.tick_size))
    validate_barrier_params(_barrier_params(
        config, runtime, {h["tag"]: int(h["ns"]) for h in config["horizons"]}))
    clock = config["clock"]
    if runtime.threshold.target_bars_per_day != clock["target_bars_per_day"]:
        _fail("runtime.threshold.target_bars_per_day",
              f"{runtime.threshold.target_bars_per_day} does not match the config "
              f"clock pin {clock['target_bars_per_day']}")
    if config["producer"]["lookback_cap_ns"] > config["features"]["max_lookback_ns"]:
        _fail("producer.lookback_cap_ns",
              "exceeds features.max_lookback_ns: retained rows could not satisfy "
              "the declared manifest look-back bound")
    if config["producer"]["lookback_cap_ns"] >= _DAY_NS:
        _fail("producer.lookback_cap_ns",
              "must be under one UTC day: a day-partitioned producer cannot honor "
              "a look-back cap that could bridge an entire excluded/uncovered day "
              "(the cap owns the cross-gap feature-window drop)")
    if clock["adaptive_threshold_update_rule"] != _ADAPTIVE_THRESHOLD_RULE:
        _fail("clock.adaptive_threshold_update_rule",
              f"unknown rule {clock['adaptive_threshold_update_rule']!r}; this "
              f"producer implements only {_ADAPTIVE_THRESHOLD_RULE!r} (T1's "
              "trailing windowed mean) — attesting a schedule under a foreign "
              "rule identity would be a false attestation")
    if clock["coverage_normalization"] != _COVERAGE_NORMALIZATION_RULE:
        _fail("clock.coverage_normalization",
              f"unknown rule {clock['coverage_normalization']!r}; this producer "
              f"implements only {_COVERAGE_NORMALIZATION_RULE!r} (certified "
              "included days are complete by the day-level gap policy and are "
              "recorded at full coverage)")
    labels = config["labels"]
    if labels["tp_multiplier"] != labels["sl_multiplier"]:
        _fail("labels", "tp_multiplier != sl_multiplier: data.labels implements "
                        "symmetric horizontal barriers only (one width_mult)")
    spread = config["reporting"]["spread_regime"]
    if spread["rule"] != _SPREAD_REGIME_RULE:
        _fail("reporting.spread_regime.rule",
              f"unknown rule {spread['rule']!r}; this producer implements only "
              f"{_SPREAD_REGIME_RULE!r}")


def _barrier_params(config: dict, runtime: RuntimeParams, horizons: dict) -> BarrierParams:
    labels = config["labels"]
    return BarrierParams(
        halflife_ns=labels["ewma_half_life_ns"],
        min_returns=runtime.min_returns,
        width_mult=float(labels["tp_multiplier"]),
        vol_floor_bps=runtime.vol_floor_bps,
        horizons=horizons,
    )


def _cost_assumption(config: dict) -> CostAssumption:
    a = CostAssumption(**config["costs"]["cost_assumption"])
    # the anti-aliasing gate: the build's declared single-venue identity must
    # exactly match the assumption before any row is priced (bars.cost / §G).
    # The declared source is the config-pinned CERTIFIED NORMALIZED contract
    # identity the cost inputs derive from — never the assumption's own field
    # echoed back, which would let a mistyped/copied source attest a foreign
    # identity (Codex round 19).
    require_assumption_identity(
        a, venue=VENUE_BINANCE, product=G0BN_INSTRUMENT["symbol"],
        source=config["source_certification"]["normalized_schema_version"])
    return a


# ------------------------------------------------------------------ day pipeline


def _segments(days: list) -> list:
    """Contiguous UTC-day runs, in order (label paths and coverage never bridge an
    excluded/missing day)."""
    segs = [[days[0]]]
    for prev, cur in zip(days, days[1:]):
        prev_d = _dt.date.fromisoformat(prev)
        if _dt.date.fromisoformat(cur) == prev_d + _dt.timedelta(days=1):
            segs[-1].append(cur)
        else:
            segs.append([cur])
    return segs


class _DropCounter:
    def __init__(self, tags) -> None:
        self.tags = tuple(tags)
        self.counts = {c: {t: 0 for t in self.tags} for c in DROP_COUNT_CATEGORIES}

    def add(self, category: str, tag: str) -> None:
        self.counts[category][tag] += 1

    def add_all(self, category: str) -> None:
        for t in self.tags:
            self.counts[category][t] += 1


class _Candidate(NamedTuple):
    """One surviving coalesced bar decision (bar-level gates passed). Deliberately
    slim — the Bar (with its full member-trade tuple) and the book reads are
    released as soon as classification prices the row, so segment-lifetime memory
    is candidates + assembled rows, never a segment of raw trades/books."""
    day: str
    t_event: int
    emitted_by_time_cap: bool
    feat: FeatureRow
    cost: CostRow
    label_mid: float


def _day_bounded_deltas(day: str, source) -> Iterator[BookDelta]:
    """The day's L2 delta stream with the certified day-partition bound enforced
    fail-closed: trades are day-bounded by bars_for_day and snapshot seeds by
    the at-or-before-open check, so an off-day delta row (a next-day spill or a
    prior-day duplicate inside a declared day object) is the one remaining way
    out-of-partition events could advance a book fold — it must never fold."""
    open_ns = _day_open_ns(day)
    end_ns = open_ns + _DAY_NS
    for e in iter_normalized_book_events(source):
        if not (open_ns <= e.origin_time < end_ns):
            raise ValueError(
                f"L2 delta object for {day} carries origin_time {e.origin_time} "
                f"outside its declared day [{open_ns}, {end_ns}); certified "
                "day-partitioned objects may not mix days")
        yield e


def _seed_day_book_events(day: str, snapshot_path, delta_path) -> Iterator[BookDelta]:
    """The day's full book event stream: snapshot seed first (state at/before the
    day open — a post-open snapshot origin is a broken normalized object), then
    the day's deltas.

    Certified-source ordering invariant (fail-closed downstream): a day's
    snapshot must carry origins at/after the PRIOR day's last consumed delta —
    it represents a later book state. A violating feed would make the first
    observable read of the day regress behind the carried prior read, and
    `bars.features.BarFeatureBuilder` raises on that regression rather than
    emitting a row from an incoherent seed (#93 reconciles the normalized-seed
    contract that guarantees this)."""
    open_ns = _day_open_ns(day)
    for e in iter_normalized_book_events(snapshot_path):
        if e.origin_time > open_ns:
            raise ValueError(
                f"snapshot object for {day} carries origin_time {e.origin_time} "
                f"after the day open {open_ns}; a day seed must be the book state "
                "at or before the open")
        yield e
    yield from _day_bounded_deltas(day, delta_path)


def _scan_day_validity(day: str, snapshot_path, delta_path) -> list:
    """Full-object drain AND true-book validity scan for one day: every event
    passes the validating reader (ordering, day bounds, per-event contract —
    the lazy folds pull only what decisions/label windows need, so this pass is
    what guarantees a bad tail row still fails the build), and the origin-axis
    intervals over which the folded true book is missing/one-sided/invalid/
    crossed are returned as half-open [start_ns, end_ns) pairs clamped to the
    day. A label window crossing such an interval has no covered true-mid
    support and must not resolve over an as-of-carried stale mid."""
    open_ns = _day_open_ns(day)
    end_ns = open_ns + _DAY_NS
    book = OrderBook()
    intervals: list = []
    # the day STARTS in the missing-book state: until the fold first yields a
    # usable top, there is no true-mid support from the open — a valid seed
    # closes this initial interval at the open itself (zero length, suppressed)
    invalid_since: int | None = open_ns
    prev_ts: int | None = None

    def assess(ts: int) -> None:
        nonlocal invalid_since
        bad = validate_book_top(book) is not None
        ts = max(ts, open_ns)  # seed events govern from the open
        if bad and invalid_since is None:
            invalid_since = ts
        elif not bad and invalid_since is not None:
            if invalid_since < ts:  # a same-instant recovery is not a gap
                intervals.append((invalid_since, ts))
            invalid_since = None

    for e in _seed_day_book_events(day, snapshot_path, delta_path):
        # validity is a per-TIMESTAMP property: the state at an instant is the
        # last event's post-state there (T2/T5 coalescing), so a remove/re-add
        # inside one origin_time must never mint a zero-length invalid interval
        if prev_ts is not None and e.origin_time != prev_ts:
            assess(prev_ts)
        book.apply(Delta(e.origin_time, e.seq, e.side, e.price, e.size))
        prev_ts = e.origin_time
    if prev_ts is not None:
        assess(prev_ts)
    if invalid_since is not None:
        intervals.append((invalid_since, end_ns))
    return intervals


def _window_hits_invalid(t_event: int, window_end: int, intervals: list) -> bool:
    """Does the forward label window (t_event, window_end] cross any invalid
    true-book interval [s, e)? The anchor instant itself is excluded — an
    invalid book AT t_event is already a bar-level T2 rejection."""
    return any(s <= window_end and e > t_event for s, e in intervals)


def _segment_mid_path(segment: list, paths_for_day) -> Iterator[tuple]:
    """The TRUE target-mid path over one contiguous segment: per day, fold the
    snapshot seed silently, emit one as-of point at the day open, then one point
    per delta event while the book is usable. Same origin-order fold as T2's
    label cut, so anchors' P0 equals the as-of path mid by construction."""
    for day in segment:
        paths = paths_for_day(day)
        open_ns = _day_open_ns(day)
        book = OrderBook()
        for e in iter_normalized_book_events(paths[L2_SNAPSHOT]):
            if e.origin_time > open_ns:
                raise ValueError(
                    f"snapshot object for {day} carries origin_time "
                    f"{e.origin_time} after the day open {open_ns}")
            book.apply(Delta(e.origin_time, e.seq, e.side, e.price, e.size))
        if validate_book_top(book) is None:
            yield (open_ns, book.mid())
        for e in _day_bounded_deltas(day, paths[L2_DELTA]):
            book.apply(Delta(e.origin_time, e.seq, e.side, e.price, e.size))
            if validate_book_top(book) is None:
                yield (e.origin_time, book.mid())


class _FrameBuild(NamedTuple):
    frame: pd.DataFrame
    row_counts: dict
    drop_counts: dict
    realized_schedule: list
    history: list           # every (day, notional, coverage) the schedule now holds
    source_sha256s: dict    # {day: {product: sha256}} of every consumed object


def _build_frame(config: dict, runtime: RuntimeParams, *, partition: str,
                 days: list, paths_for_day, schedule: ThresholdSchedule,
                 seeded_history: list, partition_start_ns: int,
                 partition_end_ns: int, extra_cols: tuple,
                 verify_sha_for_day=None) -> _FrameBuild:
    """The shared deterministic streaming pipeline (module docstring). Fails
    closed on any contract violation; per-row data drops are counted, never
    silently absorbed."""
    ladder = [(h["tag"], int(h["ns"])) for h in config["horizons"]]
    tags = [t for t, _ in ladder]
    counter = _DropCounter(tags)
    guard_ns = int(config["partition"]["partition_guard_ns"])
    staleness_cap_ns = int(config["producer"]["staleness_cap_ns"])
    lookback_cap_ns = int(config["producer"]["lookback_cap_ns"])
    boundary_spread_tick = float(
        config["reporting"]["spread_regime"]["boundary_spread_tick"])
    assumption = _cost_assumption(config)
    builder = BarFeatureBuilder(
        FeatureConfig(top_k=runtime.top_k, tick_size=runtime.tick_size))

    history = list(seeded_history)
    realized_schedule = []
    source_sha256s: dict = {}
    rows: list[dict] = []
    watermark = 0
    next_index = 0
    prev_segment_last_t: int | None = None
    # rebound at the top of each segment; classify() reads the current value.
    # The feature-window floor is the SEGMENT start, not the partition start: a
    # post-excluded-day segment's day-open snapshot originates inside the
    # excluded day, and a window reaching into that gap consumes state the
    # exclusion declared untrusted (plan §C.3 coverage integrity) even when it
    # is well inside the partition (Codex round 27).
    seg_start_ns = partition_start_ns

    def classify(day: str, bar: Bar, read, seg_records: list) -> None:
        """First-failure-wins drop accounting for one coalesced decision, in the
        pinned DROP_COUNT_CATEGORIES order. A bar whose monotone watermark lands
        at/after its emission day's end is a day-boundary truncation artifact:
        its true origin cut at t_event would need the NEXT day's events, which
        this day-scoped feed deliberately does not extend into — masking it
        mirrors the CLOSE_DAY_END rule and keeps P0 exact for every labeled row."""
        day_end_like = (bar.close_reason == CLOSE_DAY_END
                        or bar.t_event >= _day_open_ns(day) + _DAY_NS)
        if isinstance(read, SnapshotRejection):
            if bar.is_warmup:
                counter.add_all("warmup")
            elif day_end_like:
                counter.add_all("day_end_truncation")
            elif read.reason == REJECT_STALE:
                counter.add_all("staleness")
            else:
                counter.add_all("book_rejection")
            return
        if day_end_like:
            # a boundary-truncation read is computed from a knowingly truncated
            # event basis (the day-scoped feed omits next-day events observable
            # at its post-midnight watermark), so unlike ordinary rejections it
            # must NOT advance the feature builder's prior-read state — the next
            # retained bar's OFI/t_feature_start would difference against an
            # incomplete observation (Codex round 2)
            counter.add_all("warmup" if bar.is_warmup else "day_end_truncation")
            return
        feat = builder.build(bar, read.observable)
        if bar.is_warmup:
            counter.add_all("warmup")
            return
        if isinstance(feat, FeatureRejection):
            counter.add_all("feature_rejection")
            return
        if feat.t_feature_start < seg_start_ns:
            counter.add_all("before_start")
            return
        if bar.t_event - feat.t_feature_start > lookback_cap_ns:
            counter.add_all("lookback_cap")
            return
        seg_records.append(_Candidate(
            day=day, t_event=bar.t_event,
            emitted_by_time_cap=bar.emitted_by_time_cap, feat=feat,
            cost=cost_row(read, assumption=assumption),
            label_mid=read.label.mid))

    for segment in _segments(days):
        seg_start_ns = max(partition_start_ns, _day_open_ns(segment[0]))
        seg_end_ns = _day_open_ns(segment[-1]) + _DAY_NS
        seg_records: list[_Candidate] = []
        seg_invalid: list = []  # invalid true-book intervals across the segment
        # cross-day coalesce carry: each day's LAST decision is held back one day
        # so an equal-watermark first bar of the next day supersedes it (the
        # last-closing, most-informed decision — same rule as
        # coalesce_decision_bars), without retaining any day's full bar list.
        carry: tuple | None = None
        for day in segment:
            # every source open is an authorized single-venue open
            require_venue_allowed(BINANCE_SINGLE_VENUE, VENUE_BINANCE)
            paths = paths_for_day(day)
            if set(paths) != set(G0BN_DATA_SOURCES):
                raise ValueError(
                    f"day {day}: object set must be exactly the certified "
                    f"normalized products {list(G0BN_DATA_SOURCES)}; got "
                    f"{sorted(paths)}")
            if verify_sha_for_day is None:
                source_sha256s[day] = {p: _sha256_file(paths[p])
                                       for p in G0BN_DATA_SOURCES}
            else:
                source_sha256s[day] = verify_sha_for_day(day)
            # full-object semantic validation + true-book validity scan BEFORE
            # any fold: the lazy samplers/label passes pull only what decisions
            # and label windows need, so a malformed or off-day row in the
            # unpulled tail would otherwise never reach the validating reader
            # while the manifest/attestation still claim the sealed object was
            # consumed — a bad tail row is a bad object, never dead data. The
            # scan also records the origin intervals with an invalid true book,
            # which gate the per-horizon label windows below.
            seg_invalid.extend(_scan_day_validity(day, paths[L2_SNAPSHOT],
                                                  paths[L2_DELTA]))
            trades = read_normalized_trades(paths[TRADES])
            day_bars = list(coalesce_decision_bars(bars_for_day(
                trades, day=day, schedule=schedule,
                time_cap_ns=int(config["clock"]["time_cap_ns"]),
                initial_watermark_ns=watermark, start_index=next_index)))
            day_threshold = schedule.threshold_for(day)
            realized_schedule.append({"day": day,
                                      "threshold": float(day_threshold.threshold),
                                      "is_warmup": bool(day_threshold.is_warmup)})
            notional = math.fsum(t.price * t.amount for t in trades)
            schedule.record_day(day, notional, 1.0)
            history.append({"day": day, "completed_notional": float(notional),
                            "covered_fraction": 1.0})
            del trades
            if not day_bars:
                continue  # a trade-free day: the carry is held for a later tie
            watermark = day_bars[-1].t_event
            next_index = day_bars[-1].index + 1
            if prev_segment_last_t is not None \
                    and day_bars[0].t_event <= prev_segment_last_t:
                raise ValueError(
                    f"decision watermark {day_bars[0].t_event} does not "
                    f"increase past the previous segment's last decision "
                    f"{prev_segment_last_t}; the certified received-lag cap "
                    "should make cross-gap ties impossible")
            prev_segment_last_t = None  # only guards the segment's first bars
            if carry is not None:
                if day_bars[0].t_event == carry[1].t_event:
                    carry = None  # superseded by the more-informed later bar
                elif day_bars[0].t_event < carry[1].t_event:
                    raise ValueError(  # impossible under the chained watermark
                        f"decision watermark regressed across the {day} boundary")
                else:
                    classify(carry[0], carry[1], carry[2], seg_records)
                    carry = None
            reads_iter = dual_book_reads(
                _seed_day_book_events(day, paths[L2_SNAPSHOT], paths[L2_DELTA]),
                [b.t_event for b in day_bars],
                staleness_cap_ns=staleness_cap_ns, top_k=runtime.top_k)
            pairs = list(zip(day_bars, reads_iter))
            for bar, read in pairs[:-1]:
                classify(day, bar, read, seg_records)
            carry = (day, pairs[-1][0], pairs[-1][1])
            del day_bars, pairs
        if carry is not None:
            classify(carry[0], carry[1], carry[2], seg_records)
            prev_segment_last_t = carry[1].t_event
            carry = None

        # per-(candidate, horizon) survival gates first (pure arithmetic, no
        # I/O). The invalid-stretch gate deliberately does NOT run here: an
        # early-touching barrier can resolve entirely on covered points before
        # a later invalid stretch, so that check runs post-labeling against the
        # RESOLVED span (t_event, t_barrier] below (Codex round 18). Boundary
        # overhang, by contrast, stays NOMINAL and pre-label — a deliberate
        # asymmetry, not an oversight: an invalid stretch sits inside COMPLETE
        # declared coverage (the path exists, so the realized span is
        # knowable), while at a partition/segment end the path physically ends.
        # T5 refuses any window overhanging coverage_end BEFORE consuming the
        # path (data/labels.py boundary refusal; plan section E pins "the
        # caller's per-horizon prefilter owns boundary drops — T5 never opens
        # the next partition"), an unresolved candidate there is undecidable,
        # and the nominal rule keeps boundary row selection outcome-independent
        # (no touch-dependent inclusion bias at coverage edges).
        survivors: dict[str, list] = {tag: [] for tag, _ in ladder}
        for rec in seg_records:
            for tag, horizon_ns in ladder:
                upper = rec.t_event + horizon_ns + guard_ns
                if upper >= partition_end_ns:
                    counter.add("prefilter", tag)
                    continue
                if seg_end_ns < partition_end_ns and upper >= seg_end_ns:
                    counter.add("coverage_gap", tag)
                    continue
                survivors[tag].append(rec)
        live = [(tag, hns) for tag, hns in ladder if survivors[tag]]
        if not live:
            continue
        # ONE shared segment path pass, tee'd across the live horizons and
        # consumed in lockstep per candidate below: every branch's frontier
        # stays within (max horizon - min horizon) of the others, so the tee
        # buffer holds only that sliver of path points — never a replayed
        # per-horizon stream over multi-GB January objects (dead horizons get
        # no branch: an unconsumed tee branch would pin the whole stream)
        bound_ns = min(seg_end_ns, partition_end_ns)
        branches = tee(_segment_mid_path(segment, paths_for_day), len(live))
        label_iters = {
            tag: triple_barrier_labels(
                branch,
                ((rec.t_event, rec.label_mid) for rec in survivors[tag]),
                params=_barrier_params(config, runtime, {tag: hns}),
                coverage_end_ns=bound_ns)
            for (tag, hns), branch in zip(live, branches)
        }
        ptr = {tag: 0 for tag, _ in live}
        for rec in seg_records:
            for tag, _hns in live:
                s = survivors[tag]
                if ptr[tag] < len(s) and s[ptr[tag]] is rec:
                    ptr[tag] += 1
                    out = next(label_iters[tag])
                    if isinstance(out, LabelRejection):
                        counter.add("label_rejection", tag)
                        continue
                    if _window_hits_invalid(rec.t_event, out.t_barrier,
                                            seg_invalid):
                        # the RESOLUTION consumed the gapped stretch (a late
                        # touch or an as-of vertical over missing support);
                        # an early touch before the stretch resolved entirely
                        # on covered points and is kept. This span is NOT
                        # guard-extended, deliberately: the plan's ±guard_ns
                        # window geometry (§C.3) masks VENDOR SEAMS — fuzzy
                        # joints between feeds — while an invalid true-book
                        # stretch is an exact, observed interval in ONE
                        # hash-verified feed; a touch resolved on exact points
                        # before the stretch cannot be contaminated by it
                        # retroactively, anchor-side quality is owned by T2
                        # rejections and the staleness cap, and with
                        # partition_guard_ns >= the longest horizon a guard
                        # extension here would blanket-mask every nominally
                        # non-crossing row near any glitch and make the
                        # early-touch keep above unreachable by construction.
                        counter.add("coverage_gap", tag)
                        continue
                    if out.t_barrier + guard_ns >= bound_ns:
                        counter.add("actual_span", tag)
                        continue
                    rows.append(_assemble_row(rec, tag, out,
                                              boundary_spread_tick, extra_cols))

    if not rows:
        raise ValueError(
            f"the {partition} build produced no surviving rows; an empty matrix "
            "is never published (fail closed)")
    frame = _finalize_frame(rows, tags, extra_cols)
    row_counts = {tag: int((frame["horizon"] == tag).sum()) for tag in tags}
    min_rows = int(config["cv"]["n_groups"])
    thin = {tag: n for tag, n in row_counts.items() if n < min_rows}
    if thin:
        raise ValueError(
            f"the {partition} build left declared horizon(s) below the CPCV "
            f"minimum cv.n_groups={min_rows}: {thin}; every declared horizon "
            "must survive masking with at least n_groups rows (plan section on "
            "horizon survival) — a manifest declaring an unusable horizon is "
            "never published (the consumer would reject the empty slice or "
            "raise n_groups > n_samples after the fact)")
    return _FrameBuild(frame=frame, row_counts=row_counts,
                       drop_counts=counter.counts,
                       realized_schedule=realized_schedule, history=history,
                       source_sha256s=source_sha256s)


def _assemble_row(rec: _Candidate, tag: str, label: LabelRow,
                  boundary_spread_tick: float, extra_cols: tuple) -> dict:
    feat = rec.feat
    row = {name: float(value) for name, value in zip(FEATURE_COLS, feat[2:])}
    row.update({
        # the protocol pins labels.return_formula == log_mid_ratio_bps_v1
        # (eval.g0bn_config.LABEL_RETURN_FORMULA) while data.labels emits the
        # simple mid-ratio in bps of P0; log(P/P0) == log1p(simple/1e4) converts
        # the SAME physical move exactly into the pinned representation. The
        # barrier decision (label/t_barrier) is T5's touch event either way —
        # only the published magnitude changes space.
        "y_fwd_bps": 1e4 * math.log1p(float(label.y_fwd_bps) / 1e4),
        "label": int(label.label),
        "t_event": int(rec.t_event),
        "t_barrier": int(label.t_barrier),
        "t_feature_start": int(feat.t_feature_start),
        # availability_lag_ns == 0: synchronous decide-and-act (spec §3.2)
        "t_available": int(rec.t_event),
        "cost_bps": float(rec.cost.cost_bps),
        "half_spread_bps": float(rec.cost.half_spread_bps),
        "uniqueness": 0.0,  # filled per horizon after assembly
        "regime": "tight" if feat.spread_tick <= boundary_spread_tick else "wide",
        "horizon": tag,
    })
    if "latency_drift_bps" in extra_cols:
        row["latency_drift_bps"] = float(rec.cost.latency_drift_bps)
    if "emitted_by_time_cap" in extra_cols:
        row["emitted_by_time_cap"] = bool(rec.emitted_by_time_cap)
    return row


def _finalize_frame(rows: list, tags: list, extra_cols: tuple) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in FEATURE_COLS:
        frame[col] = frame[col].astype("float64")
    for col in ("y_fwd_bps", "cost_bps", "half_spread_bps", "uniqueness"):
        frame[col] = frame[col].astype("float64")
    for col in ("t_event", "t_barrier", "t_feature_start", "t_available", "label"):
        frame[col] = frame[col].astype("int64")
    if "latency_drift_bps" in extra_cols:
        frame["latency_drift_bps"] = frame["latency_drift_bps"].astype("float64")
    if "emitted_by_time_cap" in extra_cols:
        frame["emitted_by_time_cap"] = frame["emitted_by_time_cap"].astype("bool")
    frame["uniqueness"] = uniqueness_by_horizon(
        frame["t_event"].to_numpy(), frame["t_barrier"].to_numpy(),
        frame["horizon"].to_numpy(object))
    return frame


# ------------------------------------------------------------ manifest assembly


def _realized_schedule_sha256(realized_schedule: list) -> str:
    return hash_obj({"schema": REALIZED_SCHEDULE_SCHEMA,
                     "days": list(realized_schedule)})


def _bar_clock_block(config: dict, realized_schedule: list,
                     realized_schedule_sha256: str) -> dict:
    return {
        "kind": G0BN_CLOCK_KIND,
        "reference_stream": G0BN_CLOCK_REFERENCE_STREAM,
        "target_bars_per_day": config["clock"]["target_bars_per_day"],
        "time_cap_ns": config["clock"]["time_cap_ns"],
        # realized evidence, deliberately ADDITIONAL fields beyond the four
        # frozen pins (eval.g0bn_freeze compares per pinned key): plan §A
        # requires the per-day VALUES plus the content hash — "not a single
        # scalar and not the hash alone" (a hash cannot recover the thresholds
        # for a rebuild/audit)
        "threshold_schedule": [dict(s) for s in realized_schedule],
        "threshold_schedule_hash": realized_schedule_sha256,
    }


def _base_manifest(config: dict, *, dataset_id: str, sources: list,
                   extra_cols: tuple, realized_schedule: list,
                   realized_schedule_sha256: str,
                   generated_at: str, dtypes: dict | None = None) -> dict:
    if dtypes is None:
        # development: pin every emitted diagnostic dtype (rebuildable, no
        # frozen contract to reproduce — stricter is better)
        dtypes = dict(G0BN_COST_DTYPES)
        if "emitted_by_time_cap" in extra_cols:
            dtypes["emitted_by_time_cap"] = "bool"
    else:
        # holdout: reproduce the plan's frozen output-contract dtypes VERBATIM —
        # adding an unpinned entry (e.g. a bool pin for an opted-in
        # emitted_by_time_cap) would fail verify_holdout_manifest_binding's
        # exact dtypes comparison post-burn; the bool's physical type is still
        # attested via the frozen Arrow schema hash (Codex round 4)
        dtypes = dict(dtypes)
    return {
        "manifest_version": 1,
        "dataset_id": dataset_id,
        "build_id": "0" * 64,  # placeholder; derived below from the final frame
        "bar_clock": _bar_clock_block(config, realized_schedule,
                                      realized_schedule_sha256),
        "time": {"unit": "ns", "timezone": "UTC"},
        "feature_cols": list(FEATURE_COLS),
        "target_cols": list(G0BN_TARGETS),
        "reserved_cols": list(RESERVED),
        "extra_cols": list(extra_cols),
        "venues": [dict(G0BN_VENUE)],
        "horizons": {h["tag"]: h["ns"] for h in config["horizons"]},
        "sources": sources,
        "generated_at": generated_at,
        "max_lookback_ns": config["features"]["max_lookback_ns"],
        "embargo_ns": config["cv"]["embargo_ns"],
        "availability_lag_ns": 0,
        "dtypes": dtypes,
    }


def _evidence_sources(config: dict, *, partition: str) -> list:
    cert = config["source_certification"]
    sources = [
        {"name": "source_certification", "sha256": cert["certification_sha256"]},
        {"name": "coverage", "sha256": cert["coverage_sha256"]},
        dict(config["costs"]["cost_assumption"], name="cost_assumption"),
        {"name": "partition_contract", "schema": "g0bn-partition-plan-v1",
         "partition": partition,
         "partition_plan_sha256": config["partition"]["sha256"]},
        {"name": "g0bn_protocol", "protocol": PROTOCOL_ID,
         "protocol_config_sha256": config["sha256"],
         "source_certification_sha256": cert["certification_sha256"],
         "horizon_roles_sha256": _horizon_roles_sha256(config),
         "instrument": dict(G0BN_INSTRUMENT)},
    ]
    if partition == "holdout":
        sources.insert(1, {"name": "custodian_seal",
                           "sha256": cert["custodian_seal_sha256"]})
    return sources


def _finalize_build_id(manifest: dict, frame: pd.DataFrame,
                       build_params: dict) -> str:
    lrh = logical_row_sha256(frame, ordered_manifest_columns(manifest))
    bid = build_id_for(dataset_id=manifest["dataset_id"],
                       logical_row_sha256=lrh, build_params=build_params)
    manifest["build_id"] = bid
    return bid


# ------------------------------------------------------------------ development


def produce_development(config: dict, *, runtime: RuntimeParams,
                        day_objects: Mapping, matrix_path, manifest_path,
                        generated_at: str,
                        extra_cols: tuple = ("latency_drift_bps",
                                             "emitted_by_time_cap"),
                        verify_schedule_pin: bool | None = None,
                        ) -> DevelopmentBuild:
    """Deterministic development build over an EXPLICIT day/object allowlist
    (`{day: {normalized product: path}}` — never a range, glob, or discovery).
    Runs the full validate-then-write path (plan §H) and returns the logical
    development data identity trials bind to.

    Realized-schedule pin (`config.clock.development_schedule_sha256`,
    `verify_schedule_pin`): with the default None, the realized threshold
    schedule hash is verified against the config pin exactly when the build
    covers the config's FULL included-day scope — a canonical, trial-bindable
    build must realize the sealed schedule, and a wrong RuntimeParams.threshold
    fails closed before anything is published. Subset/calibration builds cannot
    match a full-scope pin by construction and skip the gate (their manifests
    still REPORT the realized hash, and their build identity commits to the
    actual runtime used). Pass False only for the pin-minting bootstrap run,
    which predates the pin and publishes the realized hash the operator seals;
    pass True to force the gate on any build. The 67-E pre-burn preflight
    additionally reconciles the canonical manifest against the sealed pin
    before any burn (the one-shot's own load-bearing pin, the development-END
    clock state, IS hash-enforced by materialize_holdout)."""
    validate_protocol_config(config)
    _validate_runtime(runtime, config)
    if verify_schedule_pin is not None and not isinstance(verify_schedule_pin,
                                                          bool):
        _fail("verify_schedule_pin",
              f"must be None or a plain bool; got "
              f"{type(verify_schedule_pin).__name__}")
    _validate_generated_at(generated_at)
    if not isinstance(day_objects, Mapping) or not day_objects:
        _fail("day_objects", "must be a non-empty {day: {product: path}} mapping "
                             "(the explicit development allowlist)")
    days = sorted(day_objects)
    included = set(config["exclusions"]["included_days"])
    for day in days:
        _dt.date.fromisoformat(day)
        if day not in included:
            _fail("day_objects", f"day {day} is not in the config's outcome-blind "
                                 "included development days; excluded or "
                                 "out-of-window days may not be built")
        paths = day_objects[day]
        if not isinstance(paths, Mapping) or set(paths) != set(G0BN_DATA_SOURCES):
            _fail("day_objects", f"day {day} must map exactly the certified "
                                 f"normalized products {list(G0BN_DATA_SOURCES)}")
        for product, path in paths.items():
            _validate_source_path(f"day_objects[{day}][{product}]", path)

    part = config["partition"]
    schedule = ThresholdSchedule(runtime.threshold)
    build = _build_frame(
        config, runtime, partition="development", days=days,
        paths_for_day=lambda day: day_objects[day], schedule=schedule,
        seeded_history=[],
        partition_start_ns=int(part["development_start_ns"]),
        partition_end_ns=int(part["development_end_ns"]),
        extra_cols=tuple(extra_cols))

    schedule_sha = _realized_schedule_sha256(build.realized_schedule)
    enforce_pin = (verify_schedule_pin if verify_schedule_pin is not None
                   else sorted(days) == sorted(included))
    if enforce_pin:
        pinned = config["clock"]["development_schedule_sha256"]
        if schedule_sha != pinned:
            _fail("realized threshold schedule",
                  f"hash {schedule_sha} does not reproduce the config's "
                  f"development_schedule_sha256 pin {pinned} for a canonical "
                  "full-scope build; a trial-bindable development matrix must "
                  "realize exactly the sealed schedule (only the pin-minting "
                  "bootstrap run may pass verify_schedule_pin=False)")
    sources = [
        {"name": product, "day": day, "sha256": build.source_sha256s[day][product]}
        for day in days for product in G0BN_DATA_SOURCES
    ] + _evidence_sources(config, partition="development")
    manifest = _base_manifest(
        config, dataset_id=G0BN_DEV_DATASET_ID, sources=sources,
        extra_cols=tuple(extra_cols),
        realized_schedule=build.realized_schedule,
        realized_schedule_sha256=schedule_sha,
        generated_at=generated_at)
    build_params = {
        "builder": PRODUCER_VERSION,
        "source_mode": BINANCE_SINGLE_VENUE,
        "partition": "development",
        "protocol_config_sha256": config["sha256"],
        "days": list(days),
        "runtime": runtime.as_dict(),
    }
    _finalize_build_id(manifest, build.frame, build_params)
    write = write_development(build.frame, manifest, build_params=build_params,
                             matrix_path=matrix_path, manifest_path=manifest_path)
    identity = development_data_identity({
        "development_dataset_id": write.dataset_id,
        "development_build_id": write.build_id,
        "development_manifest_sha256": write.manifest_sha256,
        "development_logical_row_sha256": write.logical_row_sha256,
        "partition_plan_sha256": part["sha256"],
    })
    state = _clock_state(runtime.threshold, build.history)
    return DevelopmentBuild(
        write=write, data_identity=identity, row_counts=build.row_counts,
        drop_counts=build.drop_counts,
        realized_threshold_schedule=build.realized_schedule,
        realized_threshold_schedule_sha256=schedule_sha,
        clock_state=state, clock_state_sha256=clock_state_sha256(state))


def _validate_source_path(ctx: str, path) -> None:
    p = os.fspath(path)
    if not isinstance(p, str):
        _fail(ctx, f"must be a filesystem path; got {type(path).__name__}")
    if any(g in p for g in _GLOB_CHARS):
        _fail(ctx, f"path {p!r} contains glob metacharacters; the producer accepts "
                   "exact object paths only (no ranges, globs, or discovery)")
    if not os.path.isfile(p):
        _fail(ctx, f"path {p!r} is not an existing regular file")


# ---------------------------------------------------------------- blind holdout


def _load_raw_access_claim(path, *, config: dict, plan: dict, freeze: dict) -> dict:
    """Consume the durable `g0bn-raw-access-claim-v1` the 67-E runner created via
    O_EXCL+fsync BEFORE invoking this materializer. The claim must bind exactly
    this transaction/config/plan/freeze (spec section 6.2: both claims carry the
    same universe/transaction IDs and config/freeze/plan hashes); anything else
    is an unclaimed or foreign invocation and fails closed before any source
    open."""
    ctx = "raw access claim"
    p = os.fspath(path)
    if not os.path.isfile(p):
        _fail(ctx, f"{p!r} is not an existing regular file; the materializer "
                   "cannot run unclaimed (the raw-access burn precedes the first "
                   "January source read)")
    with open(p, "r", encoding="utf-8") as f:
        claim = json.load(f)
    _dict(ctx, claim, _RAW_CLAIM_FIELDS)
    _exact(f"{ctx}.schema", claim["schema"], RAW_ACCESS_CLAIM_SCHEMA)
    _exact(f"{ctx}.holdout_universe_id", claim["holdout_universe_id"],
           plan["holdout_universe_id"])
    _exact(f"{ctx}.transaction_id", claim["transaction_id"], plan["transaction_id"])
    _exact(f"{ctx}.protocol_config_sha256", claim["protocol_config_sha256"],
           config["sha256"])
    _exact(f"{ctx}.holdout_plan_sha256", claim["holdout_plan_sha256"],
           plan["sha256"])
    _exact(f"{ctx}.freeze_sha256", claim["freeze_sha256"], freeze["sha256"])
    _validate_generated_at(claim["generated_at"])
    embedded = _sha256(f"{ctx}.sha256", claim["sha256"])
    recomputed = g0bn_artifact_sha256(claim)
    if embedded != recomputed:
        _fail(f"{ctx}.sha256", f"embedded claim sha256 does not match the "
                               f"canonical content (tampered or partial): "
                               f"{embedded} != {recomputed}")
    return claim


def _validate_object_paths(object_paths: Mapping, normalized: list) -> dict:
    """Exact one-to-one mapping between the sealed normalized allowlist and the
    caller-supplied object paths — no opens happen here. Missing, extra, or
    path-duplicated objects reject before any payload access; day/product are
    never caller-supplied (they come from the sealed allowlist entry)."""
    ctx = "object_paths"
    if not isinstance(object_paths, Mapping):
        _fail(ctx, f"must be a {{object_id: path}} mapping of exactly the sealed "
                   f"normalized allowlist; got {type(object_paths).__name__}")
    expected = {o["object_id"] for o in normalized}
    got = set(object_paths)
    missing = sorted(expected - got)
    if missing:
        _fail(ctx, f"missing sealed normalized object path(s): {missing[:3]}"
                   f"{'...' if len(missing) > 3 else ''}; the materializer "
                   "consumes the complete sealed scope, never a subset")
    extra = sorted(got - expected)
    if extra:
        _fail(ctx, f"unknown object id(s) {extra[:3]}"
                   f"{'...' if len(extra) > 3 else ''}: not in the plan's sealed "
                   "normalized allowlist (no fallback or discovered sources)")
    resolved = {}
    for obj in normalized:
        oid = obj["object_id"]
        _validate_source_path(f"{ctx}[{oid!r}]", object_paths[oid])
        if os.path.islink(os.fspath(object_paths[oid])):
            _fail(f"{ctx}[{oid!r}]",
                  f"{os.fspath(object_paths[oid])!r} is a symlink; sealed "
                  "objects are opened O_NOFOLLOW as regular files (no link "
                  "indirection into or out of custody)")
        real = os.path.realpath(os.fspath(object_paths[oid]))
        if real in resolved:
            _fail(ctx, f"objects {resolved[real]!r} and {oid!r} resolve to the "
                       f"same file {real!r}; sealed objects are distinct")
        resolved[real] = oid
    return {obj["object_id"]: os.fspath(object_paths[obj["object_id"]])
            for obj in normalized}


def _atomic_write_json(payload: dict, path) -> None:
    """Fresh atomic durable publication: exclusive temp write + fsync, atomic
    rename into place, directory fsync. A partial attestation is never observable
    at the final path, and the final artifact is never reopened here."""
    final = os.fspath(path)
    tmp = final + ".tmp"
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(final):
        raise FileExistsError(
            f"refusing to publish over an existing artifact: {final!r} (the "
            "one-shot materializer requires fresh outputs)")
    os.rename(tmp, final)
    _fsync_dir(final)


def materialize_holdout(*, config: dict, plan: dict, freeze: dict,
                        inventory: dict, runtime: RuntimeParams,
                        clock_state: dict, raw_access_claim_path,
                        object_paths: Mapping, matrix_path, manifest_path,
                        attestation_path, generated_at: str
                        ) -> HoldoutMaterialization:
    """THE sole blind-materializer boundary for the 67-E one-shot runner (spec
    section 6.3 steps 4-5). Only the runner may call it, after the raw-access
    burn; it is never exposed through a generic runner or CLI.

    Ordering contract (read spies pin it): all metadata validation runs first
    with zero data access; the durable raw-access claim is the first file read;
    every sealed object is then pinned to one held descriptor and hash-verified
    against its custody pin before any payload decode, and every decode streams
    from the same verified descriptor; the frozen recipe streams exactly once
    through `eval.writer.write_holdout` (fresh O_EXCL outputs, every hash
    computed while writing); and the attestation is atomically written and
    fsynced last. No derived matrix/manifest/parquet footer is ever reopened.

    Scope claim (least privilege): this materializer consumes and physically
    verifies the NORMALIZED layer of the sealed allowlist only — exactly the
    scope `eval.g0bn_freeze.verify_holdout_manifest_binding` audits one-to-one.
    It never receives raw object paths and must not: raw-layer custody is
    verified by the #68 seal evidence and the 67-E pre-burn preflight, and
    handing the operator-plane materializer raw payload access it does not need
    would widen the custody boundary, not tighten it. The attestation binds the
    plan hash (which pins BOTH layers' sealed hashes) but attests consumption
    of the normalized scope only. Spec section 6.3 step 4's "opens only those
    allowlisted January raw/normalized objects" (like the plan table's "accepts
    only the exact frozen raw/normalized allowlist") is an EXCLUSIVITY bound —
    nothing outside the allowlist may be opened — not an obligation to open
    every allowlisted object: T9 derives nothing from raw payloads, so opening
    them here would widen the burn surface without adding integrity the sealed
    hash chain does not already provide, and raw-layer intactness remains
    claimed (and independently auditable) through the #68 seal evidence rather
    than through this attestation."""
    validate_protocol_config(config)
    if (not isinstance(plan, dict)
            or list(plan.get("drop_count_categories", [])) != list(DROP_COUNT_CATEGORIES)):
        _fail("plan.drop_count_categories",
              f"pinned count schema "
              f"{plan.get('drop_count_categories') if isinstance(plan, dict) else plan!r} "
              f"does not match this producer's taxonomy {list(DROP_COUNT_CATEGORIES)}; "
              "the attested counts would be unclassifiable")
    # custody-anchored plan+freeze validation, and the exact manifest binding the
    # published OOS manifest must carry (validates freeze/plan/inventory/config)
    plan_binding = holdout_plan_binding(plan, freeze, config=config,
                                        inventory=inventory)
    oc = plan["output_contract"]
    extra_cols = tuple(oc["extra_cols"])
    _validate_runtime(runtime, config)
    # config-deterministic cost identity: a mistyped/copied assumption source
    # must fail in this no-I/O preflight, never after the claim read and the
    # sealed-object hash pass (Codex round 25)
    _cost_assumption(config)
    _validate_generated_at(generated_at)
    part = config["partition"]
    holdout_start_ns = int(part["holdout_start_ns"])
    holdout_end_ns = int(part["holdout_end_ns"])
    _validate_clock_state(clock_state, runtime=runtime, config=config,
                          before_ns=holdout_start_ns)

    # fresh-output preflight — pure path metadata (an allowed preflight input,
    # spec section 5.1), so it runs BEFORE the claim read: a predictable
    # output-path failure must never consume the claim or touch any source.
    # write_holdout re-checks matrix/manifest with O_EXCL.
    fresh = [os.fspath(matrix_path), os.fspath(manifest_path),
             os.fspath(attestation_path), os.fspath(attestation_path) + ".tmp"]
    resolved_outputs: dict[str, str] = {}
    for p in fresh:
        if os.path.islink(p):
            _fail("output paths",
                  f"{p!r} is a symlink; one-shot outputs are created fresh at "
                  "real paths only (a dangling link defeats the existence "
                  "preflight and would fail O_EXCL only after the raw burn)")
        real = os.path.realpath(p)
        if real in resolved_outputs:
            _fail("output paths",
                  f"{resolved_outputs[real]!r} and {p!r} resolve to the same "
                  f"file {real!r}; the matrix, manifest, attestation, and "
                  "attestation-temp paths must be pairwise distinct (an "
                  "aliasing path template would otherwise fail only mid-write, "
                  "after the raw burn, stranding an unattested artifact)")
        resolved_outputs[real] = p
        parent = os.path.dirname(real) or os.curdir
        if not os.path.isdir(parent):
            _fail("output paths",
                  f"parent directory {parent!r} of output path {p!r} is not an "
                  "existing directory; a mistyped output location must fail "
                  "before the raw burn, not mid-write after it")
        if not os.access(parent, os.W_OK | os.X_OK):
            _fail("output paths",
                  f"parent directory {parent!r} of output path {p!r} is not "
                  "writable; the one-shot publication would fail only after "
                  "the raw burn")
    existing = [p for p in fresh if os.path.exists(p)]
    if existing:
        raise FileExistsError(
            f"refusing blind materialization onto existing output path(s): "
            f"{existing}; the one-shot write requires fresh derived artifacts")

    # the durable claim is the FIRST file this materializer reads
    claim = _load_raw_access_claim(raw_access_claim_path, config=config,
                                   plan=plan, freeze=freeze)

    normalized = [o for o in plan["object_allowlist"] if o["layer"] == "normalized"]
    paths_by_id = _validate_object_paths(object_paths, normalized)

    # pin every sealed object to ONE held descriptor (the first source opens —
    # strictly after the claim read above), hash-verify the COMPLETE scope over
    # those descriptors, and decode only from the same descriptors: a foreign
    # object never reaches a parser, and a path swapped after verification can
    # never substitute unverified bytes for the parse (by-name reopen TOCTOU)
    by_day: dict[str, dict] = {}
    for obj in normalized:
        by_day.setdefault(obj["day"], {})[obj["product"]] = obj
    pinned: dict[str, _PinnedObject] = {}
    try:
        for obj in normalized:
            pinned[obj["object_id"]] = _PinnedObject(paths_by_id[obj["object_id"]])
        sealed_sha: dict[str, dict] = {}
        for day in plan["included_days"]:
            sealed_sha[day] = {}
            for product in G0BN_DATA_SOURCES:
                obj = by_day[day][product]
                actual = pinned[obj["object_id"]].sha256()
                if actual != obj["sha256"]:
                    _fail(f"object_paths[{obj['object_id']!r}]",
                          f"content hash {actual} does not match the sealed "
                          f"custody pin {obj['sha256']} (foreign or corrupted "
                          "object); no payload was decoded")
                sealed_sha[day][product] = actual

        days = list(plan["included_days"])
        schedule = ThresholdSchedule(runtime.threshold)
        for entry in clock_state["history"]:
            schedule.record_day(entry["day"], float(entry["completed_notional"]),
                                float(entry["covered_fraction"]))
        build = _build_frame(
            config, runtime, partition="holdout", days=days,
            paths_for_day=lambda day: {p: pinned[by_day[day][p]["object_id"]]
                                       for p in G0BN_DATA_SOURCES},
            schedule=schedule, seeded_history=list(clock_state["history"]),
            partition_start_ns=holdout_start_ns, partition_end_ns=holdout_end_ns,
            extra_cols=extra_cols,
            verify_sha_for_day=lambda day: dict(sealed_sha[day]))
    finally:
        for po in pinned.values():
            po.close()

    schedule_sha = _realized_schedule_sha256(build.realized_schedule)
    sources = [
        {"name": obj["product"], "object_id": obj["object_id"],
         "day": obj["day"], "sha256": obj["sha256"]}
        for obj in normalized
    ] + _evidence_sources(config, partition="holdout") + [dict(plan_binding)]
    manifest = _base_manifest(
        config, dataset_id=G0BN_OOS_DATASET_ID, sources=sources,
        extra_cols=extra_cols,
        realized_schedule=build.realized_schedule,
        realized_schedule_sha256=schedule_sha,
        generated_at=generated_at, dtypes=oc["dtypes"])
    base_params = {
        "builder": PRODUCER_VERSION,
        "source_mode": BINANCE_SINGLE_VENUE,
        "partition": "holdout",
        "protocol_config_sha256": config["sha256"],
        "days": list(days),
        "runtime": runtime.as_dict(),
    }
    build_params = oos_build_params(plan, base_params, config=config,
                                    inventory=inventory)
    _finalize_build_id(manifest, build.frame, build_params)
    # manifest-only pre-write audit against the plan's frozen output contract
    verify_holdout_manifest_binding(manifest, plan, freeze, config=config,
                                    inventory=inventory)

    write = write_holdout(build.frame, manifest, build_params=build_params,
                          matrix_path=matrix_path, manifest_path=manifest_path)
    if write.physical_schema_sha256 != oc["expected_physical_schema_sha256"]:
        _fail("physical_schema_sha256",
              f"written Arrow schema hash {write.physical_schema_sha256} does not "
              f"match the plan's frozen pin {oc['expected_physical_schema_sha256']}"
              " (post-burn: the runner records this as terminal INCONCLUSIVE)")

    end_state = _clock_state(runtime.threshold, build.history)
    counts = {"schema": COUNTS_SCHEMA, "row_count": write.row_count,
              "row_counts": build.row_counts, "drop_counts": build.drop_counts}
    cert = config["source_certification"]
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "pilot_id": config["pilot_id"],
        "holdout_universe_id": plan["holdout_universe_id"],
        "transaction_id": plan["transaction_id"],
        "protocol_config_sha256": config["sha256"],
        "source_certification_sha256": cert["certification_sha256"],
        "custodian_seal_sha256": cert["custodian_seal_sha256"],
        "partition_plan_sha256": part["sha256"],
        "holdout_plan_sha256": plan["sha256"],
        "freeze_sha256": freeze["sha256"],
        "raw_access_claim_sha256": claim["sha256"],
        "dataset_id": write.dataset_id,
        "build_id": write.build_id,
        "logical_row_sha256": write.logical_row_sha256,
        "manifest_sha256": write.manifest_sha256,
        "matrix_file_sha256": write.matrix_file_sha256,
        "physical_schema_sha256": write.physical_schema_sha256,
        # audit context, deliberately hash-bearing: the attestation is a one-shot
        # record of THIS materialization (67-E preflights/pins the output paths
        # pre-burn and binds this attestation hash into the matrix claim as
        # produced); rematerialization identity lives in build_id/logical-row/
        # counts/schedule hashes, never in the attestation self-hash
        "matrix_path": str(write.matrix_path),
        "manifest_path": str(write.manifest_path),
        "row_count": write.row_count,
        "row_counts": dict(build.row_counts),
        "drop_counts": {c: dict(t) for c, t in build.drop_counts.items()},
        "drop_count_categories": list(DROP_COUNT_CATEGORIES),
        "counts_sha256": hash_obj(counts),
        "days_built": list(days),
        "n_days_built": len(days),
        "realized_threshold_schedule": list(build.realized_schedule),
        "realized_threshold_schedule_sha256": schedule_sha,
        "realized_clock_state_sha256": clock_state_sha256(end_state),
        "generated_at": generated_at,
    }
    attestation["sha256"] = g0bn_artifact_sha256(attestation)
    _atomic_write_json(attestation, attestation_path)
    return HoldoutMaterialization(
        write=write, attestation=attestation,
        attestation_path=os.fspath(attestation_path),
        attestation_sha256=attestation["sha256"],
        row_counts=build.row_counts, drop_counts=build.drop_counts,
        realized_threshold_schedule=build.realized_schedule,
        realized_threshold_schedule_sha256=schedule_sha,
        clock_state_sha256=attestation["realized_clock_state_sha256"])
