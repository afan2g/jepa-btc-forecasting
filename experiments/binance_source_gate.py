"""Issue #64 — Binance source-quality gate: experiment-scoped library.

Everything here is EXPERIMENT-SCOPED (never imported by recon/, ingest/, eval/, or any
production script) and OFFLINE (no vendor import at module top, no network access on
import, in unit tests, or in any dry/validate/replay/compare path — the only code that
can touch the network is the `fetch` opener built lazily inside
scripts/run_binance_source_gate.py AFTER the explicit user-approval checkpoint).

Three parts:

  1. A CryptoHFTData adapter + causal replayer for the documented CommonOrderbookEvent
     hourly Parquet schema (snapshots + incremental updates with Binance first/final/
     previous update-ID semantics). It normalizes into the EXISTING internal top-K
     frame contract (recon.orderbook.OrderBook snapshot assembly: mid, microprice,
     bid_i_price/size, ask_i_price/size, sample_ts) and FAILS CLOSED on every anomaly
     preregistered in experiments/preregistration_64.json `replay_contract.cryptohftdata`
     (missing/malformed/truncated/crossed/stale snapshot, update-ID gap, incompatible
     overlap, backwards/future snapshot, identity mismatch, missing/duplicate hour,
     timescale ambiguity, ordering regressions over the preregistered bound).
  2. Crypto Lake integrity helpers for the 2026-04-01 Stage-2 certification run: input
     identity (sha256/schema/fingerprint vs the preregistered Stage-1 pin), tick-scale
     measurement, delta-stream silence, frozen-book metrics, and the Stage-2 CLI
     cross-run determinism comparator.
  3. The fixed independent-source comparison (identical 1 s exchange-time grids, tick
     space, k=10) plus the April holdout guard: `assert_report_publishable` refuses any
     report carrying an outcome-bearing key or an unbounded per-sample series, so a
     forbidden April metric cannot be published even by accident.

The PREREGISTERED constant mirrors experiments/preregistration_64.json (thresholds,
decision logic, fixture identity) and is pinned equal to the artifact by
tests/test_binance_source_gate.py — the #54 pattern (experiments/snapshot_seed.py).
No threshold below may change after any decision-bearing real-data result is seen.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from math import isfinite

import numpy as np
import pandas as pd

from experiments.snapshot_seed import frame_replay_hash  # deterministic top-K content hash
from eval.hashing import hash_obj

# ----------------------------------------------------------------------------- preregistration pin
# Mirrors experiments/preregistration_64.json; pinned equal by
# tests/test_binance_source_gate.py::TestPreregistration. Committed (633d4e4) BEFORE any
# decision-bearing real-data metric ran on this branch.
PREREGISTERED = {
    "fixture_identity": {
        "lake_day": "2026-04-01",
        "lake_n_units": 9,
        "lake_rows_total": 147377429,
        "lake_out_bytes_total": 687215789,
        "chd_probe_object": "binance_futures/2026-04-01/12/BTCUSDT_orderbook.parquet.zst",
    },
    "thresholds": {
        "lake_day_quality": {
            "crossed_usable_max": 0.01,
            "missing_usable_max": 0.02,
            "thin_usable_max": 0.10,
            "seed_crossed_frac_max": 0.05,
            "source": "frozen production bars: scripts/run_binance_recon.py::Thresholds == "
                      "run_coinbase_quality_map thresholds (docs 5a-QualityMap); NOT chosen by "
                      "this experiment",
        },
        "anomaly_caps": {
            "off_tick": "no d <= 4 renders all prices integral -> verdict capped at degraded",
            "silence_gap_s_cap": 300.0,
            "silence_cap_effect": "any book_delta_v2 inter-event gap > 300s -> verdict capped "
                                  "at degraded",
            "frozen_fraction_max": 0.02,
            "frozen_cap_effect": "frozen_fraction > 0.02 -> verdict capped at degraded",
        },
        "hard_invalidators": {
            "input_identity": "raw unit sha256 or schema fingerprint != fixture pin -> "
                              "inconclusive (wrong input)",
            "schema_drift": "observed columns incompatible with expected_schemas -> inconclusive",
            "determinism": "any cross-run inequality (excluding secs/ts keys) -> inconclusive",
            "native_conformance": "python/native frame or meta divergence when the conformance "
                                  "arm runs -> inconclusive",
        },
        "chd_window_quality": {
            "crossed_usable_max": 0.01,
            "missing_usable_max": 0.02,
            "thin_usable_max": 0.10,
            "seed_crossed_frac_max": 0.05,
            "note": "same frozen bars applied to the CryptoHFTData reconstruction on its "
                    "approved windows (seed_crossed_frac over its snapshot events)",
        },
        "comparison": {
            "joint_valid_fraction_min": 0.97,
            "touch_agreement_within_1_tick_min": 0.90,
            "mid_abs_diff_ticks_p50_max": 1.0,
            "mid_abs_diff_ticks_p99_max": 10.0,
            "basis": "both sources sampled as-of the same exchange-time 1s grid, k=10, tick "
                     "space at the shared conformance scale; agreement metrics computed over "
                     "joint-valid samples; exact-tick agreement and depth overlap recorded "
                     "descriptively",
        },
    },
    "decision_logic": {
        "lake_unit_requirements": {
            "BINANCE_FUTURES/BTC-USDT-PERP": {"topk_l2": "certified", "trades": "ok",
                                              "funding": "ok", "open_interest": "ok",
                                              "liquidations": "ok or missing-sparse"},
            "BINANCE/BTC-USDT": {"topk_l2": "certified", "trades": "ok"},
        },
        "lake_verdict": {
            "certified": "every unit meets lake_unit_requirements AND no anomaly cap fired AND "
                         "no hard invalidator fired AND determinism passed AND schema/tick "
                         "conformance passed",
            "degraded": "units complete but any topk_l2 classifies degraded, or any anomaly cap "
                        "fired (off-tick / silence / frozen), with no hard invalidator",
            "inconclusive": "any hard invalidator fired, any required unit missing or errored, "
                            "any topk_l2 classifies inconclusive (seed gate), or the run cannot "
                            "complete",
        },
        "chd_verdict": {
            "certified": "approved window(s) pass validation, the fail-closed replay completes "
                         "with zero continuity violations, and chd_window_quality bars pass",
            "degraded": "replay completes but a chd_window_quality bar fails",
            "inconclusive": "validation or fail-closed replay refuses the data (missing/"
                            "malformed snapshot, gap, overlap, identity failure), or the window "
                            "was never approved/downloaded",
        },
        "final_source_decision": {
            "lake_go": "lake_verdict == certified AND (comparison not executable OR comparison "
                       "bars pass). If comparison bars pass: 'Crypto Lake approved for the "
                       "pilot, independently validated; CryptoHFTData recorded as agreeing "
                       "fallback candidate'. If CHD data was unusable (chd inconclusive on its "
                       "own data): 'Crypto Lake approved on internal certification only; "
                       "independent validation impossible; fallback none' - recorded explicitly",
            "chd_go": "lake_verdict != certified AND chd_verdict == certified on the approved "
                      "April windows: 'CryptoHFTData approved instead; #35 source and manifest "
                      "contract must switch' (subject to the non-April escalation when April "
                      "cannot carry the decision alone)",
            "neither": "lake_verdict != certified AND chd_verdict != certified -> neither "
                       "source approved; #35 stays blocked; open a separately scoped "
                       "vendor/pivot decision",
            "disagreement": "lake_verdict == certified AND chd_verdict == certified AND "
                            "comparison bars FAIL -> the April day cannot attribute fault; NO "
                            "GO is issued for either source from April data alone; escalate "
                            "per escalation.dev_days",
        },
    },
}

PREREG_ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "preregistration_64.json")


def load_preregistration(path: str = PREREG_ARTIFACT_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- error taxonomy
class SourceGateError(RuntimeError):
    """Fail-closed refusal. `code` is the stable reason code recorded in reports."""
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class ChdValidationError(SourceGateError):
    """Schema / identity / partition / timescale refusal (before any replay)."""


class ChdSnapshotError(SourceGateError):
    """Missing / malformed / truncated / one-sided / crossed / stale / future snapshot."""


class ChdContinuityError(SourceGateError):
    """Update-ID gap, incompatible overlap, backwards ids, ordering anomaly."""


# ----------------------------------------------------------------------------- CHD schema contract
# The documented CommonOrderbookEvent fields (https://www.cryptohftdata.com/docs/python-orderbook,
# pinned verbatim in preregistration_64.json `expected_schemas.cryptohftdata_common_orderbook_event`).
CHD_REQUIRED_COLUMNS = (
    "received_time", "event_time", "transaction_time", "symbol", "event_type",
    "first_update_id", "final_update_id", "prev_final_update_id", "last_update_id",
    "side", "price", "quantity",
)
CHD_OPTIONAL_COLUMNS = ("order_count",)
CHD_EVENT_TYPES = frozenset({"snapshot", "update"})
CHD_SIDES = frozenset({"bid", "ask"})

# Preregistered timescale detection window (replay_contract.cryptohftdata.timestamp_units).
_TS_NS_LO = int(pd.Timestamp("2020-01-01", tz="UTC").value)
_TS_NS_HI = int(pd.Timestamp("2030-01-01", tz="UTC").value)
_TS_MULTIPLIERS = (1, 10**3, 10**6, 10**9)   # ns, us, ms, s -> ns

# Preregistered ordering-anomaly bound, TIGHTENED by the 2026-07-11 amendment (added before
# any CryptoHFTData object was downloaded or replayed): an event that is APPLIED to the book
# (accepted update, or snapshot seed/reset) must never carry a raw event_time behind the
# monotone watermark — an applied regression is a hard per-event refusal, because the
# aggregate bound alone would let a single time-reordered stale update mutate a freshly
# reseeded book silently (adversarial-review finding). The fractional bound below applies
# only to NON-APPLIED events (pre-snapshot skips, deduped duplicates), where a regression is
# recorded but harmless.
MAX_EVENT_TIME_REGRESSION_FRAC = 0.001

MIN_SNAPSHOT_LEVELS_PER_SIDE = 5      # production seed-validity floor (CLI --seed-min-levels)
IDENTITY_IN_HOUR_MIN_FRAC = 0.999     # >=99.9% of rows inside the nominal hour on one time axis

NS = 1_000_000_000


def normalize_epoch_ns(values: np.ndarray, *, fieldname: str) -> np.ndarray:
    """int64-ns view of an epoch column whose timescale is exchange/vendor dependent.

    Deterministic magnitude rule (preregistered): the UNIQUE multiplier in {1, 1e3, 1e6, 1e9}
    that lands the median positive value inside [2020-01-01, 2030-01-01) ns. The candidate
    ranges are ~1e6 apart so at most one multiplier can fit; none fitting refuses the file."""
    try:
        v = np.asarray(values, dtype="int64")
    except (ValueError, TypeError, OverflowError) as e:
        # null/object/non-numeric vendor timestamps must REFUSE with a stable code, never
        # escape as a raw numpy error past the SourceGateError-only refusal path
        raise ChdValidationError(
            "malformed_timestamp",
            f"{fieldname}: non-integer values ({type(e).__name__}: {e})"[:300]) from e
    pos = v[v > 0]
    if len(pos) == 0:
        raise ChdValidationError("timescale_undetectable", f"{fieldname}: no positive values")
    med = float(np.median(pos))
    for mult in _TS_MULTIPLIERS:
        if _TS_NS_LO <= med * mult < _TS_NS_HI:
            return v * np.int64(mult)
    raise ChdValidationError(
        "timescale_undetectable",
        f"{fieldname}: median {med} fits no ns/us/ms/s epoch in [2020, 2030)")


# ----------------------------------------------------------------------------- decimal / tick rules
def parse_decimal(text: str, *, field: str = "decimal") -> Decimal:
    """Vendor decimal string -> finite Decimal, fail closed: any unparseable or non-finite
    value raises ChdValidationError('malformed_decimal') so the CLI refusal path (which
    catches only SourceGateError) writes the preregistered inconclusive report instead of
    crashing on a raw decimal.InvalidOperation/ValueError."""
    try:
        d = Decimal(text)
    except (InvalidOperation, ValueError, TypeError) as e:
        raise ChdValidationError("malformed_decimal",
                                 f"unparseable {field} {text!r}") from e
    if not d.is_finite():
        raise ChdValidationError("malformed_decimal", f"non-finite {field} {text!r}")
    return d


def decimal_places(text: str) -> int:
    """Exact decimal places of a vendor decimal string after normalization (trailing zeros
    stripped): '50000.10' -> 1, '50000.05' -> 2, '50000' -> 0. Raises on a non-decimal."""
    exp = parse_decimal(text).normalize().as_tuple().exponent
    return max(0, -int(exp))


def to_ticks(text: str, scale: int) -> int:
    """Exact integer ticks of a vendor decimal string at `scale` (=10^d). Never round-trips
    through float; an off-scale price refuses rather than rounding; a malformed/non-finite
    price refuses with a stable code (never a raw decimal exception)."""
    scaled = parse_decimal(text, field="price") * scale
    ticks = int(scaled)
    if scaled != ticks:
        raise ChdValidationError("off_tick", f"{text!r} is not integral at scale {scale}")
    return ticks


def measure_float_price_scale(prices: np.ndarray, *, expected_decimals: int,
                              max_decimals: int = 4) -> dict:
    """Tick measurement for FLOAT price arrays (the Lake raw store) per the preregistered
    tick_rules.lake_measurement: measured d = smallest d in {0..max_decimals} with
    max(|p*10^d - round(p*10^d)|) <= 1e-6; the conformance scale is 10^max(d_measured,
    d_expected). `off_tick_at_expected` counts prices not integral at the EXPECTED scale."""
    p = np.asarray(prices, dtype="float64")
    if len(p) == 0 or not np.isfinite(p).all():
        return {"ok": False, "reason": "empty_or_nonfinite_prices", "measured_decimals": None,
                "conformance_scale": None, "off_tick_at_expected": None, "n_prices": int(len(p))}
    measured = None
    for d in range(max_decimals + 1):
        s = 10.0 ** d
        if np.max(np.abs(p * s - np.round(p * s))) <= 1e-6:
            measured = d
            break
    exp_s = 10.0 ** expected_decimals
    off_expected = int(np.count_nonzero(np.abs(p * exp_s - np.round(p * exp_s)) > 1e-6))
    if measured is None:
        return {"ok": False, "reason": "no_integral_scale", "measured_decimals": None,
                "conformance_scale": None, "off_tick_at_expected": off_expected,
                "n_prices": int(len(p))}
    return {"ok": True, "reason": None, "measured_decimals": measured,
            "conformance_scale": 10 ** max(measured, expected_decimals),
            "off_tick_at_expected": off_expected, "n_prices": int(len(p))}


# ----------------------------------------------------------------------------- CHD event model
@dataclass(frozen=True)
class ChdEvent:
    """One grouped CommonOrderbookEvent: a snapshot (full book) or an update (level changes).
    Levels are (price_ticks, size) with size as float64 from the exact Decimal."""
    kind: str                          # "snapshot" | "update"
    order_id: int                      # final_update_id (update) / last_update_id (snapshot)
    event_time_ns: int
    first_update_id: int | None
    final_update_id: int | None
    prev_final_update_id: int | None
    last_update_id: int | None
    bids: tuple[tuple[int, float], ...]
    asks: tuple[tuple[int, float], ...]


def validate_chd_frame(df: pd.DataFrame, *, exchange: str, symbol: str, date_iso: str,
                       hour: int) -> dict:
    """Schema + identity validation of ONE hourly CommonOrderbookEvent frame (fail closed).

    Checks (preregistered `replay_contract.cryptohftdata.identity` + expected_schemas):
      * every required column present (extras recorded, never silently consumed);
      * event_type/side vocabularies exact;
      * symbol uniform and equal to the expected symbol;
      * >= 99.9% of rows inside the nominal hour on received_time OR event_time (whichever
        axis matches is recorded; both failing refuses the file — wrong date/hour/partition).
    Returns an identity report dict; raises ChdValidationError on any refusal."""
    missing = [c for c in CHD_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ChdValidationError("schema_missing_columns", f"missing {missing}; "
                                 f"saw {list(df.columns)}")
    extras = [c for c in df.columns
              if c not in CHD_REQUIRED_COLUMNS and c not in CHD_OPTIONAL_COLUMNS]
    if len(df) == 0:
        raise ChdValidationError("empty_partition", f"{exchange}/{date_iso}/{hour:02d} is empty")
    kinds = set(df["event_type"].astype(str).unique())
    if not kinds <= CHD_EVENT_TYPES:
        raise ChdValidationError("unknown_event_type",
                                 f"unexpected event_type values {sorted(kinds - CHD_EVENT_TYPES)}")
    sides = set(df["side"].astype(str).unique())
    if not sides <= CHD_SIDES:
        raise ChdValidationError("unknown_side", f"unexpected side values {sorted(sides - CHD_SIDES)}")
    symbols = set(df["symbol"].astype(str).unique())
    if symbols != {symbol}:
        raise ChdValidationError("wrong_symbol", f"expected {{{symbol!r}}}, saw {sorted(symbols)}")

    hour_start = int(pd.Timestamp(f"{date_iso}T{hour:02d}:00:00", tz="UTC").value)
    hour_end = hour_start + 3600 * NS
    axis_frac = {}
    for axis in ("received_time", "event_time"):
        t = normalize_epoch_ns(df[axis].to_numpy(), fieldname=axis)
        axis_frac[axis] = float(np.mean((t >= hour_start) & (t < hour_end)))
    # transaction_time is nullable but part of the preregistered timestamp contract:
    # non-null values must be integer epochs at a detectable timescale — replay never
    # reads this column, so validation is the only fail-closed check it gets
    try:
        tx = pd.array(df["transaction_time"], dtype="Int64")
    except (ValueError, TypeError) as e:
        raise ChdValidationError(
            "malformed_timestamp",
            f"transaction_time: non-integer values ({type(e).__name__}: {e})"[:300]) from e
    tx_non_null = tx.dropna()
    if len(tx_non_null):
        normalize_epoch_ns(tx_non_null.to_numpy(dtype="int64"),
                           fieldname="transaction_time")
    matched = [a for a, frac in axis_frac.items() if frac >= IDENTITY_IN_HOUR_MIN_FRAC]
    if not matched:
        raise ChdValidationError(
            "wrong_partition_window",
            f"neither received_time nor event_time places >={IDENTITY_IN_HOUR_MIN_FRAC:.1%} of "
            f"rows in {date_iso}T{hour:02d}Z (fractions {axis_frac}) — wrong exchange/date/hour "
            "object or unexpected partition axis")
    return {"exchange": exchange, "symbol": symbol, "date": date_iso, "hour": hour,
            "rows": int(len(df)), "extra_columns": extras,
            "partition_axis": matched[0], "in_hour_fraction": axis_frac,
            "has_order_count": "order_count" in df.columns,
            "transaction_time_non_null": int(len(tx_non_null)),
            "event_type_rows": {k: int((df["event_type"] == k).sum()) for k in sorted(kinds)}}


def _group_events(df: pd.DataFrame, *, price_scale: int) -> list[ChdEvent]:
    """Group level rows into ChdEvents and order them by the preregistered event order:
    order_id ascending (update final_update_id / snapshot last_update_id), snapshots AFTER an
    update sharing the same id (authoritative overwrite — the Lake merge convention).

    Fail-closed grouping rules (replay_contract.cryptohftdata):
      * id columns must be present per kind (update: first/final; snapshot: last);
      * event_time must be uniform inside one event;
      * duplicate (side, price) rows inside one event are malformed;
      * byte-identical duplicate events collapse (counted by the caller via id equality);
        partially differing rows for the same event ids are a conflict."""
    # Byte-identical duplicate rows (a double-captured event) are dropped and counted per the
    # preregistered duplicates_overlap_reset rule — BEFORE grouping, so a duplicated event
    # does not masquerade as duplicate levels inside one event. Any partially-differing
    # duplicate still fails closed downstream (duplicate_level_in_event / continuity).
    n_dup_rows = int(df.duplicated(keep="first").sum())
    if n_dup_rows:
        df = df[~df.duplicated(keep="first")].reset_index(drop=True)
    ev_time = normalize_epoch_ns(df["event_time"].to_numpy(), fieldname="event_time")
    kind = df["event_type"].astype(str).to_numpy()
    side = df["side"].astype(str).to_numpy()
    price = df["price"].astype(str).to_numpy()
    qty = df["quantity"].astype(str).to_numpy()

    def _ids(col: str) -> np.ndarray:
        # nullable INT64 -> int64 with -1 sentinel for null
        return pd.array(df[col], dtype="Int64").to_numpy(dtype="int64", na_value=-1)

    first_u, final_u = _ids("first_update_id"), _ids("final_update_id")
    prev_u, last_u = _ids("prev_final_update_id"), _ids("last_update_id")

    groups: dict[tuple, dict] = {}
    for i in range(len(df)):
        k = kind[i]
        if k == "update":
            if final_u[i] < 0 or first_u[i] < 0:
                raise ChdValidationError("update_missing_ids",
                                         f"update row {i} lacks first/final_update_id")
            if first_u[i] > final_u[i]:
                # a backwards ID range is malformed regardless of how it chains — it must
                # never mutate the book (fail-closed update-ID contract)
                raise ChdValidationError(
                    "backwards_update_ids",
                    f"update row {i} has first_update_id {first_u[i]} > final_update_id "
                    f"{final_u[i]}")
            gk = ("update", int(first_u[i]), int(final_u[i]), int(prev_u[i]))
        else:
            if last_u[i] < 0:
                raise ChdValidationError("snapshot_missing_ids",
                                         f"snapshot row {i} lacks last_update_id")
            gk = ("snapshot", int(last_u[i]))
        g = groups.setdefault(gk, {"rows": [], "event_time": int(ev_time[i])})
        if g["event_time"] != int(ev_time[i]):
            raise ChdValidationError("event_time_not_uniform",
                                     f"event {gk} carries multiple event_time values")
        g["rows"].append(i)

    events: list[ChdEvent] = []
    for gk, g in groups.items():
        rows = g["rows"]
        bids: dict[int, float] = {}
        asks: dict[int, float] = {}
        for i in rows:
            ticks = to_ticks(price[i], price_scale)
            if ticks <= 0:
                # a zero/negative price is malformed vendor data in ANY event kind —
                # update levels must never mutate the book with it (snapshots already
                # refuse p <= 0 downstream via classify_chd_snapshot)
                raise ChdValidationError("malformed_price",
                                         f"non-positive price {price[i]!r} in event row")
            try:
                size = float(parse_decimal(qty[i], field="quantity"))
            except OverflowError as e:
                raise ChdValidationError("malformed_quantity",
                                         f"quantity {qty[i]!r} overflows float") from e
            if not isfinite(size) or size < 0:
                # validate at GROUPING so a pre-snapshot/skipped event cannot hide a
                # malformed level from the apply-time check (Codex round 19); zero stays
                # legal (update deletes; snapshot zero-size levels refuse downstream)
                raise ChdValidationError("malformed_quantity",
                                         f"negative/non-finite quantity {qty[i]!r}")
            book = bids if side[i] == "bid" else asks
            if ticks in book:
                raise ChdValidationError("duplicate_level_in_event",
                                         f"event {gk} has duplicate {side[i]} level at "
                                         f"{ticks} ticks")
            book[ticks] = size
        if gk[0] == "update":
            _, fu, lu, pu = gk
            events.append(ChdEvent("update", lu, g["event_time"], fu, lu,
                                   (None if pu < 0 else pu), None,
                                   tuple(sorted(bids.items(), reverse=True)),
                                   tuple(sorted(asks.items()))))
        else:
            _, l = gk
            events.append(ChdEvent("snapshot", l, g["event_time"], None, None, None, l,
                                   tuple(sorted(bids.items(), reverse=True)),
                                   tuple(sorted(asks.items()))))
    # Preregistered total order: order_id ascending; a snapshot sorts AFTER an update with the
    # same id (overwrite), and deterministically before any higher id.
    events.sort(key=lambda e: (e.order_id, 0 if e.kind == "update" else 1))
    return events, {"duplicate_rows_dropped": n_dup_rows}


def classify_chd_snapshot(ev: ChdEvent) -> str:
    """Snapshot validity per the preregistered bars (mirrors recon.reseed.classify_snapshot
    precedence at min_levels_per_side=5, on integer-tick levels): one_sided, bad_values,
    thin_depth, unsorted(inherently sorted here), crossed, else ok. Sizes must be finite>0."""
    if not ev.bids or not ev.asks:
        return "one_sided"
    for _, s in (*ev.bids, *ev.asks):
        if not isfinite(s) or s <= 0.0:
            return "bad_values"
    for p, _ in (*ev.bids, *ev.asks):
        if p <= 0:
            return "bad_values"
    if len(ev.bids) < MIN_SNAPSHOT_LEVELS_PER_SIDE or len(ev.asks) < MIN_SNAPSHOT_LEVELS_PER_SIDE:
        return "thin_depth"
    if ev.bids[0][0] >= ev.asks[0][0]:
        return "crossed"
    return "ok"


# ----------------------------------------------------------------------------- CHD causal replay
@dataclass
class ChdReplayState:
    """Book state carried across hourly partitions within one approved window. `anchored`
    is False until the FIRST update after the most recent snapshot has been accepted — every
    snapshot (seed or reset) re-arms the anchor rule, because Binance snapshots are
    out-of-band book versions whose last_update_id may fall inside an update's
    [first_update_id, final_update_id] range rather than on an event boundary."""
    bids: dict[int, float] = field(default_factory=dict)
    asks: dict[int, float] = field(default_factory=dict)
    last_final_update_id: int | None = None
    seeded: bool = False
    anchored: bool = False
    watermark_ns: int = 0


def _topk_row(state: ChdReplayState, k: int, scale: int, sample_ts: int) -> dict:
    """Assemble one sample row in the EXACT internal top-K contract
    (recon.orderbook.OrderBook.snapshot key set + NaN padding + derived mid/microprice),
    prices materialized once as ticks/scale."""
    nan = float("nan")
    bids = sorted(state.bids, reverse=True)[:k]
    asks = sorted(state.asks)[:k]
    bb = bids[0] if bids else None
    ba = asks[0] if asks else None
    if bb is not None and ba is not None:
        bbf, baf = bb / scale, ba / scale
        bs, as_ = state.bids[bb], state.asks[ba]
        m = (bbf + baf) / 2.0
        mp = (as_ * bbf + bs * baf) / (bs + as_)
    else:
        m = mp = None
    out: dict = {"mid": nan if m is None else m, "microprice": nan if mp is None else mp}
    for i in range(k):
        out[f"bid_{i}_price"] = bids[i] / scale if i < len(bids) else nan
        out[f"bid_{i}_size"] = state.bids[bids[i]] if i < len(bids) else nan
        out[f"ask_{i}_price"] = asks[i] / scale if i < len(asks) else nan
        out[f"ask_{i}_size"] = state.asks[asks[i]] if i < len(asks) else nan
    out["sample_ts"] = int(sample_ts)
    return out


def replay_chd_window(hour_frames: list[tuple[dict, pd.DataFrame]], *, market: str,
                      price_scale: int, grid: list[int], k: int = 10,
                      state: ChdReplayState | None = None) -> tuple[pd.DataFrame, dict]:
    """Causally replay validated hourly frames over `grid` (int-ns exchange-time samples,
    as-of apply-before-read) into the internal top-K contract. FAIL CLOSED on every
    preregistered anomaly; on success returns `(frame, meta)` with aggregate integrity
    metrics only.

    `hour_frames` is an ORDERED list of `(identity, frame)` for consecutive hours (identity
    from validate_chd_frame — enforced consecutive; a missing hour inside the window must be
    refused by the caller via `require_consecutive_hours`). `market` is 'futures' | 'spot'
    (preregistered continuity semantics). State may carry across the hour seam only through
    unbroken update-ID continuity."""
    if market not in ("futures", "spot"):
        raise ValueError(f"unknown market {market!r}")
    st = state or ChdReplayState()
    n = len(grid)
    si = 0
    rows: list[dict] = []

    crossed = missing = thin = 0
    counters = {"events": 0, "updates_applied": 0, "updates_skipped_pre_snapshot": 0,
                "duplicate_events_dropped": 0, "resets": 0, "delete_absent_levels": 0,
                "event_time_regressions": 0, "snapshot_reason_codes": {},
                "snapshots_applied": 0}
    seed_reason_first = None

    def emit(g: int) -> None:
        nonlocal crossed, missing, thin
        rows.append(_topk_row(st, k, price_scale, g))
        bb = max(st.bids) if st.bids else None
        ba = min(st.asks) if st.asks else None
        if bb is None or ba is None:
            missing += 1
        elif bb >= ba:
            crossed += 1
        elif len(st.bids) < k or len(st.asks) < k:
            thin += 1

    prev_event: ChdEvent | None = None
    for identity, df in hour_frames:
        events, group_meta = _group_events(df, price_scale=price_scale)
        counters["duplicate_rows_dropped"] = counters.get("duplicate_rows_dropped", 0) + \
            group_meta["duplicate_rows_dropped"]
        for ev in events:
            counters["events"] += 1
            # Monotone watermark on exchange event time (preregistered ordering axis). An
            # event that will be APPLIED must not regress behind it — checked per kind
            # below; only skipped/deduped events may regress (counted, bounded).
            regresses = ev.event_time_ns < st.watermark_ns
            t = max(ev.event_time_ns, st.watermark_ns)

            if ev.kind == "snapshot":
                reason = classify_chd_snapshot(ev)
                counters["snapshot_reason_codes"][reason] = \
                    counters["snapshot_reason_codes"].get(reason, 0) + 1
                if seed_reason_first is None:
                    seed_reason_first = reason
                if reason != "ok":
                    # NOTE (2026-07-11 amendment): every non-ok CHD snapshot is a HARD
                    # refusal — stricter than the Lake seed gate, which counts crossed
                    # candidates into seed_source_crossed_frac. Consequently, on any
                    # replay that returns, seed_source_crossed_frac is structurally 0.0
                    # and the chd_window_quality seed bar is vacuous-by-strictness.
                    raise ChdSnapshotError(f"snapshot_{reason}",
                                           f"snapshot@{ev.last_update_id} classified {reason}")
                if st.seeded:
                    if regresses:
                        raise ChdSnapshotError(
                            "backwards_snapshot",
                            f"snapshot@{ev.last_update_id} event_time regresses "
                            f"{st.watermark_ns - ev.event_time_ns} ns behind the watermark "
                            "(a past/future-misplaced snapshot must never reseed)")
                    if st.last_final_update_id is not None and \
                            ev.last_update_id < st.last_final_update_id:
                        raise ChdSnapshotError(
                            "stale_snapshot",
                            f"snapshot last_update_id {ev.last_update_id} < applied "
                            f"{st.last_final_update_id} (backwards/stale book version)")
                    counters["resets"] += 1
                # advance the sample cursor BEFORE the snapshot lands (apply-before-read)
                while si < n and grid[si] < t:
                    emit(grid[si]); si += 1
                st.bids = {p: s for p, s in ev.bids}
                st.asks = {p: s for p, s in ev.asks}
                st.last_final_update_id = ev.last_update_id
                st.seeded = True
                st.anchored = False       # every snapshot re-arms the anchor rule (see state)
                st.watermark_ns = t
                counters["snapshots_applied"] += 1
                prev_event = ev
                continue

            # ---- update event
            if not st.seeded:
                counters["updates_skipped_pre_snapshot"] += 1
                counters["event_time_regressions"] += int(regresses)
                prev_event = ev
                continue
            L = st.last_final_update_id
            assert L is not None
            # duplicates / overlap (preregistered duplicates_overlap_reset)
            if prev_event is not None and prev_event.kind == "update" and \
                    ev.first_update_id == prev_event.first_update_id and \
                    ev.final_update_id == prev_event.final_update_id:
                # a duplicate is harmless only when ALL ids (incl. prev_final_update_id)
                # AND the level payload match; same U/u with a different pu is conflicting
                # update-ID metadata, not a re-capture (Codex round 7)
                if ev.prev_final_update_id == prev_event.prev_final_update_id and \
                        (ev.bids, ev.asks) == (prev_event.bids, prev_event.asks):
                    counters["duplicate_events_dropped"] += 1
                    counters["event_time_regressions"] += int(regresses)
                    continue
                raise ChdContinuityError(
                    "conflicting_duplicate_event",
                    f"two different events share ids ({ev.first_update_id},"
                    f"{ev.final_update_id})")
            if market == "futures":
                if not st.anchored:
                    if ev.final_update_id < L:      # pre-snapshot event, superseded by the seed
                        counters["updates_skipped_pre_snapshot"] += 1
                        counters["event_time_regressions"] += int(regresses)
                        prev_event = ev
                        continue
                    ok = (ev.first_update_id <= L <= ev.final_update_id) or \
                         (ev.prev_final_update_id == L)
                    if not ok:
                        raise ChdContinuityError(
                            "seed_anchor_gap",
                            f"first applied update ({ev.first_update_id},{ev.final_update_id},"
                            f"pu={ev.prev_final_update_id}) does not anchor to snapshot {L}")
                else:
                    if ev.final_update_id <= L:     # backwards event once anchored: overlap
                        raise ChdContinuityError(
                            "incompatible_overlap",
                            f"update ({ev.first_update_id},{ev.final_update_id}) does not "
                            f"advance past applied final_update_id {L}")
                    if ev.prev_final_update_id != L:
                        raise ChdContinuityError(
                            "sequence_gap",
                            f"prev_final_update_id {ev.prev_final_update_id} != last applied "
                            f"final_update_id {L}")
            else:  # spot
                if not st.anchored:
                    if ev.final_update_id <= L:     # pre-snapshot event, superseded by the seed
                        counters["updates_skipped_pre_snapshot"] += 1
                        counters["event_time_regressions"] += int(regresses)
                        prev_event = ev
                        continue
                    if not (ev.first_update_id <= L + 1 <= ev.final_update_id):
                        raise ChdContinuityError(
                            "seed_anchor_gap",
                            f"first applied update ({ev.first_update_id},"
                            f"{ev.final_update_id}) does not straddle last_update_id+1={L + 1}")
                else:
                    if ev.final_update_id <= L:     # backwards event once anchored: overlap
                        raise ChdContinuityError(
                            "incompatible_overlap",
                            f"update ({ev.first_update_id},{ev.final_update_id}) does not "
                            f"advance past applied final_update_id {L}")
                    if ev.first_update_id != L + 1:
                        raise ChdContinuityError(
                            "sequence_gap",
                            f"first_update_id {ev.first_update_id} != previous "
                            f"final_update_id+1 = {L + 1}")
                    if ev.prev_final_update_id is not None and ev.prev_final_update_id != L:
                        raise ChdContinuityError(
                            "sequence_gap",
                            f"non-null prev_final_update_id {ev.prev_final_update_id} != {L}")

            if regresses:
                # 2026-07-11 amendment: an APPLIED update must never regress behind the
                # watermark — the aggregate bound cannot be allowed to admit an
                # out-of-order book mutation (e.g. a stale pre-reset update whose ids
                # happen to straddle the reset snapshot's book version).
                raise ChdContinuityError(
                    "ordering_anomaly",
                    f"applied update ({ev.first_update_id},{ev.final_update_id}) event_time "
                    f"regresses {st.watermark_ns - ev.event_time_ns} ns behind the watermark")
            while si < n and grid[si] < t:
                emit(grid[si]); si += 1
            for book, levels in ((st.bids, ev.bids), (st.asks, ev.asks)):
                for ticks, size in levels:
                    if size == 0.0:
                        if ticks not in book:
                            counters["delete_absent_levels"] += 1
                        book.pop(ticks, None)
                    else:
                        if not isfinite(size) or size < 0:
                            raise ChdValidationError("malformed_quantity",
                                                     f"non-finite/negative size {size}")
                        book[ticks] = size
            st.last_final_update_id = ev.final_update_id
            st.anchored = True
            st.watermark_ns = t
            counters["updates_applied"] += 1
            prev_event = ev

    while si < n:
        emit(grid[si]); si += 1

    if counters["events"] and \
            counters["event_time_regressions"] / counters["events"] > \
            MAX_EVENT_TIME_REGRESSION_FRAC:
        raise ChdContinuityError(
            "ordering_anomaly",
            f"{counters['event_time_regressions']}/{counters['events']} event_time regressions "
            f"exceed the preregistered {MAX_EVENT_TIME_REGRESSION_FRAC:.3%} bound")
    if not st.seeded:
        raise ChdSnapshotError("missing_initial_snapshot",
                               "window contains no valid snapshot and no carried state")

    frame = pd.DataFrame(rows)
    n_snap = sum(counters["snapshot_reason_codes"].values())
    crossed_snap = counters["snapshot_reason_codes"].get("crossed", 0)
    meta = {
        "market": market, "price_scale": int(price_scale), "k": int(k),
        "n_samples": int(n),
        "crossed_samples": int(crossed), "crossed_rate": (crossed / n if n else 0.0),
        "missing_book_samples": int(missing),
        "missing_book_fraction": (missing / n if n else 0.0),
        "thin_depth_samples": int(thin), "thin_depth_fraction": (thin / n if n else 0.0),
        "seed_source_crossed_frac": (crossed_snap / n_snap if n_snap else 0.0),
        "seed_reason_first": seed_reason_first,
        "counters": counters,
        "frame_replay_hash": frame_replay_hash(frame),
    }
    return frame, meta


def require_consecutive_hours(identities: list[dict]) -> None:
    """Refuse a window whose hourly partitions are not exactly consecutive (missing hour) or
    contain duplicates (two objects claiming one hour)."""
    seen = [(i["date"], i["hour"]) for i in identities]
    if len(set(seen)) != len(seen):
        raise ChdValidationError("duplicate_hour_partition", f"duplicate hours in {seen}")
    for (d0, h0), (d1, h1) in zip(seen, seen[1:]):
        t0 = pd.Timestamp(f"{d0}T{h0:02d}:00:00", tz="UTC")
        if pd.Timestamp(f"{d1}T{h1:02d}:00:00", tz="UTC") != t0 + pd.Timedelta(hours=1):
            raise ChdValidationError("missing_hour_partition",
                                     f"hours not consecutive: {(d0, h0)} -> {(d1, h1)}")


# ----------------------------------------------------------------------------- lake integrity extras
def silence_metrics(engine_time_ns: np.ndarray) -> dict:
    """Delta-stream silence per integrity_definitions.silence: inter-event gaps on the resolved
    engine-time axis (sorted ascending first — the replay order)."""
    t = np.sort(np.asarray(engine_time_ns, dtype="int64"))
    if len(t) < 2:
        return {"n_events": int(len(t)), "max_gap_s": None, "gaps_gt_10s": None,
                "gaps_gt_60s": None, "gaps_gt_300s": None, "silent_seconds_gt_10s": None}
    gaps = np.diff(t) / 1e9
    return {"n_events": int(len(t)),
            "max_gap_s": float(gaps.max()),
            "gaps_gt_10s": int((gaps > 10.0).sum()),
            "gaps_gt_60s": int((gaps > 60.0).sum()),
            "gaps_gt_300s": int((gaps > 300.0).sum()),
            "silent_seconds_gt_10s": float(gaps[gaps > 10.0].sum())}


def frozen_metrics(frame: pd.DataFrame, *, min_run_samples: int = 60) -> dict:
    """Frozen/stale-book metrics per integrity_definitions.frozen_run: maximal runs of
    consecutive grid samples whose FULL top-K state (every price/size column) is identical
    (NaN == NaN for padding). `stale_but_uncrossed` counts the frozen samples that are
    present + uncrossed at the touch."""
    cols = [c for c in frame.columns if c != "sample_ts"]
    a = frame[cols].to_numpy(dtype="float64")
    n = len(a)
    if n == 0:
        return {"n_samples": 0, "n_frozen_runs": 0, "max_frozen_run_s": 0.0,
                "frozen_fraction": 0.0, "stale_but_uncrossed_fraction": 0.0}
    eq = (a[1:] == a[:-1]) | (np.isnan(a[1:]) & np.isnan(a[:-1]))
    same = eq.all(axis=1)                      # same[i] — sample i+1 identical to sample i
    bb, ba = frame["bid_0_price"].to_numpy(), frame["ask_0_price"].to_numpy()
    valid = np.isfinite(bb) & np.isfinite(ba) & (bb < ba)
    frozen_mask = np.zeros(n, dtype=bool)
    runs = 0
    max_run = 0
    i = 0
    while i < len(same):
        if same[i]:
            j = i
            while j < len(same) and same[j]:
                j += 1
            run_len = (j - i) + 1              # samples i .. j inclusive
            if run_len >= min_run_samples:
                runs += 1
                max_run = max(max_run, run_len)
                frozen_mask[i:j + 1] = True
            i = j
        else:
            i += 1
    return {"n_samples": int(n), "n_frozen_runs": int(runs),
            "max_frozen_run_s": float(max_run),
            "frozen_fraction": float(frozen_mask.mean()),
            "stale_but_uncrossed_fraction": float((frozen_mask & valid).mean())}


# Volatile keys excluded from the cross-run comparison: {secs, ts} are wall-clock; {engine,
# price_scale} differ BY DESIGN under the 2026-07-12 cross-engine protocol amendment (run 1
# = python oracle, run 2 = native at the measured scale) — every SEMANTIC field (rows,
# sha256, classification, reasons, quality metrics, seed blocks, engine_time_col,
# dropped_rows) must still be exactly equal, which is full-day cross-engine conformance.
DETERMINISM_EXCLUDED_KEYS = ("secs", "ts", "engine", "price_scale")


def _load_raw_manifest(path: str) -> dict:
    """(output, exchange, symbol, dt) -> FULL last-wins record."""
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[(r["output"], r["exchange"], r["symbol"], r["dt"])] = r
    return recs


def _load_semantic_manifest(path: str) -> dict:
    """(output, exchange, symbol, dt) -> semantic record view (last record wins,
    DETERMINISM_EXCLUDED_KEYS removed)."""
    return {key: {k: v for k, v in r.items() if k not in DETERMINISM_EXCLUDED_KEYS}
            for key, r in _load_raw_manifest(path).items()}


def _topk_engine_provenance(path: str) -> dict:
    """'output|exchange|symbol|dt' -> [engine, price_scale] for every topk_l2 record.
    Reported by compare_stage2_manifests so the verdict can require GENUINE cross-engine
    coverage (run1 python oracle, run2 native at the measured scales) — the excluded-key
    equality alone would also pass a python-vs-python self-compare."""
    return {"|".join(key): [r.get("engine"), r.get("price_scale")]
            for key, r in _load_raw_manifest(path).items() if key[0] == "topk_l2"}


def semantic_manifest_fingerprint(path: str) -> str:
    """Content fingerprint of a Stage-2 manifest's SEMANTIC state: canonical-JSON hash of
    the per-unit last-wins records with DETERMINISM_EXCLUDED_KEYS removed. Binds a
    stage2-compare report to the exact manifest content the verdict certifies — a stale
    passing comparison from another output root cannot vouch for a different manifest."""
    recs = _load_semantic_manifest(path)
    return hash_obj({"|".join(k): v for k, v in recs.items()})


def compare_stage2_manifests(path_a: str, path_b: str) -> dict:
    """Cross-run determinism/conformance comparator for two Stage-2 processed manifests
    (preregistered determinism.stage2_cli + the 2026-07-12 amendment): per-unit records
    keyed by (output, exchange, symbol, dt) must be equal excluding
    DETERMINISM_EXCLUDED_KEYS. Returns equal/diffs plus the compared unit keys and each
    side's semantic fingerprint (consumed by the verdict binding check)."""
    a, b = _load_semantic_manifest(path_a), _load_semantic_manifest(path_b)
    diffs = []
    for key in sorted(set(a) | set(b), key=str):
        ra, rb = a.get(key), b.get(key)
        if ra is None or rb is None:
            diffs.append({"unit": list(key), "diff": "missing_in_one_run"})
        elif ra != rb:
            changed = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
            diffs.append({"unit": list(key), "diff": f"keys_differ:{changed}"})
    return {"equal": not diffs, "n_units": len(set(a) | set(b)), "diffs": diffs,
            "units": sorted([list(k) for k in set(a) | set(b)]),
            "run1_semantic_fingerprint": semantic_manifest_fingerprint(path_a),
            "run2_semantic_fingerprint": semantic_manifest_fingerprint(path_b),
            "run1_topk_engines": _topk_engine_provenance(path_a),
            "run2_topk_engines": _topk_engine_provenance(path_b)}


# ----------------------------------------------------------------------------- fixed comparison
def _ticks_col(frame: pd.DataFrame, col: str, scale: int) -> np.ndarray:
    p = frame[col].to_numpy(dtype="float64")
    t = np.round(p * scale)
    bad = np.isfinite(p) & (np.abs(p * scale - t) > 1e-6)
    if bad.any():
        raise SourceGateError("off_tick", f"{col}: {int(bad.sum())} values not integral at "
                                          f"scale {scale}")
    return np.where(np.isfinite(p), t, np.nan)


def compare_topk_frames(lake: pd.DataFrame, chd: pd.DataFrame, *, price_scale: int,
                        k: int = 10) -> dict:
    """The preregistered fixed comparison (thresholds.comparison.basis): identical
    exchange-time grids, tick space, k=10. Aggregate deviation/agreement metrics ONLY —
    never levels, paths, labels, or any outcome-bearing statistic."""
    if not np.array_equal(lake["sample_ts"].to_numpy(), chd["sample_ts"].to_numpy()):
        raise SourceGateError("grid_mismatch", "frames are not on the identical sample grid")
    n = len(lake)
    lb_, la_ = _ticks_col(lake, "bid_0_price", price_scale), _ticks_col(lake, "ask_0_price", price_scale)
    cb_, ca_ = _ticks_col(chd, "bid_0_price", price_scale), _ticks_col(chd, "ask_0_price", price_scale)
    valid_l = np.isfinite(lb_) & np.isfinite(la_) & (lb_ < la_)
    valid_c = np.isfinite(cb_) & np.isfinite(ca_) & (cb_ < ca_)
    joint = valid_l & valid_c
    nj = int(joint.sum())
    out = {
        "n_samples": int(n),
        "valid_fraction_lake": float(valid_l.mean()) if n else 0.0,
        "valid_fraction_chd": float(valid_c.mean()) if n else 0.0,
        "joint_valid_fraction": float(joint.mean()) if n else 0.0,
        "n_joint_valid": nj,
    }
    if nj:
        dbid = np.abs(lb_[joint] - cb_[joint])
        dask = np.abs(la_[joint] - ca_[joint])
        dmid = np.abs((lb_[joint] + la_[joint]) / 2.0 - (cb_[joint] + ca_[joint]) / 2.0)
        within1 = (dbid <= 1.0) & (dask <= 1.0)
        exact = (dbid == 0.0) & (dask == 0.0)
        q = lambda a, p: float(np.quantile(a, p))
        # descriptive top-K price-set overlap per side (mean shared level count)
        shared = {"bid": [], "ask": []}
        for side in ("bid", "ask"):
            lcols = [f"{side}_{i}_price" for i in range(k)]
            lt = np.round(lake[lcols].to_numpy(dtype="float64") * price_scale)
            ct = np.round(chd[lcols].to_numpy(dtype="float64") * price_scale)
            idx = np.flatnonzero(joint)
            cnt = [len(set(lt[i][np.isfinite(lt[i])]) & set(ct[i][np.isfinite(ct[i])]))
                   for i in idx]
            shared[side] = float(np.mean(cnt)) if cnt else None
        out.update({
            "touch_agreement_exact_tick": float(exact.mean()),
            "touch_agreement_within_1_tick": float(within1.mean()),
            "bid_abs_diff_ticks": {"p50": q(dbid, 0.5), "p95": q(dbid, 0.95),
                                   "p99": q(dbid, 0.99), "max": float(dbid.max())},
            "ask_abs_diff_ticks": {"p50": q(dask, 0.5), "p95": q(dask, 0.95),
                                   "p99": q(dask, 0.99), "max": float(dask.max())},
            "mid_abs_diff_ticks": {"p50": q(dmid, 0.5), "p95": q(dmid, 0.95),
                                   "p99": q(dmid, 0.99), "max": float(dmid.max())},
            "topk_shared_levels_mean": shared,
        })
    return out


def evaluate_comparison(metrics: dict) -> dict:
    """Evaluate the fixed comparison against the preregistered bars, FAIL-CLOSED: a missing
    metric fails its criterion (the #54 evaluate_preregistered convention)."""
    bars = PREREGISTERED["thresholds"]["comparison"]
    def check(name, value, bound, op):
        ok = (value is not None) and (op(value, bound))
        return {"criterion": name, "value": value, "bound": bound, "ok": bool(ok)}
    mid = metrics.get("mid_abs_diff_ticks") or {}
    checks = [
        check("joint_valid_fraction_min", metrics.get("joint_valid_fraction"),
              bars["joint_valid_fraction_min"], lambda v, b: v >= b),
        check("touch_agreement_within_1_tick_min", metrics.get("touch_agreement_within_1_tick"),
              bars["touch_agreement_within_1_tick_min"], lambda v, b: v >= b),
        check("mid_abs_diff_ticks_p50_max", mid.get("p50"),
              bars["mid_abs_diff_ticks_p50_max"], lambda v, b: v <= b),
        check("mid_abs_diff_ticks_p99_max", mid.get("p99"),
              bars["mid_abs_diff_ticks_p99_max"], lambda v, b: v <= b),
    ]
    return {"checks": checks, "pass": all(c["ok"] for c in checks)}


# ----------------------------------------------------------------------------- April holdout guard
# Key-marker denylist for anything published by this experiment (tracked evidence AND ignored
# reports). Substring match on lowercased key names; mirrors forbidden_april_metrics in the
# preregistration. The length bound stops raw per-sample series from being smuggled into
# evidence as an innocent-looking list.
FORBIDDEN_KEY_MARKERS = (
    "label", "pnl", "feature", "forecast", "cost", "notional", "interarrival",
    "trade_size", "side_ratio", "return", "volatility", "sharpe", "price_path",
    "price_mean", "price_level", "mid_mean", "mid_usd", "spread_mean", "spread_usd",
)
MAX_PUBLISHED_LIST_LEN = 200


def assert_report_publishable(obj, *, path: str = "report") -> None:
    """April holdout guard (fail closed): refuse to publish a report carrying an
    outcome-bearing key or an unbounded list (a raw per-sample series). Raises
    SourceGateError('forbidden_metric'|'unbounded_series')."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            lk = str(key).lower()
            for marker in FORBIDDEN_KEY_MARKERS:
                if marker in lk:
                    raise SourceGateError(
                        "forbidden_metric",
                        f"{path}.{key} matches forbidden marker {marker!r} "
                        "(April holdout guard)")
            assert_report_publishable(value, path=f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        if len(obj) > MAX_PUBLISHED_LIST_LEN:
            raise SourceGateError(
                "unbounded_series",
                f"{path} has {len(obj)} elements > {MAX_PUBLISHED_LIST_LEN} "
                "(no raw series in published evidence)")
        for i, value in enumerate(obj):
            assert_report_publishable(value, path=f"{path}[{i}]")


def finalize_report(report: dict) -> dict:
    """Attach report_hash (canonical-JSON, excluding itself) AFTER the publishability guard."""
    assert_report_publishable(report)
    report = dict(report)
    report["report_hash"] = hash_obj(_json_safe(report), exclude_keys=("report_hash",))
    return report


def _json_safe(obj):
    """Strict-JSON coercion (non-finite floats -> None, numpy scalars -> python) — the
    experiments/snapshot_seed.py convention."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if isfinite(obj) else None
    if hasattr(obj, "item"):
        v = obj.item()
        return _json_safe(v) if isinstance(v, float) else v
    return obj
