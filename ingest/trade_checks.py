"""Pure, source-agnostic trade-feed validation checks (docs/data.md §5b / §10 "trade validation
breadth"; plan docs/superpowers/plans/2026-07-02-trade-validation-breadth-plan.md, Phase 1a).

Validates a *normalized* trade frame — the loaded/renamed Crypto Lake or CoinAPI `trades` schema
(`origin_time`, `received_time`: datetime64[ns]; `price`, `quantity`: float; `side` ∈ {buy, sell};
`trade_id`: int64) — regardless of vendor. Trades drive the notional bar clock (spec §5.1), so a
mis-validated trade stream mis-times every bar, feature, and label; this module is the gate the bar
builder consumes.

This mirrors the repo's established pure/vendor split (`recon.stitch_policy` vs. the vendor
`scripts/run_coinbase_quality_map.py` runner): pandas/numpy ONLY — no `lakeapi`/`boto3` import, no
vendor I/O — so every check is synthetic-testable (`tests/test_trade_checks.py`). The thin Lake CLI
wrapper (`ingest/validate_trade_feeds.py`, day selection + `lakeapi.load_data`) and the gated CoinAPI
trade-fill path are Phase-1b/3b follow-ups that call THIS module unchanged.

Every `(venue, day)` gets exactly one `status` from a closed set of five: `pass`, `warn`, `fail`
(the single blocking status), and the two routed states `coinapi_fill` / `excluded`. Structural
problems (`empty_partition`, `missing_partition`, `load_error`) are reason codes, not statuses: on a
required non-fill day they resolve to `fail`; a clean "no partition" on a calendar trades-fill day
resolves to `coinapi_fill`; any excluded-calendar day resolves to `excluded` (§1).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- statuses (closed set)
PASS = "pass"
WARN = "warn"
FAIL = "fail"                       # the single BLOCKING status (§1): a consumer keying off
COINAPI_FILL = "coinapi_fill"       # status == "fail" / counts.fail can never let a structural
EXCLUDED = "excluded"               # miss escape the gate.
STATUSES = (PASS, WARN, FAIL, COINAPI_FILL, EXCLUDED)

# --------------------------------------------------------------------------- calendar routing
ROUTE_REQUIRED = "required"         # a required Lake day: the metric verdict stands
ROUTE_COINAPI_FILL = "coinapi_fill"  # a Coinbase trades-fill day → deferred to the CoinAPI path
ROUTE_EXCLUDED = "excluded"         # dropped by the usable calendar → out of scope

# --------------------------------------------------------------------------- venues & products (§2)
# Reuse the exact Crypto Lake identifiers from ingest/verify_trades_and_calendar.py — do NOT invent.
VENUES = {
    "binance_perp": ("BINANCE_FUTURES", "BTC-USDT-PERP"),
    "binance_spot": ("BINANCE", "BTC-USDT"),
    "coinbase": ("COINBASE", "BTC-USD"),
}

# --------------------------------------------------------------------------- reason codes (§6)
# Stable module-level string constants (SCREAMING_SNAKE name → snake_case value). A reason ENTRY is
# either the bare code or a `code:detail` / `metric=value>threshold` string, stable code first — the
# run_coinbase_quality_map.classify_day convention.
OK = "ok"
EMPTY_PARTITION = "empty_partition"
MISSING_PARTITION = "missing_partition"
LOAD_ERROR = "load_error"
ORIGIN_TIME_COLUMN_MISSING = "origin_time_column_missing"
ORIGIN_TIME_NULL_FRACTION_HIGH = "origin_time_null_fraction_high"
RECEIVED_TIME_FALLBACK_USED = "received_time_fallback_used"
RECEIVED_TIME_FALLBACK_UNAVAILABLE = "received_time_fallback_unavailable"
NONMONOTONIC_AFTER_SORT = "nonmonotonic_after_sort"
PRICE_OUT_OF_RANGE = "price_out_of_range"
SIZE_NONPOSITIVE = "size_nonpositive"
NOTIONAL_NONPOSITIVE = "notional_nonpositive"
ROW_COUNT_IMPLAUSIBLY_LOW = "row_count_implausibly_low"
DUPLICATE_TIMESTAMP_CLUSTER = "duplicate_timestamp_cluster"
DUPLICATE_TRADE_ID = "duplicate_trade_id"
PRICE_JUMP_EXCESS = "price_jump_excess"
PRICE_SPIKE = "price_spike"
SIZE_OUT_OF_RANGE = "size_out_of_range"
SIZE_OUT_OF_BAND = "size_out_of_band"
INTERARRIVAL_GAP_EXCESS = "interarrival_gap_excess"
MISSING_HOUR = "missing_hour"
SPARSE_HOUR = "sparse_hour"
MISSING_HOURS_EXCESS = "missing_hours_excess"
LAG_NEGATIVE = "lag_negative"
SIDE_VALUE_UNEXPECTED = "side_value_unexpected"
ROW_COUNT_LOW = "row_count_low"
COINAPI_FILL_DAY = "coinapi_fill_day"
CALENDAR_EXCLUDED_DAY = "calendar_excluded_day"

# --------------------------------------------------------------------------- quota / exit constants
# Mirrors scripts/run_coinbase_quality_map.py (the shared Lake quota pattern). Only the auto-cap
# default differs — the bounded trade sample is smaller (§7).
QUOTA_GB = 300.0                    # Crypto Lake individual-plan monthly download cap (docs/data.md)
DEFAULT_MAX_AUTO_GB = 3.0           # a request larger than this is a "broad" pull → needs --allow-broad
DEFAULT_HEADROOM_GB = 10.0          # never plan to use the last N GB of the monthly quota
QUOTA_REFUSED_EXIT = 5             # small-int exit-code convention (cf. parity 3, backfill 4, native 6)
VALIDATION_FAILED_EXIT = 7         # --strict and ≥1 blocking fail

# Provisional per-day Lake `trades` footprint by venue, GB (§6, Phase-2-measured later). Over-estimates
# so the quota gate errs high.
TRADES_GB_PER_DAY = {"binance_perp": 0.12, "binance_spot": 0.10, "coinbase": 0.05}

# Null-timestamp sentinel: lakeapi encodes an absent exchange/receipt time as an epoch value; the
# existing verify_trades_and_calendar.py bar is `< pd.Timestamp("2015-01-01")` (tz-naive, matching the
# tz-naive loaded columns).
SENTINEL_CUTOFF = pd.Timestamp("2015-01-01")

VALID_SIDES = ("buy", "sell")


@dataclass(frozen=True)
class TradeThresholds:
    """Frozen validation thresholds, emitted into `meta.thresholds` (the `Thresholds.as_dict()`
    pattern — every artifact records the knobs that produced it). Conservative first-pass values,
    Phase-2-tunable (§6)."""
    origin_time_null_max: float = 0.01     # matches verify_lake's `<0.01 → USABLE (exchange time)` bar
    min_rows_hard: int = 1000              # a `trades` day under ~1k rows is a broken partition
    dup_ts_cluster_warn: int = 50          # a >50-deep single-ns cluster is worth a look
    price_jump_warn: float = 0.10          # a broad-day p99 >10% consecutive-trade churn (volatile regime)
    price_spike_warn: float = 0.50         # a >50% single-tick abs return is almost always one corrupt print
    price_range_factor: float = 10.0       # a price outside [median/10, median×10] is grossly implausible
    size_max_btc: float = 500.0            # a single trade > 500 BTC is unusually large but can be real
    size_hard_max_btc: float = 5000.0      # a single trade > 5000 BTC (~$300M) is bar-clock-corrupting
    interarrival_gap_warn_s: float = 120.0  # a >2 min no-trade gap in a normally-active market
    sparse_hour_min_rows: int = 60         # < 1 trade/min for a whole UTC hour is sparse
    max_missing_hours: int = 1             # ≥2 fully-empty UTC hours is a data gap, not quiet
    lag_neg_frac_max: float = 0.001        # `received ≥ origin` should hold; nonzero negative is a fault
    min_rows_soft: int | None = None       # optional regime soft floor → `row_count_low` warn (off by default)

    def as_dict(self) -> dict:
        return {
            "origin_time_null_max": self.origin_time_null_max,
            "min_rows_hard": self.min_rows_hard,
            "dup_ts_cluster_warn": self.dup_ts_cluster_warn,
            "price_jump_warn": self.price_jump_warn,
            "price_spike_warn": self.price_spike_warn,
            "price_range_factor": self.price_range_factor,
            "size_max_btc": self.size_max_btc,
            "size_hard_max_btc": self.size_hard_max_btc,
            "interarrival_gap_warn_s": self.interarrival_gap_warn_s,
            "sparse_hour_min_rows": self.sparse_hour_min_rows,
            "max_missing_hours": self.max_missing_hours,
            "lag_neg_frac_max": self.lag_neg_frac_max,
            "min_rows_soft": self.min_rows_soft,
        }


THRESHOLDS = TradeThresholds()


# --------------------------------------------------------------------------- JSON safety
def _json_safe(obj):
    """Recursively coerce to strict-JSON-valid types: non-finite floats → None, numpy scalars →
    python scalars, so the artifact passes `jq empty` (AGENTS.md). Mirrors run_coinbase_quality_map."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):           # numpy scalar
        v = obj.item()
        return _json_safe(v) if isinstance(v, float) else v
    return obj


def _f(x) -> float:
    """A plain python float; NaN/inf preserved (report-time `_json_safe` nulls them)."""
    return float(x)


def _iso_z(ts) -> str | None:
    """ISO-8601 UTC string with a trailing `Z` for a tz-naive engine-clock Timestamp; None for NaT."""
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat() + "Z"


# --------------------------------------------------------------------------- timestamp / engine clock
def _null_ts_mask(series: pd.Series) -> pd.Series:
    """Rows whose timestamp is null: `NaT` OR the pre-2015 sentinel (§4/§5)."""
    return series.isna() | (series < SENTINEL_CUTOFF)


def _origin_col(df: pd.DataFrame) -> str | None:
    """The loaded exchange-time column name — `origin_time` (post-lakeapi) or raw `timestamp`."""
    if "origin_time" in df.columns:
        return "origin_time"
    if "timestamp" in df.columns:
        return "timestamp"
    return None


def _received_col(df: pd.DataFrame) -> str | None:
    if "received_time" in df.columns:
        return "received_time"
    if "receipt_timestamp" in df.columns:
        return "receipt_timestamp"
    return None


def build_engine_clock(df: pd.DataFrame):
    """The §5 engine clock: `origin_time` with a PER-ROW, UNCONDITIONAL `received_time` substitution
    for every null/sentinel-origin row (the substitution is not gated by any threshold — that would
    let sub-threshold 1970 sentinels sort to the front and read as trivially monotonic, silently
    corrupting the bar clock). Returns `(clock, origin_null_mask, unresolved_mask)` where
    `unresolved` marks rows whose origin AND received are both null (no resolvable engine time)."""
    ocol = _origin_col(df)
    origin = df[ocol]
    origin_null = _null_ts_mask(origin)
    clock = origin.copy()
    rcol = _received_col(df)
    if rcol is not None:
        # keep origin where present, else take received_time (per-row, unconditional)
        clock = origin.where(~origin_null, df[rcol])
    unresolved = _null_ts_mask(clock)
    return clock, origin_null, unresolved


def sorted_engine_clock(df: pd.DataFrame) -> pd.Series:
    """The resolved engine clock, dropped of any unresolved (still-null) rows and STABLY sorted
    ascending — `kind="mergesort"`, ties breaking by original row order (§5). Every time-based metric
    reads this, never raw `origin_time`."""
    clock, _, unresolved = build_engine_clock(df)
    clock = clock[~unresolved]
    return clock.sort_values(kind="mergesort")


def monotonic_after_sort(df: pd.DataFrame) -> bool:
    """Whether the post-fallback engine clock is non-decreasing after the stable sort, treating any
    residual `NaT`/sentinel (`< 2015-01-01`) as INVALID — an unresolvable clock never reads as
    trivially monotonic (§4 row 5, §5). The sort itself always orders finite timestamps, so this
    reduces to "every row has a resolvable engine time"; its value is that it (a) applies the sort
    (Coinbase's file order is not `origin_time`-ordered) and (b) rejects unresolved rows."""
    clock, _, unresolved = build_engine_clock(df)
    if bool(unresolved.any()):
        return False
    resolved = clock[~unresolved].sort_values(kind="mergesort")
    return bool(resolved.is_monotonic_increasing)


def was_presorted(df: pd.DataFrame) -> bool:
    """Whether the FILE arrived `origin_time`-monotonic (pre-sort) — informational (Coinbase→False,
    Binance→True). Reads raw `origin_time`, before the fallback/sort (§4 row 6)."""
    ocol = _origin_col(df)
    return bool(df[ocol].is_monotonic_increasing)


# --------------------------------------------------------------------------- per-metric functions (§4)
def dup_trade_ids(df: pd.DataFrame) -> dict:
    """Duplicate `trade_id` metrics (§4 row 8). No `trade_id` column → zeros (metric N/A)."""
    if "trade_id" not in df.columns:
        return {"trade_id_available": False, "dup_trade_id_count": 0, "dup_trade_id_frac": 0.0}
    n = len(df)
    count = int(n - df["trade_id"].nunique())
    return {"trade_id_available": True, "dup_trade_id_count": count,
            "dup_trade_id_frac": _f(count / n) if n else 0.0}


def dup_timestamp_clusters(df: pd.DataFrame) -> dict:
    """Duplicate engine-clock (`origin_time`) cluster metrics (§4 row 7): the number of distinct
    same-ns timestamps with count>1, and the max multiplicity."""
    clock = sorted_engine_clock(df)
    if clock.empty:
        return {"dup_ts_cluster_count": 0, "dup_ts_max_cluster": 0}
    counts = clock.value_counts()
    clustered = counts[counts > 1]
    return {"dup_ts_cluster_count": int(len(clustered)),
            "dup_ts_max_cluster": int(counts.max())}


def _abs_returns(df: pd.DataFrame) -> np.ndarray:
    """`abs(price.pct_change())` on the price series ordered by the engine clock (§4 row 9). Division
    warnings on a corrupt zero price are suppressed — the zero itself is caught by `price_out_of_range`
    and the resulting inf still trips `price_spike`."""
    order = sorted_engine_clock(df).index
    price = df.loc[order, "price"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.abs(np.diff(price) / price[:-1])
    return rets


def price_checks(df: pd.DataFrame, thresholds: TradeThresholds = THRESHOLDS) -> dict:
    """Price sanity metrics (§4 row 9): min/max/median, p99 AND max of consecutive abs-return, and the
    count of prices outside the robust `[median/factor, median×factor]` band (an isolated corrupt print
    the p99 misses)."""
    price = df["price"].to_numpy(dtype="float64")
    median = float(np.nanmedian(price)) if len(price) else float("nan")
    rets = _abs_returns(df)
    finite = rets[np.isfinite(rets)]
    p99 = float(np.quantile(finite, 0.99)) if finite.size else 0.0
    max_ret = float(np.nanmax(rets)) if rets.size and np.isfinite(rets).any() else (
        float("inf") if rets.size and not np.isfinite(rets).all() else 0.0)
    if math.isfinite(median) and median > 0:
        lo, hi = median / thresholds.price_range_factor, median * thresholds.price_range_factor
        out_of_band = int(np.sum((price < lo) | (price > hi)))
    else:
        out_of_band = 0
    return {
        "price_min": _f(np.nanmin(price)) if len(price) else float("nan"),
        "price_max": _f(np.nanmax(price)) if len(price) else float("nan"),
        "price_median": median,
        "price_p99_abs_ret": p99,
        "price_max_abs_ret": max_ret,
        "price_out_of_band_count": out_of_band,
        # NaN/±inf prices are invisible to nanmin/nansum/the robust band, so count them explicitly —
        # any non-finite price is invalid (§4 row 9 / §6 `price_out_of_range`) and corrupts the clock.
        "price_nonfinite_count": int(np.sum(~np.isfinite(price))) if len(price) else 0,
    }


def size_checks(df: pd.DataFrame) -> dict:
    """Size (`quantity`) sanity metrics (§4 row 10). Zero/negative sizes corrupt dollar bars."""
    q = df["quantity"].to_numpy(dtype="float64")
    n = len(q)
    nan = np.isnan(q)
    return {
        "size_min": _f(np.nanmin(q)) if n else float("nan"),
        "size_max": _f(np.nanmax(q)) if n else float("nan"),
        "size_zero_frac": _f(np.sum(q == 0.0) / n) if n else 0.0,
        # a NaN size is non-positive-equivalent (unusable) — count it with the negatives
        "size_neg_frac": _f(np.sum((q < 0.0) | nan) / n) if n else 0.0,
    }


def notional_checks(df: pd.DataFrame) -> dict:
    """Notional volume metrics (§4 row 11): Σ price×quantity and the max single-trade notional."""
    notional = (df["price"].to_numpy(dtype="float64") * df["quantity"].to_numpy(dtype="float64"))
    return {"notional_sum": _f(np.nansum(notional)) if len(notional) else 0.0,
            "notional_max_trade": _f(np.nanmax(notional)) if len(notional) else float("nan")}


def interarrival(df: pd.DataFrame) -> dict:
    """Inter-arrival gap summary on the sorted engine clock (§4 row 12): median/p95/p99/max seconds."""
    clock = sorted_engine_clock(df)
    if len(clock) < 2:
        return {"interarrival_median_s": 0.0, "interarrival_p95_s": 0.0,
                "interarrival_p99_s": 0.0, "interarrival_max_s": 0.0}
    gaps = np.diff(clock.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[ns]")
    secs = gaps.astype("float64") / 1e9
    return {"interarrival_median_s": _f(np.median(secs)),
            "interarrival_p95_s": _f(np.quantile(secs, 0.95)),
            "interarrival_p99_s": _f(np.quantile(secs, 0.99)),
            "interarrival_max_s": _f(np.max(secs))}


def hour_coverage(df: pd.DataFrame, thresholds: TradeThresholds = THRESHOLDS) -> dict:
    """UTC-hour coverage of the engine clock (§4 row 13): fully-empty hours (missing) and non-empty
    hours below `sparse_hour_min_rows` (sparse). Computed on the post-fallback clock so substituted
    rows land in their real hour, not 1970 (the P2 clock fix)."""
    clock = sorted_engine_clock(df)
    if clock.empty:
        return {"missing_hour_count": 24, "sparse_hour_count": 0,
                "missing_hours": list(range(24)), "sparse_hours": []}
    counts = clock.dt.hour.value_counts()
    per_hour = {h: int(counts.get(h, 0)) for h in range(24)}
    missing = [h for h in range(24) if per_hour[h] == 0]
    sparse = [h for h in range(24) if 0 < per_hour[h] < thresholds.sparse_hour_min_rows]
    return {"missing_hour_count": len(missing), "sparse_hour_count": len(sparse),
            "missing_hours": missing, "sparse_hours": sparse}


def lag_metrics(df: pd.DataFrame) -> dict:
    """`received − origin` lag metrics in ms (§4 row 14), over rows where BOTH clocks are non-null.
    Informational (Coinbase inherently higher); a nonzero negative fraction is a clock fault."""
    ocol, rcol = _origin_col(df), _received_col(df)
    if rcol is None:
        return {"recv_origin_lag_median_ms": None, "recv_origin_lag_p95_ms": None,
                "recv_origin_lag_neg_frac": 0.0}
    both = ~(_null_ts_mask(df[ocol]) | _null_ts_mask(df[rcol]))
    if not bool(both.any()):
        return {"recv_origin_lag_median_ms": None, "recv_origin_lag_p95_ms": None,
                "recv_origin_lag_neg_frac": 0.0}
    lag_ms = ((df.loc[both, rcol] - df.loc[both, ocol]).dt.total_seconds() * 1e3).to_numpy()
    return {"recv_origin_lag_median_ms": _f(np.median(lag_ms)),
            "recv_origin_lag_p95_ms": _f(np.quantile(lag_ms, 0.95)),
            "recv_origin_lag_neg_frac": _f(np.mean(lag_ms < 0))}


def side_values(df: pd.DataFrame) -> dict:
    """`side` value counts (§4 row 15). A value ∉ {buy, sell} is surfaced (warn)."""
    if "side" not in df.columns:
        return {"side_values": {}}
    vc = df["side"].value_counts(dropna=False)
    return {"side_values": {str(k): int(v) for k, v in vc.items()}}


# --------------------------------------------------------------------------- metric assembly
def compute_metrics(df: pd.DataFrame, thresholds: TradeThresholds = THRESHOLDS) -> dict:
    """All §4 per-frame metrics on the loaded frame (after the §5 sort). Every float is plain-python;
    report-time `_json_safe` nulls any non-finite value."""
    clock = sorted_engine_clock(df)
    _, origin_null, unresolved = build_engine_clock(df)
    rcol = _received_col(df)
    m = {
        "row_count": int(len(df)),
        "first_ts": _iso_z(clock.min()) if not clock.empty else None,
        "last_ts": _iso_z(clock.max()) if not clock.empty else None,
        "origin_time_null_frac": _f(origin_null.mean()) if len(df) else 0.0,
        "received_time_available": rcol is not None,
        "received_time_null_frac": (_f(_null_ts_mask(df[rcol]).mean())
                                    if rcol is not None and len(df) else None),
        "monotonic_after_sort": monotonic_after_sort(df),
        "was_presorted": was_presorted(df),
        "used_received_time_fallback": bool(origin_null.any()),
        "engine_clock_unresolved_count": int(unresolved.sum()),
    }
    m.update(dup_timestamp_clusters(df))
    m.update(dup_trade_ids(df))
    m.update(price_checks(df, thresholds))
    m.update(size_checks(df))
    m.update(notional_checks(df))
    m.update(interarrival(df))
    m.update(hour_coverage(df, thresholds))
    m.update(lag_metrics(df))
    m.update(side_values(df))
    return m


# --------------------------------------------------------------------------- classification (§8)
def classify(metrics: dict, thresholds: TradeThresholds = THRESHOLDS,
             route: str = ROUTE_REQUIRED) -> tuple[str, list[str]]:
    """Map §4 metrics + thresholds to `(status, reason_codes)` for a REQUIRED Lake day, or route an
    excluded/fill day to its terminal status. Fail is the single blocking status (§8): any fail-level
    reason sets `fail`; else any warn-level reason sets `warn`; else `pass`/`[ok]`. `reason_codes`
    lists fail codes first, then warn codes (stable code, then `metric=value>threshold` detail)."""
    if route == ROUTE_EXCLUDED:
        return EXCLUDED, [CALENDAR_EXCLUDED_DAY]
    if route == ROUTE_COINAPI_FILL:
        return COINAPI_FILL, [COINAPI_FILL_DAY]

    fails: list[str] = []
    warns: list[str] = []

    # --- timestamp / engine clock (§5) --------------------------------------------------------
    if not metrics["monotonic_after_sort"]:
        fails.append(NONMONOTONIC_AFTER_SORT)
    null_frac = metrics["origin_time_null_frac"]
    if metrics["engine_clock_unresolved_count"] > 0:
        fails.append(RECEIVED_TIME_FALLBACK_UNAVAILABLE)     # unresolvable at any fraction
    elif null_frac > thresholds.origin_time_null_max:
        fails.append(ORIGIN_TIME_NULL_FRACTION_HIGH)
        fails.append(f"{ORIGIN_TIME_NULL_FRACTION_HIGH}:frac={null_frac:.4f}>"
                     f"{thresholds.origin_time_null_max}")
    elif metrics["used_received_time_fallback"]:
        warns.append(RECEIVED_TIME_FALLBACK_USED)

    # --- row count ----------------------------------------------------------------------------
    if metrics["row_count"] < thresholds.min_rows_hard:
        fails.append(ROW_COUNT_IMPLAUSIBLY_LOW)
    elif thresholds.min_rows_soft is not None and metrics["row_count"] < thresholds.min_rows_soft:
        warns.append(ROW_COUNT_LOW)

    # --- price (§4 row 9) ---------------------------------------------------------------------
    pmin = metrics["price_min"]
    if (metrics["price_nonfinite_count"] > 0                  # any NaN/±inf price is invalid
            or not (isinstance(pmin, float) and math.isfinite(pmin)) or pmin <= 0.0):
        fails.append(PRICE_OUT_OF_RANGE)
    max_ret = metrics["price_max_abs_ret"]
    spike = ((isinstance(max_ret, float) and math.isfinite(max_ret)
              and max_ret > thresholds.price_spike_warn)
             or metrics["price_out_of_band_count"] > 0)
    if spike:
        fails.append(PRICE_SPIKE)
    elif metrics["price_p99_abs_ret"] > thresholds.price_jump_warn:
        warns.append(PRICE_JUMP_EXCESS)

    # --- size (§4 row 10) ---------------------------------------------------------------------
    if metrics["size_zero_frac"] > 0.0 or metrics["size_neg_frac"] > 0.0:
        fails.append(SIZE_NONPOSITIVE)
    smax = metrics["size_max"]
    if isinstance(smax, float) and math.isfinite(smax):
        if smax > thresholds.size_hard_max_btc:
            fails.append(SIZE_OUT_OF_BAND)                   # hard band takes precedence over warn
        elif smax > thresholds.size_max_btc:
            warns.append(SIZE_OUT_OF_RANGE)

    # --- notional (§4 row 11) -----------------------------------------------------------------
    nsum = metrics["notional_sum"]
    if not (isinstance(nsum, float) and math.isfinite(nsum)) or nsum <= 0.0:
        fails.append(NOTIONAL_NONPOSITIVE)

    # --- duplicate clusters / ids (§4 rows 7-8) -----------------------------------------------
    if metrics["dup_ts_max_cluster"] > thresholds.dup_ts_cluster_warn:
        warns.append(DUPLICATE_TIMESTAMP_CLUSTER)
    if metrics["dup_trade_id_count"] > 0:
        warns.append(DUPLICATE_TRADE_ID)

    # --- inter-arrival gap (§4 row 12) --------------------------------------------------------
    if metrics["interarrival_max_s"] > thresholds.interarrival_gap_warn_s:
        warns.append(INTERARRIVAL_GAP_EXCESS)

    # --- lag (§4 row 14) ----------------------------------------------------------------------
    if metrics["recv_origin_lag_neg_frac"] > thresholds.lag_neg_frac_max:
        fails.append(LAG_NEGATIVE)

    # --- side values (§4 row 15) --------------------------------------------------------------
    if any(v not in VALID_SIDES for v in metrics["side_values"]):
        warns.append(SIDE_VALUE_UNEXPECTED)

    # --- hour coverage (§4 row 13, §8) --------------------------------------------------------
    missing, sparse = metrics["missing_hours"], metrics["sparse_hours"]
    if sparse:
        warns.append(SPARSE_HOUR)
        warns.extend(f"{SPARSE_HOUR}:hour={h:02d}" for h in sparse)
    if len(missing) > thresholds.max_missing_hours:
        fails.append(MISSING_HOURS_EXCESS)
        fails.append(f"{MISSING_HOURS_EXCESS}:count={len(missing)}>{thresholds.max_missing_hours}")
    elif missing:
        warns.append(MISSING_HOUR)
        warns.extend(f"{MISSING_HOUR}:hour={h:02d}" for h in missing)

    if fails:
        return FAIL, fails + warns
    if warns:
        return WARN, warns
    return PASS, [OK]


# --------------------------------------------------------------------------- per-(venue, day) record
def _empty_metrics() -> dict:
    """A schema-consistent all-None/zero metrics block for a structural (no-frame) record."""
    return {
        "row_count": 0, "first_ts": None, "last_ts": None, "origin_time_null_frac": None,
        "received_time_available": None, "received_time_null_frac": None,
        "monotonic_after_sort": None, "was_presorted": None,
        "used_received_time_fallback": None, "engine_clock_unresolved_count": None,
        "dup_ts_cluster_count": None, "dup_ts_max_cluster": None,
        "dup_trade_id_count": None, "dup_trade_id_frac": None, "trade_id_available": None,
        "price_min": None, "price_max": None, "price_median": None,
        "price_p99_abs_ret": None, "price_max_abs_ret": None, "price_out_of_band_count": None,
        "price_nonfinite_count": None,
        "size_min": None, "size_max": None, "size_zero_frac": None, "size_neg_frac": None,
        "notional_sum": None, "notional_max_trade": None,
        "interarrival_median_s": None, "interarrival_p95_s": None,
        "interarrival_p99_s": None, "interarrival_max_s": None,
        "missing_hour_count": None, "sparse_hour_count": None,
        "missing_hours": None, "sparse_hours": None,
        "recv_origin_lag_median_ms": None, "recv_origin_lag_p95_ms": None,
        "recv_origin_lag_neg_frac": None, "side_values": None,
    }


def _record(*, day, venue, status, reason_codes, metrics, calendar_state, vendor_source):
    exch, sym = VENUES.get(venue, (None, None))
    return {"day": day, "venue": venue, "exchange": exch, "symbol": sym,
            "status": status, "reason_codes": list(reason_codes), "vendor_source": vendor_source,
            "metrics": metrics, "calendar_state": calendar_state}


def _route_structural(code: str, route: str) -> tuple[str, list[str]]:
    """Resolve a structural problem (`missing_partition`/`empty_partition`/...) against the calendar
    route: excluded → excluded; a trades-fill day → coinapi_fill (the missing Lake side is expected);
    a required day → the single blocking `fail` (§1/§8)."""
    if route == ROUTE_EXCLUDED:
        return EXCLUDED, [CALENDAR_EXCLUDED_DAY]
    if route == ROUTE_COINAPI_FILL:
        return COINAPI_FILL, [COINAPI_FILL_DAY]
    return FAIL, [code]


def validate_trade_frame(df, venue: str, day: str, thresholds: TradeThresholds = THRESHOLDS,
                         calendar_state: dict | None = None, vendor_source: str = "lake") -> dict:
    """Validate one normalized `(venue, day)` trade frame → a JSON-safe per-day record (§4/§6/§8).

    `df=None` is an ABSENT partition (`missing_partition`); an empty frame is `empty_partition`; both
    route per the calendar (a trades-fill day → `coinapi_fill`, an excluded day → `excluded`,
    otherwise `fail`). A present frame is metric-classified — unless the calendar routes the whole day
    away (fill/excluded days are deferred/out-of-scope; the calendar is authoritative for fill routing,
    §3). Pure: no vendor I/O."""
    cs = calendar_state if calendar_state is not None else {"route": ROUTE_REQUIRED}
    route = cs.get("route", ROUTE_REQUIRED)

    if df is None or len(df) == 0:
        code = MISSING_PARTITION if df is None else EMPTY_PARTITION
        status, reasons = _route_structural(code, route)
        return _record(day=day, venue=venue, status=status, reason_codes=reasons,
                       metrics=_empty_metrics(), calendar_state=cs, vendor_source=vendor_source)

    if _origin_col(df) is None:                              # present frame, no exchange-time column
        if route == ROUTE_EXCLUDED:
            status, reasons = EXCLUDED, [CALENDAR_EXCLUDED_DAY]
        elif route == ROUTE_COINAPI_FILL:
            status, reasons = COINAPI_FILL, [COINAPI_FILL_DAY]
        else:
            status, reasons = FAIL, [ORIGIN_TIME_COLUMN_MISSING]
        return _record(day=day, venue=venue, status=status, reason_codes=reasons,
                       metrics=_empty_metrics(), calendar_state=cs, vendor_source=vendor_source)

    metrics = compute_metrics(df, thresholds)
    status, reasons = classify(metrics, thresholds, route)
    return _record(day=day, venue=venue, status=status, reason_codes=reasons,
                   metrics=metrics, calendar_state=cs, vendor_source=vendor_source)


# --------------------------------------------------------------------------- calendar crossing (§8)
def load_usable_calendar(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _validate_fill_days(fill_days: dict) -> None:
    """Every `coinbase_fill_days` value must be a `{book, trades}` dict (the calendar contract) — a
    malformed entry is a hard error, never a silent mis-route (§9 case 11)."""
    for d, v in fill_days.items():
        if not (isinstance(v, dict) and "book" in v and "trades" in v):
            raise ValueError(
                f"coinbase_fill_days[{d!r}] must be a {{'book':bool,'trades':bool}} dict, got {v!r}")


def calendar_state(cal: dict | None, day_iso: str, venue: str) -> dict:
    """Cross one `(venue, day)` with `data/usable_calendar.json` → a routing state (§4 row 16, §8).

    Precedence: an excluded-calendar day → `excluded`; a Coinbase trades-fill day
    (`coinbase_fill_days[day].trades` truthy, venue `coinbase`) → `coinapi_fill`; otherwise a required
    Lake day. Fill routing is driven ONLY by the calendar flag, never inferred from metrics (§3)."""
    if cal is None:
        return {"route": ROUTE_REQUIRED, "in_usable_days": None, "is_fill_day": False,
                "excluded_reason": None, "fill": None}
    fill_days = cal.get("coinbase_fill_days") or {}
    _validate_fill_days(fill_days)
    excluded_reason = (cal.get("excluded_days_by_reason") or {}).get(day_iso)
    fill_entry = fill_days.get(day_iso)
    is_trades_fill = (venue == "coinbase" and bool((fill_entry or {}).get("trades")))
    if excluded_reason is not None:
        route = ROUTE_EXCLUDED
    elif is_trades_fill:
        route = ROUTE_COINAPI_FILL
    else:
        route = ROUTE_REQUIRED
    return {"route": route, "in_usable_days": day_iso in set(cal.get("usable_days", [])),
            "is_fill_day": is_trades_fill, "excluded_reason": excluded_reason, "fill": fill_entry}


# --------------------------------------------------------------------------- GB / quota gate (§7)
def estimate_trades_gb(venues, days, per_venue_gb: dict | None = None) -> float:
    """Conservative upper-bound Lake `trades` download estimate (GB) for `venues` × `days`."""
    per = per_venue_gb or TRADES_GB_PER_DAY
    return float(len(days)) * sum(per[v] for v in venues)


def quota_decision(*, est_gb, used_gb, quota_gb=QUOTA_GB, max_auto_gb=DEFAULT_MAX_AUTO_GB,
                   allow_broad=False, headroom_gb=DEFAULT_HEADROOM_GB) -> dict:
    """Decide whether a Lake pull of `est_gb` is allowed given current monthly `used_gb`. Two gates:
    (1) the monthly quota is a HARD external limit — refuse if the pull would leave less than
    `headroom_gb` of `quota_gb`, REGARDLESS of `allow_broad`; (2) a soft auto-cap — a pull larger than
    `max_auto_gb` is refused unless `allow_broad`. Mirrors run_coinbase_quality_map.quota_decision."""
    remaining = float(quota_gb) - float(used_gb)
    safe_remaining = remaining - float(headroom_gb)
    base = {"est_gb": float(est_gb), "used_gb": float(used_gb), "quota_gb": float(quota_gb),
            "remaining_gb": remaining, "safe_remaining_gb": safe_remaining,
            "max_auto_gb": float(max_auto_gb), "allow_broad": bool(allow_broad),
            "headroom_gb": float(headroom_gb)}
    if est_gb <= 0:                       # nothing to load — always allowed, headroom irrelevant
        return {**base, "ok": True, "reason": "ok"}
    if est_gb > safe_remaining:
        return {**base, "ok": False, "reason": "quota_headroom"}
    if est_gb > max_auto_gb and not allow_broad:
        return {**base, "ok": False, "reason": "exceeds_auto_cap"}
    return {**base, "ok": True, "reason": "ok"}


# --------------------------------------------------------------------------- report assembly (§6)
def build_report(per_day_results, *, meta: dict) -> dict:
    """Aggregate per-day records into the report: stable per-status counts, per-venue counts, the
    fail/warn day-venue lists, and the `gate` block the bar builder consumes (§6/§8).

    `gate.lake_required_pass` — no required-day `fail` (structural/metric fails only ever land on a
    required day; fill/excluded days are routed away). `gate.coinapi_fill_deferred` — the fill
    `(venue, day)` cases whose CoinAPI validation is still pending (the locked backfill gate).
    `gate.bars_ready` = `lake_required_pass` AND `coinapi_fill_deferred == []`; Phase 4 gates on
    `bars_ready`, never `lake_required_pass` alone, so a span with an unvalidated/locked fill is never
    buildable."""
    days = [dict(r) for r in per_day_results]
    counts = {s: 0 for s in STATUSES}
    by_venue = {v: {s: 0 for s in STATUSES} for v in VENUES}
    fail_day_venues: list[dict] = []
    warn_day_venues: list[dict] = []
    blocking_failures: list[dict] = []
    coinapi_fill_deferred: list[dict] = []
    for r in days:
        status = r["status"]
        counts[status] = counts.get(status, 0) + 1
        by_venue.setdefault(r["venue"], {s: 0 for s in STATUSES})
        by_venue[r["venue"]][status] = by_venue[r["venue"]].get(status, 0) + 1
        if status == FAIL:
            fail_day_venues.append({"day": r["day"], "venue": r["venue"]})
            blocking_failures.append({"day": r["day"], "venue": r["venue"],
                                      "reason_codes": list(r["reason_codes"])})
        elif status == WARN:
            warn_day_venues.append({"day": r["day"], "venue": r["venue"],
                                    "reason_codes": list(r["reason_codes"])})
        elif status == COINAPI_FILL:
            coinapi_fill_deferred.append({"day": r["day"], "venue": r["venue"]})
    lake_required_pass = not blocking_failures
    gate = {"lake_required_pass": lake_required_pass,
            "bars_ready": lake_required_pass and not coinapi_fill_deferred,
            "blocking_failures": blocking_failures,
            "coinapi_fill_deferred": coinapi_fill_deferred}
    summary = {"n_days": len({r["day"] for r in days}), "n_venues": len({r["venue"] for r in days}),
               "counts": counts, "by_venue": by_venue,
               "fail_day_venues": fail_day_venues, "warn_day_venues": warn_day_venues,
               "gate": gate}
    return {"meta": meta, "summary": summary, "days": days}


def write_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
        f.write("\n")
    return path
