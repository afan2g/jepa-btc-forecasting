"""Quota-aware multi-day Coinbase `book_delta_v2` quality map (docs/data.md §5a-Recon, §10 TODO).

Answers the open §10 question — *how many PRESENT Coinbase days reconstruct to a usable
`book_delta_v2` book after the seed/reseed policy, and which present-but-degraded or missing days
need CoinAPI fill?* — without unlocking the backfill gate. For each requested day it runs the SAME
seed/reseed reconstruction-quality path the one-day parity gate uses on the Lake side
(`recon.reseed.reconstruct_lake_l2_at_samples_seeded` + `recon.parity.frame_quality`) and classifies
the day. It is the multi-day GENERALIZATION of `scripts/run_coinbase_parity.py`, restricted to the
Lake side (no CoinAPI replay), so it needs no CoinAPI download — it only RECORDS whether a local
CoinAPI parquet / a calendar-verified fill is available per day.

This is a VALIDATION / quality-map tool, NOT a backfill:
  * It does NOT download CoinAPI and does NOT unlock the §5a backfill gate (still enforced in
    `ingest/download_coinapi.py` / `ingest/_common.py`).
  * It respects Crypto Lake's **300 GB/month** download quota (docs/data.md §2.1/§6/§8): it prints
    current `lakeapi.used_data(sess)`, estimates the requested Lake GB from the §6 measured per-day
    sizes, and REFUSES a broad pull unless `--allow-broad` is passed — and always refuses a pull that
    would breach the monthly quota headroom, override or not.
  * Default day set is small (2025-06-01 = the validated clean day → expected `lake_usable`;
    2026-04-01 = the crossed-`book`-product day, `seed_source_crossed_frac`=0.3751 of the thinned
    seed candidates (31.75% of raw product rows) → its seed
    SOURCE is too unreliable to trust → expected `inconclusive`). Add more with `--days`,
    `--days-file`, or `--include-gap-days N` (documented gap/seam days from the usable calendar). It
    writes the full report under `data/reports/` (git-ignored).

Classifications (explicit thresholds, emitted in the report JSON — see `Thresholds`):
  * lake_usable            — present, seed accepted, crossed/missing/thin all within the usable bar.
  * lake_present_degraded  — present, seed accepted, but a metric is over the usable bar → CoinAPI fill.
  * missing_needs_coinapi  — no Lake `book_delta_v2` for the day (a gap) → CoinAPI fill.
  * excluded              — out of the usable calendar for a non-Coinbase reason (e.g. a Binance gap);
                            skipped before any Lake load (saves quota).
  * inconclusive          — cannot validate the reconstruction: no/all-rejected seed, OR an accepted
                            seed whose `book` SOURCE is itself substantially crossed (e.g. 2026-04-01
                            `seed_source_crossed_frac`=0.3751), OR the Lake load failed. Needs a
                            better seed source or CoinAPI fill before a verdict.

`classification` stays the LAKE-ONLY quality verdict; the downstream CoinAPI fill/stitch decision
lives in each record's `coinapi_fill` block (`coinapi_fill_block`): the PR #13 `needs_fill`/`why`
decision plus the partial-day stitch plan from `recon.stitch_policy`
(`fill_profile`/`fill_segments`/`seams`/`seam_policy` — plan-doc Q7,
docs/superpowers/plans/2026-07-02-partial-day-fill-policy.md). On the Python engine path the plan
is derived from the per-sample validity mask (`plan_day_stitch`), and `quality` carries the
coverage metrics (`lake_present_*`/`trusted_lake_*`/`invalid_runs`); the metrics-only native
engine emits None coverage, and its fill days get a conservative full-day plan (native per-sample
metrics are the plan's Task-3 follow-up).

Usage:
  .venv/bin/python scripts/run_coinbase_quality_map.py                       # default 2-day set
  .venv/bin/python scripts/run_coinbase_quality_map.py --days 2025-06-01,2026-04-01
  .venv/bin/python scripts/run_coinbase_quality_map.py --days-file days.txt --allow-broad
  .venv/bin/python scripts/run_coinbase_quality_map.py --include-gap-days 3   # + calendar gap/seam days

Credentials: Crypto Lake AWS keys in .env (Lake-only; no COINAPI_KEY needed). Mirrors
`scripts/run_coinbase_parity.py::lake_session`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon import native as _native                                  # noqa: E402 (import-safe; no Rust needed)
from recon.ingest import shared_engine_time_col                       # noqa: E402
from recon.parity import frame_quality                               # noqa: E402
from recon.reseed import (                                           # noqa: E402
    ReseedPolicy, reconstruct_lake_l2_at_samples_seeded, snapshots_from_lake_book_df,
)
from recon.stitch_policy import (                                     # noqa: E402
    FULL_DAY_FILL, LAKE_ONLY, PARTIAL_FILL_PROFILES, REASON_CROSSED_SOURCE,
    full_day_plan, invalid_runs, plan_day_stitch, valid_mask_from_frame,
)

NS_PER_MS = 1_000_000
DAY_MS = 86_400_000

# ----------------------------------------------------------------------------- classifications
LAKE_USABLE = "lake_usable"
LAKE_PRESENT_DEGRADED = "lake_present_degraded"
MISSING_NEEDS_COINAPI = "missing_needs_coinapi"
EXCLUDED = "excluded"
INCONCLUSIVE = "inconclusive"
CLASSES = (LAKE_USABLE, LAKE_PRESENT_DEGRADED, MISSING_NEEDS_COINAPI, EXCLUDED, INCONCLUSIVE)
# The reason code that routes a day inconclusive because its accepted seed's `book` SOURCE is itself
# crossed above the bar — the case the 2026-07-01 CoinAPI cross-validation resolved to "fill".
SEED_SOURCE_UNRELIABLE = "seed_accepted_but_source_unreliable"
# The stable reason code for a thin-depth usable-bar failure — the one degraded dimension the
# top-of-book validity mask cannot see (mask predicate: best bid/ask present + uncrossed at
# min_levels_per_side=1, the shared parity-gate warmup predicate), so the fill composer must not
# keep mask-planned Lake spans on such days (Codex P2).
THIN_DEPTH_OVER_BAR = "thin_depth_over_usable_bar"

# ----------------------------------------------------------------------------- quota constants
QUOTA_GB = 300.0           # Crypto Lake individual-plan monthly download cap (docs/data.md §2.1/§8)
DEFAULT_MAX_AUTO_GB = 5.0  # a request larger than this is a "broad" pull → needs --allow-broad
DEFAULT_HEADROOM_GB = 10.0  # never plan to use the last N GB of the monthly quota (used_data lags ~60 min)
QUOTA_REFUSED_EXIT = 5     # small-int exit-code convention (cf. parity exit 3, backfill gate exit 4)
NATIVE_UNAVAILABLE_EXIT = 6  # explicit --engine native but the native extension/tick scale is unavailable

# Conservative per-day Lake footprint by product, GB (docs/data.md §6: Coinbase book_delta_v2+trades
# ~303 MB/day; the `book` 20-level snapshot product ~180 MB/day). We load book_delta_v2 (projected to
# 6 cols) + `book` (for seeding); estimate the FULL product sizes so the quota gate over-estimates
# rather than under-estimates a pull. Known gap days actually cost ~0, so this is an upper bound.
LAKE_GB_PER_DAY = {"book_delta_v2": 0.30, "book": 0.18}
LAKE_PRODUCTS = ("book_delta_v2", "book")

# Small default validation set: 2025-06-01 = the validated clean day (→ lake_usable); 2026-04-01 =
# the crossed-`book`-product day (seed_source_crossed_frac=0.3751 → inconclusive). docs §5a-QualityMap.
DEFAULT_DAYS = ("2025-06-01", "2026-04-01")


@dataclass(frozen=True)
class Thresholds:
    """Usable-day thresholds applied to the seeded reconstruction (emitted in the report JSON).

    Conservative initial values, tunable as multi-day evidence accrues. 2025-06-01 (the validated
    day) sits far inside `lake_usable`: 0.015% crossed, ~0% missing, 0% crossed seed source."""
    crossed_usable_max: float = 0.01    # ≤1% of grid samples crossed after reseed
    missing_usable_max: float = 0.02    # ≤2% of grid samples with no top-of-book on a side
    thin_usable_max: float = 0.10       # ≤10% of grid samples present+uncrossed but thin (< k/side)
    seed_crossed_frac_max: float = 0.05  # ≤5% of `book` seed candidates may be crossed; above this the
                                         # seed SOURCE is unreliable → inconclusive (2026-04-01: 0.3751)

    def as_dict(self) -> dict:
        return {"crossed_usable_max": self.crossed_usable_max,
                "missing_usable_max": self.missing_usable_max,
                "thin_usable_max": self.thin_usable_max,
                "seed_crossed_frac_max": self.seed_crossed_frac_max}


THRESHOLDS = Thresholds()


# ----------------------------------------------------------------------------- pure helpers
def build_grid(day: dt.date, grid_ms: int) -> list[int]:
    """Exchange-time sample grid (int ns) spanning the partition day at `grid_ms` spacing.
    Mirrors `scripts/run_coinbase_parity.py::build_grid` (the shared sampling convention), plus a
    divisibility guard: a `grid_ms` that does not divide the 24 h day would silently truncate the
    grid, so mask-derived stitch-plan day bounds (`last sample + grid_ns`) would stop short of the
    midnight bound synthesized full-day plans use (Codex P3) — fail fast instead."""
    if grid_ms <= 0 or DAY_MS % grid_ms:
        raise ValueError(f"grid_ms must be positive and divide the {DAY_MS} ms day evenly "
                         f"(got {grid_ms}); otherwise the grid truncates and fill-segment day "
                         "bounds fall short of midnight")
    day_open = int(pd.Timestamp(day).value)
    step_ns = grid_ms * NS_PER_MS
    n = DAY_MS // grid_ms
    return [day_open + i * step_ns for i in range(n)]


def _json_safe(obj):
    """Recursively coerce to strict-JSON-valid types: non-finite floats → None, numpy scalars →
    python scalars, so the artifact passes `jq empty` (AGENTS.md). Mirrors run_coinbase_parity."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):  # numpy scalar
        v = obj.item()
        return _json_safe(v) if isinstance(v, float) else v
    return obj


def estimate_lake_gb(n_days, *, products=LAKE_PRODUCTS, per_product_gb=None) -> float:
    """Conservative upper-bound Lake download estimate (GB) for `n_days` × `products`."""
    per = per_product_gb or LAKE_GB_PER_DAY
    return float(n_days) * sum(per[p] for p in products)


def quota_decision(*, est_gb, used_gb, quota_gb=QUOTA_GB, max_auto_gb=DEFAULT_MAX_AUTO_GB,
                   allow_broad=False, headroom_gb=DEFAULT_HEADROOM_GB) -> dict:
    """Decide whether a Lake pull of `est_gb` is allowed given current monthly `used_gb`.

    Two gates: (1) the monthly quota is a HARD external limit — refuse if the pull would leave less
    than `headroom_gb` of the `quota_gb` cap, REGARDLESS of `allow_broad` (used_data lags ~60 min, so
    we keep headroom). (2) a soft auto-cap — a pull larger than `max_auto_gb` is "broad" and refused
    unless `allow_broad`. Returns a JSON-safe decision dict (`ok`, `reason`, and the inputs)."""
    remaining = float(quota_gb) - float(used_gb)
    safe_remaining = remaining - float(headroom_gb)
    base = {"est_gb": float(est_gb), "used_gb": float(used_gb), "quota_gb": float(quota_gb),
            "remaining_gb": remaining, "safe_remaining_gb": safe_remaining,
            "max_auto_gb": float(max_auto_gb), "allow_broad": bool(allow_broad),
            "headroom_gb": float(headroom_gb)}
    if est_gb <= 0:  # nothing to load (e.g. all days excluded) — always allowed, headroom irrelevant
        return {**base, "ok": True, "reason": "ok"}
    if est_gb > safe_remaining:
        return {**base, "ok": False, "reason": "quota_headroom"}
    if est_gb > max_auto_gb and not allow_broad:
        return {**base, "ok": False, "reason": "exceeds_auto_cap"}
    return {**base, "ok": True, "reason": "ok"}


def classify_day(*, have_lake: bool, meta: dict | None, lake_q: dict | None,
                 thresholds: Thresholds = THRESHOLDS) -> tuple[str, list[str]]:
    """Classify a day from its Lake reconstruction metrics. Pure: `meta` is the seed/reseed metrics
    dict, `lake_q` is `frame_quality(...)`. Returns `(classification, reason_codes)`.

    A confident `lake_usable`/`lake_present_degraded` verdict requires a VALIDATED accepted seed AND
    a trustworthy seed SOURCE — without an accepted seed (none/all rejected), a cold-started book can
    look uncrossed for the wrong reason; and even with an accepted seed, if the `book` seed source is
    itself substantially crossed (e.g. 2026-04-01: 0.3751 of candidates), seeds during crossed episodes are
    unreliable and reseeds get blocked, so a clean-looking reconstruction can't be trusted. Both →
    `inconclusive` (the day needs a better seed source or CoinAPI fill), not silently usable."""
    if not have_lake:
        return MISSING_NEEDS_COINAPI, ["lake_book_delta_v2_absent"]
    crossed = float(lake_q["crossed_rate"])
    missing = float(lake_q["missing_book_fraction"])
    thin = float(meta.get("thin_depth_fraction") or 0.0)
    if not meta.get("seed_accepted"):
        sr = meta.get("seed_reason")
        code = "no_seed_snapshots" if sr in (None, "no_snapshots") else f"seed_rejected:{sr}"
        return INCONCLUSIVE, [code, f"crossed_rate_after={crossed:.4f}"]
    # Seed accepted, but is the seed SOURCE itself reliable? The `book` product is intermittently
    # crossed (2026-04-01: 0.3751 of candidates); above seed_crossed_frac_max the accepted seed
    # cannot be trusted.
    codes = meta.get("snapshot_reason_codes") or {}
    n_snap = sum(codes.values())
    seed_crossed_frac = (codes.get("crossed", 0) / n_snap) if n_snap else 0.0
    if seed_crossed_frac > thresholds.seed_crossed_frac_max:
        return INCONCLUSIVE, [SEED_SOURCE_UNRELIABLE,
                              f"seed_source_crossed_frac={seed_crossed_frac:.4f}>"
                              f"{thresholds.seed_crossed_frac_max}"]
    over: list[str] = []
    if crossed > thresholds.crossed_usable_max:
        over.append(f"crossed_rate_after={crossed:.4f}>{thresholds.crossed_usable_max}")
    if missing > thresholds.missing_usable_max:
        over.append(f"missing_book_fraction={missing:.4f}>{thresholds.missing_usable_max}")
    if thin > thresholds.thin_usable_max:
        # Stable code first, metric detail second — the SEED_SOURCE_UNRELIABLE pattern.
        over.append(THIN_DEPTH_OVER_BAR)
        over.append(f"thin_depth_fraction={thin:.4f}>{thresholds.thin_usable_max}")
    if over:
        return LAKE_PRESENT_DEGRADED, ["seed_accepted", *over]
    return LAKE_USABLE, ["seed_accepted", f"crossed_rate_after={crossed:.4f}",
                         f"missing_book_fraction={missing:.4f}", f"thin_depth_fraction={thin:.4f}"]


def coinapi_fill_decision(classification: str, reasons) -> dict:
    """Machine-readable CoinAPI-fill mapping for a day's `(classification, reasons)` — the contract
    fill manifests consume, so the doc-level fill policy never needs manual reinterpretation of
    reason strings (docs/data.md §5a-QualityMap "CoinAPI cross-validation").

    Returns `{"needs_fill": True|False|None, "why": <code>}`. `needs_fill=None` means the day has no
    fill decision — either `why="no_verdict"` (an unresolved `inconclusive` day: a fill manifest must
    surface it, never silently drop it) or `why="excluded_not_in_scope"` (out of the usable calendar
    for a non-Coinbase reason — not a Coinbase fill candidate at all; `build_report` buckets the two
    separately).

    `inconclusive` days carrying `seed_accepted_but_source_unreliable` map to `needs_fill=True` per
    the 2026-07-01 CoinAPI cross-validation: on 2 of the 4 such days (the extremes of the observed
    8.4–37.5% severity range) parity fails even outside the excluded crossed windows, so
    crossed-seed-source days are fill days, not rehabilitable from Lake alone. That policy is
    PROVISIONAL (2 of 4 days measured) — deliberately encoded here rather than by reclassifying the
    day, so `inconclusive` keeps meaning "no verdict from Lake alone"."""
    rs = set(reasons or ())
    if classification == MISSING_NEEDS_COINAPI:
        return {"needs_fill": True, "why": "lake_book_delta_v2_absent"}
    if classification == LAKE_PRESENT_DEGRADED:
        return {"needs_fill": True, "why": "quality_over_usable_bar"}
    if classification == INCONCLUSIVE and SEED_SOURCE_UNRELIABLE in rs:
        return {"needs_fill": True, "why": "crossed_seed_source_cross_validated_2026-07-01"}
    if classification == LAKE_USABLE:
        return {"needs_fill": False, "why": "lake_usable"}
    if classification == EXCLUDED:
        return {"needs_fill": None, "why": "excluded_not_in_scope"}
    return {"needs_fill": None, "why": "no_verdict"}


# Stitch-plan keys of the per-day `coinapi_fill` block (plan-doc Q7). Days without a stitch plan
# (no fill, no verdict, out of scope) carry them as None — stable schema either way.
_NO_PLAN_FIELDS = {"fill_profile": None, "full_day_reason": None, "fill_segments": None,
                   "seams": None, "seam_policy": None}
# Fallback full-day reason per needs-fill `why` code, used when a fill day has NO mask-derived plan
# (Lake partition absent; metrics-only native engine — per-sample coverage is the Task-3 native
# follow-up) or a mask plan with no fillable window (`lake_only` — e.g. thin-depth degradation,
# invisible to the top-of-book validity predicate): route the WHOLE day to CoinAPI, per the policy
# "full-day unless the partial policy clearly supports a narrower fill". Script-level codes reuse
# the day-level reason strings; the crossed-source code is the shared stitch-policy constant.
_FALLBACK_FULL_DAY_REASON = {
    "lake_book_delta_v2_absent": "lake_book_delta_v2_absent",
    "quality_over_usable_bar": "quality_over_usable_bar",
    "crossed_seed_source_cross_validated_2026-07-01": REASON_CROSSED_SOURCE,
}
INVALID_RUNS_CAP = 100  # invalid_runs list cap in the report (n_invalid_runs keeps the full count)


def coinapi_fill_block(classification: str, reasons, *, day: str, grid_ms: int = 1000,
                       stitch_plan: dict | None = None) -> dict:
    """The full per-day `coinapi_fill` report block: the PR #13 `needs_fill`/`why` decision
    (`coinapi_fill_decision`, unchanged) plus the partial-day stitch plan (plan-doc Q7 /
    docs/data.md §5a-QualityMap). Q2 routing applies only AFTER the day-level decision:

      * `needs_fill` in (False, None) → no stitch plan (`fill_profile: null`), whether or not a
        mask plan was computed — `lake_usable` days are Lake-only, unresolved days stay surfaced.
      * `needs_fill is True` with a mask-derived plan supporting a fill → that plan verbatim
        (`fill_profile`/`full_day_reason`/`fill_segments`/`seams`/`seam_policy` from
        `plan_day_stitch(...).as_dict()`).
      * `needs_fill is True` otherwise → a synthesized conservative full-day plan
        (`full_day_plan`) with the `_FALLBACK_FULL_DAY_REASON` code for the decision's `why`.
    """
    base = coinapi_fill_decision(classification, reasons)
    if base["needs_fill"] is not True:
        return {**base, **_NO_PLAN_FIELDS}
    plan = stitch_plan
    if plan is not None and plan["fill_profile"] != FULL_DAY_FILL and (
            plan["fill_profile"] == LAKE_ONLY or THIN_DEPTH_OVER_BAR in set(reasons or ())):
        # No mask-supported narrower fill: either the mask shows no fillable window (lake_only),
        # or the thin-depth bar failed — a depth failure the top-of-book mask cannot see, so its
        # Lake spans (even beside a real gap — Codex P2) cannot be vouched for → full-day.
        plan = None
    if plan is None:
        if stitch_plan is not None:  # reuse the mask plan's exact day bounds
            day_open, day_end = int(stitch_plan["day_open_ts"]), int(stitch_plan["day_end_ts"])
            grid_ns = int(stitch_plan["grid_ns"])
        else:
            day_open = int(pd.Timestamp(day).value)
            day_end = day_open + DAY_MS * NS_PER_MS
            grid_ns = grid_ms * NS_PER_MS
        plan = full_day_plan(day_open_ts=day_open, day_end_ts=day_end, grid_ns=grid_ns,
                             reason=_FALLBACK_FULL_DAY_REASON[base["why"]], day=day).as_dict()
    return {**base, "fill_profile": plan["fill_profile"],
            "full_day_reason": plan["full_day_reason"], "fill_segments": plan["fill_segments"],
            "seams": plan["seams"], "seam_policy": plan["seam_policy"]}


def _empty_seed_block(snapshots_present, candidates) -> dict:
    return {"snapshots_present": bool(snapshots_present), "snapshot_candidates": int(candidates),
            "seed_accepted": False, "seed_reason": None, "seed_ts": None, "reseed_count": 0,
            "reseed_blocked_invalid_snapshot": 0, "snapshot_reason_codes": {}}


# Q7 coverage keys (plan doc): None whenever no validity mask exists — missing/excluded/load-failed
# days, and the metrics-only native engine (per-sample runs are the Task-3 native follow-up).
_EMPTY_COVERAGE = {"lake_present_start_ts": None, "lake_present_end_ts": None,
                   "trusted_lake_start_ts": None, "trusted_lake_end_ts": None,
                   "n_invalid_runs": None, "invalid_runs": None}


def _empty_quality_block(k, grid_ms) -> dict:
    return {"k": int(k), "grid_ms": int(grid_ms), "n_grid": None, "engine_time_col": None,
            "crossed_rate_after": None, "crossed_samples_after": None, "crossed_rate_cold": None,
            "missing_book_fraction": None, "thin_depth_fraction": None,
            "crossed_duration_s_after": None, **_EMPTY_COVERAGE}


def _default_coinapi_block() -> dict:
    return {"parquet_local": None, "parquet_path": None, "fillable": None}


def _default_calendar_block() -> dict:
    return {"in_usable_days": None, "in_lake_all_days": None, "is_coinbase_fill_day": None,
            "excluded_reason": None}


def _seeded_reconstruct(engine, price_scale, *, df, grid, k, engine_col, snapshots, policy, frame_out):
    """Dispatch to the native or Python seeded reconstruction (identical `(frame, meta)` schema).
    Native mode consumes the verified `price_scale` (tick multiplier); Python ignores it."""
    if engine == "native":
        return _native.reconstruct_lake_l2_at_samples_seeded_native(
            df, grid, k=k, engine_time_col=engine_col, snapshots=snapshots, policy=policy,
            frame_out=frame_out, price_scale=price_scale)
    return reconstruct_lake_l2_at_samples_seeded(
        df, grid, k=k, engine_time_col=engine_col, snapshots=snapshots, policy=policy,
        frame_out=frame_out)


def _stitch_and_coverage(frame, *, meta: dict, reasons, grid_ms: int,
                         day: dt.date) -> tuple[dict, dict]:
    """Mask-derived stitch plan + Q7 coverage metrics from the materialized top-K frame (Python
    engine path only). The validity mask is the shared parity-gate predicate
    (`valid_mask_from_frame`); presence is the `frame_quality` both-sides-of-book predicate.
    `seed_source_trusted` comes from the classification's SEED_SOURCE_UNRELIABLE reason, so the
    plan and the classification can never disagree on the PR #13 crossed-source rule."""
    sf = frame.sort_values("sample_ts")
    ts = sf["sample_ts"].to_numpy(dtype=np.int64)
    grid_ns = grid_ms * NS_PER_MS
    valid = valid_mask_from_frame(sf)
    present = (sf["bid_0_price"].notna() & sf["ask_0_price"].notna()).to_numpy(dtype=bool)
    plan = plan_day_stitch(ts, valid, grid_ns=grid_ns, seed_accepted=bool(meta["seed_accepted"]),
                           seed_ts=meta["seed_ts"],
                           seed_source_trusted=SEED_SOURCE_UNRELIABLE not in reasons,
                           present=present, day=day.isoformat()).as_dict()
    runs = invalid_runs(ts, valid, grid_ns=grid_ns)
    coverage = {key: plan[key] for key in ("lake_present_start_ts", "lake_present_end_ts",
                                           "trusted_lake_start_ts", "trusted_lake_end_ts")}
    coverage.update(n_invalid_runs=len(runs),
                    invalid_runs=[[a, b] for a, b in runs[:INVALID_RUNS_CAP]])
    return plan, coverage


def _lake_quality_from_meta(meta, *, source_rows) -> dict:
    """`frame_quality(...)`-shaped dict from native metrics-only `meta` — the native quality-map path
    classifies WITHOUT materializing the top-K frame. `meta.crossed_*`/`missing_*` are pinned equal to
    `frame_quality(frame)` by the conformance tests (test_native_recon), so classification is engine-
    independent."""
    return {"n_samples": int(meta["n_samples"]),
            "crossed_samples": int(meta["crossed_samples"]),
            "crossed_rate": float(meta["crossed_rate"]),
            "missing_book_samples": int(meta["missing_book_samples"]),
            "missing_book_fraction": float(meta["missing_book_fraction"]),
            "source_rows": int(source_rows)}


def assess_lake_day(lake_delta_df, lake_book_snapshots, *, day: dt.date, k: int = 10,
                    grid_ms: int = 1000, reseed: bool = True, reseed_after_crossed_s: float = 2.0,
                    seed_min_levels: int = 5, max_spread_frac: float | None = None,
                    cold_ab: bool = True, thresholds: Thresholds = THRESHOLDS,
                    coinapi: dict | None = None, calendar: dict | None = None,
                    engine: str = "python", price_scale: int | None = None) -> dict:
    """Run the seed/reseed Lake reconstruction-quality path for one day and classify it. PURE w.r.t.
    its inputs (a pre-loaded Lake `book_delta_v2` DataFrame + optional pre-parsed `book` snapshots) —
    no vendor I/O — so the offline tests drive the exact production classification path.

    `engine` selects the replay engine: `"python"` (default; the correctness oracle, builds the top-K
    frame + `frame_quality`) or `"native"` (metrics-only — classifies from native `meta` without the
    frame, `price_scale` required). Classification is identical either way."""
    have_lake = lake_delta_df is not None and len(lake_delta_df) > 0
    n_rows = (0 if lake_delta_df is None else int(len(lake_delta_df)))
    snaps = list(lake_book_snapshots or [])
    have_snaps = bool(snaps)
    result = {
        "day": day.isoformat(),
        "lake_book_delta_v2_present": bool(have_lake),
        "lake_delta_rows": n_rows,
        "coinapi": coinapi if coinapi is not None else _default_coinapi_block(),
        "calendar": calendar if calendar is not None else _default_calendar_block(),
    }

    if not have_lake:
        cls, reasons = classify_day(have_lake=False, meta=None, lake_q=None, thresholds=thresholds)
        result.update(classification=cls, reasons=reasons,
                      seed=_empty_seed_block(have_snaps, len(snaps)),
                      quality=_empty_quality_block(k, grid_ms), stitch_plan=None)
        return result

    grid = build_grid(day, grid_ms)
    engine_col = shared_engine_time_col(lake_delta_df)
    policy = ReseedPolicy(enabled=reseed, min_levels_per_side=seed_min_levels,
                          reseed_after_crossed_s=reseed_after_crossed_s,
                          max_spread_frac=max_spread_frac)
    # Native mode classifies from metrics-only meta (no top-K frame materialized); Python builds the
    # frame and derives frame_quality. The two are pinned equal by the conformance tests.
    if engine == "native":
        _, meta = _seeded_reconstruct(engine, price_scale, df=lake_delta_df, grid=grid, k=k,
                                      engine_col=engine_col, snapshots=(snaps or None), policy=policy,
                                      frame_out=False)
        lake_q = _lake_quality_from_meta(meta, source_rows=n_rows)
    else:
        frame, meta = _seeded_reconstruct(engine, price_scale, df=lake_delta_df, grid=grid, k=k,
                                          engine_col=engine_col, snapshots=(snaps or None),
                                          policy=policy, frame_out=True)
        lake_q = frame_quality(frame, source_rows=n_rows)

    cold_rate = None
    if cold_ab and have_snaps:
        # A/B "before": the byte-identical reconstruction cold-started (no seed/reseed); metrics-only
        # (frame_out=False) so the 86,400-row frame is not re-materialized. Doubles the per-day delta
        # replay — disable with --no-cold-ab on a multi-GB-day sweep where only the after-rate matters.
        _, cold_meta = _seeded_reconstruct(engine, price_scale, df=lake_delta_df, grid=grid, k=k,
                                           engine_col=engine_col, snapshots=None, policy=policy,
                                           frame_out=False)
        cold_rate = float(cold_meta["crossed_rate"])

    cls, reasons = classify_day(have_lake=True, meta=meta, lake_q=lake_q, thresholds=thresholds)
    # Stitch plan + Q7 coverage need the per-sample validity mask, so they exist only where the
    # top-K frame was materialized (Python engine); native stays metrics-only (Task-3 follow-up) —
    # its fill days get the conservative full-day fallback when the report is built.
    stitch, coverage = None, dict(_EMPTY_COVERAGE)
    if engine != "native":
        stitch, coverage = _stitch_and_coverage(frame, meta=meta, reasons=reasons,
                                                grid_ms=grid_ms, day=day)
    result.update(
        classification=cls, reasons=reasons, stitch_plan=stitch,
        seed={
            "snapshots_present": have_snaps,
            "snapshot_candidates": len(snaps),
            "seed_accepted": bool(meta["seed_accepted"]),
            "seed_reason": meta["seed_reason"],
            "seed_ts": meta["seed_ts"],
            "reseed_count": int(meta["reseed_count"]),
            "reseed_blocked_invalid_snapshot": int(meta["reseed_blocked_invalid_snapshot"]),
            "snapshot_reason_codes": dict(meta["snapshot_reason_codes"]),
        },
        quality={
            "k": int(k), "grid_ms": int(grid_ms), "n_grid": len(grid),
            "engine_time_col": engine_col,
            "crossed_rate_after": float(lake_q["crossed_rate"]),
            "crossed_samples_after": int(lake_q["crossed_samples"]),
            "crossed_rate_cold": cold_rate,
            "missing_book_fraction": float(lake_q["missing_book_fraction"]),
            "thin_depth_fraction": float(meta["thin_depth_fraction"]),
            "crossed_duration_s_after": float(meta["crossed_duration_s"]),
            **coverage,
        },
    )
    return result


def excluded_result(day: dt.date, reasons, *, k: int = 10, grid_ms: int = 1000,
                    coinapi: dict | None = None, calendar: dict | None = None) -> dict:
    """A schema-consistent `excluded` per-day record for a day skipped before any Lake load."""
    return {
        "day": day.isoformat(), "classification": EXCLUDED, "reasons": list(reasons),
        "lake_book_delta_v2_present": None, "lake_delta_rows": None, "stitch_plan": None,
        "seed": _empty_seed_block(False, 0), "quality": _empty_quality_block(k, grid_ms),
        "coinapi": coinapi if coinapi is not None else _default_coinapi_block(),
        "calendar": calendar if calendar is not None else _default_calendar_block(),
    }


def inconclusive_load_failure(day: dt.date, err: str, *, k: int = 10, grid_ms: int = 1000,
                              coinapi: dict | None = None, calendar: dict | None = None) -> dict:
    """A schema-consistent `inconclusive` record for a day whose Lake load raised (can't be judged)."""
    return {
        "day": day.isoformat(), "classification": INCONCLUSIVE,
        "reasons": [f"lake_load_failed:{err}"],
        "lake_book_delta_v2_present": None, "lake_delta_rows": None, "stitch_plan": None,
        "seed": _empty_seed_block(False, 0), "quality": _empty_quality_block(k, grid_ms),
        "coinapi": coinapi if coinapi is not None else _default_coinapi_block(),
        "calendar": calendar if calendar is not None else _default_calendar_block(),
    }


def build_report(per_day_results, *, meta: dict) -> dict:
    """Aggregate per-day results into the report: stable per-class counts + day lists + the rows.
    Every per-day record is stamped with its full machine-readable `coinapi_fill` block
    (`coinapi_fill_block`: the PR #13 needs_fill/why decision + the Q7 stitch plan), and the
    summary carries the fill day-lists — the contract a fill manifest reads, so no consumer
    re-parses reason strings. A record's internal `stitch_plan` key (the mask-derived
    `plan_day_stitch` dict from `assess_lake_day`, Python engine only) is consumed here and never
    emitted per-day. `no_verdict` holds only genuinely unresolved days; calendar-excluded days go
    to the separate `not_in_scope` list so out-of-scope (e.g. Binance-gap) days are never mistaken
    for unresolved Coinbase fills.

    Summary extensions (plan-doc Q7 + the wiring task): `partial_fill` (day list, ⊆ `needs_fill`),
    `fill_counts` (flat counts: needs_fill, the five fill profiles, crossed-source full-days,
    no_verdict/no_fill/not_in_scope), and `full_day_reason_counts` (full-day fills by reason)."""
    days = []
    for r in per_day_results:
        rec = dict(r)
        plan = rec.pop("stitch_plan", None)
        rec["coinapi_fill"] = coinapi_fill_block(
            rec["classification"], rec.get("reasons"), day=rec["day"],
            grid_ms=(rec.get("quality") or {}).get("grid_ms") or 1000, stitch_plan=plan)
        if (plan is not None and plan["fill_profile"] != FULL_DAY_FILL
                and rec["coinapi_fill"]["fill_profile"] == FULL_DAY_FILL and rec.get("quality")):
            # A non-full-day mask plan (lake_only, or a partial plan on a thin-depth failure) was
            # overridden to a full-day fill: no Lake coverage survives a full-day route (plan-doc
            # definitions table), so the report's trusted_lake_* must be None. Presence/invalid-run
            # facts stay; copy-on-write keeps the caller's record (the mask-level view, consistent
            # with its own plan) unmutated.
            rec["quality"] = {**rec["quality"], "trusted_lake_start_ts": None,
                              "trusted_lake_end_ts": None}
        days.append(rec)
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    fill: dict[str, list] = {"needs_fill": [], "no_fill": [], "no_verdict": [], "not_in_scope": [],
                             "partial_fill": []}
    profile_counts = {p: 0 for p in (FULL_DAY_FILL, *PARTIAL_FILL_PROFILES)}
    full_day_reasons: dict[str, int] = {}
    for r in days:
        by_class.setdefault(r["classification"], []).append(r["day"])
        cf = r["coinapi_fill"]
        nf = cf["needs_fill"]
        if nf:
            fill["needs_fill"].append(r["day"])
            profile_counts[cf["fill_profile"]] += 1
            if cf["fill_profile"] in PARTIAL_FILL_PROFILES:
                fill["partial_fill"].append(r["day"])
            if cf["full_day_reason"] is not None:
                full_day_reasons[cf["full_day_reason"]] = \
                    full_day_reasons.get(cf["full_day_reason"], 0) + 1
        elif nf is False:
            fill["no_fill"].append(r["day"])
        else:
            key = ("not_in_scope" if cf["why"] == "excluded_not_in_scope" else "no_verdict")
            fill[key].append(r["day"])
    counts = {c: len(by_class[c]) for c in by_class}
    fill_counts = {"needs_fill": len(fill["needs_fill"]), **profile_counts,
                   "crossed_source_full_day": full_day_reasons.get(REASON_CROSSED_SOURCE, 0),
                   "no_verdict": len(fill["no_verdict"]), "no_fill": len(fill["no_fill"]),
                   "not_in_scope": len(fill["not_in_scope"])}
    return {"meta": meta,
            "summary": {"n_days": len(days), "counts": counts, "by_class": by_class,
                        "coinapi_fill": {**fill, "fill_counts": fill_counts,
                                         "full_day_reason_counts": full_day_reasons}},
            "days": days}


def write_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
        f.write("\n")
    return path


# ----------------------------------------------------------------------------- credentials (vendor)
def lake_session():
    """Crypto Lake boto3 session from .env subscriber keys (NOT the personal ~/.aws default).
    Lake-only: does not require COINAPI_KEY. Mirrors scripts/run_coinbase_parity.py::lake_session."""
    import boto3  # local import: importing this module must not require boto3/lakeapi
    env: dict = {}
    envpath = ROOT / ".env"
    if envpath.exists():
        for line in envpath.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip().strip('"').strip("'")
    env = {**env, **os.environ}
    try:
        return boto3.Session(
            aws_access_key_id=env["aws_access_key_id"],
            aws_secret_access_key=env["aws_secret_access_key"],
            region_name=env.get("region", "eu-west-1"),
        )
    except KeyError as e:
        raise SystemExit(
            f"Crypto Lake AWS key {e} not found in .env or environment "
            "(need aws_access_key_id and aws_secret_access_key)."
        ) from None


def lake_used_data(sess) -> dict:
    """Current monthly Crypto Lake download usage: {downloaded_gb, timeframe_days, ...}. May lag
    up to ~60 min (vendor-side) and is cached 60 s (lakeapi FSLRUCache)."""
    import lakeapi  # local import: keep module import side-effect-free / vendor-free
    return lakeapi.used_data(sess)


# ----------------------------------------------------------------------------- data loading (vendor)
def load_lake_book_delta_v2(sess, day: dt.date, exchange: str, symbol: str) -> pd.DataFrame:
    """Load one day of Crypto Lake `book_delta_v2` (projected to the 6 recon columns). Mirrors
    scripts/run_coinbase_parity.py."""
    import lakeapi  # local import
    start = dt.datetime.combine(day, dt.time())
    end = start + dt.timedelta(days=1)
    return lakeapi.load_data(
        table="book_delta_v2", start=start, end=end, symbols=[symbol], exchanges=[exchange],
        columns=["timestamp", "receipt_timestamp", "sequence_number", "side_is_bid",
                 "price", "size"],
        boto3_session=sess, drop_partition_cols=True,
    )


def load_lake_book_snapshots(sess, day: dt.date, exchange: str, symbol: str, *,
                             max_levels: int = 20, stride_ms: int = 1000):
    """Load one day of the Crypto Lake `book` (20-level snapshot) product as seed/reseed candidates,
    thinned to ~one per `stride_ms`. Mirrors scripts/run_coinbase_parity.py."""
    import lakeapi  # local import
    cols = ["timestamp"]
    for i in range(max_levels):
        cols += [f"bid_{i}_price", f"bid_{i}_size", f"ask_{i}_price", f"ask_{i}_size"]
    start = dt.datetime.combine(day, dt.time())
    end = start + dt.timedelta(days=1)
    df = lakeapi.load_data(table="book", start=start, end=end, symbols=[symbol],
                           exchanges=[exchange], columns=cols, boto3_session=sess,
                           drop_partition_cols=True)
    if not len(df):
        return []
    etc = shared_engine_time_col(df)
    return snapshots_from_lake_book_df(df, engine_time_col=etc, max_levels=max_levels,
                                       stride_ns=stride_ms * NS_PER_MS)


def _is_no_files(exc: BaseException) -> bool:
    """True if `exc` signals an ABSENT Lake partition (a gap day). lakeapi (0.22.3) raises
    `NoFilesFound` for a fully-missing partition rather than returning an empty frame, so a gap must
    be detected here and routed to missing_needs_coinapi — NOT treated as a load failure."""
    return (type(exc).__name__ == "NoFilesFound"
            or "nofilesfound" in repr(exc).lower()
            or "no files found" in str(exc).lower())


def coinapi_parquet_path(out_root: str, day: dt.date, exchange: str, symbol: str) -> str:
    return os.path.join(out_root, "limitbook_full", f"exchange={exchange}",
                        f"symbol={symbol}", f"dt={day.isoformat()}", "data.parquet")


# ----------------------------------------------------------------------------- usable calendar
def load_usable_calendar(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def calendar_context(cal: dict | None, day_iso: str) -> dict:
    if cal is None:
        return _default_calendar_block()
    return {
        "in_usable_days": day_iso in set(cal.get("usable_days", [])),
        "in_lake_all_days": day_iso in set(cal.get("lake_all_days", [])),
        "is_coinbase_fill_day": day_iso in (cal.get("coinbase_fill_days") or {}),
        "excluded_reason": (cal.get("excluded_days_by_reason") or {}).get(day_iso),
    }


def coinapi_context(cal: dict | None, day: dt.date, coinapi_root: str,
                    exchange: str, symbol: str) -> dict:
    """Whether CoinAPI parity is AVAILABLE for the day — a local parquet on disk and/or a
    calendar-verified flat-files `book` fill — without downloading anything."""
    path = coinapi_parquet_path(coinapi_root, day, exchange, symbol)
    fillable = None
    if cal is not None:
        fs = (cal.get("fill_status") or {}).get(day.isoformat())
        if fs is not None:
            b = fs.get("book")
            fillable = bool(b and b.get("present"))
    return {"parquet_local": os.path.exists(path), "parquet_path": path, "fillable": fillable}


def gap_days_from_calendar(cal: dict | None, n: int) -> list[str]:
    """Up to `n` documented Coinbase BOOK-gap days (time-sorted). `coinbase_fill_days` mixes book gaps
    with trade-only gaps (each entry is `{"book": bool, "trades": bool}` = which product Lake is missing
    that day). This runner maps `book_delta_v2` only, so we keep just `book: True` days (Lake book
    absent → expected missing_needs_coinapi); a trade-only day's Lake book is PRESENT and would waste
    quota without yielding a book-gap sample."""
    if cal is None or n <= 0:
        return []
    fill = cal.get("coinbase_fill_days") or {}
    book_gaps = [day for day, v in fill.items() if (v or {}).get("book") is True]
    return sorted(book_gaps)[:n]


# ----------------------------------------------------------------------------- day-set resolution
def resolve_days(args) -> list[dt.date]:
    if args.days_file:
        text = pathlib.Path(args.days_file).read_text()
        toks = [t.strip() for line in text.splitlines() for t in line.split(",")]
    elif args.days:
        toks = [t.strip() for t in args.days.split(",")]
    else:
        toks = list(DEFAULT_DAYS)
    days = [dt.date.fromisoformat(t) for t in toks if t]
    if args.include_gap_days:
        cal = load_usable_calendar(args.usable_calendar)
        for d in gap_days_from_calendar(cal, args.include_gap_days):
            gd = dt.date.fromisoformat(d)
            if gd not in days:
                days.append(gd)
    # de-dupe, keep order
    seen: set = set()
    out: list[dt.date] = []
    for d in days:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# ----------------------------------------------------------------------------- reporting (stdout)
def print_summary(report: dict) -> None:
    s = report["summary"]
    m = report["meta"]
    print("\n" + "=" * 74)
    print(f"  COINBASE QUALITY MAP — {s['n_days']} day(s)  (k={m.get('k')}, "
          f"grid={m.get('grid_ms')}ms)")
    print("=" * 74)
    for c in CLASSES:
        days = s["by_class"].get(c, [])
        if days:
            shown = ", ".join(days[:8]) + (" …" if len(days) > 8 else "")
            print(f"  {c:<22} {len(days):>4}   {shown}")
        else:
            print(f"  {c:<22} {len(days):>4}")
    fc = (s.get("coinapi_fill") or {}).get("fill_counts")
    if fc:
        n_partial = sum(fc[p] for p in PARTIAL_FILL_PROFILES)
        print(f"  coinapi fill: {fc['needs_fill']} day(s) to fill — {fc['full_day_fill']} "
              f"full-day ({fc['crossed_source_full_day']} crossed-source), {n_partial} partial; "
              f"{fc['no_verdict']} unresolved, {fc['no_fill']} no-fill")
    for d in report["days"]:
        q = d.get("quality", {})
        extra = ""
        if q.get("crossed_rate_after") is not None:
            extra = (f"crossed {q['crossed_rate_after']:.4%}"
                     + (f" (cold {q['crossed_rate_cold']:.2%})" if q.get("crossed_rate_cold")
                        is not None else "")
                     + f", missing {q['missing_book_fraction']:.4%}")
        print(f"    {d['day']}  {d['classification']:<22} {extra}")


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Quota-aware multi-day Coinbase book_delta_v2 quality map (validation, NOT a "
                    "backfill; does not unlock the §5a CoinAPI backfill gate)")
    ap.add_argument("--days", default=None,
                    help="explicit days, CSV YYYY-MM-DD,YYYY-MM-DD (default: the small validation set "
                         "2025-06-01,2026-04-01)")
    ap.add_argument("--days-file", default=None,
                    help="file of days (CSV and/or one-per-line); overrides --days")
    ap.add_argument("--include-gap-days", type=int, default=0,
                    help="also map the first N documented Coinbase gap/seam days from the usable "
                         "calendar (expected missing_needs_coinapi; default 0)")
    ap.add_argument("--k", type=int, default=10, help="top-K levels (default 10)")
    ap.add_argument("--grid-ms", type=int, default=1000, help="sample-grid spacing ms (default 1000)")
    ap.add_argument("--no-reseed", action="store_true",
                    help="seed once but DISABLE intraday reseed-on-crossing (A/B; §5a-Recon)")
    ap.add_argument("--no-lake-seed", action="store_true",
                    help="do NOT load the Lake `book` snapshot product (pure cold-start; every "
                         "present day then classifies inconclusive — no validated seed)")
    ap.add_argument("--no-cold-ab", action="store_true",
                    help="skip the cold-start A/B crossed-rate (halves per-day reconstruction cost)")
    ap.add_argument("--engine", choices=("auto", "python", "native"), default="auto",
                    help="Lake replay engine (docs/data.md §5a-Recon native engine). 'python' = the "
                         "correctness reference; 'native' = the recon_native Rust core (fails before "
                         "any Lake load if unavailable or the symbol lacks a verified tick scale); "
                         "'auto' (default) = native when available+verified, else Python.")
    ap.add_argument("--reseed-after-crossed-s", type=float, default=2.0,
                    help="reseed only after the book is crossed continuously ≥ this many seconds")
    ap.add_argument("--seed-min-levels", type=int, default=5,
                    help="min levels/side for a valid Lake `book` seed/reseed source (default 5)")
    ap.add_argument("--seed-max-spread-frac", type=float, default=None,
                    help="optional sane-spread guard for seed validation (off by default)")
    ap.add_argument("--book-stride-ms", type=int, default=1000,
                    help="thin the Lake `book` product to ~one seed candidate per N ms (default 1000)")
    ap.add_argument("--book-max-levels", type=int, default=20,
                    help="Lake `book` snapshot depth to load/seed from (default 20)")
    ap.add_argument("--exchange", default="COINBASE", help="Crypto Lake / partition exchange")
    ap.add_argument("--symbol", default="BTC-USD", help="Crypto Lake / partition symbol")
    ap.add_argument("--coinapi-root", default="data/raw", help="root of any LOCAL CoinAPI parquet tree")
    ap.add_argument("--usable-calendar", default="data/usable_calendar.json",
                    help="usable-calendar JSON for excluded/gap/fill context (§5b)")
    ap.add_argument("--out-dir", default="data/reports", help="report output dir (git-ignored)")
    # quota controls
    ap.add_argument("--quota-gb", type=float, default=QUOTA_GB,
                    help="Crypto Lake monthly download cap GB (default 300)")
    ap.add_argument("--max-auto-gb", type=float, default=DEFAULT_MAX_AUTO_GB,
                    help="auto-allowed request cap GB; larger needs --allow-broad (default 5)")
    ap.add_argument("--headroom-gb", type=float, default=DEFAULT_HEADROOM_GB,
                    help="quota GB to always leave unused (default 10)")
    ap.add_argument("--allow-broad", action="store_true",
                    help="override the auto cap for a deliberate broad pull (still refused if it "
                         "would breach the monthly quota headroom)")
    args = ap.parse_args(argv)
    if args.grid_ms <= 0 or DAY_MS % args.grid_ms:
        # Fail before any Lake session: a non-divisor grid truncates the day (see build_grid).
        ap.error(f"--grid-ms must be positive and divide the {DAY_MS} ms day evenly "
                 f"(got {args.grid_ms})")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    days = resolve_days(args)
    cal = load_usable_calendar(args.usable_calendar)
    if cal is None:
        print(f"NOTE: usable calendar {args.usable_calendar} not found — excluded/gap/fill context "
              "will be null (run ingest/verify_trades_and_calendar.py to produce it).",
              file=sys.stderr)

    # Resolve the replay engine BEFORE any Lake session/load. An explicit --engine native must fail
    # cleanly (nonzero, no vendor I/O) when the extension or a verified tick scale is unavailable.
    engine, price_scale, engine_note = _native.resolve_engine(
        args.engine, exchange=args.exchange, symbol=args.symbol)
    if args.engine == "native" and engine != "native":
        print(f"ERROR: {engine_note}", file=sys.stderr)
        return NATIVE_UNAVAILABLE_EXIT
    if engine_note:
        print(f"NOTE: {engine_note}", file=sys.stderr)
    print(f"Reconstruction engine: {engine}"
          + (f" (native tick scale {price_scale})" if engine == "native" else ""))

    excluded_set = set((cal or {}).get("excluded_days_by_reason", {}))
    to_load = [d for d in days if d.isoformat() not in excluded_set]
    excluded = [d for d in days if d.isoformat() in excluded_set]

    # --no-lake-seed skips the `book` snapshot load (see the per-day loop below), so estimate ONLY the
    # products actually pulled — otherwise a cold-start run is over-estimated by the `book` size and can
    # be wrongly refused at the auto cap / quota headroom.
    products = ("book_delta_v2",) if args.no_lake_seed else LAKE_PRODUCTS
    per_day_gb = sum(LAKE_GB_PER_DAY[p] for p in products)
    est_gb = estimate_lake_gb(len(to_load), products=products)
    print(f"Quality map: {len(days)} day(s) requested — {len(to_load)} to load, "
          f"{len(excluded)} excluded by calendar.")
    print(f"Estimated Crypto Lake download: ~{est_gb:.2f} GB "
          f"({len(to_load)} day(s) × {per_day_gb:.2f} GB/day [{'+'.join(products)}], "
          "conservative upper bound).")

    # Excluded days are pure calendar verdicts — build them WITHOUT a Lake session, so a calendar-only
    # run (every day excluded) works with no AWS keys and never hits the quota gate.
    results: list[dict] = []
    for d in excluded:
        di = d.isoformat()
        # d ∈ excluded ⟹ di ∈ excluded_days_by_reason (so cal is present and the key exists).
        results.append(excluded_result(
            d, cal["excluded_days_by_reason"][di], k=args.k, grid_ms=args.grid_ms,
            coinapi=coinapi_context(cal, d, args.coinapi_root, args.exchange, args.symbol),
            calendar=calendar_context(cal, di)))

    used_gb = used_after = None
    decision = {"ok": True, "reason": "no_days_to_load", "est_gb": est_gb}
    if not to_load:
        print("No days to load (all excluded, or empty set) — writing a calendar-only report; "
              "no Crypto Lake session created.")
    else:
        sess = lake_session()
        try:
            used = lake_used_data(sess)
            used_gb = float(used.get("downloaded_gb", 0.0))
            print(f"Crypto Lake usage BEFORE: {used_gb:.2f} GB / {used.get('timeframe_days')} days "
                  f"(cap {args.quota_gb:.0f} GB; may lag ~60 min).")
        except Exception as e:  # noqa: BLE001 — fail safe: cannot confirm headroom ⇒ refuse below
            print(f"WARNING: could not read lakeapi.used_data ({e!r}); assuming worst-case usage at "
                  "the monthly cap, so the quota gate will REFUSE any non-empty pull this run "
                  "regardless of --allow-broad (cannot confirm headroom). Re-run once used_data is "
                  "readable.", file=sys.stderr)
            used_gb = args.quota_gb

        decision = quota_decision(est_gb=est_gb, used_gb=used_gb, quota_gb=args.quota_gb,
                                  max_auto_gb=args.max_auto_gb, allow_broad=args.allow_broad,
                                  headroom_gb=args.headroom_gb)
        if not decision["ok"]:
            why = ("would breach the monthly quota headroom"
                   if decision["reason"] == "quota_headroom"
                   else f"exceeds the {args.max_auto_gb:.0f} GB auto cap")
            print(f"\nREFUSING Lake load: estimate ~{est_gb:.2f} GB {why} "
                  f"(remaining ~{decision['remaining_gb']:.1f} GB of {args.quota_gb:.0f} GB).\n"
                  "  • Narrow the day set (--days), or\n"
                  "  • pass --allow-broad for a deliberate pull that still fits the monthly quota "
                  "(ensure headroom; used_data lags ~60 min).", file=sys.stderr)
            return QUOTA_REFUSED_EXIT

        for d in to_load:
            di = d.isoformat()
            cctx = coinapi_context(cal, d, args.coinapi_root, args.exchange, args.symbol)
            calctx = calendar_context(cal, di)
            print(f"\nLoading Crypto Lake {args.exchange} {args.symbol} book_delta_v2 for {di} …")
            try:
                lake_df = load_lake_book_delta_v2(sess, d, args.exchange, args.symbol)
            except Exception as e:  # noqa: BLE001
                if _is_no_files(e):
                    # An ABSENT Lake partition (a gap day): lakeapi raises NoFilesFound rather than
                    # returning an empty frame, so this is missing_needs_coinapi, NOT a load failure.
                    print(f"  Lake book_delta_v2 absent for {di} (gap) → missing_needs_coinapi.")
                    results.append(assess_lake_day(pd.DataFrame(), None, day=d, k=args.k,
                                                   grid_ms=args.grid_ms, coinapi=cctx, calendar=calctx,
                                                   engine=engine, price_scale=price_scale))
                else:
                    print(f"  WARNING: Lake book_delta_v2 load failed for {di} ({e!r}) — inconclusive.",
                          file=sys.stderr)
                    results.append(inconclusive_load_failure(d, repr(e), k=args.k,
                                                             grid_ms=args.grid_ms, coinapi=cctx,
                                                             calendar=calctx))
                continue
            print(f"  Lake book_delta_v2 rows: {len(lake_df):,}")

            snaps = None
            if not args.no_lake_seed and len(lake_df):
                try:
                    snaps = load_lake_book_snapshots(sess, d, args.exchange, args.symbol,
                                                     max_levels=args.book_max_levels,
                                                     stride_ms=args.book_stride_ms)
                    print(f"  Lake book seed candidates: {len(snaps):,}")
                except Exception as e:  # noqa: BLE001 — fall back to cold-start (→ inconclusive)
                    print(f"  WARNING: could not load Lake `book` snapshots ({e!r}); cold-start (the "
                          "day will classify inconclusive without a validated seed).", file=sys.stderr)
                    snaps = None

            results.append(assess_lake_day(
                lake_df, snaps, day=d, k=args.k, grid_ms=args.grid_ms, reseed=not args.no_reseed,
                reseed_after_crossed_s=args.reseed_after_crossed_s,
                seed_min_levels=args.seed_min_levels, max_spread_frac=args.seed_max_spread_frac,
                cold_ab=not args.no_cold_ab, coinapi=cctx, calendar=calctx,
                engine=engine, price_scale=price_scale))

        try:
            used_after = float(lake_used_data(sess).get("downloaded_gb", 0.0))
        except Exception:  # noqa: BLE001
            pass

    meta = {
        "k": int(args.k), "grid_ms": int(args.grid_ms),
        "exchange": args.exchange, "symbol": args.symbol,
        "engine": engine, "engine_requested": args.engine,
        "engine_price_scale": price_scale,
        "thresholds": THRESHOLDS.as_dict(),
        "policy": {"reseed": not args.no_reseed, "reseed_after_crossed_s": args.reseed_after_crossed_s,
                   "seed_min_levels": args.seed_min_levels, "no_lake_seed": args.no_lake_seed,
                   "cold_ab": not args.no_cold_ab},
        "lake_gb_per_day": LAKE_GB_PER_DAY,
        "quota": {**decision, "used_gb_before": used_gb, "used_gb_after": used_after},
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "VALIDATION quality map (docs/data.md §5a-Recon / §10). The CoinAPI backfill gate "
                "stays LOCKED until the multi-day quality map passes; this tool does not unlock it.",
    }
    report = build_report(results, meta=meta)
    out_path = os.path.join(args.out_dir, "coinbase_quality_map.json")
    write_report(report, out_path)
    print_summary(report)
    print(f"\n  wrote {out_path}")
    if used_after is not None:
        print(f"  Crypto Lake usage AFTER: {used_after:.2f} GB (Δ ~{used_after - used_gb:+.2f} GB; "
              "may lag ~60 min).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
