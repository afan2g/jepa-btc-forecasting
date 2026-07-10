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
from recon.parity import compare_topk, lake_warmup_cutoff
from recon.reseed import (BookSnapshot, ReseedPolicy, book_snapshot, classify_snapshot,
                          reconstruct_lake_l2_at_samples_seeded)

# ---------------------------------------------------------------------- preregistration
# Preregistered BEFORE any real-data arm was run (issue #54 Phase 2). Mirrored verbatim
# in experiments/preregistration_54.json (pinned equal by test) so the bars cannot drift
# silently after results are seen. Anchors, quoted from docs/data.md:
#   * day-quality bars = the §5a-QualityMap `lake_usable` classification thresholds
#     (crossed <=1%, missing <=2%, thin <=10%);
#   * parity bars separate the clean reference class (2025-06-01: median $0.00,
#     p95/p99/max $0.48/$4.35/$66.59, corr 0.99999778, labels 0.951/0.983/0.995) from
#     the measured crossed-seed cross-validation FAILURES (p99 $25.40/$157.00,
#     >$50-spike fractions 2.2e-3/2.8e-2, 2s labels 0.832/0.917);
#   * a >$50-spike-fraction failure may be overridden ONLY by the PR-#28-style
#     volatility attribution (spikes concentrated in hours the REFERENCE itself moved),
#     documented spike-by-spike in the report — never silently.
PREREGISTERED = {
    "thresholds": {
        "day_quality": {"crossed_rate_max": 0.01,
                        "missing_book_fraction_max": 0.02,
                        "thin_depth_fraction_max": 0.10,
                        "crossed_duration_s_max": 900.0},
        "parity": {"mid_median_usd_max": 0.01,
                   "mid_signed_mean_abs_usd_max": 0.50,
                   "mid_corr_min": 0.9995,
                   "mid_p95_usd_max": 10.0,
                   "mid_p99_usd_max": 35.0,
                   "spike_gt50_fraction_max": 0.001,
                   "label_agreement_min": {"2": 0.92, "10": 0.96, "60": 0.985}},
        "clean_control_non_regression": {"crossed_rate_delta_max": 0.001,
                                         "mid_p99_delta_usd_max": 1.0,
                                         "mid_corr_delta_max": 0.0001,
                                         "label2_delta_max": 0.005},
        "economics": {"max_fraction_of_full_day_cost": 0.25},
    },
    "fixture_days": {
        "clean_control": ["2025-06-01"],
        "crossed_seed_mild": ["2024-12-04"],
        "crossed_seed_severe": ["2026-04-01"],
        "emulated_degradations": ["no_lake_book_snapshots", "leading_gap_seam",
                                  "sparse_deltas"],
    },
}

# Vendor billing facts used by the cost projection. Every number carries its source in
# BILLING_SOURCES; the flat-file rates match the #33 manifest cost model
# (scripts/review_coinbase_backfill_manifest.py BOOK_USD_PER_GB/TRADES_USD_PER_GB) and
# live 2026-06/07 usage, but their public pages were bot-gated on 2026-07-10 — treat as
# archived-capture facts pending re-verification (see BILLING_SOURCES status fields).
BILLING_FACTS = {
    "book_usd_per_gb": 1.0,
    "trades_usd_per_gb": 3.0,
    "requests_usd_per_1000": 10.0,
    "rest_usd_per_credit_first_1k_per_day": 5.26 / 1000.0,
    "rest_credit_per_100_data_items": 1,
    "rest_date_bounded_credit_cap": 10,
    "rest_history_max_levels": 20,
}
BILLING_SOURCES = [
    {"fact": "rest_credit_per_100_data_items / default limit=100 = 1 request",
     "url": "https://docs.coinapi.io/market-data/api-limits-and-billing-metrics",
     "accessed": "2026-07-10", "status": "confirmed_live_via_llms_full_txt"},
    {"fact": "rest_usd_per_credit_first_1k_per_day ($5.26 first 1,000/day, $2.63 next)",
     "url": "https://www.coinapi.io/llms-full.txt",
     "accessed": "2026-07-10", "status": "confirmed_live"},
    {"fact": "rest history endpoint is L2, max 20 levels",
     "url": "https://docs.coinapi.io/market-data/rest-api/order-book/historical-data",
     "accessed": "2026-07-10", "status": "confirmed_live_via_search_index"},
    {"fact": "flat files $1/GB limit book, $3/GB trades, $10 per 1,000 GET/LIST/HEAD",
     "url": "https://docs.coinapi.io/flat-files-api/billing",
     "accessed": "2026-07-10",
     "status": "archived_capture_2025_02_06; live page bot-gated; matches repo-measured "
               "billing (docs/data.md §2.2) and the #33 manifest cost model"},
    {"fact": "limitbook_snapshot_X product exists (top-X levels, 1 s interval, "
             "'Limit Book Data' tier)",
     "url": "https://www.coinapi.io/products/flat-files/pricing",
     "accessed": "2026-07-10", "status": "archived_capture_2025_10_11; live page "
                                         "bot-gated; availability for COINBASE unknown"},
]


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


def evaluate_arm_parity(arm_frame: pd.DataFrame, arm_meta: dict,
                        reference_frame: pd.DataFrame, *, k: int, grid_s: float,
                        injection_guard_s: float | None = None,
                        horizons_s=(2, 10, 60), band_bps: float = 0.0) -> dict:
    """Compare a seeded-arm frame against the full-day CoinAPI reference, with the SAME
    exclusion semantics as the production parity gate (`run_parity_core`):

      * `since_ts` — warm-up cutoff clamped to the accepted seed's ts (pre-seed samples
        are cold-start warm-up, not the arm's behavior);
      * residual crossed arm samples (awaiting a reseed) are excluded point-wise and
        counted, only when a seed was actually accepted;
      * `parity_guarded` — the SAME comparison additionally masking
        `injection_guard_s` after every applied snapshot (seed + reseeds). Because the
        reference and the emulated snapshots come from the SAME vendor file, samples
        right after an injection agree trivially; the guarded variant shows how much of
        the parity is genuinely carried by the Lake deltas.
    """
    seed_accepted = bool(arm_meta.get("seed_accepted"))
    seed_ts = arm_meta.get("seed_ts")
    cutoff = lake_warmup_cutoff(arm_frame)
    if seed_accepted and seed_ts is not None:
        cutoff = int(seed_ts) if cutoff is None else max(int(cutoff), int(seed_ts))
    reseed_enabled = bool(arm_meta.get("policy", {}).get("enabled", False))
    excluded = (set(arm_meta.get("crossed_sample_ts", []))
                if (seed_accepted and reseed_enabled) else set())

    parity = compare_topk(arm_frame, reference_frame, k=k, grid_s=grid_s,
                          horizons_s=horizons_s, band_bps=band_bps,
                          since_ts=cutoff, exclude_ts=excluded)

    guard: dict = {"guard_s": None, "n_guard_excluded": 0}
    parity_guarded = None
    if injection_guard_s is not None:
        guard_ns = int(injection_guard_s * 1e9)
        inj_ts = [int(t) for t in ([seed_ts] if (seed_accepted and seed_ts is not None)
                                   else [])] + [int(t) for t in
                                                arm_meta.get("reseed_ts", [])]
        ts = arm_frame["sample_ts"].astype("int64").to_numpy()
        in_guard = np.zeros(len(ts), dtype=bool)
        for t0 in inj_ts:
            in_guard |= (ts >= t0) & (ts <= t0 + guard_ns)
        guard_ts = set(int(t) for t in ts[in_guard])
        guard = {"guard_s": float(injection_guard_s),
                 "n_guard_excluded": len(guard_ts - excluded),
                 "injection_ts": inj_ts}
        parity_guarded = compare_topk(arm_frame, reference_frame, k=k, grid_s=grid_s,
                                      horizons_s=horizons_s, band_bps=band_bps,
                                      since_ts=cutoff, exclude_ts=excluded | guard_ts)

    rep = {
        "day_quality": {
            "crossed_rate": arm_meta.get("crossed_rate"),
            "crossed_samples": arm_meta.get("crossed_samples"),
            "missing_book_fraction": arm_meta.get("missing_book_fraction"),
            "thin_depth_fraction": arm_meta.get("thin_depth_fraction"),
            "crossed_duration_s": arm_meta.get("crossed_duration_s"),
        },
        "since_ts": (int(cutoff) if cutoff is not None else None),
        "excluded_crossed_ts": sorted(excluded)[:100],
        "n_excluded_crossed": len(excluded),
        "injection_guard": guard,
        "parity": parity,
        "parity_guarded": parity_guarded,
    }
    rep["report_hash"] = hash_obj(_json_safe(rep), exclude_keys=("report_hash",))
    return rep


def _json_safe(obj):
    """Strict-JSON coercion (non-finite floats -> None, numpy scalars -> python)."""
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


def evaluate_preregistered(*, day_quality: dict, parity: dict) -> dict:
    """Score one arm's metrics against the PREREGISTERED thresholds.

    Returns `{"pass": bool, "failed": [criterion, ...], "checked": {...}}`. A missing
    metric FAILS its criterion (fail-closed): an arm that could not measure a bar has
    not passed it. Clean-control non-regression and economics are day-level checks
    applied by the runner, not here.
    """
    th = PREREGISTERED["thresholds"]
    failed: list[str] = []
    checked: dict = {}

    def check(name: str, value, ok) -> None:
        checked[name] = {"value": value, "ok": bool(ok)}
        if not ok:
            failed.append(name)

    dq = th["day_quality"]
    for key, bar in (("crossed_rate", dq["crossed_rate_max"]),
                     ("missing_book_fraction", dq["missing_book_fraction_max"]),
                     ("thin_depth_fraction", dq["thin_depth_fraction_max"]),
                     ("crossed_duration_s", dq["crossed_duration_s_max"])):
        v = day_quality.get(key)
        check(f"day_quality.{key}", v, v is not None and v <= bar)

    pa = th["parity"]
    md = parity.get("mid_diff", {})
    check("parity.mid_median", md.get("median"),
          md.get("median") is not None and md["median"] <= pa["mid_median_usd_max"])
    sm = md.get("signed_mean")
    check("parity.mid_signed_mean", sm,
          sm is not None and abs(sm) <= pa["mid_signed_mean_abs_usd_max"])
    check("parity.mid_corr", md.get("corr"),
          md.get("corr") is not None and md["corr"] >= pa["mid_corr_min"])
    check("parity.mid_p95", md.get("p95"),
          md.get("p95") is not None and md["p95"] <= pa["mid_p95_usd_max"])
    check("parity.mid_p99", md.get("p99"),
          md.get("p99") is not None and md["p99"] <= pa["mid_p99_usd_max"])
    sf = parity.get("spike_fraction", {}).get(">50")
    check("parity.spike_gt50_fraction", sf,
          sf is not None and sf <= pa["spike_gt50_fraction_max"])
    for h, bar in pa["label_agreement_min"].items():
        ag = parity.get("label_agreement", {}).get(h, {}).get("agreement")
        check(f"parity.label_agreement.{h}", ag, ag is not None and ag >= bar)

    return {"pass": not failed, "failed": failed, "checked": checked,
            "thresholds": th}


def project_strategy_costs(*, full_day_book_gb: float,
                           on_demand_requests: int | None = None,
                           stream_stats: dict | None = None) -> dict:
    """Project per-day vendor cost for each bootstrap strategy against the full-day
    fill baseline, from BILLING_FACTS only (no vendor calls). Unknown billing
    granularity is carried as an explicit low/high BAND plus an assumptions list —
    never resolved optimistically.
    """
    facts = BILLING_FACTS
    get_usd = facts["requests_usd_per_1000"] / 1000.0
    credit_usd = facts["rest_usd_per_credit_first_1k_per_day"]
    full_usd = full_day_book_gb * facts["book_usd_per_gb"] + get_usd
    out: dict = {
        "billing_facts": dict(facts),
        "billing_sources": list(BILLING_SOURCES),
        "full_day_fill": {
            "gb": float(full_day_book_gb), "usd": full_usd,
            "assumptions": ["whole daily limitbook_full object at $1/GB + 1 GET",
                            "no tiered discount applied (unconfirmed)"]},
    }

    def band(low_usd: float, high_usd: float) -> dict:
        return {"low": low_usd, "high": high_usd}

    def saving(usd_band: dict) -> dict:
        # low saving uses the HIGH cost estimate (conservative), and vice versa.
        return {"low": 1.0 - usd_band["high"] / full_usd,
                "high": 1.0 - usd_band["low"] / full_usd}

    if on_demand_requests is not None:
        n = int(on_demand_requests)
        credits = band(n * 1.0, n * float(facts["rest_date_bounded_credit_cap"]))
        usd = band(credits["low"] * credit_usd, credits["high"] * credit_usd)
        out["rest_on_demand"] = {
            "n_requests": n,
            "credits_band": credits,
            "usd_band": usd,
            "saving_vs_full_day": saving(usd),
            "assumptions": [
                "1 credit minimum per request (confirmed billing rule); high bound = "
                "the documented 10-credit cap on date-bounded queries because the "
                "'data item' unit for an order-book response is UNDOCUMENTED",
                "REST /history is L2 max 20 levels; intraday availability of "
                "historical snapshots is UNVERIFIED (a daily-00:00-only reading "
                "exists) — an on-demand intraday request may be unserviceable",
                "first-1k/day credit pricing ($5.26/1k)",
            ]}

    if stream_stats is not None:
        rows = int(stream_stats["n_changed"])
        levels = int(stream_stats["max_levels"])
        # CSV row estimate: 2 ISO-8601 timestamps (~56 B) + 4*levels numeric fields at
        # ~12 B each incl. separators; gzip ratio band for repetitive numeric CSV.
        row_bytes = 56 + 48 * levels
        raw_gb = rows * row_bytes / 1e9
        gz = band(raw_gb * 0.10, raw_gb * 0.35)
        usd = band(gz["low"] * facts["book_usd_per_gb"] + get_usd,
                   gz["high"] * facts["book_usd_per_gb"] + get_usd)
        out["flatfile_snapshot_stream"] = {
            "rows": rows, "levels": levels,
            "raw_gb_estimate": raw_gb, "gz_gb_band": gz,
            "usd_band": usd,
            "saving_vs_full_day": saving(usd),
            "assumptions": [
                f"row bytes ~= 56 + 48*levels = {row_bytes} (uncompressed CSV estimate)",
                "gzip ratio band [0.10, 0.35] for repetitive numeric CSV",
                "limitbook_snapshot_X availability/size for COINBASE is UNVERIFIED "
                "(needs one S3 LIST) — billed at the Limit Book $1/GB tier + 1 GET",
                "rows = changed top-X seconds measured from the emulated stream",
            ]}
    return out


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
