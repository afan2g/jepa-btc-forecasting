"""Top-K L2 vendor-parity comparison (Crypto Lake `book_delta_v2` vs CoinAPI `limitbook_full`).

Pure pandas/numpy — no vendor I/O, JSON-serializable output — so it is unit-testable without
credentials and reusable by `scripts/run_coinbase_parity.py`. Both inputs are top-K frames on
the SAME exchange-time grid (the `recon.reconstruct` / `recon.coinapi` reconstructors emit the
identical `sample_ts, mid, microprice, {bid,ask}_i_{price,size}` schema), so a comparison
reflects genuine vendor divergence, not a sampler mismatch.

Reports (docs/data.md §5a hard gate #1): bid/ask/mid differences, per-level price & size
deltas, per-vendor crossed-book and missing-book rates, the |Δmid| spike distribution (the
known ~$249 second-scale concern — characterized, never assumed to wash out), and directional
label agreement at the project horizons (docs/experiment-plan.md ladder: 2 s / 10 s / 60 s).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPIKE_THRESHOLDS = (1.0, 5.0, 10.0, 50.0, 100.0, 200.0)  # $ |Δmid| buckets
DEFAULT_HORIZONS_S = (2, 10, 60)


def _abs_stats(x) -> dict:
    """Finite-only distribution summary (median/mean/p95/p99/max); None-filled when empty."""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "median": None, "mean": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def frame_quality(frame: pd.DataFrame, *, source_rows: int | None = None) -> dict:
    """Per-vendor data-quality summary of a reconstructed top-K frame: sample count, and the
    crossed-book and missing-book (no top-of-book on a side) rates. Used for the Crypto Lake
    side (the CoinAPI reconstructor already returns its own counters)."""
    n = len(frame)
    bid0, ask0 = frame["bid_0_price"], frame["ask_0_price"]
    both = bid0.notna() & ask0.notna()
    crossed = both & (bid0 >= ask0)
    missing = bid0.isna() | ask0.isna()
    q = {
        "n_samples": int(n),
        "crossed_samples": int(crossed.sum()),
        "crossed_rate": (float(crossed.sum() / n) if n else 0.0),
        "missing_book_samples": int(missing.sum()),
        "missing_book_fraction": (float(missing.sum() / n) if n else 0.0),
    }
    if source_rows is not None:
        q["source_rows"] = int(source_rows)
    return q


def align_grids(lake: pd.DataFrame, capi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join the two top-K frames on `sample_ts` (defensive — they are built on the same
    grid, so this is normally an identity)."""
    L = lake.set_index("sample_ts").sort_index()
    C = capi.set_index("sample_ts").sort_index()
    idx = L.index.intersection(C.index)
    return L.loc[idx], C.loc[idx]


def _signed_labels(mid: pd.Series, step: int, band_bps: float) -> pd.Series:
    """Directional label of the forward mid move over `step` grid points, with a symmetric
    no-trade band of `band_bps` bps: +1 up / -1 down / 0 inside band; NaN where no forward
    point exists."""
    fwd = mid.shift(-step) - mid
    band = band_bps * 1e-4 * mid.abs()
    lab = pd.Series(np.sign(fwd.to_numpy()), index=mid.index)
    lab = lab.where(fwd.abs() > band, 0.0)   # inside band ⇒ flat
    lab = lab.mask(fwd.isna())               # no forward point ⇒ undefined
    return lab


def label_agreement(lake_mid: pd.Series, capi_mid: pd.Series, *, grid_s: float,
                    horizons_s=DEFAULT_HORIZONS_S, band_bps: float = 0.0) -> dict:
    """Per-horizon directional-label agreement between the two vendors' mid series."""
    out: dict = {}
    for h in horizons_s:
        step = max(1, round(h / grid_s))
        ll = _signed_labels(lake_mid, step, band_bps)
        cl = _signed_labels(capi_mid, step, band_bps)
        valid = ll.notna() & cl.notna()
        nv = int(valid.sum())
        rec = {"horizon_s": float(h), "step": int(step), "n": nv}
        if nv == 0:
            rec.update(agreement=None, both_up=0, both_down=0, both_flat=0, disagree=0,
                       fwd_return_corr=None)
        else:
            a, b = ll[valid], cl[valid]
            fl = (lake_mid.shift(-step) - lake_mid)[valid]
            fc = (capi_mid.shift(-step) - capi_mid)[valid]
            rec.update(
                agreement=float((a == b).mean()),
                both_up=int(((a == 1) & (b == 1)).sum()),
                both_down=int(((a == -1) & (b == -1)).sum()),
                both_flat=int(((a == 0) & (b == 0)).sum()),
                disagree=int((a != b).sum()),
                fwd_return_corr=(float(fl.corr(fc)) if nv > 2 and fl.std() and fc.std() else None),
            )
        out[str(h)] = rec
    return out


def lake_warmup_cutoff(frame: pd.DataFrame, *, min_consecutive: int = 3,
                       min_levels_per_side: int = 1) -> int | None:
    """First `sample_ts` at which a COLD-STARTED Lake `book_delta_v2` reconstruction looks
    seeded — best bid & ask present, uncrossed, with ≥ `min_levels_per_side` levels per side —
    SUSTAINED for `min_consecutive` consecutive grid samples. Returns that cutoff ts (int), or
    None if the book never establishes a sustained seed.

    Rationale (docs/data.md §5a-Recon, §5a): `book_delta_v2` has NO per-day snapshot, so a
    day's reconstruction cold-starts from an empty book and its early top-K is warm-up, not
    real liquidity — whereas CoinAPI starts from its full opening SNAPSHOT. Comparing across
    the warm-up window would let Lake warm-up artifacts, not true vendor divergence, drive the
    parity decision. The caller excludes samples before this cutoff from the gate. This is the
    minimal seed-established proxy; a fully validated seed from the Lake `book` snapshot product
    remains the deferred §5a-Recon follow-up, and per-level coverage (below) surfaces residual
    depth gaps past the cutoff."""
    f = frame.set_index("sample_ts").sort_index() if "sample_ts" in frame.columns else frame.sort_index()
    bid_cols = [c for c in f.columns if c.startswith("bid_") and c.endswith("_price")]
    ask_cols = [c for c in f.columns if c.startswith("ask_") and c.endswith("_price")]
    bid_depth = f[bid_cols].notna().sum(axis=1)
    ask_depth = f[ask_cols].notna().sum(axis=1)
    good = ((bid_depth >= min_levels_per_side) & (ask_depth >= min_levels_per_side)
            & f["bid_0_price"].notna() & f["ask_0_price"].notna()
            & (f["bid_0_price"] < f["ask_0_price"]))
    run = 0
    for ts, g in good.items():
        run = run + 1 if g else 0
        if run >= min_consecutive:
            return int(ts)
    return None


def compare_topk(lake: pd.DataFrame, capi: pd.DataFrame, *, k: int, grid_s: float = 1.0,
                 horizons_s=DEFAULT_HORIZONS_S, band_bps: float = 0.0,
                 n_spikes: int = 25, since_ts: int | None = None,
                 exclude_ts=None) -> dict:
    """Compare two top-K L2 frames and return a JSON-serializable parity report dict.

    `since_ts` (optional): restrict the comparison to samples with `sample_ts >= since_ts`,
    used to exclude the Lake cold-start warm-up window (see `lake_warmup_cutoff`) so the gate
    reflects true Lake↔CoinAPI parity, not warm-up. `exclude_ts` (optional): a set of `sample_ts`
    to drop from the comparison — used for residual crossed Lake samples awaiting a reseed
    (docs/data.md §5a-Recon); a crossed mid is not a real vendor mid. `n_grid_full` is ALWAYS the
    true full-grid intersection (before BOTH filters), so the honest grid size is never undercounted;
    `n_excluded_crossed` reports how many `exclude_ts` samples were dropped from the compared window.
    Per-level `coverage` always reports the both-present / one-vendor-only counts so thin or
    one-sided depths are visibly marked rather than silently dropped by the NaN-skipping diff stats."""
    L, C = align_grids(lake, capi)
    n_full = len(L)
    if since_ts is not None:
        L = L[L.index >= since_ts]
        C = C[C.index >= since_ts]
    n_after_since = len(L)
    # Keep the regular (contiguous) post-warm-up grid for TIME-based labels; point-wise stats drop
    # the excluded crossed samples. Compacting the grid before labels would make _signed_labels'
    # positional shift(-step) jump the gaps → wrong physical horizon (Codex P2). For labels we mask
    # the excluded points to NaN on the regular grid instead, so a label whose origin or target is
    # excluded is undefined, not horizon-stretched.
    Lreg, Creg = L, C
    excl_mask = Lreg.index.isin(list(exclude_ts)) if exclude_ts else None
    if excl_mask is not None and excl_mask.any():
        L, C = Lreg[~excl_mask], Creg[~excl_mask]
    n = len(L)
    out: dict = {"n_grid": int(n), "n_grid_full": int(n_full), "k": int(k),
                 "grid_s": float(grid_s),
                 "since_ts": (int(since_ts) if since_ts is not None else None),
                 "n_excluded_crossed": int(n_after_since - n)}
    if n == 0:
        out["error"] = "no overlapping grid points after warm-up + crossed-exclusion"
        return out

    lake_present = L["bid_0_price"].notna() & L["ask_0_price"].notna()
    capi_present = C["bid_0_price"].notna() & C["ask_0_price"].notna()
    both = lake_present & capi_present
    out["missing_book"] = {
        "lake_fraction": float((~lake_present).mean()),
        "capi_fraction": float((~capi_present).mean()),
        "either_fraction": float((~both).mean()),
        "both_present": int(both.sum()),
    }
    out["crossed_rate"] = {
        "lake": float((lake_present & (L["bid_0_price"] >= L["ask_0_price"])).mean()),
        "capi": float((capi_present & (C["bid_0_price"] >= C["ask_0_price"])).mean()),
    }

    dmid = (L["mid"] - C["mid"])[both]
    midref = ((L["mid"] + C["mid"]) / 2.0)[both]
    lm, cm = L["mid"][both], C["mid"][both]
    out["mid_diff"] = {
        **_abs_stats(dmid.abs()),
        "signed_mean": (float(dmid.mean()) if len(dmid) else None),
        "corr": (float(lm.corr(cm)) if int(both.sum()) > 2 and lm.std() > 0 and cm.std() > 0
                 else None),
    }
    out["mid_diff_bps"] = _abs_stats((dmid.abs() / midref.replace(0.0, np.nan)) * 1e4)
    out["best_bid_diff"] = _abs_stats((L["bid_0_price"] - C["bid_0_price"])[both].abs())
    out["best_ask_diff"] = _abs_stats((L["ask_0_price"] - C["ask_0_price"])[both].abs())

    out["per_level"] = {}
    for i in range(k):
        rec: dict = {}
        coverage: dict = {}
        for side in ("bid", "ask"):
            pc, sc = f"{side}_{i}_price", f"{side}_{i}_size"
            lp, cp = L[pc].notna(), C[pc].notna()
            both_lvl = lp & cp
            coverage[side] = {
                "both_present": int(both_lvl.sum()),
                "both_fraction": (float(both_lvl.sum() / n) if n else 0.0),
                "only_lake": int((lp & ~cp).sum()),    # one-sided depth = an uncompared parity miss
                "only_capi": int((cp & ~lp).sum()),
            }
            rec[f"{side}_price"] = _abs_stats((L[pc] - C[pc]).abs())
            rec[f"{side}_size"] = _abs_stats((L[sc] - C[sc]).abs())
        rec["coverage"] = coverage
        out["per_level"][str(i)] = rec

    # |Δmid| spike population — characterized at the bar/label resolution, never assumed away.
    adm = (L["mid"] - C["mid"]).abs()
    finite = adm[np.isfinite(adm)]
    out["spike_counts"] = {f">{t:g}": int((finite > t).sum()) for t in SPIKE_THRESHOLDS}
    out["spike_fraction"] = {
        f">{t:g}": (float((finite > t).mean()) if len(finite) else 0.0) for t in SPIKE_THRESHOLDS
    }
    top = finite.nlargest(n_spikes)
    out["top_spikes"] = [
        {"sample_ts": int(ts), "lake_mid": float(L.loc[ts, "mid"]),
         "capi_mid": float(C.loc[ts, "mid"]), "abs_dmid": float(v)}
        for ts, v in top.items()
    ]

    # Labels on the REGULAR grid (excluded points masked to NaN) so horizons span true physical time.
    if excl_mask is not None and excl_mask.any():
        lake_mid_lbl = Lreg["mid"].mask(excl_mask)
        capi_mid_lbl = Creg["mid"].mask(excl_mask)
    else:
        lake_mid_lbl, capi_mid_lbl = L["mid"], C["mid"]
    out["label_agreement"] = label_agreement(
        lake_mid_lbl, capi_mid_lbl, grid_s=grid_s, horizons_s=horizons_s, band_bps=band_bps
    )
    return out
