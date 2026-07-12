"""Import-safe adapter for the optional native seed/reseed replay core (`recon_native`).

This is the Python seam for the Rust throughput engine (docs/data.md §5a-Recon; plan
`docs/superpowers/plans/2026-07-01-native-recon-engine.md`). Importing this module NEVER requires
the native extension — `native_available()` reports whether it is present, and the plain Python
reconstruction (`recon.reseed.reconstruct_lake_l2_at_samples_seeded`) remains the correctness oracle
and the default for tests/dev.

Division of labour (deliberate, plan §"Native API"):
  * **Python owns** column resolution, side decoding, and snapshot parse/thin/**classify** — it reuses
    `recon.reseed.classify_snapshot` so the non-obvious validation precedence (one_sided → bad_values →
    thin_depth → unsorted → crossed → wide_spread) and NaN handling live in ONE place. It hands Rust
    compact arrays with a precomputed `reason_code` + `is_valid` per snapshot.
  * **Rust owns** only the delta hot loop: stable `(ts,seq)` sort, delta/snapshot merge, the seed/reseed
    state machine (gated on `is_valid`), the sample loop, and crossed/missing/thin metrics — plus the
    compact per-sample `coverage` runs (invalid-run index pairs + presence bounds, plan-doc Task 3)
    that let the quality map plan partial-day CoinAPI fills without materializing the top-K frame.

The returned `(frame, meta)` is schema-identical to
`recon.reseed.reconstruct_lake_l2_at_samples_seeded` (including `meta["coverage"]`), so callers can
swap engines transparently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from recon.ingest import _pick, _require_populated
from recon.reconstruct import _decode_sides
from recon.reseed import BookSnapshot, ReseedPolicy, classify_snapshot, require_finite_deltas

# Snapshot reason-code enum — the ORDER is load-bearing: the u8 index passed to Rust and mapped back
# to a string here must agree with the Rust side (`recon_native.N_REASONS == len(REASON_CODES)`).
REASON_CODES: tuple[str, ...] = (
    "ok", "one_sided", "bad_values", "thin_depth", "unsorted", "crossed", "wide_spread",
)
_REASON_INDEX = {r: i for i, r in enumerate(REASON_CODES)}
# Sentinel matching Rust `NO_SNAPSHOTS` — seed_reason before any snapshot is seen ("no_snapshots").
_NO_SNAPSHOTS = 255
# Result-dict ABI version; must equal Rust `META_ABI`. Bump in lockstep whenever the fields
# `reconstruct_seeded` returns change. v2: per-sample coverage (`present_first_idx`/
# `present_last_idx`/`invalid_runs_idx` — plan-doc 2026-07-02 Task 3); v1 builds lack the attribute.
_META_ABI = 2


def _validate_native(mod) -> None:
    """Reject a stale/incompatible `recon_native` at import time. A build on `PYTHONPATH` that is
    missing the entry point or — worse — carries a DIFFERENT reason-code enum (e.g. an old
    `maturin develop`) would otherwise pass the `--engine native` pre-load guard and then either fail
    AFTER a Lake load or SILENTLY mis-reconstruct with misaligned seed/reseed reason codes. Verifying
    the ABI surface here makes such a module fall into the import `except` (→ treated as unavailable
    with a precise reason), preserving the "fail before any Lake load" contract."""
    missing = [a for a in ("reconstruct_seeded", "N_REASONS", "NO_SNAPSHOTS", "META_ABI")
               if not hasattr(mod, a)]
    if missing:
        raise ImportError(f"recon_native is missing required attributes {missing} "
                          "(stale/incompatible build — rebuild: maturin develop --release "
                          "-m native/recon_native/Cargo.toml)")
    if mod.N_REASONS != len(REASON_CODES):
        raise ImportError(f"recon_native.N_REASONS={mod.N_REASONS} != len(REASON_CODES)="
                          f"{len(REASON_CODES)} — reason-code enum mismatch; rebuild the extension")
    if mod.NO_SNAPSHOTS != _NO_SNAPSHOTS:
        raise ImportError(f"recon_native.NO_SNAPSHOTS={mod.NO_SNAPSHOTS} != {_NO_SNAPSHOTS} — "
                          "stale/incompatible build; rebuild the extension")
    if mod.META_ABI != _META_ABI:
        raise ImportError(f"recon_native.META_ABI={mod.META_ABI} != {_META_ABI} — result-dict "
                          "contract mismatch (coverage metrics); rebuild the extension")


try:  # the extension is optional — never fail import if it is not built (or is stale/incompatible)
    import recon_native as _rn

    _validate_native(_rn)
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 — absent, ImportError, or an ABI/version mismatch, must not crash import
    _rn = None
    _IMPORT_ERROR = exc

# Verified (exchange, symbol) -> price_scale, where native tick = round(price * price_scale). A symbol
# is only eligible for native mode if its true tick size is KNOWN and every price is an exact multiple
# of it (so tick ordering == float ordering and same-ts crossing decisions cannot drift from Python).
# Coinbase BTC-USD trades on a $0.01 quote increment => scale 100 (integer cents). The two Binance
# BTC-USDT instruments were measured by the #64 tick-scale step (issue #71): ZERO off-tick prices
# across book_delta_v2/trades/book on Lake day 2026-04-01 — every feed this tick contract governs
# (only delta and book-seed prices are tick-keyed by the replay; passthrough tables such as the
# perp liquidations prices are never tick-keyed and were not measured) — perp $0.10
# tick => 10, spot $0.01 tick => 100 (report data/reports/binance_source_quality/tick_scale.json,
# report_hash d5025c58aa48fb6b23d26f8f26cf270c42b4be33e567644f359fafd171a1d7f0, prereg commit
# 60a2b745). Extend ONLY after a product's tick size is verified the same way; unknown symbols must
# fall back to Python (plan §"Rust Data Structures").
_TICK_SCALE: dict[tuple[str, str], int] = {
    ("COINBASE", "BTC-USD"): 100,
    ("BINANCE_FUTURES", "BTC-USDT-PERP"): 10,
    ("BINANCE", "BTC-USDT"): 100,
}


def native_available() -> bool:
    """True iff the `recon_native` extension imported AND passed the ABI capability check
    (`_validate_native`) — a stale/incompatible build reports False, with the reason in
    `native_import_error()`."""
    return _rn is not None


def native_import_error() -> Exception | None:
    """The exception raised while importing `recon_native`, or None if it imported (or was absent
    without error). Useful for a precise `--engine native` failure message."""
    return _IMPORT_ERROR


def tick_scale_for(exchange: str, symbol: str) -> int | None:
    """Verified native price scale for `(exchange, symbol)`, or None if the symbol has no verified
    tick contract (=> not eligible for native mode)."""
    return _TICK_SCALE.get((str(exchange).upper(), str(symbol).upper()))


def resolve_engine(requested: str, *, exchange: str, symbol: str) -> tuple[str, int | None, str | None]:
    """Resolve the effective Lake reconstruction engine BEFORE any vendor load (plan §"Integration
    Points"). Pure — does no I/O.

    Returns `(engine, price_scale, note)` where `engine` is `"python"` or `"native"`:
      * `requested="python"` → always `("python", None, None)`.
      * `requested="native"` → `("native", scale, None)` only if the extension is importable AND the
        symbol has a verified tick scale; otherwise `("python", None, <reason>)` and the CALLER MUST
        abort (an explicit native request must never silently fall back — plan Review Checklist).
      * `requested="auto"` → native when available+verified, else `("python", None, <fallback note>)`.
    """
    requested = str(requested)
    if requested == "python":
        return "python", None, None
    scale = tick_scale_for(exchange, symbol)
    avail = native_available()
    if requested == "native":
        if not avail:
            return "python", None, (
                f"native engine requested but the recon_native extension is not importable "
                f"({native_import_error()!r}); build it with "
                f"`maturin develop --release -m native/recon_native/Cargo.toml`")
        if scale is None:
            return "python", None, (
                f"native engine requested but no verified tick scale for {exchange} {symbol} "
                f"(known: {sorted('/'.join(k) for k in _TICK_SCALE)})")
        return "native", scale, None
    # auto
    if avail and scale is not None:
        return "native", scale, None
    reason = ("recon_native not importable" if not avail
              else f"no verified tick scale for {exchange} {symbol}")
    return "python", None, f"auto: using Python engine ({reason})"


def _snapshots_to_arrays(snapshots: list[BookSnapshot], policy: ReseedPolicy) -> dict:
    """Classify each snapshot in Python (the single source of validation precedence) and flatten into
    the compact native input contract: per-snapshot `ts`/`reason`/`is_valid` + CSR-style concatenated
    bid/ask price/size arrays with per-snapshot counts. Rust re-sorts snapshots by ts (stable), so
    input order is irrelevant here."""
    n = len(snapshots)
    ts = np.empty(n, dtype=np.int64)
    reason = np.empty(n, dtype=np.uint8)
    is_valid = np.empty(n, dtype=bool)
    bid_n = np.empty(n, dtype=np.int64)
    ask_n = np.empty(n, dtype=np.int64)
    bid_px: list[float] = []
    bid_sz: list[float] = []
    ask_px: list[float] = []
    ask_sz: list[float] = []
    for i, s in enumerate(snapshots):
        r = classify_snapshot(s, min_levels_per_side=policy.min_levels_per_side,
                              max_spread_frac=policy.max_spread_frac)
        ts[i] = s.ts
        reason[i] = _REASON_INDEX[r]
        is_valid[i] = (r == "ok")
        bid_n[i] = len(s.bids)
        ask_n[i] = len(s.asks)
        for p, sz in s.bids:
            bid_px.append(p)
            bid_sz.append(sz)
        for p, sz in s.asks:
            ask_px.append(p)
            ask_sz.append(sz)
    return {
        "ts": ts, "reason": reason, "is_valid": is_valid, "bid_n": bid_n, "ask_n": ask_n,
        "bid_px": np.asarray(bid_px, dtype=np.float64), "bid_sz": np.asarray(bid_sz, dtype=np.float64),
        "ask_px": np.asarray(ask_px, dtype=np.float64), "ask_sz": np.asarray(ask_sz, dtype=np.float64),
    }


def _c_i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def _c_f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def reconstruct_lake_l2_at_samples_seeded_native(
        df: pd.DataFrame, sample_ts, *, k: int, engine_time_col: str,
        price_scale: int,
        snapshots: list[BookSnapshot] | None = None,
        policy: ReseedPolicy | None = None,
        frame_out: bool = True) -> tuple[pd.DataFrame | None, dict]:
    """Native counterpart of `recon.reseed.reconstruct_lake_l2_at_samples_seeded` — identical
    `(frame, meta)` schema. `price_scale` is the verified native tick multiplier (see
    `tick_scale_for`); the caller resolves it via `resolve_engine`.

    Raises `RuntimeError` if the extension is unavailable — callers select the engine up front (before
    any Lake load) so an explicit `--engine native` never reaches here without the extension."""
    if _rn is None:
        raise RuntimeError(
            f"recon_native extension not available ({_IMPORT_ERROR!r}); build it with "
            f"`maturin develop --release -m native/recon_native/Cargo.toml`")
    policy = policy or ReseedPolicy()
    _require_populated(df, engine_time_col)
    seq_col = _pick(df, ("sequence_number", "seq"), field="delta sequence")
    side_col = _pick(df, ("side_is_bid", "side", "is_bid"), field="delta side")
    size_col = _pick(df, ("size", "amount"), field="delta size")

    ts = _c_i64(df[engine_time_col].astype("int64").to_numpy())
    seq = _c_i64(df[seq_col].astype("int64").to_numpy())
    price = _c_f64(df["price"].astype("float64").to_numpy())
    size = _c_f64(df[size_col].astype("float64").to_numpy())
    # Same finite-values bar as the Python engine: NaN/inf book keys are engine-divergent (NaN
    # casts to tick 0 here vs a poisoned float key in Python) — fail fast, identically, on both.
    require_finite_deltas(price, size)
    # Side decode + validation stays in Python (raises on unknown encodings) — Rust takes a plain bool.
    sides = _decode_sides(df[side_col].to_numpy())
    side_is_bid = np.ascontiguousarray(sides == "bid", dtype=bool)
    sample_ts_arr = _c_i64(np.asarray(sample_ts))

    snaps = list(snapshots or [])
    S = _snapshots_to_arrays(snaps, policy)

    res = _rn.reconstruct_seeded(
        ts, seq, side_is_bid, price, size, sample_ts_arr,
        int(k), float(price_scale), bool(frame_out),
        S["ts"], S["bid_px"], S["bid_sz"], S["bid_n"], S["ask_px"], S["ask_sz"], S["ask_n"],
        S["reason"], S["is_valid"],
        bool(policy.enabled), int(policy.reseed_after_crossed_ns),
    )
    return _assemble(res, sample_ts_arr, int(k), policy, bool(frame_out))


def _coverage_from_result(res: dict) -> dict:
    """The `meta["coverage"]` block from the native result — JSON-safe ints, identical shape to the
    Python replay's block (`recon.reseed._replay_seeded`): maximal half-open `[i0, i1)` sample-INDEX
    invalid runs (the shared `valid_mask_from_frame` predicate at min_levels_per_side=1) + the
    first/last present-sample indices behind `lake_present_*`. Index pairs, not timestamps — the
    replay does not know the grid step; the quality map converts against its own grid."""
    pfi, pli = res["present_first_idx"], res["present_last_idx"]
    runs = [[int(a), int(b)] for a, b in res["invalid_runs_idx"]]
    return {"present_first_idx": (None if pfi is None else int(pfi)),
            "present_last_idx": (None if pli is None else int(pli)),
            "n_invalid_runs": len(runs), "invalid_runs_idx": runs}


def _assemble(res: dict, sample_ts_arr: np.ndarray, k: int, policy: ReseedPolicy,
              frame_out: bool) -> tuple[pd.DataFrame | None, dict]:
    """Assemble the native result dict into the Python-compatible `(frame, meta)`."""
    n = int(res["n_samples"])
    counts = res["reason_counts"]
    # Counter-equivalent: only non-zero reasons (mirrors `dict(Counter(...))`).
    snapshot_reason_codes = {REASON_CODES[i]: int(c) for i, c in enumerate(counts) if c}
    src = int(res["seed_reason_code"])
    seed_reason = "no_snapshots" if src == _NO_SNAPSHOTS else REASON_CODES[src]
    crossed = int(res["crossed_samples"])
    missing = int(res["missing_book_samples"])
    thin = int(res["thin_depth_samples"])
    cd_ns = int(res["crossed_duration_ns"])

    meta = {
        "seed_accepted": bool(res["seed_accepted"]),
        "seed_ts": (int(res["seed_ts"]) if res["seed_ts"] is not None else None),
        "seed_reason": seed_reason,
        "reseed_count": int(res["reseed_count"]),
        "reseed_ts": [int(x) for x in res["reseed_ts"]],
        "reseed_blocked_invalid_snapshot": int(res["reseed_blocked"]),
        "snapshot_reason_codes": snapshot_reason_codes,
        "n_samples": n,
        "crossed_samples": crossed,
        "crossed_rate": (float(crossed / n) if n else 0.0),
        "crossed_sample_ts": [int(x) for x in res["crossed_sample_ts"]],
        "excluded_samples": crossed,
        "crossed_duration_ns": cd_ns,
        "crossed_duration_s": float(cd_ns / 1e9),
        "missing_book_samples": missing,
        "missing_book_fraction": (float(missing / n) if n else 0.0),
        "thin_depth_samples": thin,
        "thin_depth_fraction": (float(thin / n) if n else 0.0),
        "coverage": _coverage_from_result(res),
        "policy": policy.as_dict(),
    }

    frame = None
    if frame_out:
        # Column order + float64/int64 dtypes are chosen to be byte-identical to Python's
        # `pd.DataFrame(list-of-snapshot-dicts)` (mid, microprice, then per-level, then sample_ts).
        bid_px = np.asarray(res["bid_px"], dtype=np.float64).reshape(n, k)
        bid_sz = np.asarray(res["bid_sz"], dtype=np.float64).reshape(n, k)
        ask_px = np.asarray(res["ask_px"], dtype=np.float64).reshape(n, k)
        ask_sz = np.asarray(res["ask_sz"], dtype=np.float64).reshape(n, k)
        data: dict[str, np.ndarray] = {
            "mid": np.asarray(res["mid"], dtype=np.float64),
            "microprice": np.asarray(res["microprice"], dtype=np.float64),
        }
        for i in range(k):
            data[f"bid_{i}_price"] = bid_px[:, i]
            data[f"bid_{i}_size"] = bid_sz[:, i]
            data[f"ask_{i}_price"] = ask_px[:, i]
            data[f"ask_{i}_size"] = ask_sz[:, i]
        data["sample_ts"] = np.asarray(sample_ts_arr, dtype=np.int64)
        frame = pd.DataFrame(data)
    return frame, meta
