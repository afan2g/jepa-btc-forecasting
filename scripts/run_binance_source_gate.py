#!/usr/bin/env python
"""Issue #64 — Binance source-quality gate experiment CLI (offline by default).

Subcommands (all offline, reading only local files, except `fetch`):

  verify-inputs       sha256 / footer-schema / fingerprint / row-count check of the nine
                      2026-04-01 raw units against the preregistered Stage-1 pin.
  tick-scale          preregistered tick measurement over delta/book/trade price columns.
  silence             delta-stream inter-event gap metrics (book_delta_v2).
  replay-conformance  in-process replay x2 (frame-hash determinism; --engine python|native),
                      frozen-book metrics, and (python mode) the native conformance arm.
  stage2-native-run   production Stage-2 unit processing with an explicit native-engine
                      override at the measured scales (2026-07-12 amendment).
  stage2-compare      cross-run/cross-engine comparison of two Stage-2 processed manifests.
  verdict             aggregate the preregistered certified/degraded/inconclusive verdict.
  decide              machine-enforced final_source_decision routing (fail-closed).
  chd-validate        validate one local CryptoHFTData hourly object (no network).
  chd-replay          fail-closed causal replay of local CryptoHFTData hourly objects.
  compare             fixed independent-source comparison of two top-K frames.
  fetch               APPROVAL-GATED bounded download of ONE CryptoHFTData object.
                      Refuses to run without --approved-by. Never called by tests.

Safety: no vendor/network import at module top; the ONLY network code path is inside
`cmd_fetch`, which builds its urllib opener lazily and enforces the preregistered request
bounds (byte cap, attempt cap, timeout, no overwrite). Every published report passes the
April holdout guard (`experiments.binance_source_gate.assert_report_publishable`) before it
is written. Exit codes: 0 pass · 2 setup/args error · 3 fail-closed refusal or failed check.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import binance_source_gate as bsg                             # noqa: E402

SETUP_ERROR_EXIT = 2
FAIL_EXIT = 3

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
PARQUET_MAGIC = b"PAR1"


# ----------------------------------------------------------------------------- shared helpers
def _prereg_commit() -> str | None:
    """Commit SHA that last touched the preregistration artifact (recorded in every report)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-n", "1", "--format=%H", "--",
             "experiments/preregistration_64.json"],
            capture_output=True, text=True, check=True)
        sha = out.stdout.strip()
        return sha or None
    except Exception:                                    # noqa: BLE001 — provenance is best-effort
        return None


def _write_report(out_dir: str, name: str, report: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    report = bsg.finalize_report(report)                 # April guard + report_hash
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"report: {path}")
    return path


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _slim_meta(meta: dict) -> dict:
    """Report-sized replay meta: unbounded per-sample lists dropped/capped (the #54
    _slim_meta convention) so the April guard's series bound holds."""
    m = dict(meta)
    m.pop("crossed_sample_ts", None)
    if isinstance(m.get("reseed_ts"), list):
        m["reseed_ts"] = m["reseed_ts"][:50]
    cov = m.get("coverage")
    if isinstance(cov, dict):
        cov = dict(cov)
        runs = cov.get("invalid_runs_idx")
        if isinstance(runs, list):
            cov["invalid_runs_idx"] = runs[:100]
        m["coverage"] = cov
    counters = m.get("counters")
    if isinstance(counters, dict):
        m["counters"] = dict(counters)
    return m


def _expected_decimals(prereg: dict, exchange: str, symbol: str) -> int:
    tick = prereg["tick_rules"]["expected_tick"][f"{exchange}/{symbol}"]
    d = 0
    while round(tick * 10**d) != tick * 10**d or tick * 10**d != int(tick * 10**d):
        d += 1
        if d > 8:
            raise ValueError(f"cannot derive decimals from tick {tick}")
    return d


def _read_columns(path: str, columns: list[str]):
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as pf:
        return pf.read(columns=columns).to_pandas()


# ----------------------------------------------------------------------------- verify-inputs
def cmd_verify_inputs(args) -> int:
    from ingest import lake_binance as lb
    from ingest.download_lake_binance import _sha256_file, schema_fingerprint
    import pyarrow.parquet as pq

    prereg = bsg.load_preregistration(args.prereg)
    units = prereg["fixture"]["lake"]["units"]
    results = []
    for key in sorted(units):
        u = units[key]
        path = lb.raw_parquet_path(args.raw, u["feed"], u["exchange"], u["symbol"], u["dt"])
        entry = {"unit": key, "path_exists": os.path.exists(path)}
        if not entry["path_exists"]:
            entry["ok"] = False
            results.append(entry)
            continue
        schema = pq.read_schema(path)
        with pq.ParquetFile(path) as pf:
            n_rows = pf.metadata.num_rows
        checks = {
            "sha256": _sha256_file(path) == u["sha256"],
            "schema_fingerprint": schema_fingerprint(schema) == u["schema_fingerprint"],
            "schema_cols": list(schema.names) == list(u["schema_cols"]),
            "rows": n_rows == u["rows"],
            "out_bytes": os.path.getsize(path) == u["out_bytes"],
        }
        entry.update(checks=checks, ok=all(checks.values()))
        results.append(entry)
    ok = all(e["ok"] for e in results) and len(results) == 9
    report = {"step": "verify-inputs", "prereg_commit": _prereg_commit(), "raw_root": args.raw,
              "n_units": len(results), "units": results, "pass": ok}
    _write_report(args.out, "verify_inputs.json", report)
    print(f"verify-inputs: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else FAIL_EXIT


# ----------------------------------------------------------------------------- tick-scale
def cmd_tick_scale(args) -> int:
    import numpy as np
    from ingest import lake_binance as lb

    prereg = bsg.load_preregistration(args.prereg)
    day = prereg["fixture"]["lake"]["day"]
    out = {}
    overall_ok = True
    for key, inst in lb.INSTRUMENTS.items():
        exp_d = _expected_decimals(prereg, inst.exchange, inst.symbol)
        per_feed = {}
        feeds = [("book_delta_v2", ["price"]), ("trades", ["price"])]
        for feed, cols in feeds:
            path = lb.raw_parquet_path(args.raw, feed, inst.exchange, inst.symbol, day)
            df = _read_columns(path, cols)
            per_feed[feed] = bsg.measure_float_price_scale(
                df["price"].to_numpy(dtype="float64"), expected_decimals=exp_d)
        book_path = lb.raw_parquet_path(args.raw, "book", inst.exchange, inst.symbol, day)
        import pyarrow.parquet as pq
        names = [n for n in pq.read_schema(book_path).names
                 if n.endswith("_price")]
        bdf = _read_columns(book_path, names)
        prices = bdf.to_numpy(dtype="float64").ravel()
        prices = prices[np.isfinite(prices)]
        per_feed["book"] = bsg.measure_float_price_scale(prices, expected_decimals=exp_d)
        feed_ok = all(m["ok"] for m in per_feed.values())
        overall_ok &= feed_ok
        measured = [m["measured_decimals"] for m in per_feed.values() if m["ok"]]
        conf_scale = 10 ** max([*measured, exp_d]) if feed_ok else None
        out[key] = {"exchange": inst.exchange, "symbol": inst.symbol,
                    "expected_decimals": exp_d, "per_feed": per_feed,
                    "measured_decimals_max": max(measured) if measured else None,
                    "conformance_scale": conf_scale, "ok": feed_ok}
    report = {"step": "tick-scale", "prereg_commit": _prereg_commit(), "day": day,
              "instruments": out, "pass": overall_ok}
    _write_report(args.out, "tick_scale.json", report)
    print(f"tick-scale: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else FAIL_EXIT


# ----------------------------------------------------------------------------- silence
def cmd_silence(args) -> int:
    from ingest import lake_binance as lb

    prereg = bsg.load_preregistration(args.prereg)
    day = prereg["fixture"]["lake"]["day"]
    cap = float(prereg["thresholds"]["anomaly_caps"]["silence_gap_s_cap"])
    out = {}
    ok = True
    for key, inst in lb.INSTRUMENTS.items():
        path = lb.raw_parquet_path(args.raw, "book_delta_v2", inst.exchange, inst.symbol, day)
        df = _read_columns(path, ["timestamp", "receipt_timestamp"])
        df = lb.canonicalize_time_columns(df)
        col, fallback, cleaned, dropped = lb.resolve_engine_time(df)
        metrics = bsg.silence_metrics(cleaned[0][col].astype("int64").to_numpy())
        capped = metrics["gaps_gt_300s"] is None or metrics["gaps_gt_300s"] > 0
        ok &= not capped
        out[key] = {"engine_time_col": col, "engine_time_fallback": bool(fallback),
                    "dropped_rows": dropped[0], **metrics,
                    "silence_cap_fired": bool(capped), "silence_gap_s_cap": cap}
    report = {"step": "silence", "prereg_commit": _prereg_commit(), "day": day,
              "instruments": out, "pass": ok}
    _write_report(args.out, "silence.json", report)
    print(f"silence: {'PASS' if ok else 'FAIL (cap fired)'}")
    return 0 if ok else FAIL_EXIT


# ----------------------------------------------------------------------------- replay-conformance
def cmd_replay_conformance(args) -> int:
    import datetime as dt
    import importlib.util
    import pandas as pd
    from ingest import lake_binance as lb
    from recon import native as rnative
    from recon.reseed import (ReseedPolicy, reconstruct_lake_l2_at_samples_seeded,
                              snapshots_from_lake_book_df)

    spec = importlib.util.spec_from_file_location(
        "run_binance_recon_for_gate", str(ROOT / "scripts" / "run_binance_recon.py"))
    rbr = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rbr                         # dataclasses resolve __module__
    spec.loader.exec_module(rbr)

    prereg = bsg.load_preregistration(args.prereg)
    day = prereg["fixture"]["lake"]["day"]
    inst = lb.INSTRUMENTS[args.instrument]
    policy = ReseedPolicy(enabled=True, min_levels_per_side=5, reseed_after_crossed_s=2.0,
                          max_spread_frac=None)

    delta_df = _read_columns(
        lb.raw_parquet_path(args.raw, "book_delta_v2", inst.exchange, inst.symbol, day),
        None)
    book_df = _read_columns(
        lb.raw_parquet_path(args.raw, "book", inst.exchange, inst.symbol, day), None)

    frames = [lb.canonicalize_time_columns(delta_df), lb.canonicalize_time_columns(book_df)]
    col, fallback, cleaned, dropped = lb.resolve_engine_time(*frames)
    snaps = snapshots_from_lake_book_df(cleaned[1], engine_time_col=col,
                                        max_levels=rbr.BOOK_MAX_LEVELS,
                                        stride_ns=1000 * rbr.NS_PER_MS)
    pre = rbr.preclassify_snapshots(snaps, policy)
    grid = rbr.build_grid(dt.date.fromisoformat(day), 1000)

    if args.engine == "native":
        # 2026-07-12 amendment: the in-process determinism arm runs the NATIVE engine
        # twice; the full-day cross-engine (oracle) equality is carried by stage2-compare.
        if not rnative.native_available() or not args.scale:
            print("ERROR: --engine native needs the recon_native extension and --scale",
                  file=sys.stderr)
            return SETUP_ERROR_EXIT
        def _run():
            return rnative.reconstruct_lake_l2_at_samples_seeded_native(
                cleaned[0], grid, k=10, engine_time_col=col, price_scale=int(args.scale),
                snapshots=snaps, policy=policy, frame_out=True)
    else:
        def _run():
            return reconstruct_lake_l2_at_samples_seeded(
                cleaned[0], grid, k=10, engine_time_col=col, snapshots=snaps,
                policy=policy, frame_out=True)

    t0 = time.monotonic()
    frame1, meta1 = _run()
    t_py1 = time.monotonic() - t0
    t0 = time.monotonic()
    frame2, meta2 = _run()
    t_py2 = time.monotonic() - t0

    h1, h2 = bsg.frame_replay_hash(frame1), bsg.frame_replay_hash(frame2)
    determinism_ok = (h1 == h2) and (meta1 == meta2)

    frozen = bsg.frozen_metrics(frame1)
    frozen_cap = float(prereg["thresholds"]["anomaly_caps"]["frozen_fraction_max"])
    frozen_fired = frozen["frozen_fraction"] > frozen_cap

    cls, reasons, frac = rbr.classify_replay(meta1, rbr.Thresholds())

    conformance: dict = {"ran": False, "native_available": rnative.native_available(),
                         "price_scale": args.scale, "engine": args.engine}
    if args.engine == "native":
        conformance["note"] = ("cross-engine full-day equality vs the Python-oracle run "
                               "is carried by stage2-compare (2026-07-12 amendment)")
    if args.engine == "python" and rnative.native_available() and args.scale:
        t0 = time.monotonic()
        nat_frame, nat_meta = rnative.reconstruct_lake_l2_at_samples_seeded_native(
            cleaned[0], grid, k=10, engine_time_col=col, price_scale=int(args.scale),
            snapshots=snaps, policy=policy, frame_out=True)
        t_nat = time.monotonic() - t0
        try:
            pd.testing.assert_frame_equal(frame1, nat_frame, check_dtype=True)
            frames_equal = True
            frame_diff = None
        except AssertionError as e:
            frames_equal = False
            frame_diff = str(e)[:500]
        meta_equal = (nat_meta == meta1)
        conformance.update(ran=True, frames_equal=frames_equal, meta_equal=meta_equal,
                           frame_diff=frame_diff, native_secs=round(t_nat, 3),
                           native_frame_hash=bsg.frame_replay_hash(nat_frame))
    conformance_ok = (not conformance["ran"]) or \
        (conformance["frames_equal"] and conformance["meta_equal"])

    if args.frame_out and frame1 is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        os.makedirs(os.path.dirname(args.frame_out) or ".", exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame1, preserve_index=False), args.frame_out,
                       compression="zstd")

    ok = determinism_ok and conformance_ok and not frozen_fired
    report = {
        "step": "replay-conformance", "prereg_commit": _prereg_commit(), "day": day,
        "instrument": args.instrument, "engine": args.engine, "engine_time_col": col,
        "engine_time_fallback": bool(fallback),
        "dropped_rows": {"book_delta_v2": dropped[0], "book": dropped[1]},
        "src_rows": int(len(delta_df)), "book_rows": int(len(book_df)),
        "pre_replay_seed_gate": {k: v for k, v in pre.items()},
        "classification": cls, "reasons": reasons, "seed_source_crossed_frac": frac,
        "replay_meta": _slim_meta(meta1),
        "frame_replay_hash": h1, "frame_replay_hash_run2": h2,
        "harness_determinism_ok": bool(determinism_ok),
        "replay_secs": [round(t_py1, 3), round(t_py2, 3)],
        "frozen": {**frozen, "frozen_fraction_max": frozen_cap,
                   "frozen_cap_fired": bool(frozen_fired)},
        "conformance": conformance, "conformance_ok": bool(conformance_ok),
        "pass": bool(ok),
    }
    _write_report(args.out, f"replay_conformance_{args.instrument}.json", report)
    print(f"replay-conformance[{args.instrument}]: classification={cls} "
          f"determinism={'OK' if determinism_ok else 'FAIL'} "
          f"conformance={'OK' if conformance_ok else ('SKIP' if not conformance['ran'] else 'FAIL')} "
          f"frozen_cap={'FIRED' if frozen_fired else 'ok'}")
    return 0 if ok else FAIL_EXIT


# ----------------------------------------------------------------------------- stage2-native-run
def cmd_stage2_native_run(args) -> int:
    """Stage-2 production unit processing with an EXPLICIT native-engine override at the
    measured tick scales (2026-07-12 amendment): reuses run_binance_recon's RunConfig /
    plan_units / unit_is_done / process_unit VERBATIM — only resolve_engine's registry
    lookup is bypassed, because no Binance tick scale is registered in recon/native.py
    (that registration remains a separate reviewed change). Trust in the native output
    comes from full-day cross-engine equality with the Python-oracle run
    (stage2-compare), which is a hard invalidator when violated."""
    import datetime as dt
    import importlib.util
    from threading import Lock
    from ingest import lake_binance as lb
    from recon import native as rnative
    from recon.reseed import ReseedPolicy

    if not rnative.native_available():
        print(f"ERROR: recon_native unavailable ({rnative.native_import_error()!r})",
              file=sys.stderr)
        return SETUP_ERROR_EXIT
    spec = importlib.util.spec_from_file_location(
        "run_binance_recon_for_native_run", str(ROOT / "scripts" / "run_binance_recon.py"))
    rbr = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rbr
    spec.loader.exec_module(rbr)

    prereg = bsg.load_preregistration(args.prereg)
    day = prereg["fixture"]["lake"]["day"]
    engine_by_key = {"binance-perp": ("native", int(args.perp_scale)),
                     "binance-spot": ("native", int(args.spot_scale))}
    cfg = rbr.RunConfig(
        raw_root=args.raw, out_root=args.out_root, manifest_root=args.out_root,
        overwrite=False, k=10, grid_ms=1000, book_stride_ms=1000,
        policy=ReseedPolicy(enabled=True, min_levels_per_side=5,
                            reseed_after_crossed_s=2.0, max_spread_frac=None),
        thresholds=rbr.Thresholds(), engine_by_key=engine_by_key)
    units = rbr.plan_units(list(engine_by_key), None, [day])
    lb.cleanup_tmp(cfg.out_root)
    state = rbr.processed_state(cfg.manifest_root)
    pending = [u for u in units if not rbr.unit_is_done(u, cfg, state)]

    counts = {"ok": 0, "skip": len(units) - len(pending), "missing": 0,
              "missing_required": 0, "inconclusive": 0, "degraded": 0, "error": 0}
    per_unit = []
    lock = Lock()
    for u in pending:
        res = rbr.process_unit(u, cfg, lock)
        counts[res.status] = counts.get(res.status, 0) + 1
        if res.status == "missing" and rbr.feed_miss_is_fatal(u.feed):
            counts["missing_required"] += 1
        per_unit.append({"output": u.output, "symbol": u.symbol, "status": res.status,
                         "rows": res.rows})
        print(f"{u.day}  {u.output:<14} {res.status:<12} rows={res.rows:,}")

    report = {"step": "stage2-native-run", "prereg_commit": _prereg_commit(), "day": day,
              "engine_by_key": {k: list(v) for k, v in engine_by_key.items()},
              "raw": args.raw, "out": args.out_root, "counts": counts,
              "per_unit": per_unit,
              "note": "production process_unit path, explicit native override; validity "
                      "gated by stage2-compare cross-engine equality"}
    _write_report(args.out, "stage2_native_run.json", report)
    if counts["error"] or counts["missing_required"]:
        return FAIL_EXIT
    return 0


# ----------------------------------------------------------------------------- stage2-compare
def cmd_stage2_compare(args) -> int:
    result = bsg.compare_stage2_manifests(args.run1, args.run2)
    report = {"step": "stage2-compare", "prereg_commit": _prereg_commit(),
              "run1": args.run1, "run2": args.run2, **result,
              "excluded_keys": list(bsg.DETERMINISM_EXCLUDED_KEYS), "pass": result["equal"]}
    _write_report(args.out, "stage2_determinism.json", report)
    print(f"stage2-compare: {'EQUAL' if result['equal'] else 'DIFFERS'} "
          f"({result['n_units']} units)")
    return 0 if result["equal"] else FAIL_EXIT


# ----------------------------------------------------------------------------- verdict
REQUIRED_UNITS = {
    ("BINANCE_FUTURES", "BTC-USDT-PERP"): {
        "topk_l2": "certified", "trades": "ok", "funding": "ok", "open_interest": "ok",
        "liquidations": "ok_or_missing_sparse"},
    ("BINANCE", "BTC-USDT"): {"topk_l2": "certified", "trades": "ok"},
}
REQUIRED_REPLAY_INSTRUMENTS = ("binance-perp", "binance-spot")

# The reconstruction contract a topk_l2 manifest record must have been produced under for
# its classification to count (preregistration replay_contract.lake) — a certified record
# built under different settings is NOT the preregistered measurement.
EXPECTED_TOPK_CONTRACT = {
    "k": 10,
    "grid_ms": 1000,
    "book_stride_ms": 1000,
    "schema_version": "topk_l2/1",
    "policy": {"enabled": True, "min_levels_per_side": 5, "reseed_after_crossed_s": 2.0,
               "max_spread_frac": None},
    "thresholds": {"crossed_usable_max": 0.01, "missing_usable_max": 0.02,
                   "thin_usable_max": 0.10, "seed_crossed_frac_max": 0.05},
}


def cmd_verdict(args) -> int:
    prereg = bsg.load_preregistration(args.prereg)
    day = prereg["fixture"]["lake"]["day"]

    recs: dict[tuple, dict] = {}
    with open(args.stage2_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("dt") != day:      # only fixture-day records may carry the verdict
                    continue
                recs[(r["exchange"], r["symbol"], r["output"])] = r

    unit_checks = []
    inconclusive_reasons: list[str] = []
    degraded_reasons: list[str] = []
    for (exchange, symbol), reqs in REQUIRED_UNITS.items():
        for output, want in reqs.items():
            r = recs.get((exchange, symbol, output))
            got_status = r.get("status") if r else None
            got_cls = r.get("classification") if r else None
            if r is None:
                ok = False
                inconclusive_reasons.append(f"{exchange}/{symbol}/{output}: no manifest "
                                            f"record for {day}")
            elif want == "certified":
                contract_diffs = sorted(
                    key for key, expect in EXPECTED_TOPK_CONTRACT.items()
                    if r.get(key) != expect)
                if got_cls is not None and contract_diffs:
                    # produced under a non-preregistered reconstruction contract — its
                    # classification is not the preregistered measurement
                    inconclusive_reasons.append(
                        f"{exchange}/{symbol}/{output}: manifest record contract differs "
                        f"from the preregistered one on {contract_diffs}")
                    unit_checks.append({"exchange": exchange, "symbol": symbol,
                                        "output": output, "required": want,
                                        "status": got_status, "classification": got_cls,
                                        "ok": False})
                    continue
                ok = got_status == "ok" and got_cls == "certified"
                if got_cls == "inconclusive" or got_status in ("missing", "error"):
                    inconclusive_reasons.append(
                        f"{exchange}/{symbol}/{output}: {got_status}/{got_cls} "
                        f"reasons={r.get('reasons')}")
                elif got_cls == "degraded":
                    degraded_reasons.append(
                        f"{exchange}/{symbol}/{output}: degraded reasons={r.get('reasons')}")
            elif want == "ok_or_missing_sparse":
                ok = got_status == "ok" or (got_status == "missing" and r.get("sparse_ok"))
                if not ok:
                    inconclusive_reasons.append(f"{exchange}/{symbol}/{output}: {got_status}")
            else:
                ok = got_status == "ok"
                if not ok:
                    inconclusive_reasons.append(f"{exchange}/{symbol}/{output}: {got_status}")
            unit_checks.append({"exchange": exchange, "symbol": symbol, "output": output,
                                "required": want, "status": got_status,
                                "classification": got_cls, "ok": bool(ok)})

    def _load_step(path: str, step: str) -> dict:
        rep = _read_json(path)
        if rep.get("step") != step:
            raise ValueError(f"{path} is not a {step} report")
        return rep

    verify = _load_step(args.verify_report, "verify-inputs")
    tick = _load_step(args.tick_report, "tick-scale")
    silence = _load_step(args.silence_report, "silence")
    determinism = _load_step(args.determinism_report, "stage2-compare")
    replays = {p: _load_step(p, "replay-conformance") for p in args.replay_report}

    # every required instrument must have a replay-conformance report — a missing one means
    # harness determinism / native conformance / frozen caps were never checked for it
    seen_instruments = {rep["instrument"] for rep in replays.values()}
    for instrument in REQUIRED_REPLAY_INSTRUMENTS:
        if instrument not in seen_instruments:
            inconclusive_reasons.append(
                f"hard invalidator: required replay-conformance report missing for "
                f"{instrument}")

    if not verify["pass"]:
        inconclusive_reasons.append("hard invalidator: input identity/schema mismatch")
    if not determinism["pass"]:
        inconclusive_reasons.append("hard invalidator: stage2 cross-run determinism failed")
    # Bind the determinism evidence to THIS manifest's content and coverage: a stale
    # passing comparison from another output root must never vouch for a different or
    # edited manifest (Codex P1). run1 is the manifest under verdict by convention.
    fingerprint = bsg.semantic_manifest_fingerprint(args.stage2_manifest)
    if determinism.get("run1_semantic_fingerprint") != fingerprint:
        inconclusive_reasons.append(
            "hard invalidator: determinism report is not about this manifest "
            f"(run1 fingerprint {determinism.get('run1_semantic_fingerprint')} != "
            f"{fingerprint})")
    compared_units = {tuple(u) for u in determinism.get("units", [])}
    for (exchange, symbol), reqs in REQUIRED_UNITS.items():
        for output in reqs:
            if (output, exchange, symbol, day) not in compared_units:
                inconclusive_reasons.append(
                    "hard invalidator: determinism report does not cover "
                    f"{exchange}/{symbol}/{output} for {day}")
    for path, rep in replays.items():
        if not rep["harness_determinism_ok"]:
            inconclusive_reasons.append(f"hard invalidator: harness determinism failed "
                                        f"({rep['instrument']})")
        if rep["conformance"]["ran"] and not rep["conformance_ok"]:
            inconclusive_reasons.append(f"hard invalidator: python/native divergence "
                                        f"({rep['instrument']})")
        if rep["frozen"]["frozen_cap_fired"]:
            degraded_reasons.append(f"anomaly cap: frozen_fraction "
                                    f"{rep['frozen']['frozen_fraction']:.4f} "
                                    f"({rep['instrument']})")
    if not tick["pass"]:
        degraded_reasons.append("anomaly cap: off-tick/no integral price scale")
    if not silence["pass"]:
        degraded_reasons.append("anomaly cap: delta-stream silence gap > 300s")

    if inconclusive_reasons or not all(c["ok"] or c["classification"] == "degraded"
                                       for c in unit_checks):
        # any hard invalidator, missing/errored/inconclusive unit -> inconclusive
        if inconclusive_reasons:
            verdict = "inconclusive"
            reasons = inconclusive_reasons + degraded_reasons
        else:
            verdict = "inconclusive"
            reasons = ["unit requirements unmet"] + degraded_reasons
    elif degraded_reasons or any(c["classification"] == "degraded" for c in unit_checks):
        verdict = "degraded"
        reasons = degraded_reasons
    elif all(c["ok"] for c in unit_checks):
        verdict = "certified"
        reasons = ["all unit requirements met; no caps; no invalidators; determinism and "
                   "conformance passed"]
    else:
        verdict = "inconclusive"
        reasons = ["unit requirements unmet"]

    report = {"step": "verdict", "prereg_commit": _prereg_commit(), "day": day,
              "unit_checks": unit_checks,
              "inputs": {"verify": verify["pass"], "tick": tick["pass"],
                         "silence": silence["pass"], "stage2_determinism": determinism["pass"],
                         "replays": {rep["instrument"]: rep["pass"]
                                     for rep in replays.values()}},
              "lake_verdict": verdict, "reasons": reasons,
              "decision_logic": bsg.PREREGISTERED["decision_logic"]["lake_verdict"]}
    _write_report(args.out, "lake_verdict.json", report)
    print(f"LAKE VERDICT ({day}): {verdict.upper()}")
    for r in reasons:
        print(f"  - {r}")
    return 0


# ----------------------------------------------------------------------------- decide
def cmd_decide(args) -> int:
    """Machine-enforced final_source_decision routing (preregistration decision_logic).
    Fail-closed: a missing/withheld input never improves the outcome."""
    lake = _read_json(args.lake_verdict)
    if lake.get("step") != "verdict":
        raise ValueError(f"{args.lake_verdict} is not a verdict report")
    lake_verdict = lake["lake_verdict"]

    chd_reports = [_read_json(p) for p in (args.chd_replay or [])]
    for path, rep in zip(args.chd_replay or [], chd_reports):
        if rep.get("step") != "chd-replay":
            raise ValueError(f"{path} is not a chd-replay report")
    if not chd_reports:
        chd_verdict = "inconclusive"    # never approved/downloaded — fail closed
    elif any(r.get("chd_verdict") != "certified" for r in chd_reports):
        verdicts = [r.get("chd_verdict") for r in chd_reports]
        chd_verdict = "inconclusive" if "inconclusive" in verdicts else "degraded"
    else:
        chd_verdict = "certified"

    comparison_pass = None
    if args.comparison:
        comp = _read_json(args.comparison)
        if comp.get("step") != "compare":
            raise ValueError(f"{args.comparison} is not a compare report")
        comparison_pass = bool(comp.get("pass"))

    if lake_verdict == "certified":
        if comparison_pass is True:
            decision, detail = "lake_go", (
                "Crypto Lake approved for the pilot, independently validated; "
                "CryptoHFTData recorded as agreeing fallback candidate")
        elif comparison_pass is False and chd_verdict == "certified":
            decision, detail = "disagreement", (
                "both sources certify individually but the fixed comparison bars fail: "
                "the April day cannot attribute fault — NO GO for either source from "
                "April data alone; escalate per preregistration escalation.dev_days")
        elif comparison_pass is None and chd_verdict == "inconclusive":
            decision, detail = "lake_go", (
                "Crypto Lake approved on internal certification only; independent "
                "validation not executable (CryptoHFTData window unavailable or "
                "refused fail-closed); fallback none")
        else:
            decision, detail = "escalate", (
                "ambiguous evidence combination (lake certified, chd "
                f"{chd_verdict}, comparison {comparison_pass}) — fail closed, human "
                "adjudication against the preregistered decision_logic required")
    elif chd_verdict == "certified":
        decision, detail = "chd_go", (
            "Crypto Lake did not certify; CryptoHFTData certified on the approved April "
            "windows — #35's approved source and manifest contract must switch (subject "
            "to the non-April escalation when April cannot carry the decision alone)")
    else:
        decision, detail = "neither", (
            "neither source certified — #35 stays blocked; open a separately scoped "
            "vendor/pivot decision")

    report = {"step": "decide", "prereg_commit": _prereg_commit(),
              "inputs": {"lake_verdict": lake_verdict, "chd_verdict": chd_verdict,
                         "comparison_pass": comparison_pass,
                         "n_chd_reports": len(chd_reports)},
              "decision": decision, "detail": detail,
              "decision_logic": bsg.PREREGISTERED["decision_logic"]["final_source_decision"]}
    _write_report(args.out, "final_source_decision.json", report)
    print(f"FINAL SOURCE DECISION: {decision.upper()}")
    print(f"  {detail}")
    return 0


# ----------------------------------------------------------------------------- chd-validate
def _decompress_if_zstd(path: str) -> tuple[str, dict]:
    """Return a plain-parquet path for `path` (streamed zstd decompress to a sibling
    `.parquet` if needed) plus provenance hashes. Bounded memory (1 MiB chunks)."""
    from ingest.download_lake_binance import _sha256_file
    with open(path, "rb") as f:
        magic = f.read(4)
    prov = {"object_path": path, "object_sha256": _sha256_file(path),
            "object_bytes": os.path.getsize(path)}
    if magic == PARQUET_MAGIC:
        prov.update(compression="none", parquet_path=path,
                    parquet_sha256=prov["object_sha256"])
        return path, prov
    if magic != ZSTD_MAGIC:
        raise bsg.ChdValidationError("not_zstd_or_parquet",
                                     f"{path}: magic {magic!r} is neither zstd nor parquet")
    import pyarrow as pa
    dest = path[:-len(".zst")] if path.endswith(".zst") else path + ".parquet"
    if not os.path.exists(dest):
        tmp = dest + ".tmp"
        with pa.input_stream(path, compression="zstd") as src, open(tmp, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    prov.update(compression="zstd", parquet_path=dest, parquet_sha256=_sha256_file(dest))
    return dest, prov


def _load_chd_hour(path: str, *, exchange: str, symbol: str, date_iso: str, hour: int):
    import pyarrow.parquet as pq
    parquet_path, prov = _decompress_if_zstd(path)
    with pq.ParquetFile(parquet_path) as pf:
        df = pf.read().to_pandas()
    identity = bsg.validate_chd_frame(df, exchange=exchange, symbol=symbol,
                                      date_iso=date_iso, hour=hour)
    identity["provenance"] = prov
    return identity, df


def cmd_chd_validate(args) -> int:
    try:
        identity, df = _load_chd_hour(args.file, exchange=args.exchange, symbol=args.symbol,
                                      date_iso=args.date, hour=args.hour)
    except bsg.SourceGateError as e:
        report = {"step": "chd-validate", "prereg_commit": _prereg_commit(),
                  "file": args.file, "pass": False, "refusal": e.code, "detail": str(e)[:500]}
        _write_report(args.out, f"chd_validate_{args.exchange}_{args.date}_"
                                f"{args.hour:02d}.json", report)
        print(f"chd-validate: REFUSED ({e.code})")
        return FAIL_EXIT
    report = {"step": "chd-validate", "prereg_commit": _prereg_commit(), "file": args.file,
              "identity": identity, "pass": True}
    _write_report(args.out, f"chd_validate_{args.exchange}_{args.date}_{args.hour:02d}.json",
                  report)
    print(f"chd-validate: PASS ({identity['rows']} rows, axis={identity['partition_axis']})")
    return 0


# ----------------------------------------------------------------------------- chd-replay
def cmd_chd_replay(args) -> int:
    import pandas as pd

    prereg = bsg.load_preregistration(args.prereg)
    bars = bsg.PREREGISTERED["thresholds"]["chd_window_quality"]
    hours = list(range(args.start_hour, args.start_hour + args.n_hours))
    if len(args.files) != len(hours):
        print(f"ERROR: {len(args.files)} files for {len(hours)} hours", file=sys.stderr)
        return SETUP_ERROR_EXIT
    name = f"chd_replay_{args.exchange}_{args.date}_{args.start_hour:02d}_{args.n_hours}h"
    try:
        loaded = []
        for path, hour in zip(args.files, hours):
            identity, df = _load_chd_hour(path, exchange=args.exchange, symbol=args.symbol,
                                          date_iso=args.date, hour=hour)
            loaded.append((identity, df))
        bsg.require_consecutive_hours([i for i, _ in loaded])
        start_ns = int(pd.Timestamp(f"{args.date}T{args.start_hour:02d}:00:00", tz="UTC").value)
        grid = [start_ns + i * 1_000_000_000 for i in range(args.n_hours * 3600)]
        market = "futures" if args.exchange.endswith("futures") else "spot"
        frame, meta = bsg.replay_chd_window(loaded, market=market,
                                            price_scale=int(args.scale), grid=grid, k=args.k)
    except bsg.SourceGateError as e:
        report = {"step": "chd-replay", "prereg_commit": _prereg_commit(),
                  "exchange": args.exchange, "date": args.date, "hours": hours,
                  "pass": False, "chd_verdict": "inconclusive",
                  "refusal": e.code, "detail": str(e)[:500]}
        _write_report(args.out, f"{name}.json", report)
        print(f"chd-replay: REFUSED ({e.code}) -> inconclusive")
        return FAIL_EXIT

    quality_ok = (meta["crossed_rate"] <= bars["crossed_usable_max"]
                  and meta["missing_book_fraction"] <= bars["missing_usable_max"]
                  and meta["thin_depth_fraction"] <= bars["thin_usable_max"]
                  and meta["seed_source_crossed_frac"] <= bars["seed_crossed_frac_max"])
    verdict = "certified" if quality_ok else "degraded"
    if args.frame_out:
        import pyarrow as pa
        import pyarrow.parquet as pq
        os.makedirs(os.path.dirname(args.frame_out) or ".", exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), args.frame_out,
                       compression="zstd")
    report = {"step": "chd-replay", "prereg_commit": _prereg_commit(),
              "exchange": args.exchange, "symbol": args.symbol, "date": args.date,
              "hours": hours, "market": ("futures" if args.exchange.endswith("futures")
                                         else "spot"),
              "identities": [i for i, _ in loaded],
              "meta": _slim_meta(meta), "bars": bars,
              "chd_verdict": verdict, "pass": bool(quality_ok)}
    _write_report(args.out, f"{name}.json", report)
    print(f"chd-replay: {verdict.upper()} crossed={meta['crossed_rate']:.5f} "
          f"missing={meta['missing_book_fraction']:.5f} thin={meta['thin_depth_fraction']:.5f}")
    return 0 if quality_ok else FAIL_EXIT


# ----------------------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    import pandas as pd
    import pyarrow.parquet as pq

    frames = {}
    for label, path in (("lake", args.lake_frame), ("chd", args.chd_frame)):
        with pq.ParquetFile(path) as pf:
            frames[label] = pf.read().to_pandas()
    if args.window_start and args.window_end:
        lo = int(pd.Timestamp(args.window_start, tz="UTC").value)
        hi = int(pd.Timestamp(args.window_end, tz="UTC").value)
        for label in frames:
            f = frames[label]
            frames[label] = f[(f["sample_ts"] >= lo) & (f["sample_ts"] < hi)] \
                .reset_index(drop=True)
    try:
        metrics = bsg.compare_topk_frames(frames["lake"], frames["chd"],
                                          price_scale=int(args.scale), k=args.k)
    except bsg.SourceGateError as e:
        report = {"step": "compare", "prereg_commit": _prereg_commit(), "pass": False,
                  "refusal": e.code, "detail": str(e)[:500]}
        _write_report(args.out, "comparison.json", report)
        print(f"compare: REFUSED ({e.code})")
        return FAIL_EXIT
    evaluation = bsg.evaluate_comparison(metrics)
    report = {"step": "compare", "prereg_commit": _prereg_commit(),
              "lake_frame": args.lake_frame, "chd_frame": args.chd_frame,
              "window": [args.window_start, args.window_end], "price_scale": int(args.scale),
              "metrics": metrics, "evaluation": evaluation, "pass": evaluation["pass"]}
    _write_report(args.out, "comparison.json", report)
    print(f"compare: {'PASS' if evaluation['pass'] else 'FAIL'} "
          f"joint_valid={metrics.get('joint_valid_fraction'):.4f}")
    return 0 if evaluation["pass"] else FAIL_EXIT


# ----------------------------------------------------------------------------- fetch (approval-gated)
CHD_BASE_URL = "https://api.cryptohftdata.com"
FETCH_MAX_ATTEMPTS = 3
FETCH_TIMEOUT_S = 120
FETCH_BYTE_CAP = 536_870_912          # 512 MiB — preregistered fail-closed cap


def cmd_fetch(args) -> int:
    """Download ONE CryptoHFTData object under the preregistered request bounds.

    HARD-GATED: refuses without --approved-by (the recorded approval provenance). This is
    the ONLY network code path in the experiment; it is never imported at module top and
    never exercised by tests (tests inject fake openers at the function boundary)."""
    if not args.approved_by or not args.approved_by.strip():
        print("REFUSING fetch: --approved-by is required (explicit user approval provenance; "
              "AGENTS.md vendor gate).", file=sys.stderr)
        return SETUP_ERROR_EXIT
    if os.path.exists(args.dest):
        print(f"REFUSING fetch: dest {args.dest} already exists (never overwrite raw vendor "
              "data).", file=sys.stderr)
        return SETUP_ERROR_EXIT

    import urllib.error
    import urllib.request
    url = f"{CHD_BASE_URL}/download?file={args.object}"
    attempts = 0
    got_bytes = 0
    t0 = time.monotonic()
    failure = None
    os.makedirs(os.path.dirname(args.dest) or ".", exist_ok=True)
    tmp = args.dest + ".tmp"
    while attempts < args.max_attempts:
        attempts += 1
        got_bytes = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "jepa-issue64-gate/1.0"})
            with urllib.request.urlopen(req, timeout=args.timeout) as resp, \
                    open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    got_bytes += len(chunk)
                    if got_bytes > args.byte_cap:
                        raise RuntimeError(f"byte cap {args.byte_cap} exceeded")
                    out.write(chunk)
            os.replace(tmp, args.dest)
            failure = None
            break
        except urllib.error.HTTPError as e:
            failure = f"HTTP {e.code}"
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(min(30.0, 2.0 * (2 ** (attempts - 1))))
                continue
            break                                        # other 4xx: abort, no retry
        except Exception as e:                           # noqa: BLE001 — recorded, bounded retries
            failure = f"{type(e).__name__}: {e}"[:300]
            time.sleep(min(30.0, 2.0 * (2 ** (attempts - 1))))
    if os.path.exists(tmp):
        os.remove(tmp)
    secs = round(time.monotonic() - t0, 3)
    ok = failure is None and os.path.exists(args.dest)
    sha = None
    if ok:
        from ingest.download_lake_binance import _sha256_file
        sha = _sha256_file(args.dest)
    report = {"step": "fetch", "prereg_commit": _prereg_commit(), "object": args.object,
              "url": url, "dest": args.dest, "approved_by": args.approved_by.strip(),
              "attempts": attempts, "bytes": got_bytes, "secs": secs, "sha256": sha,
              "byte_cap": args.byte_cap, "timeout_s": args.timeout,
              "failure": failure, "pass": bool(ok)}
    _write_report(args.out, f"fetch_{args.object.replace('/', '_')}.json", report)
    print(f"fetch: {'OK' if ok else f'FAILED ({failure})'} bytes={got_bytes} "
          f"attempts={attempts}")
    return 0 if ok else FAIL_EXIT


# ----------------------------------------------------------------------------- CLI wiring
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    prereg_default = str(ROOT / "experiments" / "preregistration_64.json")
    out_default = "data/reports/binance_source_quality"

    def common(p):
        p.add_argument("--prereg", default=prereg_default)
        p.add_argument("--out", default=out_default)

    p = sub.add_parser("verify-inputs"); common(p)
    p.add_argument("--raw", required=True)

    p = sub.add_parser("tick-scale"); common(p)
    p.add_argument("--raw", required=True)

    p = sub.add_parser("silence"); common(p)
    p.add_argument("--raw", required=True)

    p = sub.add_parser("replay-conformance"); common(p)
    p.add_argument("--raw", required=True)
    p.add_argument("--instrument", required=True, choices=("binance-perp", "binance-spot"))
    p.add_argument("--engine", choices=("python", "native"), default="python",
                   help="in-process replay engine (native per the 2026-07-12 amendment; "
                        "cross-engine equality is then carried by stage2-compare)")
    p.add_argument("--scale", type=int, default=None,
                   help="price scale (from tick-scale report); required for native")
    p.add_argument("--frame-out", default=None,
                   help="optional parquet path for the replayed top-K frame (ignored store)")

    p = sub.add_parser("stage2-native-run"); common(p)
    p.add_argument("--raw", required=True)
    p.add_argument("--out-root", required=True,
                   help="processed-store root for the native run (ignored path)")
    p.add_argument("--perp-scale", type=int, required=True)
    p.add_argument("--spot-scale", type=int, required=True)

    p = sub.add_parser("stage2-compare"); common(p)
    p.add_argument("--run1", required=True, help="run1 _manifest.jsonl path")
    p.add_argument("--run2", required=True, help="run2 _manifest.jsonl path")

    p = sub.add_parser("verdict"); common(p)
    p.add_argument("--stage2-manifest", required=True)
    p.add_argument("--verify-report", required=True)
    p.add_argument("--tick-report", required=True)
    p.add_argument("--silence-report", required=True)
    p.add_argument("--determinism-report", required=True)
    p.add_argument("--replay-report", action="append", required=True)

    p = sub.add_parser("decide"); common(p)
    p.add_argument("--lake-verdict", required=True)
    p.add_argument("--chd-replay", action="append", default=None,
                   help="chd-replay report path(s); omit when never approved/downloaded")
    p.add_argument("--comparison", default=None,
                   help="compare report path; omit when the comparison was not executable")

    p = sub.add_parser("chd-validate"); common(p)
    p.add_argument("--file", required=True)
    p.add_argument("--exchange", required=True)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--date", required=True)
    p.add_argument("--hour", type=int, required=True)

    p = sub.add_parser("chd-replay"); common(p)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--exchange", required=True)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--date", required=True)
    p.add_argument("--start-hour", type=int, required=True)
    p.add_argument("--n-hours", type=int, required=True)
    p.add_argument("--scale", type=int, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--frame-out", default=None)

    p = sub.add_parser("compare"); common(p)
    p.add_argument("--lake-frame", required=True)
    p.add_argument("--chd-frame", required=True)
    p.add_argument("--scale", type=int, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--window-start", default=None)
    p.add_argument("--window-end", default=None)

    p = sub.add_parser("fetch"); common(p)
    p.add_argument("--object", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--approved-by", default=None,
                   help="REQUIRED: explicit user-approval provenance string")
    p.add_argument("--byte-cap", type=int, default=FETCH_BYTE_CAP)
    p.add_argument("--max-attempts", type=int, default=FETCH_MAX_ATTEMPTS)
    p.add_argument("--timeout", type=int, default=FETCH_TIMEOUT_S)

    return ap.parse_args(argv)


COMMANDS = {
    "verify-inputs": cmd_verify_inputs,
    "tick-scale": cmd_tick_scale,
    "silence": cmd_silence,
    "replay-conformance": cmd_replay_conformance,
    "stage2-native-run": cmd_stage2_native_run,
    "stage2-compare": cmd_stage2_compare,
    "verdict": cmd_verdict,
    "decide": cmd_decide,
    "chd-validate": cmd_chd_validate,
    "chd-replay": cmd_chd_replay,
    "compare": cmd_compare,
    "fetch": cmd_fetch,
}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return COMMANDS[args.cmd](args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return SETUP_ERROR_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
