"""#54 reconciliation against the canonical #33 reviewed manifest (stdlib-only).

Buckets every book-fill day of the reviewed manifest by whether a CoinAPI
snapshot-seeded Lake replay could even in principle replace its full-day fill, and
prices the counterfactual per bucket, overall and inside the #34 Milestone-A pilot
window. READ-ONLY over the manifest; changes nothing. The output is evidence for the
#54 decision update — it does not alter the canonical manifest or fill policy.

  .venv/bin/python scripts/reconcile_snapshot_seed_54.py \
      --manifest /home/aaron/jepa-btc-forecasting/data/reports/backfill/coinbase_backfill_manifest.json \
      --out data/reports/snapshot_seed/reconciliation_54.json
"""
from __future__ import annotations

import argparse
import json
import sys

PILOT_WINDOW = ("2025-11-01", "2026-04-30")  # issue #34 Milestone A


def _bucket_for(rec: dict) -> str:
    """Addressability of a snapshot SEED/RESEED for one manifest book-fill day.

    A seed can only help where Lake `book_delta_v2` deltas exist to replay:
      * `crossed_seed_source` full-day fills — the target class (#54's question);
      * the `policy_decision` resolution days — mostly seed-source failures, grouped
        separately because one of them (lake_load_failed) is not addressable;
      * Lake-absent days and partial fills (CoinAPI covers windows where Lake has no
        deltas) — NOT addressable by any seeding strategy.
    """
    bf = rec.get("book_fill") or {}
    why = bf.get("why") or ""
    fdr = bf.get("full_day_reason") or ""
    cls = rec.get("classification") or ""
    if cls == "missing_needs_coinapi" or "missing" in why or "gap" in why:
        return "not_addressable_lake_absent"
    if fdr == "crossed_seed_source" or "crossed_seed" in why or "seed_source" in why:
        return "addressable_crossed_seed_source"
    if fdr == "policy_decision" or rec.get("resolution"):
        return "resolution_policy_days"
    if bf.get("kind") == "partial":
        return "not_addressable_partial_fills"
    return f"other_{cls or 'unclassified'}"


def reconcile_manifest(manifest: dict, *, pilot_window=PILOT_WINDOW) -> dict:
    lo, hi = pilot_window
    buckets: dict = {}
    n_days = 0
    for rec in manifest["days"]:
        bf = rec.get("book_fill") or {}
        if not bf.get("needed"):
            continue
        n_days += 1
        b = buckets.setdefault(_bucket_for(rec), {
            "days": 0, "gb": 0.0, "usd": 0.0,
            "pilot_days": 0, "pilot_gb": 0.0, "pilot_usd": 0.0})
        gb, usd = float(bf.get("gb") or 0.0), float(bf.get("usd") or 0.0)
        b["days"] += 1
        b["gb"] = round(b["gb"] + gb, 4)
        b["usd"] = round(b["usd"] + usd, 4)
        if lo <= rec["day"] <= hi:
            b["pilot_days"] += 1
            b["pilot_gb"] = round(b["pilot_gb"] + gb, 4)
            b["pilot_usd"] = round(b["pilot_usd"] + usd, 4)
    cs = manifest.get("cost_summary", {})
    return {
        "issue": 54,
        "pilot_window": list(pilot_window),
        "buckets": buckets,
        "totals": {"book_fill_days": n_days,
                   "gross_usd": cs.get("gross_usd"),
                   "book_usd": cs.get("book_usd")},
        "note": ("'addressable' = a snapshot-seeded Lake replay could in principle "
                 "replace the full-day fill (Lake deltas present, failure mode is the "
                 "seed source); whether it ACTUALLY may is the #54 experiment verdict "
                 "— a NO-GO leaves every bucket on the canonical manifest unchanged."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="#54 vs #33 manifest reconciliation")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pilot-start", default=PILOT_WINDOW[0])
    ap.add_argument("--pilot-end", default=PILOT_WINDOW[1])
    args = ap.parse_args(argv)
    with open(args.manifest) as f:
        manifest = json.load(f)
    out = reconcile_manifest(manifest,
                             pilot_window=(args.pilot_start, args.pilot_end))
    text = json.dumps(out, indent=2, allow_nan=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
