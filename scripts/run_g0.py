"""G0 orchestrator CLI (issue #52; staged protocol §2-§3).

Subcommands mirror the protocol's phases and enforce its access boundaries:

  g0cb             development-only Coinbase screen. There are NO holdout arguments, and
                   the manifest/contract binding is checked BEFORE the matrix file is
                   opened — an April/holdout input fails before any data loading.
  g0xv-dev         the unified matched multi-arm development study. Accepts
                   development-partition builds only (checked per arm before any load);
                   April cannot be opened before freeze because no holdout input exists
                   on this command.
  freeze           build + write the hash-pinned selection artifact from the g0xv-dev
                   result (winner, gate rules, trade thresholds, exact April scope,
                   sources, splits, trial history) — the precondition for ANY holdout
                   access.
  holdout-open     open the one-time consumption transaction for a frozen artifact.
  holdout-validate record #48's exact-scope trade-validation outcome (once).
  holdout-score    fit the frozen winner on pre-holdout rows and score the holdout once
                   (only after a PASSed validation); --verify-only re-computes an
                   already-consumed score and checks it reproduces the recorded hash.

Usage: .venv/bin/python scripts/run_g0.py <subcommand> ... (see --help per subcommand).
Results/ledgers/records are JSON under git-ignored output paths chosen by the caller.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.consumption import (load_record, open_transaction,           # noqa: E402
                              record_path_for, record_trade_validation)
from eval.freeze import build_freeze_artifact, load_freeze, write_freeze  # noqa: E402
from eval.g0 import (g0cb_manifest_prechecks, run_g0cb_study,          # noqa: E402
                     run_g0xv_development)
from eval.hashing import hash_obj                                      # noqa: E402
from eval.holdout import (preflight_holdout_inputs, score_fixed_holdout,  # noqa: E402
                          verify_frozen_dev_matrix)
from eval.ledger import TrialLedger, _json_safe                        # noqa: E402
from eval.manifest import load_manifest                                # noqa: E402
from eval.partition import load_partition_contract, require_binding    # noqa: E402


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(obj, path, *, quiet: bool = False) -> None:
    # Atomic like every other writer in this layer (tmp + rename): a mid-dump failure
    # must never leave a truncated result artifact behind.
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".result-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    if not quiet:
        print(f"wrote {path}")


def _preflight_out(path) -> None:
    """Fail BEFORE an irreversible step if the result cannot land at the EXACT output
    leaf — the holdout scorer consumes the one-time transaction before its result is
    persisted. A writable parent is not enough (the leaf may be a directory or otherwise
    unreplaceable), so this performs the same atomic write the result will use, leaving
    a placeholder the result then replaces."""
    if os.path.isdir(path):
        raise ValueError(f"output path {path} is a directory, not a writable file path")
    try:
        _write_json({"status": "preflight",
                     "note": "placeholder written before the one-time holdout "
                             "consumption; replaced by the score result"},
                    path, quiet=True)
    except OSError as e:
        raise ValueError(f"output path {path} is not writable: {e}") from None


def _default_read_matrix(path):
    import pandas as pd
    return pd.read_parquet(path)


def _load_ledger(path) -> TrialLedger:
    return TrialLedger.load(path) if path and pathlib.Path(path).exists() else TrialLedger()


# ------------------------------------------------------------------------- subcommands
def cmd_g0cb(args, read_matrix) -> int:
    manifest = load_manifest(args.manifest)
    contract = load_partition_contract(args.contract)
    # Fail-before-load boundary: a holdout-bound or cross-venue manifest is rejected
    # here, before the matrix file is ever opened.
    g0cb_manifest_prechecks(manifest, contract)
    gate = _read_json(args.gate_json) if args.gate_json else None
    variants = _read_json(args.variants_json) if args.variants_json else None
    ledger = _load_ledger(args.ledger)
    matrix = read_matrix(args.matrix)
    try:
        res = run_g0cb_study(matrix, manifest, contract, gate=gate, ledger=ledger,
                             variants=variants)
    finally:
        # Every attempted candidate is trial history, even when a later candidate or
        # check aborts the run — the ledger must never lose attempts to an exception.
        ledger.save(args.ledger)
    _write_json(res, args.out)
    print(f"G0-CB (development-only): pass={res['g0cb_pass']} "
          f"trials={res['ledger']['n_effective_trials']}")
    return 0


def cmd_g0xv_dev(args, read_matrix) -> int:
    contract = load_partition_contract(args.contract)
    # --arm NAME MANIFEST MATRIX (nargs=3): no separator characters to collide with
    # ':' or '=' in real paths (ISO-timestamped run dirs contain ':').
    parsed = [(a[0], a[1], a[2]) for a in args.arm]
    if any(not p for triple in parsed for p in triple):
        raise ValueError("--arm requires three non-empty values: NAME MANIFEST MATRIX")
    if not args.prior_ledger and not args.no_prior_history:
        raise ValueError(
            "g0xv-dev requires the persisted G0-CB trial history via --prior-ledger "
            "(the unified DSR count must include every prior attempt); pass "
            "--no-prior-history ONLY if no G0-CB screen was ever run for this pilot")
    manifests = {}
    for name, manifest_path, _ in parsed:
        man = load_manifest(manifest_path)
        # Every arm must be a development-partition build; checked before ANY matrix load
        # so a holdout input cannot be opened pre-freeze.
        require_binding(man, contract, "development")
        manifests[name] = man
    arms = [{"name": name, "manifest": manifests[name], "matrix": read_matrix(mx)}
            for name, _, mx in parsed]
    gate = _read_json(args.gate_json) if args.gate_json else None
    variants = _read_json(args.variants_json) if args.variants_json else None
    ledger = _load_ledger(args.ledger)
    priors = [TrialLedger.load(p) for p in (args.prior_ledger or [])]
    try:
        res = run_g0xv_development(arms, contract, gate=gate, ledger=ledger,
                                   prior_ledgers=priors, variants=variants)
    finally:
        ledger.save(args.ledger)   # attempted trials survive an aborted run
    _write_json(res, args.out)
    print(f"G0-XV development: pass={res['g0xv_dev_pass']} "
          f"winner={res['winner']['identity_sha256'][:12] if res['winner'] else None} "
          f"trials={res['ledger']['n_effective_trials']} "
          f"(imported {res['ledger']['n_imported_trials']})")
    return 0


def cmd_freeze(args, read_matrix) -> int:
    dev_result = _read_json(args.dev_result)
    contract = load_partition_contract(args.contract)
    ledger = TrialLedger.load(args.ledger)
    thresholds = _read_json(args.thresholds_json)
    scope = _read_json(args.scope_json)
    generated_at = args.generated_at or dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds")
    artifact = build_freeze_artifact(dev_result, contract=contract, ledger=ledger,
                                     trade_validation_thresholds=thresholds,
                                     holdout_scope=scope, generated_at=generated_at)
    write_freeze(artifact, args.out)
    print(f"froze selection {artifact['sha256'][:16]}... "
          f"(winner {artifact['winner']['arm']}/{artifact['winner']['config']}/"
          f"{artifact['winner']['horizon']}, {len(scope['days'])} scope day(s))")
    return 0


def cmd_holdout_open(args, read_matrix) -> int:
    freeze = load_freeze(args.freeze)
    record = open_transaction(args.records_dir, freeze)
    print(f"opened one-time holdout transaction "
          f"{record_path_for(args.records_dir, freeze)} "
          f"(holdout {record['holdout_id'][:16]}..., artifact {freeze['sha256'][:16]}...)")
    return 0


def cmd_holdout_validate(args, read_matrix) -> int:
    freeze = load_freeze(args.freeze)
    report = _read_json(args.report_json)
    for key in ("scope_days", "scope_venues", "thresholds", "passed"):
        if key not in report:
            raise ValueError(f"validation report missing {key!r} (need scope_days, "
                             "scope_venues, thresholds, passed)")
    if not isinstance(report["passed"], bool):
        # NEVER coerce: bool("false") is True, so a truthy string from external tooling
        # would permanently record a PASS on the single gate protecting the holdout.
        raise ValueError(f"validation report 'passed' must be a JSON boolean, got "
                         f"{report['passed']!r}")
    record = record_trade_validation(
        args.records_dir, freeze_artifact=freeze, scope_days=report["scope_days"],
        scope_venues=report["scope_venues"], thresholds=report["thresholds"],
        passed=report["passed"], report_sha256=hash_obj(_json_safe(report)))
    print(f"trade validation recorded: state={record['state']}")
    return 0 if record["state"] == "validated" else 3


def cmd_holdout_score(args, read_matrix) -> int:
    freeze = load_freeze(args.freeze)
    record = load_record(record_path_for(args.records_dir, freeze))
    # Fail-before-load boundary: no holdout bytes are read unless the one-time
    # transaction is in the right state for this invocation and every manifest binds the
    # pinned contract partitions.
    if freeze["sha256"] != record["artifact_sha256"]:
        raise ValueError("freeze artifact does not match the transaction's pinned "
                         "artifact")
    need = "scored" if args.verify_only else "validated"
    if record["state"] != need:
        raise ValueError(f"holdout transaction is {record['state']!r}; this invocation "
                         f"requires {need!r} — refusing before any holdout data is read")
    _preflight_out(args.out)   # scoring consumes the transaction; --out must work
    contract = load_partition_contract(args.contract)
    dev_manifest = load_manifest(args.dev_manifest)
    holdout_manifest = load_manifest(args.holdout_manifest)
    # EVERY data-free frozen-pin check runs before any matrix is opened: a validated
    # transaction with a wrong/stale build must not be able to re-open the holdout
    # matrix through repeated failing invocations.
    preflight_holdout_inputs(freeze, contract=contract, dev_manifest=dev_manifest,
                             holdout_manifest=holdout_manifest)
    dev_matrix = read_matrix(args.dev_matrix)
    # The dev matrix's frozen content pins are verified BEFORE the holdout matrix is
    # opened: tampered/stale dev rows must not re-open outcome-bearing holdout data
    # through repeated failing invocations.
    verify_frozen_dev_matrix(freeze, contract=contract, dev_matrix=dev_matrix,
                             dev_manifest=dev_manifest)
    holdout_matrix = read_matrix(args.holdout_matrix)
    res = score_fixed_holdout(freeze_artifact=freeze, records_dir=args.records_dir,
                              contract=contract, dev_matrix=dev_matrix,
                              dev_manifest=dev_manifest, holdout_matrix=holdout_matrix,
                              holdout_manifest=holdout_manifest,
                              verify_only=args.verify_only)
    _write_json(res, args.out)
    if args.verify_only:
        print(f"verify-only: reproduces_recorded_score="
              f"{res['reproduces_recorded_score']}")
        return 0 if res["reproduces_recorded_score"] else 4
    print(f"holdout scored ONCE: net={res['metrics']['net_pnl']:.1f} "
          f"trades={res['metrics']['n_trades']} (transaction consumed)")
    return 0


# ------------------------------------------------------------------------------ parser
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="G0 evaluator orchestrator (issue #52). G0-CB/G0-XV development "
                    "cannot open holdout data; holdout scoring requires the frozen "
                    "artifact + PASSed one-time trade validation.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("g0cb", help="development-only Coinbase screen (no holdout args)")
    p.add_argument("--matrix", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--ledger", required=True, help="trial ledger JSON (created/appended)")
    p.add_argument("--gate-json", default=None)
    p.add_argument("--variants-json", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_g0cb)

    p = sub.add_parser("g0xv-dev", help="unified matched multi-arm development study")
    p.add_argument("--arm", action="append", required=True, nargs=3,
                   metavar=("NAME", "MANIFEST", "MATRIX"),
                   help="arm name, manifest JSON path, matrix path (repeat per arm)")
    p.add_argument("--contract", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--prior-ledger", action="append", default=None,
                   help="prior trial ledgers (the G0-CB history); repeatable and "
                        "REQUIRED unless --no-prior-history")
    p.add_argument("--no-prior-history", action="store_true",
                   help="explicitly assert no prior G0-CB trial history exists for "
                        "this pilot (otherwise --prior-ledger is required)")
    p.add_argument("--gate-json", default=None)
    p.add_argument("--variants-json", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_g0xv_dev)

    p = sub.add_parser("freeze", help="build the hash-pinned selection artifact")
    p.add_argument("--dev-result", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--thresholds-json", required=True)
    p.add_argument("--scope-json", required=True)
    p.add_argument("--generated-at", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("holdout-open", help="open the one-time consumption transaction")
    p.add_argument("--freeze", required=True)
    p.add_argument("--records-dir", required=True,
                   help="directory of holdout consumption records; the file name is "
                        "derived from the holdout identity (one transaction per holdout)")
    p.set_defaults(fn=cmd_holdout_open)

    p = sub.add_parser("holdout-validate",
                       help="record the exact-scope trade-validation outcome (once)")
    p.add_argument("--freeze", required=True)
    p.add_argument("--records-dir", required=True)
    p.add_argument("--report-json", required=True,
                   help="validation report with scope_days, scope_venues, passed (a "
                        "strict JSON boolean)")
    p.set_defaults(fn=cmd_holdout_validate)

    p = sub.add_parser("holdout-score",
                       help="fit frozen winner pre-holdout, score the holdout once")
    p.add_argument("--freeze", required=True)
    p.add_argument("--records-dir", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--dev-matrix", required=True)
    p.add_argument("--dev-manifest", required=True)
    p.add_argument("--holdout-matrix", required=True)
    p.add_argument("--holdout-manifest", required=True)
    p.add_argument("--verify-only", action="store_true",
                   help="re-compute an already-consumed score and check it reproduces "
                        "the recorded result hash (no record mutation)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_holdout_score)
    return ap.parse_args(argv)


def main(argv=None, read_matrix=None) -> int:
    args = parse_args(argv)
    try:
        return args.fn(args, read_matrix or _default_read_matrix)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
