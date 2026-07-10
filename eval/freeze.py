"""Hash-pinned G0-XV freeze artifact (issue #52).

The freeze is the ONE object that authorizes touching the fixed holdout, and it is built
strictly from development evidence: winner + configuration, the resolved numerical gate
rules, the frozen trade-validation thresholds, the EXACT holdout scope (explicit day
list — never a range, glob, or generic selector), the pinned sources (partition
contract, arm manifests, matched row/split hashes), and the complete trial history
(ledger hash + effective count, G0-CB history included). Its content hash is what the
one-time consumption record (eval.consumption) and the holdout scorer (eval.holdout)
verify — a stale, edited, or regenerated artifact no longer matches and every downstream
step fails closed."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile

from eval.hashing import hash_obj
from eval.ledger import TrialLedger, _json_safe
from eval.partition import contract_hash, validate_partition_contract

FREEZE_VERSION = 1
_VOLATILE = ("sha256", "generated_at")   # excluded from the pinned content hash

_SCOPE_FIELDS = ("days", "venues", "dataset_id", "build_id")


def _canonical_day(token) -> str:
    """One explicit ISO day. Anything else — ranges ('2026-04-01..30'), globs ('*'),
    months ('2026-04'), non-strings — is a generic selector and is rejected: the holdout
    transaction consumes an exact, enumerated scope only."""
    if not isinstance(token, str):
        raise ValueError(f"holdout scope days must be explicit ISO date strings, got "
                         f"{token!r}")
    try:
        day = dt.date.fromisoformat(token)
    except ValueError:
        raise ValueError(f"holdout scope day {token!r} is not an explicit YYYY-MM-DD "
                         "date; generic day selectors are rejected") from None
    if day.isoformat() != token:
        raise ValueError(f"holdout scope day {token!r} is not canonical YYYY-MM-DD")
    return token


def validate_holdout_scope(scope: dict, contract: dict) -> dict:
    """Exact-scope validation: explicit sorted unique days inside the contract's holdout
    window, explicit venue keys, and the holdout dataset/build identity."""
    validate_partition_contract(contract)
    if not isinstance(scope, dict) or set(scope) != set(_SCOPE_FIELDS):
        raise ValueError(f"holdout scope must have exactly the fields {_SCOPE_FIELDS}")
    days = scope["days"]
    if not isinstance(days, list) or not days:
        raise ValueError("holdout scope days must be a non-empty explicit list")
    canon = [_canonical_day(d) for d in days]
    if sorted(set(canon)) != canon:
        raise ValueError("holdout scope days must be sorted and unique")
    lo = dt.datetime.fromtimestamp(contract["holdout_start_ns"] / 1e9,
                                   tz=dt.timezone.utc).date()
    hi = dt.datetime.fromtimestamp(contract["holdout_end_ns"] / 1e9,
                                   tz=dt.timezone.utc).date()
    outside = [d for d in canon if not (lo <= dt.date.fromisoformat(d) < hi)]
    if outside:
        raise ValueError(f"holdout scope days outside the contract holdout window "
                         f"[{lo}, {hi}): {outside}")
    venues = scope["venues"]
    if (not isinstance(venues, list) or not venues
            or any(not isinstance(v, str) or not v or "*" in v or "?" in v
                   for v in venues)):
        raise ValueError("holdout scope venues must be a non-empty list of explicit venue "
                         "keys (no wildcards)")
    for k in ("dataset_id", "build_id"):
        if not isinstance(scope[k], str) or not scope[k]:
            raise ValueError(f"holdout scope {k} must be a non-empty string")
    return scope


def _validate_thresholds(thresholds: dict) -> dict:
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("trade_validation_thresholds must be a non-empty dict (the "
                         "frozen thresholds #48's exact-scope validator applies)")
    bad = {k: v for k, v in thresholds.items()
           if not isinstance(v, (int, float, str, bool))}
    if bad:
        raise ValueError(f"trade_validation_thresholds must be scalar knobs: {bad}")
    return thresholds


def build_freeze_artifact(dev_result: dict, *, contract: dict, ledger: TrialLedger,
                          trade_validation_thresholds: dict, holdout_scope: dict,
                          generated_at: str) -> dict:
    """Freeze the G0-XV selection BEFORE any outcome-bearing holdout access. Only a
    passing development study with a selected winner authorizes a holdout transaction —
    a FAIL or blocking/inconclusive study freezes nothing (stop-or-pivot instead)."""
    # Strict-JSON copy first: a legitimately passing multi-horizon study may carry NaN
    # in a secondary horizon (unavailable PBO / empty noise band); hashing must treat it
    # exactly like the JSON round-trip the CLI performs (NaN -> null), not crash.
    dev_result = _json_safe(dev_result)
    if dev_result.get("protocol") != "g0xv-development":
        raise ValueError("freeze requires a g0xv-development result (G0-CB is "
                         "development-only and never authorizes holdout access)")
    if not dev_result.get("g0xv_dev_pass") or not dev_result.get("winner"):
        raise ValueError("a failed or inconclusive G0-XV development study does not "
                         "authorize holdout consumption; record a stop/pivot decision "
                         "instead of freezing")
    if dev_result["partition_contract_sha256"] != contract_hash(contract):
        raise ValueError("dev result pins a different partition contract than the one "
                         "supplied; refusing a stale/substituted contract")
    if ledger.ledger_hash() != dev_result["ledger"]["ledger_sha256"]:
        raise ValueError("ledger has changed since the development study ran (its hash "
                         "no longer matches the dev result); re-run development so every "
                         "trial is inside the frozen history")
    validate_holdout_scope(holdout_scope, contract)
    _validate_thresholds(trade_validation_thresholds)

    artifact = {
        "freeze_version": FREEZE_VERSION,
        "protocol": "g0xv-freeze",
        "winner": dict(dev_result["winner"]),
        "gate": dict(dev_result["gate"]),
        "trade_validation_thresholds": dict(trade_validation_thresholds),
        "holdout_scope": {**holdout_scope, "days": list(holdout_scope["days"]),
                          "venues": list(holdout_scope["venues"])},
        "holdout_window": {"holdout_start_ns": contract["holdout_start_ns"],
                           "holdout_end_ns": contract["holdout_end_ns"]},
        "sources": {
            "partition_contract_sha256": dev_result["partition_contract_sha256"],
            "arm_manifests": {arm: echo["manifest_sha256"]
                              for arm, echo in dev_result["arms"].items()},
            # Full per-arm content pins (reserved + that arm's feature VALUES): the
            # holdout refit must consume exactly the matrix the winner was selected on,
            # features included — the reserved-only row hash cannot see feature
            # substitution.
            "arm_matrix_hashes": {arm: echo["matrix_content_sha256"]
                                  for arm, echo in dev_result["arms"].items()},
            "row_content_sha256": dev_result["matched"]["row_content_sha256"],
            "split_sha256": dev_result["matched"]["split_sha256"],
        },
        "splits": {k: dev_result["matched"][k]
                   for k in ("n_groups", "k", "embargo_ns", "n_rows")},
        "trial_history": {
            "ledger_sha256": dev_result["ledger"]["ledger_sha256"],
            "n_effective_trials": dev_result["ledger"]["n_effective_trials"],
            "n_imported_trials": dev_result["ledger"]["n_imported_trials"],
        },
        "dev_result_sha256": hash_obj(dev_result),
        "generated_at": generated_at,
    }
    artifact["sha256"] = hash_obj(artifact, exclude_keys=_VOLATILE)
    return artifact


def freeze_hash(artifact: dict) -> str:
    return hash_obj(artifact, exclude_keys=_VOLATILE)


def verify_freeze(artifact: dict) -> dict:
    """Recompute the content hash and compare to the embedded pin; fail closed on any
    edit. Returns the artifact for chaining."""
    if not isinstance(artifact, dict) or artifact.get("freeze_version") != FREEZE_VERSION:
        raise ValueError(f"unsupported freeze artifact (freeze_version="
                         f"{artifact.get('freeze_version') if isinstance(artifact, dict) else None!r})")
    if artifact.get("sha256") != freeze_hash(artifact):
        raise ValueError("freeze artifact content does not match its embedded sha256 "
                         "(edited or corrupted); refusing to trust it")
    return artifact


def write_freeze(artifact: dict, path) -> None:
    verify_freeze(artifact)
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".freeze-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_freeze(path) -> dict:
    with open(path) as f:
        artifact = json.load(f)
    return verify_freeze(artifact)
