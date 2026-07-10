"""One-time holdout consumption record (issue #52; staged protocol §3).

The fixed holdout is consumed by exactly ONE transaction, in exactly this order:

    open_transaction (freeze pinned)  ->  record_trade_validation (exact scope, once)
        -> PASS -> record_holdout_score (once)  -> consumed
        -> FAIL -> G0-XV is blocking/inconclusive; scoring is refused forever

Both #48's manifest-authorized trade validator and the holdout scorer update this record.
Every update re-verifies the pinned freeze-artifact hash, so a stale artifact, a
regenerated freeze, a retry after failure, a partial-scope substitution, or a generic day
selector is rejected. A consumed or failed transaction can never be reused or replaced:
the record file name is DERIVED from the holdout identity (window bounds + dataset), so
even a regenerated freeze artifact over the same holdout maps to the same record and hits
the same one-time gate. No API mutates thresholds, exclusions, candidates, or holdout
dates — those live only in the frozen artifact."""
from __future__ import annotations

import json
import os
import tempfile

from eval.freeze import _canonical_day, verify_freeze
from eval.hashing import hash_obj

RECORD_VERSION = 1

STATE_FROZEN = "frozen"
STATE_VALIDATED = "validated"
STATE_VALIDATION_FAILED = "validation_failed"
STATE_SCORED = "scored"
STATES = (STATE_FROZEN, STATE_VALIDATED, STATE_VALIDATION_FAILED, STATE_SCORED)


def _record_hash(record: dict) -> str:
    return hash_obj(record, exclude_keys=("record_sha256",))


def _write(record: dict, path, *, must_create: bool) -> dict:
    record = dict(record)
    record["record_sha256"] = _record_hash(record)
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if must_create:
        # O_EXCL: a record for this holdout already existing means the transaction was
        # already opened (and possibly consumed/failed) — reuse/replacement is rejected.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
    else:
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".holdout-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    return record


def holdout_id(freeze_artifact: dict) -> str:
    """The holdout's transaction identity: the contract's holdout WINDOW plus the holdout
    dataset. Deliberately independent of the freeze hash — a regenerated selection
    artifact over the same holdout must map to the SAME transaction, not a fresh one."""
    win = freeze_artifact["holdout_window"]
    return hash_obj({"holdout_start_ns": win["holdout_start_ns"],
                     "holdout_end_ns": win["holdout_end_ns"],
                     "dataset_id": freeze_artifact["holdout_scope"]["dataset_id"]})


def record_path_for(records_dir, freeze_artifact: dict) -> str:
    return os.path.join(records_dir, f"holdout-{holdout_id(freeze_artifact)[:16]}.json")


def load_record(path) -> dict:
    with open(path) as f:
        record = json.load(f)
    if record.get("record_version") != RECORD_VERSION:
        raise ValueError(f"unsupported holdout record version "
                         f"{record.get('record_version')!r}")
    if record.get("record_sha256") != _record_hash(record):
        raise ValueError("holdout consumption record does not match its embedded hash "
                         "(edited or corrupted); refusing to trust it")
    if record.get("state") not in STATES:
        raise ValueError(f"holdout record in unknown state {record.get('state')!r}")
    return record


def open_transaction(records_dir, freeze_artifact: dict) -> dict:
    """Open THE holdout transaction for a verified freeze artifact. The record path is
    derived from the holdout identity inside `records_dir`, so any prior transaction for
    this holdout — consumed, failed, or merely opened, even under a regenerated freeze —
    already occupies the path and the open fails: never reused, never replaced."""
    verify_freeze(freeze_artifact)
    path = record_path_for(records_dir, freeze_artifact)
    if os.path.exists(path):
        raise ValueError(f"holdout consumption record already exists at {path}; the "
                         "holdout transaction is one-time and cannot be reused or "
                         "replaced")
    record = {
        "record_version": RECORD_VERSION,
        "holdout_id": holdout_id(freeze_artifact),
        "artifact_sha256": freeze_artifact["sha256"],
        "holdout_scope": freeze_artifact["holdout_scope"],
        "trade_validation_thresholds": freeze_artifact["trade_validation_thresholds"],
        "state": STATE_FROZEN,
        "history": [{"step": 0, "event": "opened",
                     "artifact_sha256": freeze_artifact["sha256"]}],
    }
    return _write(record, path, must_create=True)


def _require_artifact(record: dict, freeze_artifact: dict) -> None:
    verify_freeze(freeze_artifact)
    if freeze_artifact["sha256"] != record["artifact_sha256"]:
        raise ValueError(
            f"freeze artifact {freeze_artifact['sha256'][:12]}... does not match the "
            f"transaction's pinned artifact {record['artifact_sha256'][:12]}...; stale or "
            "regenerated selection artifacts cannot touch the holdout")


def _exact_scope(record: dict, scope_days, scope_venues) -> None:
    days = [_canonical_day(d) for d in (scope_days or [])]
    frozen_days = record["holdout_scope"]["days"]
    if sorted(set(days)) != days:
        raise ValueError("scope days must be sorted, unique, explicit ISO days")
    if days != frozen_days:
        raise ValueError(f"scope days {days} do not exactly match the frozen holdout "
                         f"scope {frozen_days}; partial-scope or substituted requests "
                         "are rejected")
    venues = list(scope_venues or [])
    if venues != record["holdout_scope"]["venues"]:
        raise ValueError(f"scope venues {venues} do not exactly match the frozen scope "
                         f"{record['holdout_scope']['venues']}")


def record_trade_validation(records_dir, *, freeze_artifact: dict, scope_days,
                            scope_venues, passed: bool, report_sha256: str) -> dict:
    """Record the ONE #48 exact-scope trade-validation outcome. Allowed only while the
    transaction is `frozen`; every retry — after a pass, after a fail, with a different
    artifact, or with any scope deviation — is rejected. A FAIL makes G0-XV
    blocking/inconclusive; nothing here (or anywhere) can then change thresholds,
    exclusions, candidates, or holdout dates to try again."""
    path = record_path_for(records_dir, freeze_artifact)
    record = load_record(path)
    _require_artifact(record, freeze_artifact)
    if record["state"] != STATE_FROZEN:
        raise ValueError(f"trade validation already recorded (state={record['state']!r}); "
                         "the holdout transaction accepts exactly one validation attempt")
    _exact_scope(record, scope_days, scope_venues)
    if not isinstance(passed, bool):
        raise ValueError("passed must be a bool")
    if not isinstance(report_sha256, str) or not report_sha256:
        raise ValueError("report_sha256 must be the non-empty hash of the validation "
                         "report artifact")
    record["state"] = STATE_VALIDATED if passed else STATE_VALIDATION_FAILED
    record["history"].append({"step": len(record["history"]),
                              "event": "trade_validation", "passed": passed,
                              "report_sha256": report_sha256,
                              "artifact_sha256": freeze_artifact["sha256"]})
    return _write(record, path, must_create=False)


def record_holdout_score(records_dir, *, freeze_artifact: dict,
                         result_sha256: str) -> dict:
    """Record the ONE fixed model score. Requires a PASSed trade validation; a failed
    validation blocks scoring permanently, and a second score is rejected."""
    path = record_path_for(records_dir, freeze_artifact)
    record = load_record(path)
    _require_artifact(record, freeze_artifact)
    if record["state"] == STATE_VALIDATION_FAILED:
        raise ValueError("trade validation FAILED: G0-XV is blocking/inconclusive and the "
                         "holdout cannot be scored; thresholds, exclusions, candidates, "
                         "and holdout dates cannot be changed to retry")
    if record["state"] == STATE_SCORED:
        raise ValueError("holdout already scored; the transaction is consumed and cannot "
                         "be reused")
    if record["state"] != STATE_VALIDATED:
        raise ValueError("holdout scoring requires a recorded PASSing trade validation "
                         f"first (state={record['state']!r})")
    if not isinstance(result_sha256, str) or not result_sha256:
        raise ValueError("result_sha256 must be the non-empty hash of the score result")
    record["state"] = STATE_SCORED
    record["history"].append({"step": len(record["history"]), "event": "holdout_score",
                              "result_sha256": result_sha256,
                              "artifact_sha256": freeze_artifact["sha256"]})
    return _write(record, path, must_create=False)
