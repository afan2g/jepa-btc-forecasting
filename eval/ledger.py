"""Deterministic G0 trial ledger (issue #52).

Every attempted candidate is a TRIAL: identity = (protocol, arm, dataset/build, ordered
feature list, model config, horizon, variant + params). G0-CB persists every attempt;
G0-XV imports that history so the effective DSR trial count covers the COMPLETE search —
arms, builds, models, horizons, and post-hoc variants — not just the configs of one
`run_study` call (the unreachable `extra_trials` default this replaces). Re-registering an
identical trial with an identical result is an idempotent no-op (deterministic reruns);
the same identity with a DIFFERENT result fails closed (non-determinism or tampering)."""
from __future__ import annotations

import json
import math
import os
import tempfile

from eval.hashing import hash_obj

LEDGER_VERSION = 1
PROTOCOLS = ("g0cb", "g0xv")

IDENTITY_FIELDS = ("protocol", "arm", "dataset_id", "build_id", "feature_cols",
                   "config", "horizon", "variant", "variant_params")


def _json_safe(obj):
    """Strict-JSON copy: non-finite floats -> None (canonical hashing forbids NaN)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def trial_identity(*, protocol: str, arm: str, dataset_id: str, build_id: str,
                   feature_cols, config: str, horizon: str, variant: str = "base",
                   variant_params: dict | None = None) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    for name, val in (("arm", arm), ("dataset_id", dataset_id), ("build_id", build_id),
                      ("config", config), ("horizon", horizon), ("variant", variant)):
        if not isinstance(val, str) or not val:
            raise ValueError(f"trial identity {name} must be a non-empty string")
    feats = list(feature_cols)
    if not feats or any(not isinstance(c, str) or not c for c in feats):
        raise ValueError("trial identity feature_cols must be a non-empty list of strings")
    params = dict(variant_params or {})
    return {"protocol": protocol, "arm": arm, "dataset_id": dataset_id,
            "build_id": build_id, "feature_cols": feats, "config": config,
            "horizon": horizon, "variant": variant, "variant_params": _json_safe(params)}


def identity_hash(identity: dict) -> str:
    if set(identity) != set(IDENTITY_FIELDS):
        raise ValueError(f"trial identity must have exactly the fields {IDENTITY_FIELDS}")
    return hash_obj(identity)


class TrialLedger:
    """Append-only, file-backed trial history. The ledger hash is order-independent
    (sorted by identity hash) so identical trial SETS pin identically regardless of
    registration order."""

    def __init__(self, entries: list | None = None):
        self._entries: list[dict] = []
        self._by_identity: dict[str, dict] = {}
        for e in entries or []:
            self._append(e)

    # ------------------------------------------------------------------ construction
    def _append(self, entry: dict) -> dict:
        ih, rh = entry["identity_sha256"], entry["result_sha256"]
        if identity_hash(entry["identity"]) != ih:
            raise ValueError("ledger entry identity_sha256 does not match its identity "
                             "(tampered or corrupted ledger)")
        if hash_obj(entry["result"]) != rh:
            raise ValueError("ledger entry result_sha256 does not match its result "
                             "(tampered or corrupted ledger)")
        existing = self._by_identity.get(ih)
        if existing is not None:
            if existing["result_sha256"] != rh:
                raise ValueError(
                    f"trial {ih[:12]}... already registered with a DIFFERENT result; "
                    "a re-run of the same trial identity must be deterministic — "
                    "refusing to overwrite trial history")
            return existing
        rec = {"identity": entry["identity"], "identity_sha256": ih,
               "result": entry["result"], "result_sha256": rh,
               "seq": len(self._entries)}
        self._entries.append(rec)
        self._by_identity[ih] = rec
        return rec

    def register(self, identity: dict, result: dict) -> dict:
        result = _json_safe(dict(result))
        entry = {"identity": identity, "identity_sha256": identity_hash(identity),
                 "result": result, "result_sha256": hash_obj(result), "seq": None}
        return self._append(entry)

    def import_history(self, other: "TrialLedger") -> int:
        """Merge another ledger's entries (e.g. the G0-CB history into the G0-XV ledger).
        Returns how many entries were new; identity/result conflicts fail closed."""
        added = 0
        for e in other.entries():
            before = len(self._entries)
            self._append(e)
            added += int(len(self._entries) > before)
        return added

    # --------------------------------------------------------------------- inspection
    def entries(self) -> list[dict]:
        return list(self._entries)

    def n_effective_trials(self) -> int:
        return len(self._entries)

    def ledger_hash(self) -> str:
        pairs = sorted((e["identity_sha256"], e["result_sha256"]) for e in self._entries)
        return hash_obj({"ledger_version": LEDGER_VERSION, "trials": pairs})

    # -------------------------------------------------------------------- persistence
    def save(self, path) -> None:
        payload = {"ledger_version": LEDGER_VERSION,
                   "n_effective_trials": self.n_effective_trials(),
                   "ledger_sha256": self.ledger_hash(),
                   "entries": self._entries}
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".ledger-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
                f.write("\n")
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path) -> "TrialLedger":
        with open(path) as f:
            payload = json.load(f)
        if payload.get("ledger_version") != LEDGER_VERSION:
            raise ValueError(f"unsupported ledger_version {payload.get('ledger_version')!r}")
        ledger = cls(sorted(payload["entries"], key=lambda e: e["seq"]))
        if ledger.ledger_hash() != payload.get("ledger_sha256"):
            raise ValueError("ledger file ledger_sha256 does not match its entries "
                             "(tampered or corrupted ledger)")
        return ledger
