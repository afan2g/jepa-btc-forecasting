"""Append-only G0-BN attempt ledger (`g0bn-ledger-v1`; issue #88, slice 67-B).

Implements spec section 4.2 (docs/superpowers/specs/2026-07-13-g0bn-protocol.md):
the separate G0-BN trial ledger stores each canonical `g0bn-trial-v1` identity plus
a hash-chained ordered execution-event history (starts, aborts, completions).

- Effective N counts unique canonical trial identities, not process executions; a
  completed or aborted-only identity counts exactly once. Any changed
  identity-bearing input (feature subset, parameter, seed, preprocessing, horizon,
  variant) is a DIFFERENT identity and increases N — recordable even though only
  the 15 base identities are eligible for v1 selection (eligibility is enforced at
  enumeration/selection, never here).
- An exact deterministic rerun is an idempotent execution event: it appends to the
  history but must reproduce the existing result hash and never increases N.
- A conflicting completed result for an existing identity fails closed and appends
  nothing (no failed or weak identity or event is ever overwritten or replaced).
- The ledger hashes BOTH the ordered event history (chain) and the canonical
  identity/result set (order-independent), and never imports G0-CB/G0-XV entries:
  there is no import path, and every stored identity must validate as
  `g0bn-trial-v1`, which structurally rejects legacy ledger records.
"""
from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import tempfile

from eval.g0bn_identity import trial_id as _trial_id
from eval.hashing import hash_obj

LEDGER_SCHEMA = "g0bn-ledger-v1"
EVENT_KINDS = ("started", "aborted", "completed")


def _require_strict_json(path: str, obj) -> None:
    """Fail closed on non-finite floats anywhere in a result payload: canonical JSON
    forbids NaN/Infinity, and silently coercing (the legacy ledger's _json_safe) would
    let two different in-memory results share one persisted hash."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"{path} keys must be non-empty strings; got {k!r}")
            _require_strict_json(f"{path}.{k}", v)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _require_strict_json(f"{path}[{i}]", v)
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"{path} must be finite (strict canonical JSON); got {obj!r}")
    elif obj is not None and not isinstance(obj, (bool, int, float, str)):
        raise ValueError(f"{path} must be JSON-encodable; got {type(obj).__name__}")


def _fsync_dir(path) -> None:
    fd = os.open(os.path.dirname(os.path.abspath(os.fspath(path))) or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class G0BNLedger:
    """Separate append-only G0-BN trial ledger.

    Bind a `path` to make the ledger DURABLE: every successful record_* call
    atomically rewrites and fsyncs the file before returning, so a process crash
    during a later candidate fit cannot erase an already-attempted identity from
    effective N (spec section 4.2 — the attempt history is append-only evidence,
    not best-effort state). A rejected call appends nothing and leaves the file
    byte-identical. `run_g0bn_development` refuses an unbound ledger; the unbound
    form exists for read-only reconciliation and unit tests.

    The first registered identity pins the protocol config: once any
    `g0bn-trial-v1` identity is registered, the v1 protocol config is immutable
    and any identity under a different `protocol_config_sha256` fails closed
    instead of registering as an ordinary new trial (spec section 3.1)."""

    def __init__(self, path=None):
        self._identities: dict[str, dict] = {}   # trial_id -> record, first-seen order
        self._events: list[dict] = []
        self._protocol_config_sha256: str | None = None
        self._path: str | None = None
        self._lock_fd: int | None = None
        if path is not None:
            path = os.fspath(path)
            if os.path.exists(path):
                raise ValueError(f"ledger file {path!r} already exists; resume it "
                                 "with G0BNLedger.load(path) — binding a fresh "
                                 "ledger over an existing history would fork it")
            self._acquire_lock(path)
            self._path = path

    @staticmethod
    def _open_lock(path: str) -> int:
        """Exclusive per-path writer lock, held for the ledger's bound lifetime.
        Without it, two live writers on one path would each autopersist their own
        stale in-memory history and the last os.replace() would silently discard
        the other's append-only events — a lost-update that undercounts effective
        N. Contention fails closed; there is no waiting or stealing."""
        fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ValueError(
                f"ledger {path!r} is write-locked by another live ledger instance "
                "(fcntl lock contention); concurrent writers would silently "
                "discard each other's events — close the other instance first, or "
                "load read-only with bind=False") from None
        return fd

    def _acquire_lock(self, path: str) -> None:
        self._lock_fd = self._open_lock(path)

    def close(self) -> None:
        """Release the writer lock and unbind. The in-memory history stays
        readable, but the instance is no longer durable and no longer persists."""
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None
        self._path = None

    def __del__(self):  # pragma: no cover - GC timing dependent
        try:
            self.close()
        except Exception:
            pass

    def is_durable(self) -> bool:
        return self._path is not None

    def protocol_config_sha256(self) -> str | None:
        """The immutable protocol config this ledger is pinned to (None until the
        first identity registers)."""
        return self._protocol_config_sha256

    def _persist(self) -> None:
        if self._path is not None:
            self.save(self._path)

    # ------------------------------------------------------------------ event plumbing
    def genesis_sha256(self) -> str:
        return hash_obj({"schema": LEDGER_SCHEMA, "chain": "genesis"})

    def _chain_head(self) -> str:
        return self._events[-1]["sha256"] if self._events else self.genesis_sha256()

    def _append_event(self, tid: str, kind: str, payload: dict,
                      recorded_at: str | None) -> dict:
        event = {
            "ordinal": len(self._events),
            "trial_id": tid,
            "event": kind,
            "payload": payload,
            "recorded_at": recorded_at,
            "prev_sha256": self._chain_head(),
        }
        event["sha256"] = hash_obj(event)
        self._events.append(event)
        return event

    def _register_identity(self, identity: dict) -> str:
        # Validation (exact g0bn-trial-v1 field set, embedded-hash consistency) runs
        # inside trial_id(); a legacy G0-CB/G0-XV identity shape fails here.
        tid = _trial_id(identity)
        config_sha = identity["protocol_config_sha256"]
        if self._protocol_config_sha256 is None:
            self._protocol_config_sha256 = config_sha
        elif config_sha != self._protocol_config_sha256:
            raise ValueError(
                f"protocol config drift fails closed (spec section 3.1): this ledger "
                f"is pinned to config {self._protocol_config_sha256[:12]}... by its "
                f"first registered trial, and the v1 config is immutable once any "
                f"identity is registered; got {config_sha[:12]}... — an edited "
                "config is a new protocol version, never a new trial")
        if tid not in self._identities:
            self._identities[tid] = {
                "trial_id": tid,
                "identity": copy.deepcopy(identity),
                "result": None,
                "result_sha256": None,
            }
        return tid

    # ---------------------------------------------------------------------- recording
    def record_start(self, identity: dict, *, recorded_at: str | None = None) -> str:
        """Record one execution start. First contact with a new identity registers it
        (it counts in effective N from this point, even if it later only aborts)."""
        tid = self._register_identity(identity)
        self._append_event(tid, "started", {}, recorded_at)
        self._persist()
        return tid

    def record_abort(self, identity: dict, *, error: str,
                     recorded_at: str | None = None) -> str:
        """Record an infrastructure/execution abort. An aborted-only unique identity
        still counts once in effective N (spec section 4.2) but supplies no result."""
        if not isinstance(error, str) or not error.strip():
            raise ValueError("abort error must be a non-empty string describing the "
                             "failure (aborted events are permanent evidence)")
        tid = self._register_identity(identity)
        self._append_event(tid, "aborted", {"error": error}, recorded_at)
        self._persist()
        return tid

    def record_completion(self, identity: dict, result: dict, *,
                          recorded_at: str | None = None) -> str:
        """Record a completed execution. The first completion pins the identity's
        immutable result; every later completion must reproduce the same result hash
        (idempotent deterministic rerun, or retry after an abort with every
        identity-bearing input unchanged — the canonical trial_id guarantees that).
        A different result for the same identity fails closed without appending."""
        if not isinstance(result, dict):
            raise ValueError(f"result must be a dict; got {type(result).__name__}")
        _require_strict_json("result", result)
        result = copy.deepcopy(result)
        rh = hash_obj(result)
        tid = self._register_identity(identity)
        rec = self._identities[tid]
        if rec["result_sha256"] is not None and rec["result_sha256"] != rh:
            raise ValueError(
                f"trial {tid[:12]}... already completed with a DIFFERENT result "
                f"({rec['result_sha256'][:12]}... != {rh[:12]}...); an exact rerun of "
                "the same canonical identity must be deterministic — refusing to "
                "overwrite or fork the append-only trial history")
        self._append_event(tid, "completed", {"result_sha256": rh}, recorded_at)
        if rec["result_sha256"] is None:
            rec["result"] = result
            rec["result_sha256"] = rh
        self._persist()
        return tid

    # --------------------------------------------------------------------- inspection
    def trial_ids(self) -> list[str]:
        return list(self._identities)

    def identity_for(self, tid: str) -> dict:
        return copy.deepcopy(self._identities[tid]["identity"])

    def result_for(self, tid: str):
        rec = self._identities[tid]
        return None if rec["result"] is None else copy.deepcopy(rec["result"])

    def result_sha256_for(self, tid: str):
        return self._identities[tid]["result_sha256"]

    def scored_trial_ids(self) -> list[str]:
        """Successfully scored unique identities (a completed result exists), in
        first-registration order."""
        return [tid for tid, rec in self._identities.items()
                if rec["result_sha256"] is not None]

    def events(self) -> list[dict]:
        return copy.deepcopy(self._events)

    def n_effective_trials(self) -> int:
        """Effective N: unique canonical trial identities, including aborted-only
        ones. Execution events never change this count."""
        return len(self._identities)

    # ------------------------------------------------------------------------ hashing
    def history_sha256(self) -> str:
        """Chain head of the ordered execution-event history."""
        return self._chain_head()

    def identity_set_sha256(self) -> str:
        """Order-independent canonical identity/result-set hash (sorted by trial_id),
        so identical trial SETS pin identically regardless of registration order."""
        pairs = sorted((tid, rec["result_sha256"])
                       for tid, rec in self._identities.items())
        return hash_obj({"schema": LEDGER_SCHEMA, "trials": [list(p) for p in pairs]})

    def ledger_sha256(self) -> str:
        return hash_obj({
            "schema": LEDGER_SCHEMA,
            "n_effective_trials": self.n_effective_trials(),
            "identity_set_sha256": self.identity_set_sha256(),
            "history_sha256": self.history_sha256(),
        })

    # -------------------------------------------------------------------- persistence
    def save(self, path) -> None:
        """Atomic fsynced write. The target path's writer lock is mandatory: our
        own bound path is already held for our lifetime; any other target is
        locked transiently, so a stale inspection snapshot (or any unbound
        instance) can never replace a live writer's file and silently delete
        append-only events."""
        path_str = os.fspath(path)
        transient_fd = None
        if self._path is None or (os.path.abspath(path_str)
                                  != os.path.abspath(self._path)):
            transient_fd = self._open_lock(path_str)
        try:
            self._write_file(path_str)
        finally:
            if transient_fd is not None:
                fcntl.flock(transient_fd, fcntl.LOCK_UN)
                os.close(transient_fd)

    def _write_file(self, path: str) -> None:
        payload = {
            "schema": LEDGER_SCHEMA,
            "identities": list(self._identities.values()),
            "events": self._events,
            "n_effective_trials": self.n_effective_trials(),
            "identity_set_sha256": self.identity_set_sha256(),
            "history_sha256": self.history_sha256(),
            "ledger_sha256": self.ledger_sha256(),
        }
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".g0bn-ledger-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            _fsync_dir(path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path, *, bind: bool = True) -> "G0BNLedger":
        """Load and fully verify a ledger file.

        `bind=True` (default) RESUMES the ledger: the exclusive writer lock is
        acquired BEFORE the file is read (a consistent snapshot — no
        check-then-write race with a live writer) and every later record_* call
        keeps persisting to the same file. `bind=False` is read-only inspection:
        no lock, not durable, never persists."""
        path_str = os.fspath(path)
        lock_fd = cls._open_lock(path_str) if bind else None
        try:
            ledger = cls._parse_file(path_str)
        except BaseException:
            if lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            raise
        if bind:
            ledger._lock_fd = lock_fd
            ledger._path = path_str
        return ledger

    @classmethod
    def _parse_file(cls, path: str) -> "G0BNLedger":
        with open(path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or payload.get("schema") != LEDGER_SCHEMA:
            raise ValueError(f"unsupported ledger schema {payload.get('schema')!r} "
                             f"(expected {LEDGER_SCHEMA!r}); G0-BN never loads or "
                             "imports a legacy G0-CB/G0-XV ledger")
        ledger = cls()
        for rec in payload.get("identities", []):
            # Re-validates every identity as g0bn-trial-v1 and re-derives its id, so
            # tampered identities and legacy entry shapes both fail closed.
            before = len(ledger._identities)
            tid = ledger._register_identity(rec["identity"])
            if len(ledger._identities) == before:
                raise ValueError(f"ledger file contains duplicate identity records "
                                 f"for trial {tid[:12]}... (a crafted duplicate could "
                                 "silently supersede an earlier result)")
            if tid != rec.get("trial_id"):
                raise ValueError(f"ledger identity record claims trial_id "
                                 f"{rec.get('trial_id')!r} but its identity hashes to "
                                 f"{tid} (tampered or corrupted ledger)")
            if rec.get("result") is not None:
                _require_strict_json("result", rec["result"])
                if hash_obj(rec["result"]) != rec.get("result_sha256"):
                    raise ValueError("ledger result_sha256 does not match its result "
                                     "(tampered or corrupted ledger)")
                stored = ledger._identities[tid]
                stored["result"] = copy.deepcopy(rec["result"])
                stored["result_sha256"] = rec["result_sha256"]
            elif rec.get("result_sha256") is not None:
                raise ValueError("ledger identity record carries a result_sha256 "
                                 "without its result (tampered or corrupted ledger)")
        prev = ledger.genesis_sha256()
        completed_tids = set()
        event_tids = set()
        for i, event in enumerate(payload.get("events", [])):
            body = {k: v for k, v in event.items() if k != "sha256"}
            expected = dict(body)
            if (event.get("ordinal") != i or event.get("prev_sha256") != prev
                    or event.get("event") not in EVENT_KINDS
                    or event.get("trial_id") not in ledger._identities
                    or hash_obj(expected) != event.get("sha256")):
                raise ValueError(f"ledger event {i} breaks the hash chain or "
                                 "references an unknown trial (tampered or corrupted "
                                 "ledger)")
            if event["event"] == "completed":
                rec = ledger._identities[event["trial_id"]]
                pinned = event["payload"].get("result_sha256")
                # A completion without a pinned non-null result hash is impossible
                # evidence: None == None must not let a crafted completed event
                # pass against a nulled identity record.
                if (not isinstance(pinned, str)
                        or rec["result_sha256"] is None
                        or pinned != rec["result_sha256"]):
                    raise ValueError(f"ledger completed event {i} does not pin the "
                                     "identity's immutable non-null result hash "
                                     "(tampered or corrupted ledger)")
                completed_tids.add(event["trial_id"])
            event_tids.add(event["trial_id"])
            ledger._events.append(copy.deepcopy(event))
            prev = event["sha256"]
        # Every registration path appends an event, so an identity record with no
        # event at all is fabricated: it would inflate effective N (and therefore
        # the DSR n_trials benchmark) without any append-only execution evidence.
        eventless = [tid for tid in ledger._identities if tid not in event_tids]
        if eventless:
            raise ValueError(f"ledger identity records with no execution event in "
                             f"the history: {[t[:12] for t in eventless[:3]]} "
                             "(tampered or corrupted ledger)")
        # Every pinned result must be witnessed by at least one completed event in
        # the chained history: a crafted identity record carrying a result that no
        # execution event ever produced is a fabricated outcome, not a record.
        unwitnessed = [tid for tid, rec in ledger._identities.items()
                       if rec["result_sha256"] is not None
                       and tid not in completed_tids]
        if unwitnessed:
            raise ValueError(f"ledger identity records carry results with no "
                             f"completed event in the history: "
                             f"{[t[:12] for t in unwitnessed[:3]]} (tampered or "
                             "corrupted ledger)")
        for field in ("n_effective_trials", "identity_set_sha256", "history_sha256",
                      "ledger_sha256"):
            recomputed = getattr(ledger, field)() if field != "n_effective_trials" \
                else ledger.n_effective_trials()
            if payload.get(field) != recomputed:
                raise ValueError(f"ledger file {field} does not match its entries "
                                 "(tampered or corrupted ledger)")
        return ledger
