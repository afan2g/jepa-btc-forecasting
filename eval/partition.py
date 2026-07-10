"""Hash-pinned partition contract for the staged G0 pilot (issue #52; staged protocol §3).

The producer (issue #37) partitions the pilot by SUPPORT SPAN, not by t_event calendar
date: a development row exists only if its complete guarded forward support ends before
the holdout boundary, and it records the bounds, horizon map, guard, rule version, and
per-horizon boundary-drop counts in this contract artifact. The evaluator NEVER trusts a
matrix to be span-safe: it re-validates every row against the pinned contract before any
fit/CPCV/PBO (fail closed), and it re-checks the conservative prefilter itself so an
early-resolving barrier (small t_barrier) cannot smuggle a boundary row past the rule —
deciding that row would have required opening the holdout partition, which development
must never do. The symmetric rule at the holdout's far edge keeps April labels out of May.
"""
from __future__ import annotations

import json

import pandas as pd

from eval.hashing import hash_obj

PARTITION_CONTRACT_VERSION = 1

# The one prefilter rule these evaluator semantics implement. A contract claiming any
# other rule string is a different (unsupported) partitioning discipline -> fail closed.
PREFILTER_RULE = "t_event + horizons[horizon] + guard_ns < boundary_ns"

REQUIRED_FIELDS = (
    "partition_contract_version", "rule_version", "prefilter_rule",
    "dev_start_ns", "holdout_start_ns", "holdout_end_ns", "guard_ns",
    "horizons", "boundary_drop_counts",
)
OPTIONAL_FIELDS = ("generated_at",)
PARTITIONS = ("development", "holdout")

# The manifest pins its contract via a sources entry (v1 manifest source dicts accept
# extra keys, so no manifest schema change is needed — backward compatible with G1).
BINDING_SOURCE_NAME = "partition_contract"


def _ns_int(name: str, val, *, minimum: int = 0) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise ValueError(f"partition contract: {name} must be an int >= {minimum} (nanoseconds)")
    return val


def _drop_counts(name: str, val, horizons: dict) -> dict:
    if not isinstance(val, dict):
        raise ValueError(f"partition contract: {name} must be a dict of horizon tag -> count")
    if set(val) != set(horizons):
        raise ValueError(f"partition contract: {name} keys {sorted(val)} must exactly match "
                         f"declared horizons {sorted(horizons)}")
    for tag, n in val.items():
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"partition contract: {name}[{tag!r}] must be an int >= 0")
    return val


def validate_partition_contract(contract: dict) -> dict:
    """Schema-level validation (no data); fail closed on anything unknown or ambiguous."""
    if not isinstance(contract, dict):
        raise ValueError("partition contract must be a dict")
    missing = [k for k in REQUIRED_FIELDS if k not in contract]
    if missing:
        raise ValueError(f"partition contract missing required fields: {missing}")
    unknown = set(contract) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise ValueError(f"unknown partition contract keys (misspelled?): {sorted(unknown)}")
    v = contract["partition_contract_version"]
    if isinstance(v, bool) or not isinstance(v, int) or v != PARTITION_CONTRACT_VERSION:
        raise ValueError(f"unsupported partition_contract_version {v!r}; "
                         f"this code supports {PARTITION_CONTRACT_VERSION}")
    if not isinstance(contract["rule_version"], str) or not contract["rule_version"]:
        raise ValueError("partition contract: rule_version must be a non-empty string")
    if contract["prefilter_rule"] != PREFILTER_RULE:
        raise ValueError(f"partition contract: unsupported prefilter_rule "
                         f"{contract['prefilter_rule']!r}; this evaluator implements "
                         f"{PREFILTER_RULE!r} and fails closed on anything else")
    dev = _ns_int("dev_start_ns", contract["dev_start_ns"])
    h0 = _ns_int("holdout_start_ns", contract["holdout_start_ns"])
    h1 = _ns_int("holdout_end_ns", contract["holdout_end_ns"])
    if not dev < h0 < h1:
        raise ValueError("partition contract: require dev_start_ns < holdout_start_ns "
                         "< holdout_end_ns")
    _ns_int("guard_ns", contract["guard_ns"])
    horizons = contract["horizons"]
    if (not isinstance(horizons, dict) or not horizons
            or any(not isinstance(t, str) or not t for t in horizons)
            or any(isinstance(ns, bool) or not isinstance(ns, int) or ns <= 0
                   for ns in horizons.values())):
        raise ValueError("partition contract: horizons must be a non-empty mapping of "
                         "tag -> positive int nanoseconds")
    counts = contract["boundary_drop_counts"]
    if not isinstance(counts, dict) or set(counts) != set(PARTITIONS):
        raise ValueError(f"partition contract: boundary_drop_counts must have exactly "
                         f"the partitions {PARTITIONS}")
    for part in PARTITIONS:
        _drop_counts(f"boundary_drop_counts[{part!r}]", counts[part], horizons)
    return contract


def contract_hash(contract: dict) -> str:
    """Deterministic content hash; generated_at (the only volatile field) is excluded so
    identical rebuilds pin identically."""
    validate_partition_contract(contract)
    return hash_obj(contract, exclude_keys=("generated_at",))


def load_partition_contract(path) -> dict:
    with open(path) as f:
        contract = json.load(f)
    return validate_partition_contract(contract)


# --------------------------------------------------------------------- manifest binding
def contract_binding(manifest: dict) -> dict:
    """The manifest's partition-contract binding: exactly one sources entry named
    `partition_contract`, carrying the pinned contract sha256, this build's partition,
    and the per-horizon boundary-drop counts the build reconciled against."""
    entries = [s for s in manifest.get("sources", [])
               if isinstance(s, dict) and s.get("name") == BINDING_SOURCE_NAME]
    if len(entries) != 1:
        raise ValueError(f"manifest must pin exactly one {BINDING_SOURCE_NAME!r} sources "
                         f"entry; found {len(entries)}")
    b = entries[0]
    sha = b.get("sha256")
    if not isinstance(sha, str) or not sha:
        raise ValueError("partition-contract binding must carry a non-empty 'sha256'")
    if b.get("partition") not in PARTITIONS:
        raise ValueError(f"partition-contract binding 'partition' must be one of "
                         f"{PARTITIONS}, got {b.get('partition')!r}")
    if not isinstance(b.get("boundary_drop_counts"), dict):
        raise ValueError("partition-contract binding must echo this build's per-horizon "
                         "'boundary_drop_counts' (reconciled against the contract)")
    return b


def require_binding(manifest: dict, contract: dict, expected_partition: str) -> dict:
    """Fail-closed reconciliation of a build manifest against the pinned contract:
    the binding must name the expected partition, pin the exact contract hash, echo the
    contract's drop counts for that partition, and declare only contract horizons."""
    if expected_partition not in PARTITIONS:
        raise ValueError(f"expected_partition must be one of {PARTITIONS}")
    b = contract_binding(manifest)
    if b["partition"] != expected_partition:
        raise ValueError(f"manifest is bound to partition {b['partition']!r}; this path "
                         f"accepts only {expected_partition!r} builds")
    actual = contract_hash(contract)
    if b["sha256"] != actual:
        raise ValueError(f"manifest pins partition contract {b['sha256'][:12]}..., but the "
                         f"supplied contract hashes to {actual[:12]}...; refusing a "
                         "stale/substituted contract")
    expected_counts = contract["boundary_drop_counts"][expected_partition]
    if b["boundary_drop_counts"] != expected_counts:
        raise ValueError(f"manifest boundary_drop_counts {b['boundary_drop_counts']} do not "
                         f"reconcile to the contract's {expected_partition} counts "
                         f"{expected_counts}")
    bad_h = {t: ns for t, ns in manifest["horizons"].items()
             if contract["horizons"].get(t) != ns}
    if bad_h:
        raise ValueError(f"manifest horizons not declared identically in the partition "
                         f"contract: {bad_h}")
    return b


# ------------------------------------------------------------------------ span validation
_I64_MIN = -(2**63) + 1


def _span_violations(matrix: pd.DataFrame, contract: dict, *, lo_ns: int,
                     boundary_ns: int) -> dict:
    """Per-horizon violation counts of the span rules against [lo_ns, boundary_ns):
    (a) t_event before the partition start; (b) the conservative prefilter
    t_event + horizon + guard >= boundary (checked INDEPENDENTLY of t_barrier, so an early
    barrier cannot bypass it); (c) the actual guarded label span t_barrier + guard >=
    boundary. Returns {tag: {"before_start": n, "prefilter": n, "actual_span": n}}.

    The thresholds are computed in PYTHON ints and the comparisons rearranged so no array
    arithmetic can overflow: numpy int64 addition wraps SILENTLY, so a schema-valid
    contract with a huge guard/horizon (or a garbage t_event near the int64 max) under
    `te + horizon + guard >= boundary` would wrap negative and ADMIT holdout rows. With
    `te >= boundary - horizon - guard` nothing overflows; a threshold clamped up to the
    int64 floor means nothing can be span-safe, which correctly counts every row."""
    guard = contract["guard_ns"]
    horizons = contract["horizons"]
    undeclared = sorted(set(matrix["horizon"].unique()) - set(horizons))
    if undeclared:
        raise ValueError(f"matrix horizon tags not declared in the partition contract: "
                         f"{undeclared}")
    out = {}
    for tag, sub in matrix.groupby("horizon", observed=True):
        te = sub["t_event"].to_numpy()
        tb = sub["t_barrier"].to_numpy()
        pre_thresh = max(int(boundary_ns) - int(horizons[tag]) - int(guard), _I64_MIN)
        act_thresh = max(int(boundary_ns) - int(guard), _I64_MIN)
        v = {
            "before_start": int((te < lo_ns).sum()),
            "prefilter": int((te >= pre_thresh).sum()),
            "actual_span": int((tb >= act_thresh).sum()),
        }
        if any(v.values()):
            out[str(tag)] = v
    return out


def validate_development_span(matrix: pd.DataFrame, contract: dict) -> None:
    """Reject a development artifact unless EVERY row's complete guarded support stays
    strictly before the holdout boundary (staged protocol §3). Fail closed with per-horizon
    counts; the evaluator never silently drops rows — a violating build goes back to the
    producer."""
    validate_partition_contract(contract)
    viol = _span_violations(matrix, contract, lo_ns=contract["dev_start_ns"],
                            boundary_ns=contract["holdout_start_ns"])
    if viol:
        raise ValueError(
            "development rows violate the span-safe partition rule "
            f"(guard_ns={contract['guard_ns']}, holdout_start_ns="
            f"{contract['holdout_start_ns']}): {viol}; the conservative prefilter "
            "(t_event + horizon + guard < holdout_start) binds regardless of t_barrier")


def validate_holdout_span(matrix: pd.DataFrame, contract: dict) -> None:
    """The symmetric future-boundary rule for the fixed holdout: every holdout row starts
    at/after holdout_start_ns and its complete guarded support ends strictly before
    holdout_end_ns (April labels must not silently consume May)."""
    validate_partition_contract(contract)
    viol = _span_violations(matrix, contract, lo_ns=contract["holdout_start_ns"],
                            boundary_ns=contract["holdout_end_ns"])
    if viol:
        raise ValueError(
            "holdout rows violate the span-safe partition rule (guard_ns="
            f"{contract['guard_ns']}, holdout_end_ns={contract['holdout_end_ns']}): {viol}")
