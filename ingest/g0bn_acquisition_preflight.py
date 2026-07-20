"""Deterministic, no-I/O G0-BN acquisition and custody preflight (issue #102, 68-A).

The preflight #68 must pass before it may request the bounded Crypto Lake
Binance BTC-USDT-perpetual dataset: it plans exactly the 92-day window
2025-11-01..2026-01-31 for `BINANCE_FUTURES`/`BTC-USDT-PERP` and only the
#64-certified raw products (`book`, `book_delta_v2`, `trades`), splits the 61
development days from the 31 sealed January holdout days into DISJOINT
raw/normalized destinations, reconciles units/bytes/quota/disk/retry/resume/
locking against the real Stage-1 downloader, and emits an INERT human approval
packet plus the custodian/operator handoff contract.

NO vendor I/O, ever. This module never directly imports boto3/lakeapi/pyarrow
or any network client, never constructs a vendor session, and never opens raw
data. (pyarrow loads transitively through the reused eval.* validation code but
no vendor-download SDK or network client ever does — the network-isolation
tests pin exactly that.)
Every byte/cost figure is an EX-ANTE projection — a pinned per-product constant
(`ingest.lake_binance.LAKE_GB_PER_DAY`, measured/derived from the 2026-04-01
certification evidence, docs/data.md section 6) times a day count. Measured
January byte sizes and record counts are activity proxies (spec section 5.1)
and are structurally impossible here: every artifact is validated with an exact
field set, so there is nowhere to smuggle one in. The commands in the approval
packet stay inert until a human approves them on issue #68; this preflight
replaces the full-archive planner projections that
docs/superpowers/plans/2026-07-10-staged-signal-acquisition.md section 6
forbids #68 from reusing.

Custody plane (spec section 5.1): the January holdout is owned and sealed by a
custodian identity distinct from the experiment operator. This module defines
the effective ACL/IAM/bucket-policy evidence contract
(`g0bn-permission-policy-evidence-v1` — developer-run `chmod` is rejected as
custody), emits the outcome-blind custodian inventory in exactly the 8-field
shape `eval.g0bn_freeze.validate_custody_inventory` consumes, and seals it with
the REUSED `g0bn-custodian-seal-content-v1` commitment
(`eval.g0bn_freeze.custodian_seal_content_sha256`). Validation is delegated to
`validate_custody_inventory` — never duplicated or weakened; the handoff
validator only ADDS the evidence<->inventory couplings the freeze module leaves
to #68.

Boundaries owned elsewhere: #68 (human-approved execution of the packet
commands), #93 (native<->normalized product-name reconciliation before the
custodian seals normalized objects), #94/T9 (materialization), #69 (the
one-shot transaction and its raw/matrix access burns).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import re
import sys

# Repo root on sys.path so `python ingest/g0bn_acquisition_preflight.py` works like the
# sibling downloader; `python -m ingest.g0bn_acquisition_preflight` needs no help.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import-safe by construction: ingest.lake_binance / ingest.download_lake_binance keep
# every vendor dependency (boto3/lakeapi/pyarrow) out of module scope, and the eval.*
# modules are local validation/hashing code. The network-isolation tests pin this.
from eval.g0bn_config import (  # noqa: E402
    CERTIFIED_L2_DELTA_PRODUCT,
    CERTIFIED_L2_SNAPSHOT_PRODUCT,
    CERTIFIED_PROVIDER,
    CERTIFIED_TRADE_PRODUCT,
    DEV_DAYS,
    INSTRUMENT,
    PILOT_ID,
    PROTOCOL_ID,
    _day,
    _dict,
    _exact,
    _fail,
    _sha256,
    _str,
    _validate_generated_at,
    g0bn_artifact_sha256,
)
from eval.g0bn_freeze import (  # noqa: E402
    HOLDOUT_DAYS,
    OBJECT_LAYERS,
    SEAL_CONTENT_SCHEMA,
    _OBJECT_FIELDS,
    custodian_seal_content_sha256,
    validate_custody_inventory,
)
from eval.g0bn_identity import G0BN_HOLDOUT_UNIVERSE_ID, G0BN_TRANSACTION_ID  # noqa: E402
from eval.hashing import hash_obj  # noqa: E402
from eval.writer import G0BN_DATA_SOURCES  # noqa: E402
from ingest import download_lake_binance as dl  # noqa: E402
from ingest import lake_binance as lb  # noqa: E402

ACQUISITION_PLAN_SCHEMA = "g0bn-acquisition-plan-v1"
APPROVAL_PACKET_SCHEMA = "g0bn-acquisition-approval-packet-v1"
PERMISSION_EVIDENCE_SCHEMA = "g0bn-permission-policy-evidence-v1"

INSTRUMENT_KEY = "binance-perp"
EXCHANGE = INSTRUMENT["exchange"]      # BINANCE_FUTURES
SYMBOL = INSTRUMENT["symbol"]          # BTC-USDT-PERP

# Exactly the three #64-certified raw products, in the canonical (alphabetical)
# product order the sealed-allowlist sort uses. `book` is the Stage-2 seed the
# downloader schedules automatically whenever book_delta_v2 is selected, so the
# downloader is invoked with FEEDS_ARGUMENT and transfers exactly PRODUCTS.
PRODUCTS = tuple(sorted((CERTIFIED_L2_SNAPSHOT_PRODUCT, CERTIFIED_L2_DELTA_PRODUCT,
                         CERTIFIED_TRADE_PRODUCT)))
FEEDS_ARGUMENT = "book_delta_v2,trades"
FALLBACK_POLICY = "none"               # #64: lake_go, fallback none (docs/data.md §3)

# Disjoint per-partition stores, all under git-ignored roots. The holdout stores
# are the future CUSTODIAN-owned destinations; provisioning real custodian-owned
# storage is an explicit unresolved assumption in the approval packet.
DEV_RAW_ROOT = "data/raw/lake_g0bn_dev"
HOLDOUT_RAW_ROOT = "data/raw/lake_g0bn_holdout"
DEV_NORMALIZED_ROOT = "data/processed/g0bn_dev"
HOLDOUT_NORMALIZED_ROOT = "data/processed/g0bn_holdout"
DEV_REPORT_ROOT = "data/reports/g0bn_acquisition/development"
HOLDOUT_REPORT_ROOT = "data/reports/g0bn_acquisition/holdout"
DEFAULT_OUT_DIR = "data/reports/g0bn_acquisition_preflight"
DEV_DAYS_FILE = DEFAULT_OUT_DIR + "/dev_days.txt"
HOLDOUT_DAYS_FILE = DEFAULT_OUT_DIR + "/holdout_days.txt"

EXPENSIVE_COMPUTE_LOCK = "/tmp/jepa-expensive-compute.lock"
LOCK_WRAPPER = f"flock -w 14400 {EXPENSIVE_COMPUTE_LOCK}"

# Per-command --max-gb cap: the ceiling of the partition estimate with 10% slack.
# If the downloader's own pre-transfer estimate drifts above the cap it refuses
# (exit 4) — the cap is a stop condition, not a target.
CAP_SAFETY_FACTOR = 1.1
# Free-disk requirement: raw estimate with 25% slack (the trades/book constants
# are derived, not measured) plus a flat allowance for the Stage-2 processed
# stores (topk_l2 measured well under 0.15 GB/day on the certification day).
RAW_DISK_SAFETY_FACTOR = 1.25
NORMALIZED_DISK_ALLOWANCE_GB = 20.0

SETUP_ERROR_EXIT = 2       # bad inputs / failed validation (mirrors the downloader)
PREFLIGHT_GATE_EXIT = 4    # resource-gate refusal (mirrors the broad-pull gate)

# The config's oos.raw_access_boundary spelling (spec section 5.1): the operator
# may not open a January payload or Parquet footer before the #69 raw claim.
RELEASE_BOUNDARY = "before_first_january_source_or_footer_read_v1"

NO_VENDOR_IO_STATEMENT = (
    "This packet is inert. Generating it performed no vendor I/O — no Crypto "
    "Lake, S3, HTTP, or SDK call — and none of the listed commands may run "
    "before explicit human approval on issue #68."
)

# Ex-ante estimate provenance and quota assumption, pinned as constants so the
# validator can reject a rewritten disclosure the same way it rejects a
# rewritten number (the approver reads these verbatim).
ESTIMATE_BASIS = (
    "ingest.lake_binance.LAKE_GB_PER_DAY (docs/data.md section 6): "
    "book_delta_v2 573.8 MB/day MEASURED on the 2026-04-01 certification day; "
    "trades and the book seed are DERIVED conservative constants — never "
    "measured January values"
)
QUOTA_ASSUMPTION = (
    "no other Crypto Lake pull shares this monthly quota window; the operator "
    "re-checks lakeapi.used_data immediately before execution (an "
    "approval-gated live call that is NOT part of this preflight)"
)

# Custody infrastructure this preflight can DEFINE but not create. The packet
# carries these verbatim; the #102 PR reports them as unresolved.
UNRESOLVED_CUSTODY_ASSUMPTIONS = (
    "A custodian OS/service identity distinct from the experiment operator must "
    "be provisioned to own the holdout destinations exclusively before the "
    "holdout download runs; it does not exist yet.",
    "Effective ACL/IAM/bucket-policy evidence (g0bn-permission-policy-evidence-v1) "
    "must be captured from the real storage layer and hashed into "
    "permission_policy_sha256; developer-run chmod on developer-owned files is "
    "not independent custody.",
    "The native-to-normalized product-name reconciliation (#93) must fix how "
    "Stage-2 outputs map onto the sealed normalized products "
    "(binance_futures_l2_snapshot/_l2_delta/_trades) before the custodian seals "
    "normalized objects.",
    "The coverage artifact hashed into coverage_sha256 must be custodian-produced, "
    "outcome-blind day-level coverage/continuity metadata only (no byte sizes, "
    "no record counts).",
    "Sealed-object checksums come from the write-time store manifests; deriving "
    "them by re-reading January payloads or Parquet footers is forbidden for the "
    "operator identity.",
    "The permission-policy evidence's resource scope must cover the EXACT "
    "holdout destinations — the raw and normalized stores, their "
    "_manifest.jsonl files, AND the holdout report root — because Stage-1/"
    "Stage-2 run reports and store manifests record January rows and byte "
    "sizes (activity proxies) that must stay operator-read-denied inside "
    "custody until the #69 raw-access burn. The operator must also be denied "
    "policy mutation, or the captured evidence cannot bind future access.",
)

_PLAN_FIELDS = (
    "schema", "protocol_id", "pilot_id", "holdout_universe_id", "transaction_id",
    "provider", "fallback_policy", "instrument", "instrument_key",
    "products", "feeds_argument",
    "window", "development", "holdout", "units", "n_units",
    "estimates", "quota", "disk", "execution", "custody",
    "generated_at", "sha256",
)

_PARTITION_FIELDS = (
    "partition", "days", "n_days", "n_units", "raw_root", "normalized_root",
    "raw_manifest", "normalized_manifest", "report_root", "days_file",
    "days_file_sha256", "estimated_gb",
)

_UNIT_FIELDS = ("partition", "day", "product", "raw_path")

_ESTIMATE_FIELDS = ("gb_per_day_by_product", "gb_per_day", "total_gb", "basis",
                    "ex_ante_rule")

_QUOTA_FIELDS = ("quota_gb_per_month", "headroom_gb", "safe_window_gb",
                 "fits_single_window", "assumption")

_DISK_FIELDS = ("raw_safety_factor", "normalized_allowance_gb", "required_free_gb")

_EXECUTION_FIELDS = (
    "downloader", "recon_runner", "retries", "backoff_base_s", "backoff_cap_s",
    "atomic_write_rule", "resume_rule", "quota_gate_rule",
    "expensive_compute_lock", "jobs",
)

_CUSTODY_FIELDS = (
    "custodian_identity", "operator_identity", "raw_products",
    "normalized_products", "holdout_custody_scope", "seal_content_schema",
    "permission_evidence_schema", "release_boundary", "local_chmod_is_custody",
    "unresolved_assumptions",
)

_EVIDENCE_FIELDS = (
    "schema", "mechanism", "custodian_identity", "operator_identity", "resource",
    "covered_paths", "custodian_owns_objects", "operator_payload_read_denied",
    "operator_footer_read_denied", "operator_storage_listing_denied",
    "operator_policy_write_denied", "effective_policy_capture", "captured_at",
)

_CAPTURE_FIELDS = ("method", "command", "policy_document_sha256")

_PACKET_FIELDS = (
    "schema", "protocol_id", "pilot_id", "holdout_universe_id", "transaction_id",
    "plan_sha256", "inert", "approval_authority", "window", "request_totals",
    "caps", "vendor_cost", "commands", "stop_conditions", "custody",
    "no_vendor_io_statement", "generated_at", "sha256",
)

# Effective-permission mechanisms that ARE independent custody (a policy layer
# the operator cannot rewrite) vs the developer-local fiddling spec section 5.1
# explicitly rejects. Anything not on the first list fails closed.
INDEPENDENT_CUSTODY_MECHANISMS = ("aws_s3_bucket_policy", "aws_iam_policy",
                                  "distinct_os_user_acl")
LOCAL_ONLY_MECHANISMS = ("chmod", "posix_mode_bits", "umask", "developer_copy")

# Identities are interpolated into the packet's copy-paste shell commands and
# into canonical hashes, so they must be shell-inert single tokens.
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*\Z")


def _identity(path: str, value) -> str:
    _str(path, value)
    if not _IDENTITY_RE.match(value):
        _fail(path, f"must be a single shell-inert token matching "
                    f"[A-Za-z0-9][A-Za-z0-9._@-]* (it is interpolated into "
                    f"approved commands and canonical hashes); got {value!r}")
    return value


# ------------------------------------------------------------------ destination safety
# Planned paths are embedded UNQUOTED in the packet's human-approved shell
# commands, so every /-segment must be a shell-inert token — this subsumes
# whitespace and rejects `;`, `$`, backticks, quotes, globs, and '.'/'..'.
_PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _safe_repo_data_root(path: str, value, *, required_prefix: str) -> str:
    """A destination is safe only as a normalized RELATIVE path under one of the
    git-ignored data roots — no absolute paths, no '..' escapes, no aliasing,
    and only shell-inert segment tokens."""
    _str(path, value)
    if os.path.isabs(value) or "\\" in value:
        _fail(path, f"must be a relative repo path under {required_prefix}/; "
                    f"got {value!r}")
    if not all(_PATH_SEGMENT_RE.match(part) for part in value.split("/")):
        _fail(path, f"every path segment must be a shell-inert token matching "
                    f"[A-Za-z0-9][A-Za-z0-9._-]* (paths are interpolated into "
                    f"approved shell commands); got {value!r}")
    if posixpath.normpath(value) != value or \
            any(part in ("..", ".", "") for part in value.split("/")):
        _fail(path, f"must be a normalized relative path under {required_prefix}/ "
                    f"(no '..', '.', or empty segments); got {value!r}")
    if not value.startswith(required_prefix + "/"):
        _fail(path, f"must live under the git-ignored {required_prefix}/ root; "
                    f"got {value!r}")
    return value


def _require_disjoint(roots: dict) -> None:
    """No destination may equal, contain, or live inside another: development and
    sealed-holdout stores that nest or alias would let one identity's run write
    into (or read from) the other's custody scope."""
    items = sorted(roots.items())
    for i, (name_a, a) in enumerate(items):
        for name_b, b in items[i + 1:]:
            if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                _fail("destinations",
                      f"{name_a} ({a!r}) and {name_b} ({b!r}) must be disjoint "
                      "(development and sealed-holdout destinations may never "
                      "nest or alias)")


# ------------------------------------------------------------------- plan construction
def _feeds() -> list:
    return FEEDS_ARGUMENT.split(",")


def _days_file_content(days) -> str:
    return "".join(f"{d}\n" for d in days)


def _days_file_sha256(days) -> str:
    """Content commitment for a --days-file: the file is execution-time state
    (its CONTENT drives what a later run would transfer), so the plan pins the
    canonical content hash and the packet instructs verifying it before any
    download runs."""
    return hashlib.sha256(_days_file_content(days).encode()).hexdigest()


def _partition_block(name: str, days, raw_root: str, normalized_root: str,
                     report_root: str, days_file: str) -> dict:
    return {
        "partition": name,
        "days": list(days),
        "n_days": len(days),
        "n_units": len(days) * len(PRODUCTS),
        "raw_root": raw_root,
        "normalized_root": normalized_root,
        "raw_manifest": raw_root + "/" + lb.MANIFEST_NAME,
        "normalized_manifest": normalized_root + "/" + lb.MANIFEST_NAME,
        "report_root": report_root,
        "days_file": days_file,
        "days_file_sha256": _days_file_sha256(days),
        "estimated_gb": lb.estimate_gb(INSTRUMENT_KEY, _feeds(), len(days)),
    }


def _units(dev_block: dict, holdout_block: dict) -> list:
    """The full deterministic unit list in canonical (day, product) order —
    chronological because the development days all precede the holdout days and
    PRODUCTS is the canonical product sort."""
    units = []
    for block in (dev_block, holdout_block):
        for day in block["days"]:
            for product in PRODUCTS:
                units.append({
                    "partition": block["partition"],
                    "day": day,
                    "product": product,
                    "raw_path": lb.raw_parquet_path(block["raw_root"], product,
                                                    EXCHANGE, SYMBOL, day),
                })
    return units


def build_acquisition_plan(*, custodian_identity: str, operator_identity: str,
                           dev_raw_root: str = DEV_RAW_ROOT,
                           holdout_raw_root: str = HOLDOUT_RAW_ROOT,
                           dev_normalized_root: str = DEV_NORMALIZED_ROOT,
                           holdout_normalized_root: str = HOLDOUT_NORMALIZED_ROOT,
                           generated_at: str | None = None) -> dict:
    """Build (and fail-closed validate) the deterministic `g0bn-acquisition-plan-v1`.

    Identical inputs produce an identical plan and identical `sha256`
    (`generated_at` is hash-excluded like every G0-BN artifact). All byte
    figures are ex-ante constants x day counts; nothing here touches a vendor,
    the network, or any data file."""
    dev = _partition_block("development", DEV_DAYS, dev_raw_root,
                           dev_normalized_root, DEV_REPORT_ROOT, DEV_DAYS_FILE)
    holdout = _partition_block("holdout", HOLDOUT_DAYS, holdout_raw_root,
                               holdout_normalized_root, HOLDOUT_REPORT_ROOT,
                               HOLDOUT_DAYS_FILE)
    units = _units(dev, holdout)
    n_days = len(DEV_DAYS) + len(HOLDOUT_DAYS)
    total_gb = lb.estimate_gb(INSTRUMENT_KEY, _feeds(), n_days)
    safe_window_gb = lb.QUOTA_GB - lb.DEFAULT_HEADROOM_GB
    per = lb.LAKE_GB_PER_DAY[(EXCHANGE, SYMBOL)]
    plan = {
        "schema": ACQUISITION_PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "pilot_id": PILOT_ID,
        "holdout_universe_id": G0BN_HOLDOUT_UNIVERSE_ID,
        "transaction_id": G0BN_TRANSACTION_ID,
        "provider": CERTIFIED_PROVIDER,
        "fallback_policy": FALLBACK_POLICY,
        "instrument": dict(INSTRUMENT),
        "instrument_key": INSTRUMENT_KEY,
        "products": list(PRODUCTS),
        "feeds_argument": FEEDS_ARGUMENT,
        "window": {"start_day": DEV_DAYS[0], "end_day": HOLDOUT_DAYS[-1],
                   "n_days": n_days},
        "development": dev,
        "holdout": holdout,
        "units": units,
        "n_units": len(units),
        "estimates": {
            "gb_per_day_by_product": {p: per[p] for p in PRODUCTS},
            "gb_per_day": lb.estimate_gb(INSTRUMENT_KEY, _feeds(), 1),
            "total_gb": total_gb,
            "basis": ESTIMATE_BASIS,
            "ex_ante_rule": "pinned_constants_times_day_count_v1",
        },
        "quota": {
            "quota_gb_per_month": lb.QUOTA_GB,
            "headroom_gb": lb.DEFAULT_HEADROOM_GB,
            "safe_window_gb": safe_window_gb,
            "fits_single_window": total_gb <= safe_window_gb,
            "assumption": QUOTA_ASSUMPTION,
        },
        "disk": {
            "raw_safety_factor": RAW_DISK_SAFETY_FACTOR,
            "normalized_allowance_gb": NORMALIZED_DISK_ALLOWANCE_GB,
            "required_free_gb": total_gb * RAW_DISK_SAFETY_FACTOR
                                + NORMALIZED_DISK_ALLOWANCE_GB,
        },
        "execution": {
            "downloader": "ingest/download_lake_binance.py",
            "recon_runner": "scripts/run_binance_recon.py",
            "retries": dl.DEFAULT_RETRIES,
            "backoff_base_s": dl.DEFAULT_BACKOFF_BASE_S,
            "backoff_cap_s": dl.DEFAULT_BACKOFF_CAP_S,
            "atomic_write_rule": "stream_to_tmp_then_os_replace_v1",
            "resume_rule": "skip_final_parquet_and_sparse_accepted_v1",
            "quota_gate_rule": "check_broad_gate_before_any_transfer_v1",
            "expensive_compute_lock": EXPENSIVE_COMPUTE_LOCK,
            "jobs": 1,
        },
        "custody": {
            "custodian_identity": custodian_identity,
            "operator_identity": operator_identity,
            "raw_products": list(PRODUCTS),
            "normalized_products": list(G0BN_DATA_SOURCES),
            # EVERY holdout destination is custody-scoped, including the report
            # root: Stage-1/Stage-2 run reports and the store manifests record
            # January rows/bytes (spec-5.1 activity proxies), so they must be
            # custodian-owned and operator-read-denied like the stores.
            "holdout_custody_scope": [holdout_raw_root, holdout_normalized_root,
                                      HOLDOUT_REPORT_ROOT],
            "seal_content_schema": SEAL_CONTENT_SCHEMA,
            "permission_evidence_schema": PERMISSION_EVIDENCE_SCHEMA,
            "release_boundary": RELEASE_BOUNDARY,
            "local_chmod_is_custody": False,
            "unresolved_assumptions": list(UNRESOLVED_CUSTODY_ASSUMPTIONS),
        },
        "generated_at": generated_at,
    }
    plan["sha256"] = g0bn_artifact_sha256(plan)
    return validate_acquisition_plan(plan)


def _validate_partition(path: str, block, name: str, days_expected, *,
                        report_root: str, days_file: str) -> None:
    _dict(path, block, _PARTITION_FIELDS)
    _exact(f"{path}.partition", block["partition"], name)
    _exact(f"{path}.days", block["days"], list(days_expected))
    _exact(f"{path}.n_days", block["n_days"], len(days_expected))
    _exact(f"{path}.n_units", block["n_units"], len(days_expected) * len(PRODUCTS))
    _safe_repo_data_root(f"{path}.raw_root", block["raw_root"],
                         required_prefix="data/raw")
    _safe_repo_data_root(f"{path}.normalized_root", block["normalized_root"],
                         required_prefix="data/processed")
    # report_root/days_file have no build parameter, so they are PINNED exactly:
    # a rehashed plan redirecting the inert commands' --report-dir/--days-file
    # (the days-file CONTENT, not the pinned day arrays, drives what a later
    # execution would transfer) must fail here, not at human review.
    _exact(f"{path}.report_root", block["report_root"], report_root)
    _exact(f"{path}.days_file", block["days_file"], days_file)
    _exact(f"{path}.days_file_sha256", block["days_file_sha256"],
           _days_file_sha256(days_expected))
    _safe_repo_data_root(f"{path}.report_root", block["report_root"],
                         required_prefix="data/reports")
    _safe_repo_data_root(f"{path}.days_file", block["days_file"],
                         required_prefix="data/reports")
    _exact(f"{path}.raw_manifest", block["raw_manifest"],
           block["raw_root"] + "/" + lb.MANIFEST_NAME)
    _exact(f"{path}.normalized_manifest", block["normalized_manifest"],
           block["normalized_root"] + "/" + lb.MANIFEST_NAME)
    _exact(f"{path}.estimated_gb", block["estimated_gb"],
           lb.estimate_gb(INSTRUMENT_KEY, _feeds(), len(days_expected)))


def validate_acquisition_plan(plan: dict) -> dict:
    """Strict fail-closed validation of one `g0bn-acquisition-plan-v1`.

    Exact nested field sets everywhere (extra dates, products, providers,
    fallback sources, and activity-proxy fields are structurally rejected);
    every derived value (units, estimates, quota, disk, manifests) is
    RECOMPUTED from the pinned constants and compared exactly, so a measured
    or hand-edited number can never pose as the deterministic projection; the
    embedded self-hash is verified last."""
    _dict("g0bn acquisition plan", plan, _PLAN_FIELDS)
    _exact("schema", plan["schema"], ACQUISITION_PLAN_SCHEMA)
    _exact("protocol_id", plan["protocol_id"], PROTOCOL_ID)
    _exact("pilot_id", plan["pilot_id"], PILOT_ID)
    _sha256("holdout_universe_id", plan["holdout_universe_id"])
    _exact("holdout_universe_id", plan["holdout_universe_id"],
           G0BN_HOLDOUT_UNIVERSE_ID)
    _sha256("transaction_id", plan["transaction_id"])
    _exact("transaction_id", plan["transaction_id"], G0BN_TRANSACTION_ID)
    _exact("provider", plan["provider"], CERTIFIED_PROVIDER)
    _exact("fallback_policy", plan["fallback_policy"], FALLBACK_POLICY)
    _exact("instrument", plan["instrument"], INSTRUMENT)
    _exact("instrument_key", plan["instrument_key"], INSTRUMENT_KEY)
    registry = lb.INSTRUMENTS[INSTRUMENT_KEY]
    if (registry.exchange, registry.symbol) != (EXCHANGE, SYMBOL):
        _fail("instrument",
              f"identity drift: ingest.lake_binance.INSTRUMENTS[{INSTRUMENT_KEY!r}] "
              f"names ({registry.exchange!r}, {registry.symbol!r}) but the "
              f"protocol instrument is ({EXCHANGE!r}, {SYMBOL!r})")
    _exact("products", plan["products"], list(PRODUCTS))
    _exact("feeds_argument", plan["feeds_argument"], FEEDS_ARGUMENT)
    _exact("window", plan["window"],
           {"start_day": DEV_DAYS[0], "end_day": HOLDOUT_DAYS[-1],
            "n_days": len(DEV_DAYS) + len(HOLDOUT_DAYS)})

    dev, holdout = plan["development"], plan["holdout"]
    _validate_partition("development", dev, "development", DEV_DAYS,
                        report_root=DEV_REPORT_ROOT, days_file=DEV_DAYS_FILE)
    _validate_partition("holdout", holdout, "holdout", HOLDOUT_DAYS,
                        report_root=HOLDOUT_REPORT_ROOT,
                        days_file=HOLDOUT_DAYS_FILE)
    _require_disjoint({
        "development.raw_root": dev["raw_root"],
        "development.normalized_root": dev["normalized_root"],
        "development.report_root": dev["report_root"],
        "development.days_file": dev["days_file"],
        "holdout.raw_root": holdout["raw_root"],
        "holdout.normalized_root": holdout["normalized_root"],
        "holdout.report_root": holdout["report_root"],
        "holdout.days_file": holdout["days_file"],
    })

    expected_units = _units(dev, holdout)
    if not isinstance(plan["units"], list):
        _fail("units", "must be the ordered deterministic unit array")
    for i, unit in enumerate(plan["units"]):
        _dict(f"units[{i}]", unit, _UNIT_FIELDS)
    _exact("units", plan["units"], expected_units)
    _exact("n_units", plan["n_units"], len(expected_units))
    downloader_units = dl.plan_units([INSTRUMENT_KEY], FEEDS_ARGUMENT,
                                     dev["days"] + holdout["days"])
    if {(u.feed, u.day) for u in downloader_units} != \
            {(u["product"], u["day"]) for u in plan["units"]} or \
            len(downloader_units) != len(plan["units"]):
        _fail("units",
              "do not reconcile with ingest.download_lake_binance.plan_units "
              "for the same instrument/feeds/days — the inert commands would "
              "not transfer exactly the planned units")

    est = plan["estimates"]
    _dict("estimates", est, _ESTIMATE_FIELDS)
    per = lb.LAKE_GB_PER_DAY[(EXCHANGE, SYMBOL)]
    _exact("estimates.gb_per_day_by_product", est["gb_per_day_by_product"],
           {p: per[p] for p in PRODUCTS})
    _exact("estimates.gb_per_day", est["gb_per_day"],
           lb.estimate_gb(INSTRUMENT_KEY, _feeds(), 1))
    total_expected = lb.estimate_gb(INSTRUMENT_KEY, _feeds(),
                                    len(DEV_DAYS) + len(HOLDOUT_DAYS))
    _exact("estimates.total_gb", est["total_gb"], total_expected)
    _exact("estimates.basis", est["basis"], ESTIMATE_BASIS)
    _exact("estimates.ex_ante_rule", est["ex_ante_rule"],
           "pinned_constants_times_day_count_v1")

    quota = plan["quota"]
    _dict("quota", quota, _QUOTA_FIELDS)
    _exact("quota.quota_gb_per_month", quota["quota_gb_per_month"], lb.QUOTA_GB)
    _exact("quota.headroom_gb", quota["headroom_gb"], lb.DEFAULT_HEADROOM_GB)
    safe_window = lb.QUOTA_GB - lb.DEFAULT_HEADROOM_GB
    _exact("quota.safe_window_gb", quota["safe_window_gb"], safe_window)
    if not total_expected <= safe_window:
        _fail("quota", f"the {len(DEV_DAYS) + len(HOLDOUT_DAYS)}-day estimate "
                       f"({total_expected:.2f} GB) no longer fits one monthly "
                       f"quota window with headroom ({safe_window:.0f} GB); the "
                       "single-window plan is invalid")
    _exact("quota.fits_single_window", quota["fits_single_window"], True)
    _exact("quota.assumption", quota["assumption"], QUOTA_ASSUMPTION)

    disk = plan["disk"]
    _dict("disk", disk, _DISK_FIELDS)
    _exact("disk.raw_safety_factor", disk["raw_safety_factor"],
           RAW_DISK_SAFETY_FACTOR)
    _exact("disk.normalized_allowance_gb", disk["normalized_allowance_gb"],
           NORMALIZED_DISK_ALLOWANCE_GB)
    _exact("disk.required_free_gb", disk["required_free_gb"],
           total_expected * RAW_DISK_SAFETY_FACTOR + NORMALIZED_DISK_ALLOWANCE_GB)

    _exact("execution", plan["execution"], {
        "downloader": "ingest/download_lake_binance.py",
        "recon_runner": "scripts/run_binance_recon.py",
        "retries": dl.DEFAULT_RETRIES,
        "backoff_base_s": dl.DEFAULT_BACKOFF_BASE_S,
        "backoff_cap_s": dl.DEFAULT_BACKOFF_CAP_S,
        "atomic_write_rule": "stream_to_tmp_then_os_replace_v1",
        "resume_rule": "skip_final_parquet_and_sparse_accepted_v1",
        "quota_gate_rule": "check_broad_gate_before_any_transfer_v1",
        "expensive_compute_lock": EXPENSIVE_COMPUTE_LOCK,
        "jobs": 1,
    })

    custody = plan["custody"]
    _dict("custody", custody, _CUSTODY_FIELDS)
    for k in ("custodian_identity", "operator_identity"):
        _identity(f"custody.{k}", custody[k])
    if custody["custodian_identity"] == custody["operator_identity"]:
        _fail("custody", "custodian_identity and operator_identity must be "
                         "distinct (separation of custody; spec section 5.1)")
    _exact("custody.raw_products", custody["raw_products"], list(PRODUCTS))
    _exact("custody.normalized_products", custody["normalized_products"],
           list(G0BN_DATA_SOURCES))
    _exact("custody.holdout_custody_scope", custody["holdout_custody_scope"],
           [holdout["raw_root"], holdout["normalized_root"],
            holdout["report_root"]])
    _exact("custody.seal_content_schema", custody["seal_content_schema"],
           SEAL_CONTENT_SCHEMA)
    _exact("custody.permission_evidence_schema",
           custody["permission_evidence_schema"], PERMISSION_EVIDENCE_SCHEMA)
    _exact("custody.release_boundary", custody["release_boundary"],
           RELEASE_BOUNDARY)
    _exact("custody.local_chmod_is_custody", custody["local_chmod_is_custody"],
           False)
    # The disclosures the approver reads are pinned like every other builder
    # constant: a rehashed plan that guts or rewrites them must fail closed.
    _exact("custody.unresolved_assumptions", custody["unresolved_assumptions"],
           list(UNRESOLVED_CUSTODY_ASSUMPTIONS))

    if plan["generated_at"] is not None:
        _validate_generated_at(plan["generated_at"])
    embedded = _sha256("sha256", plan["sha256"])
    recomputed = g0bn_artifact_sha256(plan)
    if embedded != recomputed:
        _fail("sha256", f"embedded acquisition-plan sha256 does not match the "
                        f"canonical content (tampered or stale): "
                        f"{embedded} != {recomputed}")
    return plan


# ----------------------------------------------------------------- approval packet
def _locked(command: str) -> str:
    return f"{LOCK_WRAPPER} {command}"


def _download_command(plan: dict, block: dict, cap_gb: float) -> str:
    ex = plan["execution"]
    return _locked(
        f".venv/bin/python {ex['downloader']} "
        f"--instrument {plan['instrument_key']} "
        f"--feeds {plan['feeds_argument']} "
        f"--days-file {block['days_file']} "
        f"--out {block['raw_root']} "
        f"--report-dir {block['report_root']}/download "
        # NO --allow-broad: it would bypass exactly the est_gb > max_gb soft
        # gate (lake_binance.check_broad_gate), turning the advertised cap into
        # a no-op. The approved cap is expressed by RAISING --max-gb instead,
        # so any estimate drift above it refuses before transfer (Codex P1).
        f"--max-gb {cap_gb:g} "
        f"--retries {ex['retries']} --jobs {ex['jobs']} --resume")


def _recon_command(plan: dict, block: dict) -> str:
    ex = plan["execution"]
    return _locked(
        f".venv/bin/python {ex['recon_runner']} "
        f"--instrument {plan['instrument_key']} "
        f"--feeds {plan['feeds_argument']} "
        f"--days-file {block['days_file']} "
        f"--raw {block['raw_root']} "
        f"--out {block['normalized_root']} "
        f"--report-dir {block['report_root']}/recon "
        f"--engine native --k 10 --grid-s 1.0 --jobs {ex['jobs']} --resume")


def build_approval_packet(plan: dict, *, generated_at: str | None = None) -> dict:
    """Derive the INERT `g0bn-acquisition-approval-packet-v1` from a validated
    plan. Everything in the packet is a pure function of the plan, so
    `validate_approval_packet` can reject any tampered command by rebuilding."""
    validate_acquisition_plan(plan)
    if generated_at is not None:
        _validate_generated_at(generated_at)
    dev, holdout = plan["development"], plan["holdout"]
    custody = plan["custody"]
    c, o = custody["custodian_identity"], custody["operator_identity"]
    dev_cap = float(math.ceil(dev["estimated_gb"] * CAP_SAFETY_FACTOR))
    holdout_cap = float(math.ceil(holdout["estimated_gb"] * CAP_SAFETY_FACTOR))
    safe_window = lb.QUOTA_GB - lb.DEFAULT_HEADROOM_GB
    if dev_cap + holdout_cap > safe_window:
        _fail("caps", f"the per-command caps ({dev_cap:g} + {holdout_cap:g} GB) "
                      f"exceed the safe monthly window ({safe_window:.0f} GB)")
    commands = [
        {
            "step": "regenerate-preflight",
            "purpose": "Reproduce this plan/packet byte-identically and refresh "
                       "the day files; verify the printed plan sha256 equals "
                       "plan_sha256 in this packet before anything else runs.",
            "run_as": "operator",
            "command": f".venv/bin/python -m ingest.g0bn_acquisition_preflight "
                       f"plan --custodian-identity {c} --operator-identity {o} "
                       f"--out-dir {DEFAULT_OUT_DIR}",
            "approval_required": True,
            "notes": "Offline: no vendor I/O. Also re-checks free disk against "
                     "disk.required_free_gb (exit 4 on breach).",
        },
        {
            "step": "development-download",
            "purpose": f"Stage-1 pull of the {dev['n_units']} development units "
                       f"({dev['n_days']} days x {len(plan['products'])} "
                       f"products), estimated {dev['estimated_gb']:.2f} GB.",
            "run_as": "operator",
            "command": _download_command(plan, dev, dev_cap),
            "approval_required": True,
            "notes": "First vendor-I/O step. Resumable and atomic; hard-stops "
                     "on quota/auth (exit 2), partial (exit 3), gate (exit 4). "
                     "--allow-broad is deliberately absent so --max-gb remains "
                     "a binding refusal. Before running, verify sha256sum of "
                     "the days file equals the plan's "
                     "development.days_file_sha256.",
        },
        {
            "step": "development-recon",
            "purpose": "Stage-2 reconstruction/normalization of the development "
                       "days from the local raw store (quota-free).",
            "run_as": "operator",
            "command": _recon_command(plan, dev),
            "approval_required": True,
            "notes": "Local-only compute; no vendor I/O. Certified outputs only "
                     "(a failed day is an explicit exclusion, never repaired).",
        },
        {
            "step": "holdout-download",
            "purpose": f"Stage-1 pull of the {holdout['n_units']} sealed-holdout "
                       f"units ({holdout['n_days']} days), estimated "
                       f"{holdout['estimated_gb']:.2f} GB, into the "
                       "custodian-owned raw store.",
            "run_as": "custodian",
            "command": _download_command(plan, holdout, holdout_cap),
            "approval_required": True,
            "notes": "Runs under the custodian identity only, after the "
                     "permission-policy evidence exists. The operator never "
                     "reads this store, and the report dir and store manifest "
                     "are custody-internal too (their run records carry "
                     "January rows/bytes — activity proxies). Before running, "
                     "verify sha256sum of the days file equals the plan's "
                     "holdout.days_file_sha256.",
        },
        {
            "step": "holdout-recon",
            "purpose": "Custodian-run Stage-2 normalization of the holdout days "
                       "into the custodian-owned normalized store.",
            "run_as": "custodian",
            "command": _recon_command(plan, holdout),
            "approval_required": True,
            "notes": "Custodian-only. January outputs stay inside custody; the "
                     "operator receives outcome-blind inventory metadata only. "
                     "The recon report dir and normalized-store manifest carry "
                     "January rows/bytes and are custody-internal.",
        },
        {
            "step": "custody-seal",
            "purpose": "Custodian deterministically mints the outcome-blind "
                       "inventory and its g0bn-custodian-seal-content-v1 "
                       "commitment. The seal hash from this step is then pinned "
                       "into the protocol config's source_certification "
                       "(handoff step 4) — the config cannot pre-exist its own "
                       "seal pin, hence mint-then-verify.",
            "run_as": "custodian",
            "command": ".venv/bin/python -m ingest.g0bn_acquisition_preflight "
                       f"build-inventory --custodian-identity {c} "
                       f"--operator-identity {o} "
                       "--coverage-sha256 <coverage_sha256> "
                       "--evidence <permission_evidence.json> "
                       "--objects <sealed_objects.json> "
                       "--excluded-days <excluded_days.json> "
                       "--out <custody_inventory.json>",
            "approval_required": True,
            "notes": "Offline, custodian-only. <...> placeholders are "
                     "custodian-produced artifacts; object checksums come from "
                     "the write-time store manifests (no payload/footer reads). "
                     "This step's output is PROVISIONAL until custody-seal-"
                     "verify passes. Sealing normalized objects is blocked on "
                     "the #93 product-name reconciliation.",
        },
        {
            "step": "custody-seal-verify",
            "purpose": "MANDATORY: after the minted seal is pinned into the "
                       "protocol config, re-emit the inventory against that "
                       "config so eval.g0bn_freeze.validate_custody_inventory "
                       "(the sole consumer gate) accepts it and the seal "
                       "reproduces identically.",
            "run_as": "custodian",
            "command": ".venv/bin/python -m ingest.g0bn_acquisition_preflight "
                       f"build-inventory --custodian-identity {c} "
                       f"--operator-identity {o} "
                       "--coverage-sha256 <coverage_sha256> "
                       "--evidence <permission_evidence.json> "
                       "--objects <sealed_objects.json> "
                       "--excluded-days <excluded_days.json> "
                       "--config <g0bn_protocol_config.json> "
                       f"--plan {DEFAULT_OUT_DIR}/g0bn_acquisition_plan.json "
                       "--out <custody_inventory.json>",
            "approval_required": True,
            "notes": "Offline, custodian-only. Must reproduce the custody-seal "
                     "step's custodian_seal_sha256 byte-identically; a "
                     "mismatch or validation failure means the seal is NOT "
                     "final and nothing downstream may consume it. --plan "
                     "makes the handoff verify that the evidence covers every "
                     "planned holdout destination "
                     "(custody.holdout_custody_scope).",
        },
    ]
    packet = {
        "schema": APPROVAL_PACKET_SCHEMA,
        "protocol_id": plan["protocol_id"],
        "pilot_id": plan["pilot_id"],
        "holdout_universe_id": plan["holdout_universe_id"],
        "transaction_id": plan["transaction_id"],
        "plan_sha256": plan["sha256"],
        "inert": True,
        "approval_authority": "explicit human approval on GitHub issue #68 "
                              "(no roadmap date, issue, or agent may authorize "
                              "vendor I/O)",
        "window": dict(plan["window"]),
        "request_totals": {
            "n_units": plan["n_units"],
            "development_units": dev["n_units"],
            "holdout_units": holdout["n_units"],
            "products_per_day": len(plan["products"]),
            "vendor_note": "one Crypto Lake partition fetch per (product, day) "
                           "unit; a vendor partition may span multiple S3 "
                           "objects, so S3 GET counts can exceed the unit count",
        },
        "caps": {
            "development_max_gb": dev_cap,
            "holdout_max_gb": holdout_cap,
            "cap_rule": "ceil_of_partition_estimate_times_1.1_v1",
            "quota_gb_per_month": lb.QUOTA_GB,
            "headroom_gb": lb.DEFAULT_HEADROOM_GB,
            "jobs": 1,
            # --allow-broad would disable the est_gb > max_gb refusal, so the
            # cap is expressed by raising --max-gb and the flag is forbidden.
            "allow_broad_forbidden": True,
        },
        "vendor_cost": {
            "vendor_cost_model": "Crypto Lake individual subscription "
                                 "(~$64/month, already active) with a 300 "
                                 "GB/month download quota; no per-GB and no AWS "
                                 "egress charge (docs/data.md sections 2.1/8, "
                                 "checked 2026-06-30)",
            "incremental_usd_within_quota": 0.0,
            "quota_consumed_gb_estimate": plan["estimates"]["total_gb"],
        },
        "commands": commands,
        "stop_conditions": [
            "Vendor quota/credit exhaustion or auth failure: the downloader "
            "hard-stops with exit 2; do not retry, report on #68.",
            "Any errored or required-missing unit: the downloader stops with "
            "exit 3; only a --resume rerun of the same approved scope is "
            "permitted, never a scope change.",
            "Broad-pull or quota-headroom gate: the downloader refuses with "
            "exit 4; never override the quota-headroom refusal (300 GB/month "
            "cap with 10 GB headroom).",
            "Estimated GB above the per-command --max-gb cap: the downloader "
            "refuses before any transfer; stop and re-approve on #68.",
            "Free disk below disk.required_free_gb at start: do not begin the "
            "download.",
            "Vendor schema drift (missing engine-time column): the unit fails "
            "loudly; stop and report before rerunning.",
            "A missing required-feed day is recorded and becomes an explicit "
            "outcome-blind exclusion; source substitution or a fallback source "
            "is forbidden.",
            "Only the exact commands in this packet may run; any other date, "
            "feed, instrument, or destination needs a new approval.",
            f"Every download/recon command serializes under {LOCK_WRAPPER}.",
            "January payloads and Parquet footers are never opened by the "
            "operator identity; custody sealing precedes any #69 access burn.",
            "Holdout run reports and store manifests record January rows/bytes "
            "(activity proxies): everything under custody.holdout_custody_scope "
            "is custodian-owned and operator-read-denied until after the #69 "
            "raw-access burn.",
        ],
        "custody": {
            "custodian_identity": c,
            "operator_identity": o,
            "handoff": [
                "Provision the custodian identity and custodian-owned holdout "
                "storage; capture g0bn-permission-policy-evidence-v1 from the "
                "effective ACL/IAM/bucket policy (developer chmod is not "
                "custody). The evidence resource scope must cover every path "
                "in custody.holdout_custody_scope — stores, manifests, and "
                "report root — and must deny the operator policy mutation.",
                "The custodian runs the holdout download and recon commands; "
                "the operator never reads the holdout stores.",
                "The custodian builds the outcome-blind 8-field inventory and "
                "mints the g0bn-custodian-seal-content-v1 commitment "
                "(custody-seal command; provisional until verified).",
                "The minted seal/coverage/permission hashes and both identities "
                "are pinned into the protocol config's source_certification, "
                "then custody-seal-verify must reproduce the identical seal "
                "against that config; "
                "eval.g0bn_freeze.validate_custody_inventory is the sole "
                "consumer gate — this preflight adds couplings, never a "
                "substitute.",
                "The operator receives inventory metadata only; January "
                "payload/footer reads stay forbidden until the #69 raw-access "
                "burn.",
            ],
            "seal_content_schema": SEAL_CONTENT_SCHEMA,
            "permission_evidence_schema": PERMISSION_EVIDENCE_SCHEMA,
            "release_boundary": RELEASE_BOUNDARY,
            "unresolved_assumptions": list(custody["unresolved_assumptions"]),
        },
        "no_vendor_io_statement": NO_VENDOR_IO_STATEMENT,
        "generated_at": generated_at,
    }
    packet["sha256"] = g0bn_artifact_sha256(packet)
    return packet


def validate_approval_packet(packet: dict, plan: dict) -> dict:
    """Fail-closed packet validation: verify the self-hash, then rebuild the
    packet from the validated plan and require exact equality — a tampered
    command, cap, stop condition, or smuggled field can never survive."""
    # Shape first (canonical_json on a non-JSON value would raise TypeError,
    # not the module's fail-closed ValueError), hash second, rebuild last.
    _dict("g0bn approval packet", packet, _PACKET_FIELDS)
    embedded = _sha256("sha256", packet.get("sha256"))
    recomputed = g0bn_artifact_sha256(packet)
    if embedded != recomputed:
        _fail("sha256", f"embedded approval-packet sha256 does not match the "
                        f"canonical content (tampered or stale): "
                        f"{embedded} != {recomputed}")
    expected = build_approval_packet(plan, generated_at=packet.get("generated_at"))
    _exact("g0bn approval packet", packet, expected)
    return packet


def render_approval_packet(packet: dict) -> str:
    """Human-readable Markdown for the approval packet (deterministic)."""
    lines = [
        "# G0-BN bounded acquisition — human approval packet (INERT)",
        "",
        f"> {packet['no_vendor_io_statement']}",
        "",
        f"- packet schema: `{packet['schema']}`",
        f"- acquisition plan sha256: `{packet['plan_sha256']}`",
        f"- packet sha256: `{packet['sha256']}`",
        f"- transaction: `{packet['transaction_id']}`",
        f"- approval authority: {packet['approval_authority']}",
        "",
        "## Scope",
        "",
        f"- window: {packet['window']['start_day']} .. "
        f"{packet['window']['end_day']} ({packet['window']['n_days']} days)",
        f"- units: {packet['request_totals']['n_units']} total = "
        f"{packet['request_totals']['development_units']} development + "
        f"{packet['request_totals']['holdout_units']} holdout "
        f"({packet['request_totals']['products_per_day']} products/day)",
        f"- {packet['request_totals']['vendor_note']}",
        "",
        "## Caps and quota",
        "",
        f"- development --max-gb: {packet['caps']['development_max_gb']:g} GB; "
        f"holdout --max-gb: {packet['caps']['holdout_max_gb']:g} GB "
        f"({packet['caps']['cap_rule']})",
        f"- monthly quota {packet['caps']['quota_gb_per_month']:g} GB with "
        f"{packet['caps']['headroom_gb']:g} GB headroom; single window; "
        f"jobs={packet['caps']['jobs']}",
        f"- cost: {packet['vendor_cost']['vendor_cost_model']}; incremental "
        f"vendor charge within quota: "
        f"${packet['vendor_cost']['incremental_usd_within_quota']:g} "
        f"for ~{packet['vendor_cost']['quota_consumed_gb_estimate']:.2f} GB",
        "",
        "## Commands (inert until #68 approval)",
        "",
    ]
    for c in packet["commands"]:
        lines += [
            f"### {c['step']} (run as: {c['run_as']})",
            "",
            c["purpose"],
            "",
            "```bash",
            c["command"],
            "```",
            "",
            f"_{c['notes']}_",
            "",
        ]
    lines += ["## Stop conditions", ""]
    lines += [f"- {s}" for s in packet["stop_conditions"]]
    lines += ["", "## Custody handoff", ""]
    lines += [f"{i}. {step}" for i, step in enumerate(packet["custody"]["handoff"], 1)]
    lines += [
        "",
        f"- custodian: `{packet['custody']['custodian_identity']}` / operator: "
        f"`{packet['custody']['operator_identity']}`",
        f"- seal content schema: `{packet['custody']['seal_content_schema']}`; "
        f"evidence schema: `{packet['custody']['permission_evidence_schema']}`",
        f"- release boundary: `{packet['custody']['release_boundary']}`",
        "",
        "## Unresolved custody assumptions",
        "",
    ]
    lines += [f"- {a}" for a in packet["custody"]["unresolved_assumptions"]]
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------- custody evidence and inventory
def validate_permission_policy_evidence(evidence: dict) -> dict:
    """Fail-closed validation of the effective ACL/IAM/bucket-policy evidence
    (spec section 5.1). The evidence must name an independent-custody mechanism
    — a policy layer the operator cannot rewrite — and must assert custodian
    ownership plus operator payload/footer denial. Developer-local permission
    fiddling (chmod & co.) is rejected by name."""
    path = "permission-policy evidence"
    _dict(path, evidence, _EVIDENCE_FIELDS)
    _exact(f"{path}.schema", evidence["schema"], PERMISSION_EVIDENCE_SCHEMA)
    mechanism = evidence["mechanism"]
    _str(f"{path}.mechanism", mechanism)
    if mechanism in LOCAL_ONLY_MECHANISMS:
        _fail(f"{path}.mechanism",
              f"{mechanism!r} is developer-local permission fiddling, not "
              "independent custody (spec section 5.1: a developer copying files "
              "or running chmod is not separation of custody)")
    if mechanism not in INDEPENDENT_CUSTODY_MECHANISMS:
        _fail(f"{path}.mechanism",
              f"{mechanism!r} is not a recognized independent-custody mechanism "
              f"(allowed: {INDEPENDENT_CUSTODY_MECHANISMS})")
    for k in ("custodian_identity", "operator_identity"):
        _identity(f"{path}.{k}", evidence[k])
    if evidence["custodian_identity"] == evidence["operator_identity"]:
        _fail(path, "custodian_identity and operator_identity must be distinct "
                    "(separation of custody; spec section 5.1)")
    _str(f"{path}.resource", evidence["resource"])
    # The storage-layer `resource` names the policy target in that layer's own
    # vocabulary (bucket/prefix/mount); `covered_paths` attests, in PLAN
    # vocabulary, exactly which planned destinations that policy covers — the
    # handoff validator checks it against custody.holdout_custody_scope so
    # evidence captured for the wrong prefix or a subset cannot vouch.
    covered = evidence["covered_paths"]
    if not isinstance(covered, list) or not covered:
        _fail(f"{path}.covered_paths",
              "must be a non-empty array of the planned destination paths the "
              "captured policy covers")
    for i, p in enumerate(covered):
        _str(f"{path}.covered_paths[{i}]", p)
    if len(set(covered)) != len(covered):
        _fail(f"{path}.covered_paths", "must be unique")
    _exact(f"{path}.custodian_owns_objects",
           evidence["custodian_owns_objects"], True)
    _exact(f"{path}.operator_payload_read_denied",
           evidence["operator_payload_read_denied"], True)
    _exact(f"{path}.operator_footer_read_denied",
           evidence["operator_footer_read_denied"], True)
    # Storage-level listing is itself an activity-proxy channel: S3 ListObjects
    # and `ls -l` both return per-object byte sizes, which spec section 5.1
    # keeps inside custody until the raw-access burn. The operator's only
    # January metadata channel is the sealed outcome-blind inventory.
    if evidence["operator_storage_listing_denied"] is not True:
        _fail(f"{path}.operator_storage_listing_denied",
              "must be exactly true: storage listing exposes per-object January "
              "byte sizes (activity proxies that stay inside custody until the "
              "raw-access burn; spec section 5.1) — the sealed inventory is the "
              "operator's only metadata channel")
    # A policy the operator can rewrite proves nothing: the point-in-time
    # capture hashed into permission_policy_sha256 only binds future access if
    # the operator cannot regrant themselves (spec section 5.1 'effective'
    # ACL/IAM/bucket-policy evidence).
    if evidence["operator_policy_write_denied"] is not True:
        _fail(f"{path}.operator_policy_write_denied",
              "must be exactly true: an operator who can modify the "
              "ACL/IAM/bucket policy can regrant read access after the "
              "evidence capture, so the pinned permission_policy_sha256 would "
              "not bind effective access (spec section 5.1)")
    capture = evidence["effective_policy_capture"]
    cpath = f"{path}.effective_policy_capture"
    _dict(cpath, capture, _CAPTURE_FIELDS)
    _str(f"{cpath}.method", capture["method"])
    _str(f"{cpath}.command", capture["command"])
    _sha256(f"{cpath}.policy_document_sha256", capture["policy_document_sha256"])
    try:
        _validate_generated_at(evidence["captured_at"])
    except ValueError:
        _fail(f"{path}.captured_at",
              f"must be an ISO-8601 timestamp with an explicit timezone; got "
              f"{evidence['captured_at']!r}")
    return evidence


def permission_policy_evidence_sha256(evidence: dict) -> str:
    """The canonical evidence hash that becomes `permission_policy_sha256` in
    the inventory and the config's source_certification pin."""
    return hash_obj(validate_permission_policy_evidence(evidence))


def _object_sort_key(obj: dict) -> tuple:
    return (obj["day"], obj["layer"], obj["product"], obj["object_id"])


def build_custody_inventory(*, custodian_identity: str, operator_identity: str,
                            coverage_sha256: str, evidence: dict,
                            excluded_days: dict, objects: list,
                            config: dict | None = None) -> dict:
    """Emit the outcome-blind #68 custodian inventory in EXACTLY the 8-field
    shape `eval.g0bn_freeze.validate_custody_inventory` consumes, sealed with
    the reused `g0bn-custodian-seal-content-v1` commitment.

    `included_days` is DERIVED as the full January window minus the explicit
    exclusions, so the 31-day accounting is complete by construction. Objects
    are canonicalized to the (day, layer, product, object_id) sort and pinned
    to the exact 5-field object shape (`byte_size`/`record_count`-style
    activity proxies are structurally rejected). Object checksums are caller
    inputs (from the write-time store manifests) — this function never reads a
    data file. When `config` is provided the emitted inventory is validated
    through the real consumer contract; construction alone never substitutes
    for that validation."""
    validate_permission_policy_evidence(evidence)
    _identity("custodian_identity", custodian_identity)
    _identity("operator_identity", operator_identity)
    if custodian_identity == operator_identity:
        _fail("custody inventory",
              "custodian_identity and operator_identity must be distinct "
              "(separation of custody; spec section 5.1)")
    for k, want in (("custodian_identity", custodian_identity),
                    ("operator_identity", operator_identity)):
        if evidence[k] != want:
            _fail("custody inventory",
                  f"permission-policy evidence identity {k}={evidence[k]!r} "
                  f"does not match the inventory identity {want!r}")
    _sha256("coverage_sha256", coverage_sha256)
    if not isinstance(excluded_days, dict):
        _fail("excluded_days", "must be a map of day -> {reason, evidence_sha256}")
    window = set(HOLDOUT_DAYS)
    for day, entry in excluded_days.items():
        epath = f"excluded_days[{day!r}]"
        _day(epath, day)
        if day not in window:
            _fail(epath, f"day {day} is outside the January holdout window "
                         f"[{HOLDOUT_DAYS[0]}, {HOLDOUT_DAYS[-1]}]")
        _dict(epath, entry, ("reason", "evidence_sha256"))
        _str(f"{epath}.reason", entry["reason"])
        _sha256(f"{epath}.evidence_sha256", entry["evidence_sha256"])
    included = [d for d in HOLDOUT_DAYS if d not in excluded_days]
    if not isinstance(objects, list) or not objects:
        _fail("objects", "must be a non-empty array of sealed object entries")
    canonical = []
    for i, obj in enumerate(objects):
        opath = f"objects[{i}]"
        _dict(opath, obj, _OBJECT_FIELDS)
        # Value typing before the canonical sort: mixed-type values would make
        # the tuple sort raise instead of failing closed, and a config-less
        # mint must still never seal a malformed body. Product/day/completeness
        # semantics stay with the consumer gate (validate_custody_inventory).
        _str(f"{opath}.object_id", obj["object_id"])
        if obj["layer"] not in OBJECT_LAYERS:
            _fail(f"{opath}.layer",
                  f"must be one of {OBJECT_LAYERS}; got {obj['layer']!r}")
        _str(f"{opath}.product", obj["product"])
        _day(f"{opath}.day", obj["day"])
        _sha256(f"{opath}.sha256", obj["sha256"])
        canonical.append({k: obj[k] for k in _OBJECT_FIELDS})
    canonical.sort(key=_object_sort_key)
    inventory = {
        "coverage_sha256": coverage_sha256,
        "permission_policy_sha256": permission_policy_evidence_sha256(evidence),
        "custodian_identity": custodian_identity,
        "operator_identity": operator_identity,
        "included_days": included,
        "excluded_days": {d: dict(excluded_days[d]) for d in sorted(excluded_days)},
        "objects": canonical,
    }
    inventory["custodian_seal_sha256"] = custodian_seal_content_sha256(inventory)
    if config is not None:
        validate_custody_inventory(inventory, config)
    return inventory


def validate_custody_handoff(inventory: dict, evidence: dict, config: dict, *,
                             plan: dict | None = None) -> dict:
    """The full custodian->operator handoff gate: the REUSED
    `validate_custody_inventory` consumer contract first (identities, day
    accounting, canonical allowlist, seal-content commitment against the
    config's pins), then the couplings only this module defines — the
    permission-policy evidence must be valid, hash to the inventory's
    `permission_policy_sha256`, and name the same identities. With `plan`
    (the packet's custody-seal-verify step always passes it), the evidence
    must additionally attest coverage of EVERY planned holdout destination
    (`custody.holdout_custody_scope`): a policy captured for the wrong prefix
    or only a subset of the stores/manifests/report root cannot vouch."""
    validate_custody_inventory(inventory, config)
    validate_permission_policy_evidence(evidence)
    expected = hash_obj(evidence)
    if inventory["permission_policy_sha256"] != expected:
        _fail("custody handoff",
              f"inventory.permission_policy_sha256 does not pin this "
              f"permission-policy evidence "
              f"({inventory['permission_policy_sha256']} != {expected})")
    for k in ("custodian_identity", "operator_identity"):
        if evidence[k] != inventory[k]:
            _fail("custody handoff",
                  f"evidence {k} {evidence[k]!r} does not match the sealed "
                  f"inventory identity {inventory[k]!r}")
    if plan is not None:
        validate_acquisition_plan(plan)
        for k in ("custodian_identity", "operator_identity"):
            if plan["custody"][k] != inventory[k]:
                _fail("custody handoff",
                      f"plan custody {k} {plan['custody'][k]!r} does not match "
                      f"the sealed inventory identity {inventory[k]!r}")
        missing = [p for p in plan["custody"]["holdout_custody_scope"]
                   if p not in evidence["covered_paths"]]
        if missing:
            _fail("custody handoff",
                  f"the permission-policy evidence does not cover the planned "
                  f"holdout custody scope: uncovered destination(s) {missing}; "
                  "a policy captured for the wrong prefix or a subset of the "
                  "stores/manifests/report root cannot vouch for custody")
    return inventory


# ----------------------------------------------------------------------------- CLI
def _enclosing_repo_root(real_path: str):
    """The nearest ancestor of `real_path` that looks like a checkout of THIS
    project: a `.git` entry (dir in a primary checkout, file in a linked
    worktree) plus the repo's AGENTS.md. The second marker keeps a stray
    unrelated `.git` (e.g. junk at /tmp/.git) from swallowing every temp path;
    protecting arbitrary foreign repos is not this guard's job. Returns None
    outside any project checkout; works for not-yet-created leaf paths."""
    cur = real_path
    while True:
        if os.path.exists(os.path.join(cur, ".git")) and \
                os.path.exists(os.path.join(cur, "AGENTS.md")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _check_out_dir(out_dir: str, *, what: str = "--out-dir") -> str:
    """Generated operational artifacts must never land in a tracked repo path.
    Anchors: this module's own checkout, the checkout enclosing the TARGET path
    (catches a sibling checkout/worktree of this repo), and the checkout
    enclosing the cwd (catches a repo-relative path from another checkout's
    root). Inside any anchor, only the git-ignored data/reports/ or data/tmp/
    roots are allowed; paths outside every checkout (e.g. a test tmpdir) are
    fine."""
    resolved = os.path.abspath(out_dir)
    real = os.path.realpath(resolved)
    allowed = ("data/reports", "data/tmp")
    anchors = {os.path.realpath(_ROOT),
               _enclosing_repo_root(real),
               _enclosing_repo_root(os.path.realpath(os.getcwd()))}
    for anchor in anchors - {None}:
        if real == anchor or real.startswith(anchor + os.sep):
            rel = os.path.relpath(real, anchor).replace(os.sep, "/")
            if not any(rel == a or rel.startswith(a + "/") for a in allowed):
                raise ValueError(
                    f"{what} {out_dir!r} is inside the repository working tree "
                    f"at {anchor!r} but not under an ignored reports root "
                    "(data/reports/ or data/tmp/); generated operational "
                    "artifacts must stay out of git")
    return resolved


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def _free_gb(path: str) -> float:
    """Free space (GB) on the filesystem holding `path`. The disk gate measures
    the REPO filesystem (the data/raw|data/processed destinations are
    repo-relative), never the report out-dir, which may sit on another volume."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def _cmd_plan(args) -> int:
    try:
        out_dir = _check_out_dir(args.out_dir)
        plan = build_acquisition_plan(
            custodian_identity=args.custodian_identity,
            operator_identity=args.operator_identity,
            generated_at=args.generated_at)
        packet = build_approval_packet(plan, generated_at=args.generated_at)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return SETUP_ERROR_EXIT
    try:
        os.makedirs(out_dir, exist_ok=True)
        _write_json(os.path.join(out_dir, "g0bn_acquisition_plan.json"), plan)
        _write_json(os.path.join(out_dir, "g0bn_approval_packet.json"), packet)
        with open(os.path.join(out_dir, "g0bn_approval_packet.md"), "w") as f:
            f.write(render_approval_packet(packet))
        for name, days in (("dev_days.txt", plan["development"]["days"]),
                           ("holdout_days.txt", plan["holdout"]["days"])):
            with open(os.path.join(out_dir, name), "w") as f:
                f.write(_days_file_content(days))
        # The packet commands reference the PLAN-PINNED day-file paths, not
        # --out-dir, so refresh those too (cwd-relative, always under the
        # ignored data/reports/ root by validation): a custom --out-dir run
        # must never leave the referenced paths missing or stale.
        for block in (plan["development"], plan["holdout"]):
            pinned = os.path.abspath(block["days_file"])
            if os.path.realpath(pinned) != os.path.realpath(
                    os.path.join(out_dir, os.path.basename(block["days_file"]))):
                os.makedirs(os.path.dirname(pinned), exist_ok=True)
                with open(pinned, "w") as f:
                    f.write(_days_file_content(block["days"]))
        # Measure the volume that will actually hold the raw/normalized stores:
        # data/ is commonly a symlink/mount onto a bigger disk than the repo.
        data_root = os.path.join(_ROOT, "data")
        free = _free_gb(os.path.realpath(data_root)
                        if os.path.isdir(data_root) else _ROOT)
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return SETUP_ERROR_EXIT
    print(f"G0-BN acquisition preflight (INERT, no vendor I/O) -> {out_dir}")
    print(f"  plan sha256:   {plan['sha256']}")
    print(f"  packet sha256: {packet['sha256']}")
    print(f"  units: {plan['n_units']} "
          f"(dev {plan['development']['n_units']} / "
          f"holdout {plan['holdout']['n_units']}), "
          f"estimated {plan['estimates']['total_gb']:.2f} GB")
    print(f"  {NO_VENDOR_IO_STATEMENT}")
    required = plan["disk"]["required_free_gb"]
    print(f"  disk (repo filesystem): free {free:.1f} GB vs required "
          f"{required:.1f} GB")
    if free < required:
        print(f"PREFLIGHT GATE: free disk {free:.1f} GB on the repo filesystem "
              f"is below the required {required:.1f} GB — do not start the "
              f"download (exit {PREFLIGHT_GATE_EXIT}).", file=sys.stderr)
        return PREFLIGHT_GATE_EXIT
    return 0


def _cmd_build_inventory(args) -> int:
    try:
        out_path = os.path.join(
            _check_out_dir(os.path.dirname(os.path.abspath(args.out)) or ".",
                           what="--out"),
            os.path.basename(args.out))
        evidence = _read_json(args.evidence)
        excluded_days = _read_json(args.excluded_days)
        objects = _read_json(args.objects)
        config = _read_json(args.config) if args.config else None
        plan = _read_json(args.plan) if args.plan else None
        if plan is not None and config is None:
            raise ValueError("--plan requires --config: the scope-coverage "
                             "check runs inside the config-anchored handoff "
                             "validation")
        inventory = build_custody_inventory(
            custodian_identity=args.custodian_identity,
            operator_identity=args.operator_identity,
            coverage_sha256=args.coverage_sha256,
            evidence=evidence, excluded_days=excluded_days, objects=objects,
            config=config)
        if config is not None:
            validate_custody_handoff(inventory, evidence, config, plan=plan)
        _write_json(out_path, inventory)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return SETUP_ERROR_EXIT
    validated = ("validated against the supplied protocol config"
                 if config else
                 "PROVISIONAL — not yet validated against a protocol config "
                 "(pin the seal into source_certification, then rerun with "
                 "--config; see the packet's custody-seal-verify step)")
    print(f"custody inventory -> {out_path}")
    label = "" if config else " (PROVISIONAL)"
    print(f"  custodian_seal_sha256{label}: {inventory['custodian_seal_sha256']}")
    print(f"  included days: {len(inventory['included_days'])}, "
          f"objects: {len(inventory['objects'])}; {validated}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="g0bn_acquisition_preflight",
        description="Deterministic no-I/O G0-BN acquisition/custody preflight "
                    "(issue #102). Plans, validates, and emits inert approval "
                    "artifacts; performs no vendor I/O.")
    sub = ap.add_subparsers(dest="command", required=True)
    plan_p = sub.add_parser("plan", help="build the acquisition plan, approval "
                                         "packet, and day files")
    plan_p.add_argument("--custodian-identity", required=True)
    plan_p.add_argument("--operator-identity", required=True)
    plan_p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"ignored output dir (default {DEFAULT_OUT_DIR})")
    plan_p.add_argument("--generated-at", default=None,
                        help="optional ISO-8601 timestamp recorded (hash-excluded)"
                             " in the artifacts; omit for byte-identical output")
    inv_p = sub.add_parser("build-inventory",
                           help="custodian-only: emit the sealed outcome-blind "
                                "inventory from local metadata (no data reads)")
    inv_p.add_argument("--custodian-identity", required=True)
    inv_p.add_argument("--operator-identity", required=True)
    inv_p.add_argument("--coverage-sha256", required=True)
    inv_p.add_argument("--evidence", required=True,
                       help="path to the g0bn-permission-policy-evidence-v1 JSON")
    inv_p.add_argument("--objects", required=True,
                       help="path to the sealed-object entries JSON "
                            "(object_id/layer/product/day/sha256 only)")
    inv_p.add_argument("--excluded-days", required=True,
                       help="path to the {day: {reason, evidence_sha256}} JSON")
    inv_p.add_argument("--config", default=None,
                       help="optional validated g0bn-protocol-config-v1 JSON; "
                            "when given, the inventory is validated through "
                            "eval.g0bn_freeze.validate_custody_inventory")
    inv_p.add_argument("--plan", default=None,
                       help="optional g0bn-acquisition-plan-v1 JSON (requires "
                            "--config); the evidence must then cover every "
                            "planned holdout destination")
    inv_p.add_argument("--out", required=True, help="output inventory JSON path")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        return _cmd_plan(args)
    return _cmd_build_inventory(args)


if __name__ == "__main__":
    raise SystemExit(main())
