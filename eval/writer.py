"""Deterministic ModelMatrix + feature-manifest publication (T8, issue #87).

This module is the explicit writer API around eval.manifest / eval.matrix:

- development/rebuildable builds run BOTH `eval.manifest.validate_frame` and
  `eval.matrix.validate_matrix` before any byte is written (plan §H: a bad build never
  reaches data/processed/);
- the G0-BN blind holdout write performs only the structural checks needed to produce
  and hash a deterministic artifact, opens fresh outputs, closes/fsyncs them, and
  returns every attestation input WITHOUT ever reopening a derived matrix/parquet/
  footer (spec §6.3; formal value validation belongs to the post-matrix-burn scorer);
- canonical logical-row/build identity is content (manifest-ordered columns, rows in
  (t_event, horizon) order, value bytes) — never Parquet file bytes. The matrix-file
  and physical-schema hashes are separate audit/attestation outputs (plan §I): two
  physical encodings of the same logical rows share logical/build identity while their
  file hashes differ.

G0-BN manifests (docs/superpowers/specs/2026-07-13-g0bn-protocol.md §2/§7,
docs/feature-manifest.md) carry exactly one `partition_contract` and one
`g0bn_protocol` source binding; a holdout manifest additionally carries exactly one
`g0bn_holdout_plan` binding. `classify_manifest` is the manifest-only preflight the
generic-runner guard (67-D / #90) consumes before any parquet loader is invoked.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bars.cost import CostAssumption, VENUE_BINANCE, validate_cost_assumption
from eval.hashing import canonical_json, canonical_row_order, hash_obj, matrix_content_hash
from eval.manifest import feature_list, validate_frame, validate_manifest, write_manifest
from eval.manifest import manifest_sha256 as _manifest_sha256
from eval.matrix import TIMING_COLS, validate_matrix

# --------------------------------------------------------------------- G0-BN constants
# Staged dataset identities (docs/feature-manifest.md): development and holdout are
# physically/logically separate builds and must never share an ID or binding set.
G0BN_DEV_DATASET_ID = "binance_single_venue_g0bn_dev"
G0BN_OOS_DATASET_ID = "binance_single_venue_g0bn_oos"

# Exact single-instrument identity (spec §2.1). The validator admits this object and
# nothing else — a second venue, spot, another asset, or an extra key fails closed.
G0BN_INSTRUMENT = {
    "exchange": "BINANCE_FUTURES",
    "native_symbol": "BTCUSDT",
    "symbol": "BTC-USDT-PERP",
    "contract_type": "linear_perpetual",
    "base_asset": "BTC",
    "quote_asset": "USDT",
    "settlement_asset": "USDT",
}
# The manifest venue entry (plan §H): exactly one, no role key — the same instrument
# supplies features and targets, so a signal/target split would be meaningless.
G0BN_VENUE = {"exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP"}

# Ordered feature registry (spec §3.3). Order is decision-bearing: it enters the
# candidate/protocol hashes, so a permutation is a different (rejected) manifest.
G0BN_FEATURES = (
    "ofi_integrated", "microprice_dev", "queue_imb", "spread_tick", "cvd",
    "depth_imbalance", "book_slope", "vwap_minus_mid", "trade_count", "signed_vol",
    "aggressor_imb", "largest_print", "event_intensity", "rv_intrabar", "mae_intrabar",
    "elapsed_ns", "tod_sin", "tod_cos",
)
G0BN_TARGETS = ("y_fwd_bps", "label")
G0BN_HORIZONS_NS = {"2s": 2_000_000_000, "10s": 10_000_000_000, "60s": 60_000_000_000}
# Binary64 storage pins (spec §8.2): the scorer performs no float32 round trip, so the
# manifest must declare — and the frame must physically carry — float64 exactly.
G0BN_COST_DTYPES = {"cost_bps": "float64", "half_spread_bps": "float64",
                    "latency_drift_bps": "float64"}

PARTITION_BINDING = "partition_contract"
PROTOCOL_BINDING = "g0bn_protocol"
HOLDOUT_PLAN_BINDING = "g0bn_holdout_plan"
_BINDING_NAMES = (PARTITION_BINDING, PROTOCOL_BINDING, HOLDOUT_PLAN_BINDING)

_PARTITIONS = ("development", "holdout")
_PARTITION_PLAN_SCHEMA = "g0bn-partition-plan-v1"
_PROTOCOL_ID = "g0bn-v1"
_ONE_SHOT_PROTOCOL = "g0bn-one-shot-v1"
_CONSUMPTION_SCHEMA = "g0bn-consumption-v1"

# Manifest-level source taxonomy for G0-BN builds (spec §2.1, plan §H): the certified
# Binance Futures L2 snapshot/delta + trade products, their evidence artifacts, and
# T7's evidenced cost assumption. Anything else (Coinbase, CoinAPI, spot, funding/OI/
# liquidations/basis, stitch plans) fails before any loader. Verifying entry CONTENT
# against the #64/#68 evidence is #67's job; the manifest gate is name-level.
G0BN_DATA_SOURCES = ("binance_futures_l2_snapshot", "binance_futures_l2_delta",
                     "binance_futures_trades")
COST_ASSUMPTION_SOURCE = "cost_assumption"
_G0BN_EVIDENCE_SOURCES = ("source_certification", "custodian_seal", "coverage")

_HEX64 = re.compile(r"[0-9a-f]{64}")

# Hash domain tags: logical-row/build/physical-schema identities live in different
# namespaces so equal input bytes can never alias across identity kinds.
_LOGICAL_ROWS_TAG = "model-matrix-logical-rows-v1"
_BUILD_ID_TAG = "model-matrix-build-v1"
_PHYSICAL_SCHEMA_TAG = "model-matrix-physical-schema-v1"

_FLOAT64_RESERVED = ("y_fwd_bps", "cost_bps", "half_spread_bps", "uniqueness")

# Binding binary64 reconciliation tolerance (spec §8.2 math.isclose policy).
_COST_TOL = 1e-12


def _hex64_field(ctx: str, field: str, val) -> str:
    if not isinstance(val, str) or not _HEX64.fullmatch(val):
        raise ValueError(f"{ctx}: {field} must be 64 lowercase hex characters, got {val!r}")
    return val


def _exact_fields(ctx: str, entry: dict, required: tuple) -> None:
    missing = sorted(set(required) - set(entry))
    unknown = sorted(set(entry) - set(required))
    if missing or unknown:
        raise ValueError(f"{ctx} binding requires exactly the fields {sorted(required)}; "
                         f"missing {missing}, unknown {unknown}")


# --------------------------------------------------------------------- G0-BN bindings

def _validate_partition_binding(b: dict) -> str:
    _exact_fields(PARTITION_BINDING, b,
                  ("name", "schema", "partition", "partition_plan_sha256"))
    if b["schema"] != _PARTITION_PLAN_SCHEMA:
        raise ValueError(f"{PARTITION_BINDING}: schema must be {_PARTITION_PLAN_SCHEMA!r}, "
                         f"got {b['schema']!r}")
    if b["partition"] not in _PARTITIONS:
        raise ValueError(f"{PARTITION_BINDING}: partition must be one of {_PARTITIONS}, "
                         f"got {b['partition']!r}")
    _hex64_field(PARTITION_BINDING, "partition_plan_sha256", b["partition_plan_sha256"])
    return b["partition"]


def _validate_protocol_binding(b: dict) -> None:
    _exact_fields(PROTOCOL_BINDING, b,
                  ("name", "protocol", "protocol_config_sha256",
                   "source_certification_sha256", "horizon_roles_sha256", "instrument"))
    if b["protocol"] != _PROTOCOL_ID:
        raise ValueError(f"{PROTOCOL_BINDING}: protocol must be {_PROTOCOL_ID!r}, "
                         f"got {b['protocol']!r}")
    for field in ("protocol_config_sha256", "source_certification_sha256",
                  "horizon_roles_sha256"):
        _hex64_field(PROTOCOL_BINDING, field, b[field])
    if b["instrument"] != G0BN_INSTRUMENT:
        raise ValueError(f"{PROTOCOL_BINDING}: instrument must be exactly the G0-BN "
                         f"Binance BTC-USDT linear-perpetual identity {G0BN_INSTRUMENT}; "
                         f"got {b['instrument']!r}")


def _validate_holdout_plan_binding(b: dict) -> None:
    _exact_fields(HOLDOUT_PLAN_BINDING, b,
                  ("name", "protocol", "consumption_schema", "holdout_universe_id",
                   "transaction_id", "holdout_plan_sha256", "freeze_sha256"))
    if b["protocol"] != _ONE_SHOT_PROTOCOL:
        raise ValueError(f"{HOLDOUT_PLAN_BINDING}: protocol must be "
                         f"{_ONE_SHOT_PROTOCOL!r}, got {b['protocol']!r}")
    if b["consumption_schema"] != _CONSUMPTION_SCHEMA:
        raise ValueError(f"{HOLDOUT_PLAN_BINDING}: consumption_schema must be "
                         f"{_CONSUMPTION_SCHEMA!r}, got {b['consumption_schema']!r}")
    for field in ("holdout_universe_id", "transaction_id", "holdout_plan_sha256",
                  "freeze_sha256"):
        _hex64_field(HOLDOUT_PLAN_BINDING, field, b[field])


def _validate_cost_assumption_source(entry: dict) -> None:
    """The manifest's serialized CostAssumption must reconstruct under T7's identity
    contract AND match the G0-BN venue/product exactly (spec §8.2)."""
    _exact_fields(COST_ASSUMPTION_SOURCE, entry,
                  ("name", "venue", "product", "source", "version", "taker_fee_bps",
                   "base_slippage_bps", "drift_policy"))
    try:
        validate_cost_assumption(CostAssumption(
            venue=entry["venue"], product=entry["product"], source=entry["source"],
            version=entry["version"], taker_fee_bps=entry["taker_fee_bps"],
            base_slippage_bps=entry["base_slippage_bps"],
            drift_policy=entry["drift_policy"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{COST_ASSUMPTION_SOURCE}: {exc}") from None
    if entry["venue"] != VENUE_BINANCE:
        raise ValueError(f"{COST_ASSUMPTION_SOURCE}: G0-BN requires venue "
                         f"{VENUE_BINANCE!r}, got {entry['venue']!r}")
    if entry["product"] != G0BN_INSTRUMENT["symbol"]:
        raise ValueError(f"{COST_ASSUMPTION_SOURCE}: G0-BN requires product "
                         f"{G0BN_INSTRUMENT['symbol']!r}, got {entry['product']!r}")


def _sources_by_name(manifest: dict) -> dict:
    named: dict[str, list] = {}
    for s in manifest["sources"]:
        if not isinstance(s, dict):
            raise ValueError("G0-BN source entries must be structured dicts with a "
                             f"'name'; got string source {s!r}")
        named.setdefault(s["name"], []).append(s)
    return named


def validate_g0bn_manifest(manifest: dict) -> dict:
    """Full manifest-level G0-BN contract (spec §2/§7, plan §H): exact bindings, exact
    single-venue template, exact ordered feature registry, float64 cost pins, and
    development/holdout isolation. Purely dict-level — never opens data. Returns the
    manifest unchanged for chaining."""
    validate_manifest(manifest)
    ds = manifest["dataset_id"]
    if ds not in (G0BN_DEV_DATASET_ID, G0BN_OOS_DATASET_ID):
        raise ValueError(f"G0-BN dataset_id must be {G0BN_DEV_DATASET_ID!r} "
                         f"(development) or {G0BN_OOS_DATASET_ID!r} (holdout); got {ds!r}")
    _hex64_field("G0-BN manifest", "build_id", manifest["build_id"])

    named = _sources_by_name(manifest)
    for name in (PARTITION_BINDING, PROTOCOL_BINDING):
        if len(named.get(name, [])) != 1:
            raise ValueError(f"G0-BN manifest requires exactly one {name!r} source "
                             f"binding; got {len(named.get(name, []))}")
    partition = _validate_partition_binding(named[PARTITION_BINDING][0])
    _validate_protocol_binding(named[PROTOCOL_BINDING][0])

    n_plan = len(named.get(HOLDOUT_PLAN_BINDING, []))
    if partition == "development":
        if ds != G0BN_DEV_DATASET_ID:
            raise ValueError(f"partition/dataset_id mismatch: a development binding "
                             f"requires dataset_id {G0BN_DEV_DATASET_ID!r}, got {ds!r}")
        if n_plan:
            raise ValueError("a development G0-BN manifest must not carry a "
                             "g0bn_holdout_plan binding (development/holdout isolation)")
    else:
        if ds != G0BN_OOS_DATASET_ID:
            raise ValueError(f"partition/dataset_id mismatch: a holdout binding "
                             f"requires dataset_id {G0BN_OOS_DATASET_ID!r}, got {ds!r}")
        if n_plan != 1:
            raise ValueError(f"a holdout G0-BN manifest requires exactly one "
                             f"g0bn_holdout_plan binding; got {n_plan}")
        _validate_holdout_plan_binding(named[HOLDOUT_PLAN_BINDING][0])

    allowed = (set(_BINDING_NAMES) | set(G0BN_DATA_SOURCES)
               | set(_G0BN_EVIDENCE_SOURCES) | {COST_ASSUMPTION_SOURCE})
    unknown = sorted(set(named) - allowed)
    if unknown:
        raise ValueError(f"forbidden/unknown G0-BN source entries: {unknown}; "
                         f"allowed source names: {sorted(allowed)}")
    for name in G0BN_DATA_SOURCES:
        if not named.get(name):
            raise ValueError(f"G0-BN manifest requires at least one {name!r} source entry")
    if len(named.get("source_certification", [])) != 1:
        raise ValueError("G0-BN manifest requires exactly one source_certification "
                         "entry (the #64 evidence pin)")
    n_seal = len(named.get("custodian_seal", []))
    if partition == "holdout" and n_seal != 1:
        raise ValueError("a holdout G0-BN manifest requires exactly one custodian_seal "
                         f"source entry (the #68 January seal); got {n_seal}")
    if n_seal > 1:
        raise ValueError(f"at most one custodian_seal source entry is allowed; got {n_seal}")
    for name, entries in named.items():
        if name in _BINDING_NAMES or name == COST_ASSUMPTION_SOURCE:
            continue
        for entry in entries:
            if "sha256" not in entry:
                raise ValueError(f"G0-BN source {name!r} entry requires an explicit "
                                 "sha256 pin (every referenced artifact carries one)")
            _hex64_field(name, "sha256", entry["sha256"])
    n_cost = len(named.get(COST_ASSUMPTION_SOURCE, []))
    if n_cost != 1:
        raise ValueError("G0-BN manifest requires exactly one cost_assumption source "
                         f"(the evidenced T7 CostAssumption); got {n_cost}")
    _validate_cost_assumption_source(named[COST_ASSUMPTION_SOURCE][0])

    if manifest["venues"] != [G0BN_VENUE]:
        raise ValueError(f"G0-BN venues must be exactly [{G0BN_VENUE!r}] "
                         f"(single venue, no role key); got {manifest['venues']!r}")
    if list(manifest["feature_cols"]) != list(G0BN_FEATURES):
        raise ValueError("G0-BN feature_cols must be exactly the ordered spec-§3.3 "
                         f"registry {list(G0BN_FEATURES)}; got {manifest['feature_cols']!r}")
    if list(manifest["target_cols"]) != list(G0BN_TARGETS):
        raise ValueError(f"G0-BN target_cols must be exactly {list(G0BN_TARGETS)} in "
                         f"order; got {manifest['target_cols']!r}")
    if manifest["horizons"] != G0BN_HORIZONS_NS:
        raise ValueError(f"G0-BN horizons must be exactly {G0BN_HORIZONS_NS}; "
                         f"got {manifest['horizons']!r}")
    if "latency_drift_bps" not in manifest.get("extra_cols", []):
        raise ValueError("G0-BN manifests must declare latency_drift_bps in extra_cols "
                         "(required non-feature diagnostic; docs/feature-manifest.md)")
    dtypes = manifest.get("dtypes")
    if dtypes is None:
        raise ValueError("G0-BN manifests require a dtypes map pinning the float64 "
                         "cost columns")
    for col, want in G0BN_COST_DTYPES.items():
        if dtypes.get(col) != want:
            raise ValueError(f"G0-BN dtypes must pin {col} = float64 exactly (binary64, "
                             f"no float32 downcast); got {dtypes.get(col)!r}")
    if manifest["embargo_ns"] != manifest["max_lookback_ns"]:
        raise ValueError("G0-BN requires embargo_ns == max_lookback_ns exactly "
                         "(t_barrier already carries the realized label span; spec §9)")
    if manifest.get("availability_lag_ns", 0) != 0:
        raise ValueError("G0-BN requires availability_lag_ns == 0 (synchronous "
                         "decide-and-act; lag features upstream)")
    return manifest


# --------------------------------------------------------------------- classification

@dataclass(frozen=True)
class ManifestClass:
    """Manifest-only preflight result (spec §7) for the generic-runner guard (#90):
    holdout_bound=True means no generic API/CLI may pass this manifest to any parquet
    loader — only the dedicated one-shot scorer after its matrix-access burn."""
    dataset_id: str
    is_g0bn: bool
    partition: str | None
    holdout_bound: bool


def classify_manifest(manifest: dict) -> ManifestClass:
    """Classify a manifest WITHOUT touching parquet. Non-G0-BN manifests pass through;
    any manifest carrying a G0-BN marker (a binding or a G0-BN dataset_id) must satisfy
    the full binding contract or this raises — a missing/ambiguous partition binding on
    a G0-BN build is a rejection, never a fallback to generic handling."""
    validate_manifest(manifest)
    ds = manifest["dataset_id"]
    has_binding = any(isinstance(s, dict) and s.get("name") in _BINDING_NAMES
                      for s in manifest["sources"])
    if not has_binding and ds not in (G0BN_DEV_DATASET_ID, G0BN_OOS_DATASET_ID):
        return ManifestClass(dataset_id=ds, is_g0bn=False, partition=None,
                             holdout_bound=False)
    validate_g0bn_manifest(manifest)
    partition = next(s["partition"] for s in manifest["sources"]
                     if isinstance(s, dict) and s.get("name") == PARTITION_BINDING)
    return ManifestClass(dataset_id=ds, is_g0bn=True, partition=partition,
                         holdout_bound=partition == "holdout")


# --------------------------------------------------------------------- identities

def ordered_manifest_columns(manifest: dict) -> list[str]:
    """The canonical published column order: the manifest's explicit declaration order
    (feature_cols, then reserved_cols, then extra_cols). Never inferred from a frame."""
    validate_manifest(manifest)
    return (list(manifest["feature_cols"]) + list(manifest["reserved_cols"])
            + list(manifest.get("extra_cols", [])))


def _logical_dtype(s: pd.Series) -> str:
    """Normalized logical dtype label — pandas version/backend spellings (object vs str
    string columns) must not change logical identity; value bytes are hashed separately
    by eval.hashing._column_bytes."""
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int64"
    if pd.api.types.is_float_dtype(s):
        return "float64"
    return "str"


def logical_row_sha256(frame: pd.DataFrame, ordered_cols) -> str:
    """Canonical logical-row hash: the manifest-ordered (name, logical dtype) schema
    plus value bytes in canonical (t_event, horizon) row order. Parquet metadata, row
    groups, compression, and writer versions cannot change it (plan §I). Duplicate
    (t_event, horizon) keys are rejected: they would make the canonical order non-total
    and the hash order-dependent."""
    cols = list(ordered_cols)
    dup_labels = sorted(set(frame.columns[frame.columns.duplicated()]))
    if dup_labels:
        raise ValueError(f"duplicate columns in frame: {dup_labels}")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"logical_row_sha256: columns missing from frame: {missing}")
    for key in ("t_event", "horizon"):
        if key not in frame.columns:
            raise ValueError(f"logical_row_sha256: frame lacks canonical sort key {key!r}")
    categorical = [c for c in cols if isinstance(frame[c].dtype, pd.CategoricalDtype)]
    if categorical:
        # A categorical sorts by category order (not value) and byte-encodes via codes,
        # so identical logical content would silently hash differently.
        raise ValueError(f"categorical columns are not canonical-hashable: {categorical}; "
                         "cast them to their value dtype before hashing")
    if frame.duplicated(subset=["t_event", "horizon"]).any():
        raise ValueError("duplicate (t_event, horizon) rows: the canonical row order "
                         "requires the producer's uniqueness invariant")
    schema = [[c, _logical_dtype(frame[c])] for c in cols]
    return hash_obj({"schema": _LOGICAL_ROWS_TAG, "columns": schema,
                     "rows_sha256": matrix_content_hash(frame, cols)})


def build_id_for(*, dataset_id: str, logical_row_sha256: str, build_params: dict) -> str:
    """Deterministic build identity: canonical logical rows + dataset identity + ALL
    build parameters. `generated_at` is excluded by contract (plan §I: the only field
    allowed to differ between otherwise-identical builds), so identical rebuilds share
    a build_id. Never a hash of Parquet file bytes."""
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"dataset_id must be a non-empty string, got {dataset_id!r}")
    _hex64_field("build_id_for", "logical_row_sha256", logical_row_sha256)
    if not isinstance(build_params, dict):
        raise ValueError("build_params must be a dict of explicit build parameters")
    if "generated_at" in build_params:
        raise ValueError("build_params must not contain generated_at: identical "
                         "rebuilds must share a build_id (plan §I)")
    try:
        canonical_json(build_params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"build_params must be canonical-JSON-encodable (strict JSON, "
                         f"no NaN/Infinity/non-primitive values): {exc}") from None
    return hash_obj({"schema": _BUILD_ID_TAG, "dataset_id": dataset_id,
                     "logical_row_sha256": logical_row_sha256,
                     "build_params": build_params})


def _physical_schema_sha256(schema: pa.Schema) -> str:
    """Arrow physical-schema hash over ordered (name, type, nullable) only — pandas/
    pyarrow version metadata is excluded so the attestation is environment-stable, and
    encodings/compression stay file-hash territory."""
    fields = [[f.name, str(f.type), bool(f.nullable)] for f in schema]
    return hash_obj({"schema": _PHYSICAL_SCHEMA_TAG, "fields": fields})


# --------------------------------------------------------------------- write plumbing

class _HashingSink:
    """Write-only sequential sink: forwards bytes to the raw binary file while hashing
    them, so matrix_file_sha256 exists at close WITHOUT reopening the artifact (spec
    §6.3: the blind materializer never rereads a derived parquet/footer). Reads and
    seeks are disabled — a seeking writer would silently corrupt the streamed hash."""

    def __init__(self, raw):
        self._raw = raw
        self._hash = hashlib.sha256()

    def write(self, data):
        chunk = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        self._hash.update(chunk)
        return self._raw.write(chunk)

    def flush(self):
        self._raw.flush()

    def tell(self):
        return self._raw.tell()

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def seek(self, *args, **kwargs):
        raise io.UnsupportedOperation("write-only hashing sink: seek would corrupt the "
                                      "streamed matrix_file_sha256")

    def read(self, *args, **kwargs):
        raise io.UnsupportedOperation("write-only hashing sink: no reads")

    @property
    def closed(self):
        return self._raw.closed

    def close(self):
        # The owning `with open(...)` closes the raw file; pyarrow closing the wrapper
        # after flushing its buffers must not double-close the descriptor.
        pass

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _fsync_dir(path) -> None:
    fd = os.open(os.path.dirname(os.path.abspath(os.fspath(path))) or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_parquet(table: pa.Table, path, parquet_options, *, exclusive: bool,
                   fsync: bool) -> tuple[str, str]:
    opts = dict(parquet_options or {})
    with open(path, "xb" if exclusive else "wb") as raw:
        sink = _HashingSink(raw)
        pq.write_table(table, sink, **opts)
        raw.flush()
        if fsync:
            os.fsync(raw.fileno())
    if fsync:
        _fsync_dir(path)
    return sink.hexdigest(), _physical_schema_sha256(table.schema)


@dataclass(frozen=True)
class WriteResult:
    """Everything T9's attestation and #67's identities need, computed during the
    write: canonical logical/build identity (modeling content), the byte-level
    matrix-file hash (custody/audit only — never a trial identity), the Arrow
    physical-schema hash, and the canonical manifest hash."""
    dataset_id: str
    build_id: str
    logical_row_sha256: str
    manifest_sha256: str
    matrix_file_sha256: str
    physical_schema_sha256: str
    row_count: int
    matrix_path: str
    manifest_path: str


def _check_write_schema(frame: pd.DataFrame, manifest: dict) -> None:
    """Structural/physical checks BOTH modes need to emit a deterministic, attestable
    artifact: exact declared column set, exact physical dtypes (binary64 floats, int64
    ns timing, no float32 anywhere in features/costs), and declared-dtype pins.
    Value/timing/leakage validation deliberately does NOT live here — development runs
    validate_frame + validate_matrix, the blind holdout write defers values to the
    post-matrix-burn scorer."""
    dup_labels = sorted(set(frame.columns[frame.columns.duplicated()]))
    if dup_labels:
        raise ValueError(f"duplicate columns in frame: {dup_labels}")
    cols = ordered_manifest_columns(manifest)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"declared columns missing from frame: {missing} — the "
                         "manifest must describe exactly the published matrix")
    declared = set(cols)
    undeclared = [c for c in frame.columns if c not in declared]
    if undeclared:
        raise ValueError(f"frame columns not declared in the manifest: {undeclared}; "
                         "declare them in extra_cols or drop them")
    for c in manifest["feature_cols"]:
        if str(frame[c].dtype) != "float64":
            raise ValueError(f"feature column {c!r} must be float64 (binary64) exactly, "
                             f"got {frame[c].dtype}")
    for c in TIMING_COLS:
        if str(frame[c].dtype) != "int64":
            raise ValueError(f"timing column {c!r} must be int64 nanoseconds exactly, "
                             f"got {frame[c].dtype}")
    for c in _FLOAT64_RESERVED:
        if str(frame[c].dtype) != "float64":
            raise ValueError(f"reserved column {c!r} must be float64 (binary64) exactly, "
                             f"got {frame[c].dtype}")
    if str(frame["label"].dtype) != "int64":
        raise ValueError(f"label must be int64, got {frame['label'].dtype}")
    for c in ("regime", "horizon"):
        if (pd.api.types.is_numeric_dtype(frame[c])
                or isinstance(frame[c].dtype, pd.CategoricalDtype)):
            raise ValueError(f"{c!r} must be a plain string tag column (categoricals "
                             f"break canonical ordering), got {frame[c].dtype}")
    for col, spec in manifest.get("dtypes", {}).items():
        if frame[col].dtype != pd.api.types.pandas_dtype(spec):
            raise ValueError(f"dtype mismatch for {col!r}: manifest pins {spec}, frame "
                             f"has {frame[col].dtype}")


def _cost_assumption_entry(manifest: dict) -> dict:
    return next(s for s in manifest["sources"]
                if isinstance(s, dict) and s.get("name") == COST_ASSUMPTION_SOURCE)


def _check_g0bn_development_values(frame: pd.DataFrame, manifest: dict) -> None:
    """Development-only value gates for the G0-BN cost diagnostics: latency_drift_bps
    is a finite non-negative non-feature diagnostic, and T7's realized identity
    cost_bps = 2*taker_fee_bps + base_slippage_bps + latency_drift_bps must reconcile
    under the binding binary64 tolerance (spec §8.2). The blind holdout write must not
    inspect these values; its scorer re-runs this after the matrix-access burn."""
    drift = frame["latency_drift_bps"].to_numpy(np.float64)
    if not np.isfinite(drift).all() or (drift < 0).any():
        raise ValueError("latency_drift_bps must be finite and non-negative "
                         "(required non-feature diagnostic)")
    cost = _cost_assumption_entry(manifest)
    expected = (2.0 * float(cost["taker_fee_bps"]) + float(cost["base_slippage_bps"])
                + drift)
    actual = frame["cost_bps"].to_numpy(np.float64)
    tol = np.maximum(_COST_TOL * np.maximum(np.abs(actual), np.abs(expected)), _COST_TOL)
    if not (np.abs(actual - expected) <= tol).all():
        raise ValueError("cost_bps does not reconcile with 2*taker_fee_bps + "
                         "base_slippage_bps + latency_drift_bps under the binding "
                         "1e-12 binary64 tolerance (spec §8.2)")


def _publish(frame: pd.DataFrame, manifest: dict, *, build_params: dict, matrix_path,
             manifest_path, parquet_options, exclusive: bool, fsync: bool) -> WriteResult:
    cols = ordered_manifest_columns(manifest)
    lrh = logical_row_sha256(frame, cols)
    bid = build_id_for(dataset_id=manifest["dataset_id"], logical_row_sha256=lrh,
                       build_params=build_params)
    if manifest["build_id"] != bid:
        raise ValueError(f"manifest build_id {manifest['build_id']!r} does not match "
                         f"the recomputed build identity {bid!r} (canonical logical "
                         "rows + build params; plan §I)")
    if exclusive:
        # Preflight BOTH destinations before writing either: the matrix parquet is
        # written first, so without this an existing manifest_path would raise only
        # AFTER a fresh derived January parquet was created and fsynced — a stray
        # published holdout matrix from a failed one-shot write (spec §5.1/§6.3). The
        # §6.2 process-owner lock guarantees no concurrent writer, so an up-front check
        # plus each create's O_EXCL fully closes the partial-artifact window.
        existing = [str(p) for p in (matrix_path, manifest_path) if os.path.exists(p)]
        if existing:
            raise FileExistsError(
                f"refusing a blind holdout write to existing output path(s): {existing}; "
                "the one-shot materializer requires fresh outputs and never overwrites "
                "a derived January artifact")
    published = canonical_row_order(frame)[cols]
    table = pa.Table.from_pandas(published, preserve_index=False)
    file_sha, schema_sha = _write_parquet(table, matrix_path, parquet_options,
                                          exclusive=exclusive, fsync=fsync)
    man_sha = write_manifest(manifest, manifest_path, exclusive=exclusive, fsync=fsync)
    if fsync:
        _fsync_dir(manifest_path)
    return WriteResult(dataset_id=manifest["dataset_id"], build_id=bid,
                       logical_row_sha256=lrh, manifest_sha256=man_sha,
                       matrix_file_sha256=file_sha, physical_schema_sha256=schema_sha,
                       row_count=len(published), matrix_path=str(matrix_path),
                       manifest_path=str(manifest_path))


# --------------------------------------------------------------------- public writers

def write_development(frame: pd.DataFrame, manifest: dict, *, build_params: dict,
                      matrix_path, manifest_path, parquet_options: dict | None = None
                      ) -> WriteResult:
    """Validate-then-publish for development and other rebuildable builds (plan §H):
    BOTH validate_frame and validate_matrix must pass before any byte reaches disk, so
    a bad build never gets published. Refuses every holdout-bound manifest — the blind
    holdout write/attestation path is write_holdout, with no generic override."""
    cls = classify_manifest(manifest)
    if cls.holdout_bound:
        raise ValueError("write_development refuses holdout-bound manifests: the blind "
                         "holdout write/attestation path is write_holdout (spec §6.3), "
                         "and generic development writes are never "
                         "transaction-authorized")
    if isinstance(build_params, dict) and "holdout_plan_sha256" in build_params:
        raise ValueError("development build_params must not bind holdout_plan_sha256 "
                         "(development/holdout identity isolation)")
    validate_frame(frame, manifest)
    validate_matrix(frame, feature_list(manifest))
    _check_write_schema(frame, manifest)
    if cls.is_g0bn:
        _check_g0bn_development_values(frame, manifest)
    return _publish(frame, manifest, build_params=build_params,
                    matrix_path=matrix_path, manifest_path=manifest_path,
                    parquet_options=parquet_options, exclusive=False, fsync=False)


def write_holdout(frame: pd.DataFrame, manifest: dict, *, build_params: dict,
                  matrix_path, manifest_path, parquet_options: dict | None = None
                  ) -> WriteResult:
    """T9's blind write/attestation path (spec §6.3 steps 4-5). Only the structural
    checks needed to produce and hash a deterministic artifact run here; formal value
    validation belongs to the post-matrix-burn scorer, and a bad value there is
    terminal INCONCLUSIVE, not a rebuild path. Outputs are opened fresh (O_EXCL),
    closed and fsynced, and every attestation hash in the WriteResult is computed
    WHILE writing — the derived matrix/parquet/footer is never reopened."""
    cls = classify_manifest(manifest)
    if not cls.is_g0bn or cls.partition != "holdout":
        raise ValueError("write_holdout accepts only a G0-BN holdout manifest; "
                         "development/rebuildable builds must use write_development")
    plan = next(s for s in manifest["sources"]
                if isinstance(s, dict) and s.get("name") == HOLDOUT_PLAN_BINDING)
    if (not isinstance(build_params, dict)
            or build_params.get("holdout_plan_sha256") != plan["holdout_plan_sha256"]):
        raise ValueError("holdout build_params must bind holdout_plan_sha256 equal to "
                         "the g0bn_holdout_plan binding: the plan hash enters the OOS "
                         "build parameters and breaks the freeze/build identity cycle "
                         "(spec §5.2)")
    _check_write_schema(frame, manifest)
    return _publish(frame, manifest, build_params=build_params,
                    matrix_path=matrix_path, manifest_path=manifest_path,
                    parquet_options=parquet_options, exclusive=True, fsync=True)
