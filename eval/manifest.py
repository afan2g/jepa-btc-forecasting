"""Feature-manifest contract (v1): the versioned, self-describing record of a bar/feature
dataset build. Training selects features EXPLICITLY from `feature_cols` — never inferred
from "all non-reserved columns" (AGENTS.md). validate_manifest() checks the manifest
itself; validate_frame() checks a ModelMatrix-shaped frame against it (columns, timing,
leakage). See docs/feature-manifest.md."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from eval.matrix import RESERVED

MANIFEST_VERSION = 1

REQUIRED_FIELDS = (
    "manifest_version", "dataset_id", "build_id", "bar_clock", "time",
    "feature_cols", "target_cols", "reserved_cols", "venues", "horizons",
    "sources", "generated_at", "max_lookback_ns", "embargo_ns",
)
OPTIONAL_FIELDS = ("extra_cols", "dtypes", "availability_lag_ns", "as_of_ns", "gate")

# A feature named like a label/outcome is label-derived by construction (future return,
# forward mid, barrier touch, ...) and can never be a valid model input, whatever the
# manifest claims. Backward-looking names ("ret_30s") are deliberately NOT flagged.
LEAKY_NAME_PATTERNS = ("fwd", "future", "forward", "barrier", "label", "target", "outcome")

VENUE_ROLES = ("signal", "target")

# Fields that only exist in versioned manifests (required AND optional — declaring dtypes
# or as_of_ns on a legacy dict must not be silently ignored). eval.runner uses this to
# refuse a manifest that carries the v1 contract but lost its 'manifest_version' key (e.g.
# a typo) instead of downgrading it to the unvalidated legacy path.
V1_ONLY_FIELDS = (frozenset(REQUIRED_FIELDS) | frozenset(OPTIONAL_FIELDS)) - {
    "manifest_version", "feature_cols", "max_lookback_ns", "embargo_ns", "gate",
}

_TIMING_COLS = ("t_event", "t_barrier", "t_feature_start", "t_available")
# Required in the frame even when targets are optional: the timing/horizon checks need them.
_STRUCTURAL_COLS = frozenset(_TIMING_COLS) | {"horizon"}

# The only core reserved columns that ARE labels. Declaring any other core column
# (cost/weight/tag/timing) as a target would make it optional on the require_targets=False
# path, letting frames validate while missing required non-label reserved data.
CORE_LABEL_COLS = ("y_fwd_bps", "label")


def _leaky_names(cols) -> list[str]:
    out = []
    for c in cols:
        lc = c.lower()
        if lc == "y" or lc.startswith("y_") or any(p in lc for p in LEAKY_NAME_PATTERNS):
            out.append(c)
    return out


def _str_list(name: str, val, *, allow_empty: bool = False) -> list[str]:
    if (not isinstance(val, list) or (not val and not allow_empty)
            or any(not isinstance(c, str) or not c for c in val)):
        raise ValueError(f"{name} must be a non-empty list of non-empty strings")
    dups = sorted({c for c in val if val.count(c) > 1})
    if dups:
        raise ValueError(f"duplicate columns in {name}: {dups}")
    return val


def _ns_int(name: str, val, *, minimum: int = 0) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise ValueError(f"{name} must be an int >= {minimum} (nanoseconds)")
    return val


def validate_manifest(manifest: dict) -> dict:
    """Schema-level validation (no data). Fail closed on anything unknown or ambiguous;
    returns the manifest unchanged for chaining."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    missing = [k for k in REQUIRED_FIELDS if k not in manifest]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")
    unknown = set(manifest) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise ValueError(f"unknown manifest keys (misspelled?): {sorted(unknown)}")

    v = manifest["manifest_version"]
    if isinstance(v, bool) or not isinstance(v, int) or v != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest_version {v!r}; this code supports {MANIFEST_VERSION}")
    for k in ("dataset_id", "build_id"):
        if not isinstance(manifest[k], str) or not manifest[k]:
            raise ValueError(f"{k} must be a non-empty string")

    clock = manifest["bar_clock"]
    if not isinstance(clock, dict) or not isinstance(clock.get("kind"), str) or not clock["kind"]:
        raise ValueError("bar_clock must be a dict with a non-empty 'kind' (e.g. dollar/volume/time)")

    time = manifest["time"]
    if not isinstance(time, dict):
        raise ValueError("time must be a dict with 'unit' and 'timezone'")
    if time.get("unit") != "ns":
        raise ValueError("time.unit must be 'ns' (int64 nanoseconds; v1 supports nothing else)")
    if time.get("timezone") != "UTC":
        raise ValueError("time.timezone must be 'UTC'")
    unknown_time = set(time) - {"unit", "timezone"}
    if unknown_time:
        raise ValueError(f"unknown time keys: {sorted(unknown_time)}")

    feats = _str_list("feature_cols", manifest["feature_cols"])
    targets = _str_list("target_cols", manifest["target_cols"])
    reserved = _str_list("reserved_cols", manifest["reserved_cols"])
    extra = _str_list("extra_cols", manifest.get("extra_cols", []), allow_empty=True)

    core_missing = [c for c in RESERVED if c not in reserved]
    if core_missing:
        raise ValueError(f"reserved_cols must include the core reserved registry; missing: {core_missing}")
    not_reserved = [c for c in targets if c not in reserved]
    if not_reserved:
        raise ValueError(f"target_cols must be a subset of reserved_cols; not reserved: {not_reserved}")
    non_label = sorted(set(targets) & (set(RESERVED) - set(CORE_LABEL_COLS)))
    if non_label:
        raise ValueError(f"core non-label reserved columns cannot be target_cols: {non_label}")

    overlap_t = sorted(set(feats) & set(targets))
    if overlap_t:
        raise ValueError(f"feature_cols overlap target_cols (labels are never features): {overlap_t}")
    overlap_r = sorted(set(feats) & set(reserved))
    if overlap_r:
        raise ValueError(f"feature_cols include reserved columns: {overlap_r}")
    overlap_e = sorted(set(extra) & (set(feats) | set(reserved)))
    if overlap_e:
        raise ValueError(f"extra_cols overlap feature/reserved columns: {overlap_e}")
    leaky = _leaky_names(feats)
    if leaky:
        raise ValueError(f"leaky feature names (future/forward/barrier/label/target-derived): {leaky}; "
                         "features must be computable from data at or before t_event")

    venues = manifest["venues"]
    if not isinstance(venues, list) or not venues:
        raise ValueError("venues must be a non-empty list of {exchange, symbol[, role]} dicts")
    for ven in venues:
        if (not isinstance(ven, dict)
                or not isinstance(ven.get("exchange"), str) or not ven["exchange"]
                or not isinstance(ven.get("symbol"), str) or not ven["symbol"]):
            raise ValueError("each venue entry must include non-empty 'exchange' and 'symbol'")
        if "role" in ven and ven["role"] not in VENUE_ROLES:
            raise ValueError(f"venue role must be one of {VENUE_ROLES}, got {ven['role']!r}")

    horizons = manifest["horizons"]
    if (not isinstance(horizons, dict) or not horizons
            or any(not isinstance(t, str) or not t for t in horizons)
            or any(isinstance(ns, bool) or not isinstance(ns, int) or ns <= 0
                   for ns in horizons.values())):
        raise ValueError("horizons must be a non-empty mapping of tag -> positive int nanoseconds")

    sources = manifest["sources"]
    ok = isinstance(sources, list) and sources and all(
        (isinstance(s, str) and s)
        or (isinstance(s, dict) and isinstance(s.get("name"), str) and s["name"])
        for s in sources)
    if not ok:
        raise ValueError("sources must be a non-empty list of names or dicts with a non-empty 'name'")

    gen = manifest["generated_at"]
    dt = None
    if isinstance(gen, str):
        try:
            dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
        except ValueError:
            dt = None
    if dt is None or dt.tzinfo is None:
        raise ValueError("generated_at must be an ISO-8601 timestamp with an explicit timezone")

    lb = _ns_int("max_lookback_ns", manifest["max_lookback_ns"])
    emb = _ns_int("embargo_ns", manifest["embargo_ns"])
    if emb < lb:
        raise ValueError("embargo_ns must be >= max_lookback_ns (embargo must cover the look-back)")

    if "dtypes" in manifest:
        dtypes = manifest["dtypes"]
        if not isinstance(dtypes, dict):
            raise ValueError("dtypes must be a dict of column -> dtype string")
        undeclared = sorted(set(dtypes) - set(feats) - set(reserved) - set(extra))
        if undeclared:
            raise ValueError(f"dtype expectations for undeclared columns: {undeclared}")
        for col, spec in dtypes.items():
            try:
                pd.api.types.pandas_dtype(spec)
            except (TypeError, ValueError):
                raise ValueError(f"invalid dtype expectation for {col!r}: {spec!r}") from None

    if "availability_lag_ns" in manifest:
        _ns_int("availability_lag_ns", manifest["availability_lag_ns"])
    if "as_of_ns" in manifest:
        if isinstance(manifest["as_of_ns"], bool) or not isinstance(manifest["as_of_ns"], int):
            raise ValueError("as_of_ns must be an int (nanoseconds)")
    if "gate" in manifest and not isinstance(manifest["gate"], dict):
        raise ValueError("gate must be a dict (contents validated by eval.runner.resolve_gate)")
    return manifest


def validate_frame(df: pd.DataFrame, manifest: dict, *, require_targets: bool = True) -> None:
    """Check a ModelMatrix-shaped frame against the manifest: declared columns, timing,
    horizon and availability consistency. require_targets=False is the unsupervised/JEPA
    path where label columns may be absent."""
    validate_manifest(manifest)
    dups = sorted(set(df.columns[df.columns.duplicated()]))
    if dups:
        raise ValueError(f"duplicate columns in frame: {dups}")
    if not len(df):
        raise ValueError("frame is empty")
    feats, targets = manifest["feature_cols"], manifest["target_cols"]
    known = set(feats) | set(manifest["reserved_cols"]) | set(manifest.get("extra_cols", []))

    missing_f = [c for c in feats if c not in df.columns]
    if missing_f:
        raise ValueError(f"manifest features missing from frame: {missing_f}")
    if require_targets:
        missing_t = [c for c in targets if c not in df.columns]
        if missing_t:
            raise ValueError(f"target columns missing from frame: {missing_t}")
    for c in RESERVED:
        if not require_targets and c in targets and c not in _STRUCTURAL_COLS:
            continue
        if c not in df.columns:
            raise ValueError(f"frame missing reserved column {c!r}")
    unknown = [c for c in df.columns if c not in known]
    if unknown:
        raise ValueError(f"unknown frame columns not declared in the manifest: {unknown}; "
                         "declare them in extra_cols or drop them")

    for c in _TIMING_COLS:
        # pd.NA comparisons are skipped by Series.all() (fail-open), and datetime64 math
        # would raise a confusing TypeError below — require non-null integer ns up front.
        if not pd.api.types.is_integer_dtype(df[c]):
            raise ValueError(f"timing column {c!r} must be integer nanoseconds, "
                             f"got {df[c].dtype}")
        if df[c].isna().any():
            raise ValueError(f"timing column {c!r} contains nulls")

    if not (df["t_barrier"] >= df["t_event"]).all():
        raise ValueError("invalid span: require t_barrier >= t_event")
    if not (df["t_feature_start"] <= df["t_event"]).all():
        raise ValueError("invalid timing: require t_feature_start <= t_event")
    observed_lb = int((df["t_event"] - df["t_feature_start"]).max())
    if observed_lb > manifest["max_lookback_ns"]:
        raise ValueError(f"observed look-back {observed_lb} exceeds declared max_lookback_ns "
                         f"{manifest['max_lookback_ns']}")

    horizons = manifest["horizons"]
    tags = list(df["horizon"].unique())
    nonstr = [t for t in tags if not isinstance(t, str)]
    if nonstr:
        # str() coercion here could false-match a manifest key (e.g. int 10 vs "10")
        raise ValueError(f"frame horizon tags must be strings, got: {nonstr}")
    undeclared_h = sorted(set(tags) - set(horizons))
    if undeclared_h:
        raise ValueError(f"frame horizon tags not declared in the manifest: {undeclared_h}")
    # observed=True: a categorical horizon column must not yield empty groups for unused
    # categories (KeyError on tags the frame never contains)
    for tag, sub in df.groupby("horizon", observed=True):
        # barrier touch may come early, but never past the declared vertical barrier
        if not (sub["t_barrier"] - sub["t_event"] <= horizons[tag]).all():
            raise ValueError(f"t_barrier exceeds declared horizon for tag {tag!r}")

    lag = manifest.get("availability_lag_ns", 0)
    delta = df["t_available"] - df["t_event"]
    if not (delta >= 0).all():
        raise ValueError("invalid timing: require t_available >= t_event")
    if not (delta <= lag).all():
        raise ValueError(f"t_available exceeds t_event + availability_lag_ns ({lag}); features must "
                         "be available at or before the sample timestamp")
    if "as_of_ns" in manifest and int(df["t_available"].max()) > manifest["as_of_ns"]:
        raise ValueError("frame claims availability after the manifest as_of_ns snapshot")

    for col, spec in manifest.get("dtypes", {}).items():
        if col in df.columns and df[col].dtype != pd.api.types.pandas_dtype(spec):
            raise ValueError(f"dtype mismatch for {col!r}: manifest expects {spec}, "
                             f"frame has {df[col].dtype}")


def feature_list(manifest: dict) -> list[str]:
    """The model-ready feature list: exactly manifest feature_cols, in order, as a copy."""
    validate_manifest(manifest)
    return list(manifest["feature_cols"])


def target_list(manifest: dict) -> list[str]:
    validate_manifest(manifest)
    return list(manifest["target_cols"])


def load_manifest(path) -> dict:
    with open(path) as f:
        man = json.load(f)
    return validate_manifest(man)


def unsafe_infer_feature_cols(df: pd.DataFrame, *, extra_cols: tuple = ()) -> list[str]:
    """DELIBERATE escape hatch for exploration only (hence the name): every non-reserved,
    non-extra column in frame order. Never a training input path — write the result into a
    manifest and pre-register it. Refuses (fails closed) if any candidate is leaky-named;
    columns named in extra_cols are excluded without leak-screening."""
    if isinstance(extra_cols, str):
        raise ValueError("extra_cols must be a sequence of column names, not a string")
    nonstr = [c for c in df.columns if not isinstance(c, str)]
    if nonstr:
        raise ValueError(f"column labels must be strings, got: {nonstr}")
    dups = sorted(set(df.columns[df.columns.duplicated()]))
    if dups:
        raise ValueError(f"duplicate columns in frame: {dups}")
    skip = set(RESERVED) | set(extra_cols)
    candidates = [c for c in df.columns if c not in skip]
    leaky = _leaky_names(candidates)
    if leaky:
        raise ValueError(f"refusing to infer: leaky-named columns in frame: {leaky}; "
                         "rename them or pass them via extra_cols")
    return candidates
