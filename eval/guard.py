"""67-D generic-runner holdout guard (issue #90; spec section 7 of
docs/superpowers/specs/2026-07-13-g0bn-protocol.md).

Generic APIs and CLIs are never transaction-authorized. Every generic route from a
manifest to matrix bytes — pandas/pyarrow loaders, Parquet footers, caller-supplied
loaders, or an already-loaded frame — runs `preflight_generic_manifest` FIRST.
`eval.writer.classify_manifest` stays the authoritative G0-BN classifier: a marked
manifest with missing/duplicate/malformed/ambiguous bindings raises there, never
falling back to generic handling. The raw marker scans here additionally reject each
holdout marker INDEPENDENTLY and before any schema validation, so renaming/removing
sibling bindings (or tearing the manifest apart) cannot downgrade a holdout artifact
into a generic build. There is deliberately no override/force/validation-only
parameter: the dedicated one-shot scorer (spec section 6.3) is the only authorized
post-burn holdout consumer, and it does not route through this module. The legacy
G0/G0-XV holdout path is likewise its own dedicated machinery
(scripts/run_g0.py holdout-score), so a legacy `partition_contract` binding naming
`holdout` refuses here too — only its development bindings flow through generic paths.
"""
from __future__ import annotations

import pandas as pd

from eval.writer import (
    G0BN_OOS_DATASET_ID,
    HOLDOUT_PLAN_BINDING,
    PARTITION_BINDING,
    ManifestClass,
    classify_manifest,
)


def _refusal(marker: str) -> ValueError:
    return ValueError(
        f"holdout guard: {marker}; generic runners and CLIs are never "
        "transaction-authorized to open a holdout-bound build — only the dedicated "
        "one-shot scorer may consume it, after its matrix-access burn (spec section 7)")


def preflight_generic_manifest(manifest: dict) -> ManifestClass:
    """Manifest-only preflight for every generic runner/CLI entry point: raises before
    the caller opens the matrix, a Parquet footer, or any loader. Each holdout marker
    rejects independently via raw pre-validation scans; a G0-BN-marked manifest must
    then satisfy the full binding contract via classify_manifest (missing/duplicate/
    ambiguous bindings raise there). Non-G0-BN manifests — including legacy G0/G0-XV
    development bindings — and development G0-BN manifests pass through, returned
    classified."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    sources = manifest.get("sources")
    # Raw scans, deliberately shape-tolerant: a torn/malformed manifest carrying any
    # holdout marker must surface the stable refusal, not an incidental schema error.
    # Non-dict entries are skipped here; their rejection belongs to classify_manifest.
    for s in (sources if isinstance(sources, list) else ()):
        if not isinstance(s, dict):
            continue
        if s.get("name") == PARTITION_BINDING and s.get("partition") == "holdout":
            raise _refusal("the partition_contract binding declares partition='holdout'")
        if s.get("name") == HOLDOUT_PLAN_BINDING:
            raise _refusal(f"the manifest carries a {HOLDOUT_PLAN_BINDING} binding")
    if manifest.get("dataset_id") == G0BN_OOS_DATASET_ID:
        raise _refusal(f"dataset_id is the one-shot OOS identity {G0BN_OOS_DATASET_ID!r}")
    cls = classify_manifest(manifest)
    if cls.holdout_bound:
        # Unreachable through today's classifier (a holdout verdict requires the
        # partition binding the raw scan already tripped on); kept so a holdout marker
        # a future classifier learns first still refuses here.
        raise _refusal("classify_manifest reports a holdout-bound build")
    return cls


def guarded_read_matrix(matrix_path, manifest: dict, *, loader=None):
    """The one authorized generic route from (matrix_path, manifest) to matrix bytes:
    classify and reject FIRST, open the file only afterwards — with whatever loader the
    caller supplies (default pandas.read_parquet)."""
    preflight_generic_manifest(manifest)
    return (pd.read_parquet if loader is None else loader)(matrix_path)
