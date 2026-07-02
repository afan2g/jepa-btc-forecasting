"""Manifest-driven G1 runner. The gate block is REQUIRED (pre-registration) and the
RESOLVED config is returned so every run is reproducible from its own output."""
from __future__ import annotations
import pandas as pd
from eval.manifest import (V1_ONLY_FIELDS, feature_list, leaky_feature_names,
                           target_list, validate_frame)
from eval.study import run_study

DEFAULT_GATE = {"n_groups": 6, "k": 2, "min_trades": 30, "min_eff_trades": 10.0,
                "min_sample_sharpe": 0.0, "dsr_thresh": 0.95, "pbo_thresh": 0.5}

# evaluate_config trains on exactly these (y_fwd_bps regression, label classification).
# A manifest declaring anything else would misdescribe what the study consumed.
BASELINE_TARGETS = frozenset(("y_fwd_bps", "label"))

_LEGACY_KEYS = ("feature_cols", "embargo_ns", "max_lookback_ns")


def resolve_gate(manifest: dict) -> dict:
    """Require the pre-registered 'gate' block; reject unknown (misspelled) keys; fill
    defaults; return the RESOLVED config."""
    if "gate" not in manifest:
        raise ValueError("manifest must include a pre-registered 'gate' block")
    unknown = set(manifest["gate"]) - set(DEFAULT_GATE)
    if unknown:
        raise ValueError(f"unknown gate keys (misspelled?): {sorted(unknown)}")
    return {**DEFAULT_GATE, **manifest["gate"]}


def run_from_manifest(matrix: pd.DataFrame, manifest: dict) -> dict:
    out = {}
    if "manifest_version" in manifest:
        # v1+ manifests are schema-validated and checked against the matrix up front.
        validate_frame(matrix, manifest)
        feats = feature_list(manifest)               # validated copy, manifest order
        targets = set(target_list(manifest))
        if targets != BASELINE_TARGETS:
            raise ValueError(f"the LightGBM baseline consumes exactly "
                             f"{sorted(BASELINE_TARGETS)} as targets; manifest declares "
                             f"{sorted(targets)}")
        if manifest.get("availability_lag_ns", 0) != 0:
            raise ValueError("the LightGBM baseline is synchronous (t_available == t_event); "
                             "availability_lag_ns > 0 is reserved for future consumers — "
                             "lag features upstream instead")
        # validate_frame checked frame tags are declared; check the converse so declared
        # horizons cannot silently vanish from the per-horizon results.
        missing_h = sorted(set(manifest["horizons"]) - set(matrix["horizon"].unique()))
        if missing_h:
            raise ValueError(f"manifest horizons missing from the matrix: {missing_h}; "
                             "the manifest must describe this exact build")
        out["manifest"] = {k: manifest[k] for k in
                           ("dataset_id", "build_id", "generated_at",
                            "embargo_ns", "max_lookback_ns")}
        out["manifest"]["feature_cols"] = feats
    else:
        markers = V1_ONLY_FIELDS & set(manifest)
        if markers:
            # a typo in 'manifest_version' itself must not silently select the
            # unvalidated legacy path
            raise ValueError(f"manifest carries v1 contract fields {sorted(markers)} but no "
                             "'manifest_version'; add manifest_version=1")
        missing = [k for k in _LEGACY_KEYS if k not in manifest]
        if missing:
            raise ValueError(f"legacy manifest missing required keys: {missing}")
        feats = list(manifest["feature_cols"])
        leaky = leaky_feature_names(feats)
        if leaky:
            # legacy dicts skip validate_frame; never skip the leak screen
            # (phase 3 removes this whole branch)
            raise ValueError(f"leaky feature names (label-derived): {leaky}")
    gate = resolve_gate(manifest)
    emb, lb = manifest["embargo_ns"], manifest["max_lookback_ns"]
    horizons = {}
    # observed=True: a categorical horizon column must not yield empty subframes for
    # unused categories (run_study crashes on an empty matrix under pandas 2.x defaults)
    for h, sub in matrix.groupby("horizon", observed=True):
        horizons[str(h)] = run_study(sub.reset_index(drop=True), feats, cost_default=None,
                                     embargo_ns=emb, max_lookback_ns=lb, **gate)
    out.update({"gate": gate, "horizons": horizons})
    return out
