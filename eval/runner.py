"""Manifest-driven G1 runner. The gate block is REQUIRED (pre-registration) and the
RESOLVED config is returned so every run is reproducible from its own output."""
from __future__ import annotations
import pandas as pd
from eval.study import run_study

DEFAULT_GATE = {"n_groups": 6, "k": 2, "min_trades": 30, "min_eff_trades": 10.0,
                "min_sample_sharpe": 0.0, "dsr_thresh": 0.95, "pbo_thresh": 0.5}


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
    gate = resolve_gate(manifest)
    feats = manifest["feature_cols"]
    emb, lb = manifest["embargo_ns"], manifest["max_lookback_ns"]
    horizons = {}
    for h, sub in matrix.groupby("horizon"):
        horizons[str(h)] = run_study(sub.reset_index(drop=True), feats, cost_default=None,
                                     embargo_ns=emb, max_lookback_ns=lb, **gate)
    return {"gate": gate, "horizons": horizons}
