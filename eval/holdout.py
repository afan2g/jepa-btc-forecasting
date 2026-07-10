"""Fixed-holdout fit + one-time scorer (issue #52) — reachable ONLY through G0-XV.

`fit_frozen_config` trains the frozen winner on pre-holdout matched development rows
only. `score_fixed_holdout` performs the first and only modeling evaluation of the
holdout: it requires the verified freeze artifact, the one-time consumption record in the
`validated` state (i.e. #48's exact-scope trade validation PASSed), the exact pinned
partition contract / dev build / dev row universe, and a holdout build whose label
support ends inside the contract window — then scores once and marks the transaction
consumed. G0-CB has no path here (it can produce neither a freeze artifact nor a
validated consumption record), and holdout metrics are terminal evidence: nothing feeds
them back into selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from eval.baseline import CONFIGS, fit_model
from eval.consumption import (STATE_SCORED, STATE_VALIDATED, STATE_VALIDATION_FAILED,
                              load_record, record_holdout_score, record_path_for)
from eval.cost import net_pnl, weighted_sharpe
from eval.freeze import verify_freeze
from eval.hashing import canonical_row_order, hash_obj, matrix_content_hash
from eval.g0 import _require_expected_target
from eval.manifest import feature_list, target_list, validate_frame
from eval.runner import BASELINE_TARGETS
from eval.matrix import RESERVED, validate_matrix
from eval.partition import (contract_hash, require_binding, validate_development_span,
                            validate_holdout_span)


def fit_frozen_config(dev_matrix: pd.DataFrame, feature_cols, config: str) -> dict:
    """Fit the frozen model configuration on development rows ONLY (the caller passes the
    winner-horizon slice of the matched pre-holdout matrix). Returns the fitted predictor
    plus the training-derived scale — nothing about the holdout enters the fit."""
    if config not in CONFIGS:
        raise ValueError(f"unknown frozen config {config!r}; supported: {CONFIGS}")
    feature_cols = list(feature_cols)
    validate_matrix(dev_matrix, feature_cols)
    X = dev_matrix[feature_cols].to_numpy(float)
    y = dev_matrix["y_fwd_bps"].to_numpy(float)
    lab = dev_matrix["label"].to_numpy(int)
    w = dev_matrix["uniqueness"].to_numpy(float)
    scale = float(y.std() + 1e-9)
    return {"config": config, "feature_cols": feature_cols,
            "predict": fit_model(config, X, y, lab, w, scale),
            "scale": scale, "n_train": int(len(dev_matrix))}


def _score(fitted: dict, holdout_matrix: pd.DataFrame) -> dict:
    validate_matrix(holdout_matrix, fitted["feature_cols"])
    fc = np.asarray(fitted["predict"](holdout_matrix[fitted["feature_cols"]]
                                      .to_numpy(float)), float)
    y = holdout_matrix["y_fwd_bps"].to_numpy(float)
    cost = holdout_matrix["cost_bps"].to_numpy(float)
    half = holdout_matrix["half_spread_bps"].to_numpy(float)
    w = holdout_matrix["uniqueness"].to_numpy(float)
    pnl, traded, gross = net_pnl(fc, y, cost_bps=cost, half_spread_bps=half)
    n_tr = int(traded.sum())
    pred_sign = np.sign(fc[traded]).astype(int)
    real_sign = np.sign(y[traded]).astype(int)
    mcc = (float(matthews_corrcoef(real_sign, pred_sign))
           if n_tr > 1 and len(np.unique(pred_sign)) > 1 else 0.0)
    return {"net_pnl": float(pnl.sum()), "gross_pnl": float(gross.sum()),
            "cost_wall": float(gross.sum() - pnl.sum()),
            "trade_sharpe": weighted_sharpe(pnl, w, traded=traded),
            "sample_sharpe": weighted_sharpe(pnl, w, trade_only=False),
            "n_trades": n_tr, "t_eff": float(w[traded].sum()),
            "turnover": float(n_tr / max(len(holdout_matrix), 1)),
            "mcc": mcc, "n_rows": int(len(holdout_matrix))}


def _utc_days(t_event: pd.Series) -> list[str]:
    return sorted(pd.to_datetime(t_event, unit="ns", utc=True)
                  .dt.strftime("%Y-%m-%d").unique().tolist())


def preflight_holdout_inputs(freeze_artifact: dict, *, contract: dict,
                             dev_manifest: dict, holdout_manifest: dict) -> None:
    """Every frozen-pin check that needs NO matrix data: contract pin, partition
    bindings, the frozen winner-arm dev build, the holdout dataset/build scope, and the
    winner's feature availability. Callers (the CLI in particular) run this BEFORE
    opening any matrix file, so a validated transaction with mismatched inputs cannot
    repeatedly re-open the holdout matrix through failing invocations."""
    verify_freeze(freeze_artifact)
    if contract_hash(contract) != freeze_artifact["sources"]["partition_contract_sha256"]:
        raise ValueError("partition contract does not match the frozen source pin")
    winner = freeze_artifact["winner"]
    require_binding(dev_manifest, contract, "development")
    require_binding(holdout_manifest, contract, "holdout")
    for man, side in ((dev_manifest, "dev"), (holdout_manifest, "holdout")):
        targets = set(target_list(man))
        if targets != BASELINE_TARGETS:
            raise ValueError(f"the {side} manifest must declare exactly "
                             f"{sorted(BASELINE_TARGETS)} as target_cols (the outcomes "
                             f"the scorer consumes); it declares {sorted(targets)}")
        # Same pinned target venue as every G0 path: the one-time April score must be
        # against Coinbase BTC-USD labels/costs, never a substituted target market.
        _require_expected_target(man, f"the {side} manifest")
    frozen_dev_manifest = freeze_artifact["sources"]["arm_manifests"][winner["arm"]]
    if hash_obj(dev_manifest) != frozen_dev_manifest:
        raise ValueError(f"dev manifest is not the frozen {winner['arm']!r} arm build "
                         "the winner was selected on")
    scope = freeze_artifact["holdout_scope"]
    if (holdout_manifest["dataset_id"] != scope["dataset_id"]
            or holdout_manifest["build_id"] != scope["build_id"]):
        raise ValueError("holdout manifest dataset/build does not match the frozen "
                         "holdout scope")
    missing = [c for c in winner["feature_cols"]
               if c not in feature_list(holdout_manifest)]
    if missing:
        raise ValueError(f"holdout build lacks frozen winner features: {missing}")


def verify_frozen_dev_matrix(freeze_artifact: dict, *, contract: dict,
                             dev_matrix: pd.DataFrame, dev_manifest: dict) -> None:
    """Every dev-matrix pin, runnable WITHOUT holdout data: frame/span validity, the
    frozen matched reserved-row hash, and the winner arm's full feature-content hash.
    The CLI runs this between the dev read and the holdout read, so tampered/stale dev
    rows cannot repeatedly re-open the holdout matrix through failing invocations."""
    winner = freeze_artifact["winner"]
    validate_frame(dev_matrix, dev_manifest)
    validate_development_span(dev_matrix, contract)
    if matrix_content_hash(dev_matrix, list(RESERVED)) \
            != freeze_artifact["sources"]["row_content_sha256"]:
        raise ValueError("development matrix reserved-row content does not match the "
                         "frozen matched row universe; the frozen winner must be refit on "
                         "exactly the rows it was selected on")
    # The reserved-row hash cannot see feature substitution (arms share reserved columns
    # but differ in features), so the refit input is ALSO pinned by the per-arm FULL
    # content hash frozen at selection time — features recomputed after the freeze
    # (e.g. with holdout knowledge) fail here.
    frozen_full = freeze_artifact["sources"]["arm_matrix_hashes"][winner["arm"]]
    if matrix_content_hash(dev_matrix,
                           list(RESERVED) + feature_list(dev_manifest)) != frozen_full:
        raise ValueError(f"development matrix FEATURE content does not match the frozen "
                         f"{winner['arm']!r} arm build; the frozen winner must be refit "
                         "on exactly the feature values it was selected on")


def score_fixed_holdout(*, freeze_artifact: dict, records_dir, contract: dict,
                        dev_matrix: pd.DataFrame, dev_manifest: dict,
                        holdout_matrix: pd.DataFrame, holdout_manifest: dict,
                        verify_only: bool = False) -> dict:
    """The one-time fixed-holdout model score. Every input is verified against the frozen
    artifact before any outcome-bearing computation; on success the consumption record
    moves to `scored`. `verify_only=True` re-computes an ALREADY-SCORED transaction and
    reports whether it reproduces the recorded result hash — it never scores an
    unconsumed holdout, never mutates the record, and returns NO metrics unless the
    recorded score is reproduced (otherwise repeated verify calls with perturbed inputs
    would be an iterate-against-holdout oracle)."""
    verify_freeze(freeze_artifact)
    record = load_record(record_path_for(records_dir, freeze_artifact))
    if freeze_artifact["sha256"] != record["artifact_sha256"]:
        raise ValueError("freeze artifact does not match the holdout transaction's pinned "
                         "artifact; stale/substituted artifacts cannot score the holdout")
    if verify_only:
        if record["state"] != STATE_SCORED:
            raise ValueError("verify_only reproduces an already-consumed score; this "
                             f"transaction is {record['state']!r}, not scored")
    elif record["state"] == STATE_VALIDATION_FAILED:
        raise ValueError("trade validation FAILED: G0-XV is blocking/inconclusive and "
                         "the holdout cannot be scored; thresholds, exclusions, "
                         "candidates, and holdout dates cannot be changed to retry")
    elif record["state"] == STATE_SCORED:
        raise ValueError("holdout already scored; the transaction is consumed and "
                         "cannot be reused")
    elif record["state"] != STATE_VALIDATED:
        raise ValueError(
            "holdout scoring requires the one-time transaction in the 'validated' state "
            f"(exact-scope trade validation PASSed); it is {record['state']!r}")

    preflight_holdout_inputs(freeze_artifact, contract=contract,
                             dev_manifest=dev_manifest,
                             holdout_manifest=holdout_manifest)
    winner = freeze_artifact["winner"]
    scope = freeze_artifact["holdout_scope"]

    verify_frozen_dev_matrix(freeze_artifact, contract=contract, dev_matrix=dev_matrix,
                             dev_manifest=dev_manifest)
    validate_frame(holdout_matrix, holdout_manifest)
    validate_holdout_span(holdout_matrix, contract)
    dup = holdout_matrix.duplicated(subset=["t_event", "horizon"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (t_event, horizon) holdout rows; "
                         "one decision per instant (coalesce rule) — duplicates would "
                         "double-count the terminal metrics")
    days = _utc_days(holdout_matrix["t_event"])
    if days != scope["days"]:
        raise ValueError(f"holdout matrix days {days} do not exactly match the frozen "
                         f"scope {scope['days']}; partial or substituted holdout builds "
                         "are rejected")

    # Canonical row order before fitting: the content pins are order-insensitive, so
    # the same frozen rows in a different parquet order pass verification — but an
    # order-sensitive fit (LightGBM binning paths) must still reproduce the identical
    # score from the frozen artifact.
    dev_matrix = canonical_row_order(dev_matrix)
    holdout_matrix = canonical_row_order(holdout_matrix)
    h = winner["horizon"]
    dev_slice = dev_matrix[dev_matrix["horizon"] == h].reset_index(drop=True)
    hold_slice = holdout_matrix[holdout_matrix["horizon"] == h].reset_index(drop=True)
    if not len(dev_slice) or not len(hold_slice):
        raise ValueError(f"winner horizon {h!r} missing from the dev or holdout build")
    # The exact-scope rule applies to the slice actually scored, not just the union of
    # horizons: a multi-horizon build missing winner-horizon rows on a frozen day would
    # otherwise consume the transaction on a partial holdout.
    slice_days = _utc_days(hold_slice["t_event"])
    if slice_days != scope["days"]:
        raise ValueError(f"winner-horizon {h!r} holdout rows cover days {slice_days}, "
                         f"not the frozen scope {scope['days']}; partial winner-horizon "
                         "coverage is rejected")

    fitted = fit_frozen_config(dev_slice, winner["feature_cols"], winner["config"])
    metrics = _score(fitted, hold_slice)
    result = {"protocol": "g0xv-holdout", "artifact_sha256": freeze_artifact["sha256"],
              "winner": dict(winner), "metrics": metrics,
              "n_train_rows": fitted["n_train"],
              # Audit pin of what was actually consumed. The holdout build cannot be
              # content-pinned at freeze time (it must not have been readable), so the
              # scored content is recorded here and enters result_sha256 -> the record.
              "holdout_matrix_sha256": matrix_content_hash(
                  holdout_matrix, list(RESERVED) + feature_list(holdout_manifest))}
    result_sha256 = hash_obj(result)

    if verify_only:
        recorded = [e for e in record["history"] if e["event"] == "holdout_score"]
        reproduces = bool(recorded and recorded[-1]["result_sha256"] == result_sha256)
        out = {"protocol": "g0xv-holdout",
               "artifact_sha256": freeze_artifact["sha256"],
               "reproduces_recorded_score": reproduces}
        if reproduces:
            # Only the already-recorded evidence is echoed; non-reproducing inputs get
            # NO metrics (that would be a fresh outcome-bearing read of the holdout).
            out["winner"] = dict(winner)
            out["metrics"] = metrics
        return out

    record_holdout_score(records_dir, freeze_artifact=freeze_artifact,
                         result_sha256=result_sha256)
    result["consumed"] = True
    result["result_sha256"] = result_sha256
    return result
