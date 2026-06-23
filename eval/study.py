"""Study: run the ladder as TRIAL CONFIGS, compute DSR (best config vs trial dispersion,
with an EFFECTIVE sample size) and PBO (CSCV over configs x OOS-sample matrix), and the
G1 gate (FAIL-CLOSED when PBO is unavailable) with per-regime breakdown and gross/net."""
from __future__ import annotations
import numpy as np
import pandas as pd
from eval.matrix import validate_matrix
from eval.baseline import evaluate_config, CONFIGS
from eval.cost import weighted_sharpe
from eval.stats import deflated_sharpe, pbo


def run_study(matrix: pd.DataFrame, feature_cols, *, cost_default, n_groups: int, k: int,
              embargo_ns: int, max_lookback_ns: int, configs=CONFIGS, extra_trials: int = 0,
              dsr_thresh: float = 0.95, pbo_thresh: float = 0.5, min_trades: int = 30,
              min_eff_trades: float = 10.0, min_sample_sharpe: float = 0.0):
    validate_matrix(matrix, feature_cols)
    if embargo_ns < max_lookback_ns:
        raise ValueError(f"embargo_ns ({embargo_ns}) must cover max_lookback_ns ({max_lookback_ns})")
    # Cross-check the DECLARED max_lookback_ns against the matrix's ACTUAL per-row look-back
    # (t_event - t_feature_start). The embargo is sized to max_lookback_ns; if that scalar
    # understates the true feature window, a post-test train row's features reach back into
    # the test label span -> silent look-ahead leakage that inflates the gate. Fail closed
    # using the ground-truth column the matrix already carries.
    observed_lookback = int((matrix["t_event"] - matrix["t_feature_start"]).max())
    if max_lookback_ns < observed_lookback:
        raise ValueError(f"max_lookback_ns ({max_lookback_ns}) understates the matrix's actual "
                         f"per-row look-back ({observed_lookback} = max(t_event - t_feature_start)); "
                         f"the embargo would not cover the feature window -> look-back leakage")

    results = {c: evaluate_config(matrix, feature_cols, c, n_groups=n_groups, k=k,
                                  embargo_ns=embargo_ns) for c in configs}
    naive = results["naive"]
    candidates = [r for c, r in results.items() if c != "naive"]

    # DSR per config: trade-level Sharpe vs across-trial dispersion; T = effective trade
    # count. The pre-registered min_trades floor guards against few-trade flukes.
    trial_sharpes = np.array([r.mean_fold_sharpe for r in results.values()])
    sr_std = float(trial_sharpes.std() + 1e-9)
    n_trials = max(2, len(results) + extra_trials)
    dsr_by = {r.name: deflated_sharpe(sr_hat=r.mean_fold_sharpe, sr_trials_std=sr_std,
                                      n_trials=n_trials, T=max(int(round(r.t_eff)), 2),
                                      skew=r.skew, kurt=r.kurt) for r in results.values()}

    # PBO over the configs x common-OOS-sample matrix (selection overfitting). Fail-closed.
    # Weight blocks by uniqueness so overlapping-label clusters don't skew the rankings.
    w = matrix["uniqueness"].to_numpy(float)
    M = np.column_stack([r.per_sample_pnl for r in results.values()])
    rows = np.isfinite(M).all(axis=1)
    pbo_available = bool(rows.sum() >= 32)
    pbo_val = float(pbo(M[rows], s=8, weights=w[rows])) if pbo_available else float("nan")

    def _solo(r):  # per-candidate gate (multiple testing handled by DSR n_trials + PBO)
        return bool(r.net_pnl > 0 and dsr_by[r.name] > dsr_thresh
                    and r.n_trades >= min_trades and r.t_eff >= min_eff_trades
                    and r.sample_sharpe >= min_sample_sharpe
                    and r.net_pnl > naive.net_pnl)
    passing = [r for r in candidates if _solo(r)]
    g1 = bool(passing and pbo_available and pbo_val < pbo_thresh)
    g1_inconclusive = bool(passing and not pbo_available)        # would pass but PBO uncomputable
    winner = (max(passing, key=lambda r: r.net_pnl) if passing
              else max(candidates, key=lambda r: r.net_pnl))

    # Per-regime: slice the WINNER's OOS PnL (no refit); sample/time-level Sharpe.
    per_regime = {}
    for reg, ii in matrix.groupby("regime").indices.items():   # .indices = positional rows (index-agnostic)
        p = winner.per_sample_pnl[ii]
        per_regime[str(reg)] = {"net_pnl": float(np.nansum(p)),
                                "sample_sharpe": weighted_sharpe(np.nan_to_num(p), w[ii], trade_only=False),
                                "n": int(np.isfinite(p).sum())}

    def _row(r):
        return {"net_pnl": r.net_pnl, "gross_pnl": r.gross_pnl,
                "cost_wall": r.gross_pnl - r.net_pnl, "trade_sharpe": r.mean_fold_sharpe,
                "sample_sharpe": r.sample_sharpe, "dsr": dsr_by[r.name], "n_trades": r.n_trades,
                "turnover": r.turnover, "mcc": r.mcc, "passes_solo": _solo(r)}
    return {
        "g1_pass": g1,
        "g1_inconclusive": g1_inconclusive,
        "winner": winner.name if passing else None,
        "pbo": pbo_val, "pbo_available": pbo_available,
        "best": {"name": winner.name, "net_pnl": winner.net_pnl, "gross_pnl": winner.gross_pnl,
                 "cost_wall": winner.gross_pnl - winner.net_pnl, "sharpe": winner.mean_fold_sharpe,
                 "trade_sharpe": winner.mean_fold_sharpe, "sample_sharpe": winner.sample_sharpe,
                 "dsr": dsr_by[winner.name], "pbo": pbo_val, "turnover": winner.turnover,
                 "mcc": winner.mcc, "n_trades": winner.n_trades},
        "rungs": {r.name: _row(r) for r in results.values()},
        "per_regime": per_regime,
    }
