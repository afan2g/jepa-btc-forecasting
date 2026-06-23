"""Baseline ladder configs + per-config CPCV evaluation (keeps the fold distribution)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import matthews_corrcoef
import lightgbm as lgb
from data.cv import cpcv_splits
from eval.cost import net_pnl, weighted_sharpe

CONFIGS = ("naive", "ridge", "lgbm_reg", "lgbm_clf")


@dataclass
class ConfigResult:
    name: str
    fold_sharpes: np.ndarray        # one OOS Sharpe per CPCV fold (the distribution)
    per_sample_pnl: np.ndarray      # OOS net PnL per sample (nan if never tested)
    mean_fold_sharpe: float         # TRADE-level Sharpe (feeds DSR; gate also requires min_trades)
    sample_sharpe: float            # sample/time-level Sharpe incl. no-trade zeros (honest headline)
    net_pnl: float
    gross_pnl: float                # PnL before costs (gross - net = the cost wall)
    n_trades: int
    turnover: float                 # trades / OOS samples
    t_eff: float                    # effective sample size = Σ uniqueness over trades
    mcc: float                      # sign(forecast) vs sign(realized) over trades
    skew: float
    kurt: float


def _fit_predict(model, Xtr, ytr, ltr, Xte, wtr, scale):
    if len(Xtr) < 2:        # degenerate fold (empty/near-empty train after purge) -> no trades
        return np.zeros(len(Xte))
    if model == "naive":
        return np.zeros(len(Xte))
    if model == "ridge":
        return Ridge(alpha=1.0).fit(Xtr, ytr, sample_weight=wtr).predict(Xte)
    if model == "lgbm_reg":
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                              min_child_samples=50, subsample=0.8, verbose=-1)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m.predict(Xte)
    if model == "lgbm_clf":
        if len(np.unique(ltr)) < 2:        # single-class fold (heavy-flat regime) -> no trades
            return np.zeros(len(Xte))
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               min_child_samples=50, subsample=0.8, verbose=-1)
        m.fit(Xtr, ltr, sample_weight=wtr)
        proba = m.predict_proba(Xte); cls = list(m.classes_)
        p_up = proba[:, cls.index(1)] if 1 in cls else 0.0
        p_dn = proba[:, cls.index(-1)] if -1 in cls else 0.0
        return (p_up - p_dn) * scale          # signed score -> bps
    raise ValueError(f"unknown model {model!r}")


def evaluate_config(matrix: pd.DataFrame, feature_cols, model: str, *,
                    n_groups: int, k: int, embargo_ns: int) -> ConfigResult:
    X = matrix[feature_cols].to_numpy(float)
    y = matrix["y_fwd_bps"].to_numpy(float)
    lab = matrix["label"].to_numpy(int)
    cost = matrix["cost_bps"].to_numpy(float)
    half = matrix["half_spread_bps"].to_numpy(float)
    w = matrix["uniqueness"].to_numpy(float)
    te_t, t0, t1 = (matrix["t_event"].to_numpy(), matrix["t_event"].to_numpy(),
                    matrix["t_barrier"].to_numpy())
    acc_fc = np.zeros(len(matrix)); cnt = np.zeros(len(matrix)); fold_sharpes = []
    for tr, te in cpcv_splits(te_t, t0, t1, n_groups=n_groups, k=k, embargo_ns=embargo_ns):
        scale = float(y[tr].std() + 1e-9) if len(tr) else 0.0  # empty-fold std would be nan
        fc = _fit_predict(model, X[tr], y[tr], lab[tr], X[te], w[tr], scale)
        fpnl, _, _ = net_pnl(fc, y[te], cost_bps=cost[te], half_spread_bps=half[te])
        fold_sharpes.append(weighted_sharpe(fpnl, w[te]))
        acc_fc[te] += fc; cnt[te] += 1
    seen = cnt > 0
    fc_ps = np.full(len(matrix), np.nan); fc_ps[seen] = acc_fc[seen] / cnt[seen]
    # Per-sample PnL from the averaged OOS forecast -> consistent pnl/gross/traded/mcc.
    pnl_s, traded_s, gross_s = net_pnl(fc_ps[seen], y[seen],
                                       cost_bps=cost[seen], half_spread_bps=half[seen])
    per_sample = np.full(len(matrix), np.nan); per_sample[seen] = pnl_s
    n_tr = int(traded_s.sum())
    t_eff = float(w[seen][traded_s].sum())
    pred_sign = np.sign(fc_ps[seen][traded_s]).astype(int)
    real_sign = np.sign(y[seen][traded_s]).astype(int)
    mcc = float(matthews_corrcoef(real_sign, pred_sign)) if n_tr > 1 and len(np.unique(pred_sign)) > 1 else 0.0
    pnl_traded = pnl_s[traded_s]
    sample_sharpe = weighted_sharpe(pnl_s, w[seen], trade_only=False)  # incl. no-trade zeros
    fs = np.array(fold_sharpes, float)
    return ConfigResult(
        name=model, fold_sharpes=fs, per_sample_pnl=per_sample,
        mean_fold_sharpe=float(fs.mean()), sample_sharpe=sample_sharpe,
        net_pnl=float(np.nansum(per_sample)), gross_pnl=float(gross_s.sum()), n_trades=n_tr,
        turnover=float(n_tr / max(int(seen.sum()), 1)), t_eff=t_eff, mcc=mcc,
        skew=float(pd.Series(pnl_traded).skew()) if n_tr > 2 else 0.0,
        kurt=float(pd.Series(pnl_traded).kurtosis() + 3.0) if n_tr > 3 else 3.0,
    )
