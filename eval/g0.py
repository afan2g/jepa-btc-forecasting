"""G0 development studies (issue #52; staged protocol §2).

Two development-only entry points over the SAME candidate engine:

- `run_g0cb_study`: the Coinbase-only preliminary screen. Its signature has NO holdout
  parameter, it rejects a manifest bound to the holdout partition before touching any
  matrix data, and it fail-closes on any row whose guarded support reaches the holdout
  boundary. Every attempted (config, horizon, variant) is persisted to the trial ledger.
- `run_g0xv_development`: ONE unified study over matched arms (Coinbase-only control,
  Binance-only, combined, ...). All arms must share identical reserved rows, labels,
  costs, regime tags, and CPCV splits (content-hash verified, fail closed). DSR uses the
  COMPLETE effective trial count — every registered candidate across arms, builds,
  models, horizons, and variants, plus the imported G0-CB history. PBO is computed from
  the common development-OOS candidate-PnL matrix spanning ALL arms' candidates; an
  unavailable PBO fails closed (blocking/inconclusive, never a pass).

Neither function can open holdout data: they accept development-partition builds only and
re-validate every row's guarded span against the pinned partition contract. Holdout
scoring lives in `eval.holdout` and is reachable only through a frozen selection artifact
plus a PASSed one-time consumption transaction (`eval.freeze` / `eval.consumption`).

These results are DEVELOPMENT evidence. They are not formal G1 (`eval.runner` remains the
unchanged per-manifest G1 path) and must never be reported as the project-defining gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.baseline import CONFIGS, evaluate_config
from eval.cost import weighted_sharpe
from eval.hashing import canonical_row_order, hash_obj, matrix_content_hash, split_hash
from eval.ledger import TRIAL_PROTOCOLS, TrialLedger, identity_hash, trial_identity
from eval.manifest import feature_list, target_list, validate_frame
from eval.matrix import RESERVED, validate_matrix
from eval.partition import contract_hash, require_binding, validate_development_span
from eval.runner import BASELINE_TARGETS, DEFAULT_GATE
from eval.stats import deflated_sharpe, pbo
from eval.study import LGBM_RUNGS

CONTROL_ARM = "coinbase_only"
BINANCE_ARM = "binance_only"
COMBINED_ARM = "combined"
# The staged protocol preregisters THREE arms (§2): omitting one would remove its
# candidates from the common PBO matrix and the effective DSR count.
REQUIRED_ARMS = (CONTROL_ARM, BINANCE_ARM, COMBINED_ARM)
G0CB_PROTOCOL = "g0cb"
G0XV_PROTOCOL = "g0xv"

# The preregistered G0-XV development gate block: the G1-style solo/DSR/PBO knobs plus the
# combined-vs-control bootstrap noise band (staged protocol §2 authorization rule).
DEFAULT_XV_GATE = {**DEFAULT_GATE, "noise_band_n_boot": 2000, "noise_band_block": 16,
                   "noise_band_alpha": 0.05, "noise_band_seed": 0}

_PBO_MIN_ROWS = 32          # same fail-closed floor as eval.study
_PBO_BLOCKS = 8


def resolve_g0_gate(gate: dict | None, *, defaults: dict) -> dict:
    """Reject unknown (misspelled) keys and fill defaults; returns the RESOLVED block."""
    gate = gate or {}
    unknown = set(gate) - set(defaults)
    if unknown:
        raise ValueError(f"unknown gate keys (misspelled?): {sorted(unknown)}")
    return {**defaults, **gate}


# ----------------------------------------------------------------------- arm preparation
def _echo_manifest(manifest: dict) -> dict:
    out = {k: manifest[k] for k in ("dataset_id", "build_id", "generated_at",
                                    "embargo_ns", "max_lookback_ns")}
    out["feature_cols"] = feature_list(manifest)
    out["manifest_sha256"] = hash_obj(manifest)
    return out


def _arm_echo(prep: dict) -> dict:
    """Manifest identity plus the FULL matrix content pin (reserved + this arm's feature
    values). The reserved-only matched-row hash proves arms align; this hash is what lets
    the holdout scorer prove the frozen winner is refit on exactly the selected data —
    feature values included."""
    echo = _echo_manifest(prep["manifest"])
    echo["matrix_content_sha256"] = matrix_content_hash(
        prep["matrix"], list(RESERVED) + prep["feature_cols"])
    return echo


def _prepare_development_input(matrix: pd.DataFrame, manifest: dict, contract: dict,
                               *, arm: str) -> dict:
    """Fail-closed development-input validation, shared by G0-CB and every G0-XV arm.
    Binding is checked FIRST (no data touched) so a holdout-bound manifest is rejected
    before any matrix work; then schema/frame validation, the span-safe partition rule,
    and the one-decision-per-(t_event, horizon) invariant."""
    require_binding(manifest, contract, "development")
    validate_frame(matrix, manifest)
    targets = set(target_list(manifest))
    if targets != BASELINE_TARGETS:
        raise ValueError(f"G0 candidates consume exactly {sorted(BASELINE_TARGETS)} as "
                         f"targets; manifest declares {sorted(targets)}")
    if manifest.get("availability_lag_ns", 0) != 0:
        raise ValueError("the G0 baseline ladder is synchronous (t_available == t_event); "
                         "lag features upstream instead")
    missing_h = sorted(set(manifest["horizons"]) - set(matrix["horizon"].unique()))
    if missing_h:
        raise ValueError(f"manifest horizons missing from the matrix: {missing_h}; "
                         "the manifest must describe this exact build")
    validate_development_span(matrix, contract)
    # validate_frame checks columns/timing; validate_matrix adds the VALUE-domain gate
    # invariants (non-negative costs, label in {-1,0,1}, uniqueness in (0,1], finite
    # study inputs) that run_study enforces on the G1 path — a negative cost row would
    # invert the no-trade band and silently inflate a development pass.
    validate_matrix(matrix, feature_list(manifest))
    m = canonical_row_order(matrix)
    dup = m.duplicated(subset=["t_event", "horizon"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (t_event, horizon) rows; the producer "
                         "must emit one decision per instant (coalesce rule)")
    return {"arm": arm, "manifest": manifest, "matrix": m,
            "feature_cols": feature_list(manifest)}


def _resolve_variants(feature_cols: list[str], variants) -> list[dict]:
    """Explicit candidate variants. The base variant is always evaluated; each extra
    variant is a REGISTERED TRIAL with its own identity (post-hoc variations are counted,
    never free). v1 supports feature-subset variants (ordered subset of the manifest
    list); any other variation must arrive as a new manifest/build."""
    out = [{"name": "base", "feature_cols": list(feature_cols), "params": {}}]
    seen = {"base"}
    for v in variants or []:
        if not isinstance(v, dict) or not isinstance(v.get("name"), str) or not v["name"]:
            raise ValueError("each variant must be a dict with a non-empty 'name'")
        unknown = set(v) - {"name", "feature_cols"}
        if unknown:
            raise ValueError(f"unknown variant keys (unsupported variation?): {sorted(unknown)}; "
                             "non-feature variations must be registered as a new build")
        if v["name"] in seen:
            raise ValueError(f"duplicate variant name {v['name']!r}")
        seen.add(v["name"])
        feats = list(v.get("feature_cols", feature_cols))
        bad = [c for c in feats if c not in feature_cols]
        if not feats or bad:
            raise ValueError(f"variant {v['name']!r} feature_cols must be a non-empty "
                             f"subset of the manifest feature list; invalid: {bad}")
        if len(set(feats)) != len(feats):
            raise ValueError(f"variant {v['name']!r} feature_cols contain duplicates; "
                             "they would double-weight columns (rejected before any "
                             "candidate is evaluated)")
        out.append({"name": v["name"], "feature_cols": feats,
                    "params": {"feature_cols": feats}})
    return out


# --------------------------------------------------------------------- candidate engine
def _ledger_result(r) -> dict:
    return {"net_pnl": float(r.net_pnl), "gross_pnl": float(r.gross_pnl),
            "trade_sharpe": float(r.trade_sharpe), "sample_sharpe": float(r.sample_sharpe),
            "mean_fold_sharpe": float(r.mean_fold_sharpe), "n_trades": int(r.n_trades),
            "t_eff": float(r.t_eff), "turnover": float(r.turnover), "mcc": float(r.mcc),
            "skew": float(r.skew), "kurt": float(r.kurt)}


def _evaluate_arm_candidates(prep: dict, *, protocol: str, gate: dict, ledger: TrialLedger,
                             variants, configs) -> list[dict]:
    """Evaluate every (horizon, variant, config) candidate of one prepared arm via CPCV
    and register each as a ledger trial. Naive is evaluated once per (arm, horizon) — its
    forecast is feature-independent, so re-registering it per variant would inflate the
    trial count without a new attempt."""
    if "naive" not in configs:
        raise ValueError("the candidate ladder must include 'naive' (the cost-aware "
                         "do-nothing benchmark every solo gate compares against)")
    man, m = prep["manifest"], prep["matrix"]
    var_list = _resolve_variants(prep["feature_cols"], variants)
    emb = man["embargo_ns"]
    out = []
    for tag, sub in m.groupby("horizon", observed=True):
        sub = sub.reset_index(drop=True)
        for var in var_list:
            for config in configs:
                if config == "naive" and var["name"] != "base":
                    continue
                r = evaluate_config(sub, var["feature_cols"], config,
                                    n_groups=gate["n_groups"], k=gate["k"], embargo_ns=emb)
                # The resolved gate is part of the trial identity: a post-hoc threshold
                # or split change is ANOTHER TRIAL (staged protocol §2) and must add
                # multiplicity when history is carried, not silently reuse identities.
                ident = trial_identity(
                    protocol=protocol, arm=prep["arm"], dataset_id=man["dataset_id"],
                    build_id=man["build_id"], feature_cols=var["feature_cols"],
                    config=config, horizon=str(tag), variant=var["name"],
                    variant_params={**var["params"], "gate": dict(sorted(gate.items()))})
                entry = ledger.register(ident, _ledger_result(r))
                out.append({"arm": prep["arm"], "horizon": str(tag), "variant": var["name"],
                            "config": config, "identity": ident,
                            "identity_sha256": entry["identity_sha256"], "result": r,
                            "uniqueness": sub["uniqueness"].to_numpy(float),
                            "regime": sub["regime"].astype(str).to_numpy(object)})
    return out


def _solo_pass(cand: dict, dsr: float, naive: dict, gate: dict) -> bool:
    r, nr = cand["result"], naive["result"]
    return bool(r.net_pnl > 0 and dsr > gate["dsr_thresh"]
                and r.n_trades >= gate["min_trades"] and r.t_eff >= gate["min_eff_trades"]
                and r.sample_sharpe >= gate["min_sample_sharpe"]
                and r.net_pnl > nr.net_pnl)


def _horizon_pool(candidates: list[dict], tag: str) -> list[dict]:
    pool = [c for c in candidates if c["horizon"] == tag]
    # Deterministic candidate order for the PBO matrix and every reported list.
    pool.sort(key=lambda c: (c["arm"], c["variant"], c["config"]))
    return pool


def _naive_for(cand: dict, candidates: list[dict]) -> dict:
    for c in candidates:
        if (c["arm"] == cand["arm"] and c["horizon"] == cand["horizon"]
                and c["variant"] == "base" and c["config"] == "naive"):
            return c
    raise ValueError(f"no naive benchmark for arm {cand['arm']!r} horizon "
                     f"{cand['horizon']!r}")   # unreachable: ladder requires naive


def _pool_dsr(pool: list[dict], n_trials: int) -> dict:
    sharpes = np.array([c["result"].trade_sharpe for c in pool])
    sr_std = float(sharpes.std() + 1e-9)
    return {c["identity_sha256"]: deflated_sharpe(
        sr_hat=c["result"].trade_sharpe, sr_trials_std=sr_std,
        n_trials=max(2, n_trials), T=max(int(round(c["result"].t_eff)), 2),
        skew=c["result"].skew, kurt=c["result"].kurt) for c in pool}


def _pool_pbo(pool: list[dict]) -> dict:
    """PBO over the COMMON development-OOS candidate-PnL matrix: one column per registered
    candidate in this horizon pool (across every arm/variant/config), rows = the matched
    development rows finite in ALL columns, uniqueness-weighted, t_event-ordered."""
    M = np.column_stack([c["result"].per_sample_pnl for c in pool])
    w = pool[0]["uniqueness"]                    # matched arms: identical by content hash
    finite = np.where(np.isfinite(M).all(axis=1))[0]
    available = bool(finite.size >= _PBO_MIN_ROWS and M.shape[1] >= 2)
    val = float(pbo(M[finite], s=_PBO_BLOCKS, weights=w[finite])) if available else float("nan")
    return {"pbo": val, "pbo_available": available, "pbo_n_rows": int(finite.size),
            "pbo_candidates": [c["identity_sha256"] for c in pool]}


def _candidate_row(cand: dict, dsr: float, solo: bool) -> dict:
    r = cand["result"]
    # Per-regime stratification (experiment-plan cross-cutting discipline): a pass
    # driven entirely by one spread/volatility slice must be visible in the evidence.
    # Slices the candidate's OOS per-sample PnL by the matched regime tag (no refit).
    p, w, reg = r.per_sample_pnl, cand["uniqueness"], cand["regime"]
    per_regime = {}
    for tag in sorted(set(reg)):
        ii = np.where(reg == tag)[0]
        pi = p[ii]
        per_regime[str(tag)] = {
            "net_pnl": float(np.nansum(pi)),
            "sample_sharpe": weighted_sharpe(np.nan_to_num(pi), w[ii], trade_only=False),
            "n": int(np.isfinite(pi).sum())}
    return {"arm": cand["arm"], "config": cand["config"], "variant": cand["variant"],
            "net_pnl": r.net_pnl, "gross_pnl": r.gross_pnl,
            "cost_wall": r.gross_pnl - r.net_pnl, "trade_sharpe": r.trade_sharpe,
            "sample_sharpe": r.sample_sharpe, "dsr": dsr, "n_trades": r.n_trades,
            "t_eff": r.t_eff, "turnover": r.turnover, "mcc": r.mcc,
            "per_regime": per_regime, "passes_solo": solo}


# ------------------------------------------------------------------------------- G0-CB
# The protocol's one trading venue (spec §2): every G0 build labels and trades Coinbase
# BTC-USD. A manifest declaring any other (or an extra) target venue is a different
# experiment and fails closed.
TARGET_EXCHANGE = "COINBASE"
TARGET_SYMBOL = "BTC-USD"
# The preregistered signal markets (docs/data.md §5b / trade-validation plan §2): a
# cross-venue arm carrying any other signal venue is a different acquisition experiment.
ALLOWED_SIGNAL_VENUES = (("BINANCE_FUTURES", "BTC-USDT-PERP"), ("BINANCE", "BTC-USDT"))


def _require_expected_target(manifest: dict, context: str) -> None:
    """Exactly ONE target venue, and it must be the protocol's Coinbase BTC-USD — a
    manifest that marks another venue (or several) as 'target' would invalidate the
    Coinbase-only screen and the combined-vs-control comparison."""
    targets = [v for v in manifest["venues"] if v.get("role") == "target"]
    if (len(targets) != 1 or targets[0].get("exchange") != TARGET_EXCHANGE
            or targets[0].get("symbol") != TARGET_SYMBOL):
        raise ValueError(f"{context} must declare exactly one target venue "
                         f"{TARGET_EXCHANGE}/{TARGET_SYMBOL}, got: {targets}")


def _require_target_only_venues(manifest: dict, context: str) -> None:
    """Fail CLOSED on venue roles: `role` is optional in the manifest schema, so an
    omitted or mislabeled role must not slip a signal venue past a target-venue-only
    path — every venue must declare role == 'target' explicitly, and the single target
    must be the protocol's Coinbase BTC-USD."""
    non_target = [v for v in manifest["venues"] if v.get("role") != "target"]
    if non_target:
        raise ValueError(f"{context} is target-venue-only; every venue must declare "
                         f"role 'target' explicitly, got: {non_target}")
    _require_expected_target(manifest, context)


def g0cb_manifest_prechecks(manifest: dict, contract: dict) -> None:
    """Everything G0-CB can reject WITHOUT touching matrix data — run by the CLI before
    the matrix file is opened, and again inside run_g0cb_study. A holdout-bound manifest
    or a cross-venue build fails here, before any data loading."""
    require_binding(manifest, contract, "development")
    _require_target_only_venues(manifest, "G0-CB")


def run_g0cb_study(matrix: pd.DataFrame, manifest: dict, contract: dict, *,
                   gate: dict | None = None, ledger: TrialLedger | None = None,
                   variants=None, configs=CONFIGS) -> dict:
    """Development-only Coinbase screen (G0-CB, issue #47). There is deliberately no
    holdout parameter: this mode cannot accept, load, or score holdout data. The manifest
    must bind the DEVELOPMENT partition of the pinned contract (checked before any data
    work) and must not declare a signal venue (Coinbase-own-book only). Every attempted
    candidate/variant is persisted to `ledger` — the trial history #48/#52 later fold
    into the G0-XV effective DSR count. The result is a development diagnosis, NOT formal
    G1 and NOT a fixed-holdout claim."""
    gate = resolve_g0_gate(gate, defaults=DEFAULT_GATE)
    g0cb_manifest_prechecks(manifest, contract)          # before any matrix access
    ledger = ledger if ledger is not None else TrialLedger()
    prep = _prepare_development_input(matrix, manifest, contract, arm=CONTROL_ARM)
    candidates = _evaluate_arm_candidates(prep, protocol=G0CB_PROTOCOL, gate=gate,
                                          ledger=ledger, variants=variants, configs=configs)
    n_eff = ledger.n_effective_trials()

    horizons = {}
    for tag in sorted({c["horizon"] for c in candidates}):
        pool = _horizon_pool(candidates, tag)
        dsr_by = _pool_dsr(pool, n_eff)
        pbo_block = _pool_pbo(pool)
        solo_by = {c["identity_sha256"]:
                   _solo_pass(c, dsr_by[c["identity_sha256"]], _naive_for(c, pool), gate)
                   for c in pool}
        gate_pool = [c for c in pool if c["config"] in LGBM_RUNGS]
        passing = [c for c in gate_pool if solo_by[c["identity_sha256"]]]
        h_pass = bool(passing and pbo_block["pbo_available"]
                      and pbo_block["pbo"] < gate["pbo_thresh"])
        best = max(passing, key=lambda c: c["result"].net_pnl) if passing else None
        horizons[tag] = {
            **pbo_block,
            "candidates": {c["identity_sha256"]:
                           _candidate_row(c, dsr_by[c["identity_sha256"]],
                                          solo_by[c["identity_sha256"]]) for c in pool},
            "pass": h_pass,
            "inconclusive": bool(passing and not pbo_block["pbo_available"]),
            "best": best["identity_sha256"] if best else None,
        }
    return {
        "protocol": "g0cb-development",
        "development_only": True,
        "g1_claim": False,          # never the project-defining gate (staged protocol §2)
        "gate": gate,
        "manifest": _arm_echo(prep),
        "partition_contract_sha256": contract_hash(contract),
        "horizons": horizons,
        "g0cb_pass": bool(any(h["pass"] for h in horizons.values())),
        "ledger": {"n_effective_trials": n_eff, "ledger_sha256": ledger.ledger_hash()},
    }


# ------------------------------------------------------------------------------- G0-XV
def _validate_matched_arms(preps: list[dict], gate: dict) -> dict:
    """All arms must be the SAME matched row universe: identical reserved content
    (labels, costs, timing, uniqueness, regime, horizon), identical row count, identical
    embargo, and therefore identical CPCV splits. Fail closed on any mismatch — per-arm
    significance over misaligned rows is meaningless (staged protocol §2)."""
    reserved = list(RESERVED)
    ref = preps[0]
    ref_hash = matrix_content_hash(ref["matrix"], reserved)
    emb = ref["manifest"]["embargo_ns"]
    ref_split = split_hash(ref["matrix"], n_groups=gate["n_groups"], k=gate["k"],
                           embargo_ns=emb)
    for p in preps[1:]:
        if len(p["matrix"]) != len(ref["matrix"]):
            raise ValueError(f"arm {p['arm']!r} has {len(p['matrix'])} rows; arm "
                             f"{ref['arm']!r} has {len(ref['matrix'])} — arms must share "
                             "one matched row universe")
        if p["manifest"]["embargo_ns"] != emb:
            raise ValueError(f"arm {p['arm']!r} embargo_ns {p['manifest']['embargo_ns']} "
                             f"!= {emb}; matched arms must share one embargo or their "
                             "CPCV splits differ")
        h = matrix_content_hash(p["matrix"], reserved)
        if h != ref_hash:
            raise ValueError(f"arm {p['arm']!r} reserved-column content differs from arm "
                             f"{ref['arm']!r} (row/label/cost/regime hash mismatch); "
                             "arms must be identical outside feature_cols")
        s = split_hash(p["matrix"], n_groups=gate["n_groups"], k=gate["k"], embargo_ns=emb)
        if s != ref_split:
            raise ValueError(f"arm {p['arm']!r} CPCV split hash differs")   # defensive
    return {"row_content_sha256": ref_hash, "split_sha256": ref_split,
            "n_rows": int(len(ref["matrix"])), "embargo_ns": int(emb),
            "n_groups": int(gate["n_groups"]), "k": int(gate["k"])}


def bootstrap_delta_band(delta, *, n_boot: int, block: int, alpha: float, seed: int):
    """Circular block-bootstrap (lo, hi) quantile band for the TOTAL paired net-PnL delta
    Σ(combined_i − control_i) over the common development-OOS rows (t_event-ordered).
    Preregistered params live in the gate block; deterministic via `seed`."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 100:
        raise ValueError("noise_band_n_boot must be an int >= 100")
    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise ValueError("noise_band_block must be an int >= 1")
    if not isinstance(alpha, float) or not (0.0 < alpha < 1.0):
        raise ValueError("noise_band_alpha must be a float in (0, 1)")
    d = np.asarray(delta, float)
    n = len(d)
    if n == 0 or block > n:
        # Too few common paired rows to assess noise at the preregistered block size: a
        # silently clamped block would degenerate to a zero-width band (a bare sign
        # check). NaN band -> beats_control False -> fail closed.
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)                     # ceil
    # Resamples are computed in bounded batches: materializing all n_boot * n indices
    # at once would allocate billions of int64 slots on a real six-month matched
    # matrix. Batching is deterministic — the RNG stream is consumed in the same order
    # regardless of batch boundaries.
    offsets = np.arange(block)[None, None, :]
    batch = max(1, 2_000_000 // max(n, 1))
    sums = np.empty(n_boot)
    pos = 0
    while pos < n_boot:
        b = min(batch, n_boot - pos)
        starts = rng.integers(0, n, size=(b, n_blocks))
        idx = (starts[:, :, None] + offsets) % n
        sums[pos:pos + b] = d[idx.reshape(b, -1)[:, :n]].sum(axis=1)
        pos += b
    lo, hi = np.quantile(sums, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _noise_band_block(pool: list[dict], *, gate: dict, control_arm: str,
                      combined_arm: str) -> dict:
    """The combined-vs-matched-control authorization condition: the combined arm's best
    gate-rung candidate must beat its SAME-(config, variant) control twin by more than
    the preregistered bootstrap noise band on paired per-row development-OOS net PnL."""
    combined = [c for c in pool if c["arm"] == combined_arm and c["config"] in LGBM_RUNGS]
    if not combined:
        return {"combined_candidate": None, "control_candidate": None,
                "delta_net_pnl": float("nan"), "band_low": float("nan"),
                "band_high": float("nan"), "beats_control": False}
    best = max(combined, key=lambda c: c["result"].net_pnl)
    twin = next(c for c in pool if c["arm"] == control_arm
                and c["config"] == best["config"] and c["variant"] == best["variant"])
    a, b = best["result"].per_sample_pnl, twin["result"].per_sample_pnl
    finite = np.isfinite(a) & np.isfinite(b)
    delta = a[finite] - b[finite]
    lo, hi = bootstrap_delta_band(delta, n_boot=gate["noise_band_n_boot"],
                                  block=gate["noise_band_block"],
                                  alpha=gate["noise_band_alpha"],
                                  seed=gate["noise_band_seed"])
    return {"combined_candidate": best["identity_sha256"],
            "control_candidate": twin["identity_sha256"],
            "delta_net_pnl": float(delta.sum()), "band_low": lo, "band_high": hi,
            "beats_control": bool(np.isfinite(lo) and lo > 0)}


def run_g0xv_development(arms: list[dict], contract: dict, *, gate: dict | None = None,
                         ledger: TrialLedger | None = None, prior_ledgers=(),
                         control_arm: str = CONTROL_ARM, combined_arm: str = COMBINED_ARM,
                         variants=None, configs=CONFIGS) -> dict:
    """The unified matched G0-XV development study (issue #48's evidence base). `arms` is
    a list of {"name", "manifest", "matrix"} over ONE matched development row universe;
    it must include the Coinbase-only control and the combined arm. This is deliberately
    NOT three independent `run_from_manifest` studies: one ledger covers every registered
    (arm, build, model config, horizon, variant) candidate, `prior_ledgers` (the G0-CB
    history) enter the effective DSR trial count, and PBO runs over the common cross-arm
    candidate-PnL matrix BEFORE any arm or winner is selected. Per-arm significance
    cannot authorize the archive when this unified study fails."""
    gate = resolve_g0_gate(gate, defaults=DEFAULT_XV_GATE)
    if not isinstance(arms, list) or len(arms) < 2:
        raise ValueError("G0-XV needs the matched multi-arm universe (>= 2 arms)")
    names = [a.get("name") for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm names: {sorted(names)}")
    registered = {control_arm, BINANCE_ARM, combined_arm}
    for required in sorted(registered):
        if required not in names:
            raise ValueError(f"G0-XV requires arm {required!r}: the staged protocol "
                             "preregisters the coinbase_only control, binance_only, and "
                             "combined arms; omitting one removes its candidates from "
                             "the common PBO matrix and the effective DSR count")
    extra = sorted(set(names) - registered)
    if extra:
        raise ValueError(f"unregistered G0-XV arms: {extra}; the staged protocol fixes "
                         "the study to exactly the three preregistered arms — a post-hoc "
                         "arm cannot enter the ledger or be frozen as spend-gate "
                         "evidence")
    for a in arms:
        if a["name"] == control_arm:
            # The control must genuinely be the target-venue-only build, not any
            # development manifest labeled 'coinbase_only' — the combined-vs-control
            # authorization test is meaningless against a non-Coinbase control.
            _require_target_only_venues(a["manifest"], "the G0-XV control arm")
        else:
            venues = a["manifest"]["venues"]
            unroled = [v for v in venues if v.get("role") not in ("signal", "target")]
            if unroled:
                raise ValueError(f"G0-XV arm {a['name']!r} venues must declare an "
                                 f"explicit signal/target role, got: {unroled}")
            signals = [v for v in venues if v["role"] == "signal"]
            if not signals:
                raise ValueError(f"G0-XV arm {a['name']!r} declares no signal venue; "
                                 "cross-venue arms must declare their signal venue "
                                 "explicitly")
            # ... and the signals must be the PREREGISTERED Binance markets: a
            # Kraken/OKX signal targeting Coinbase would freeze the wrong acquisition
            # experiment as Binance spend-gate evidence.
            bad = [v for v in signals
                   if (v.get("exchange"), v.get("symbol")) not in ALLOWED_SIGNAL_VENUES]
            if bad:
                raise ValueError(f"G0-XV arm {a['name']!r} signal venues must be the "
                                 f"preregistered Binance markets "
                                 f"{ALLOWED_SIGNAL_VENUES}, got: {bad}")
            # Cross-venue arms still label/trade the SAME protocol target venue.
            _require_expected_target(a["manifest"], f"G0-XV arm {a['name']!r}")

    ledger = ledger if ledger is not None else TrialLedger()
    # n_imported counts the TRIAL entries carried in from the supplied prior histories
    # (deduplicated, verdict entries excluded) — NOT how many were newly inserted, which
    # would report 0 on an idempotent rerun over an already-populated ledger and change
    # the frozen evidence hash.
    imported_ids: set = set()
    for prior in prior_ledgers:
        ledger.import_history(prior)
        imported_ids.update(e["identity_sha256"] for e in prior.entries()
                            if e["identity"]["protocol"] in TRIAL_PROTOCOLS)
    n_imported = len(imported_ids)

    preps = [_prepare_development_input(a["matrix"], a["manifest"], contract,
                                        arm=a["name"]) for a in arms]
    matched = _validate_matched_arms(preps, gate)

    # The combined arm must genuinely combine BOTH component feature groups (producer
    # contract: its feature list is the union of the control and Binance-only lists) —
    # an arm merely NAMED 'combined' but carrying only one side would make the
    # combined-vs-control authorization test meaningless.
    by_arm = {p["arm"]: p for p in preps}
    # The ablation arms must be genuinely disjoint: a control (target-book) feature
    # inside the Binance-only arm would let the "Binance-only" solo gate be driven by
    # Coinbase features, hollowing out the required three-way comparison.
    overlap = set(by_arm[control_arm]["feature_cols"]) \
        & set(by_arm[BINANCE_ARM]["feature_cols"])
    if overlap:
        raise ValueError(f"the {control_arm!r} and {BINANCE_ARM!r} feature sets must be "
                         f"disjoint; shared features: {sorted(overlap)}")
    union = set(by_arm[control_arm]["feature_cols"]) | set(by_arm[BINANCE_ARM]["feature_cols"])
    combined_feats = set(by_arm[combined_arm]["feature_cols"])
    if combined_feats != union:
        raise ValueError(
            f"the {combined_arm!r} arm's feature set must be exactly the union of the "
            f"{control_arm!r} and {BINANCE_ARM!r} feature sets; missing: "
            f"{sorted(union - combined_feats)}, extra: {sorted(combined_feats - union)}")

    candidates = []
    for prep in preps:
        candidates.extend(_evaluate_arm_candidates(
            prep, protocol=G0XV_PROTOCOL, gate=gate, ledger=ledger,
            variants=variants, configs=configs))
    n_eff = ledger.n_effective_trials()

    horizons = {}
    for tag in sorted({c["horizon"] for c in candidates}):
        pool = _horizon_pool(candidates, tag)
        dsr_by = _pool_dsr(pool, n_eff)
        # Matched arms share identical reserved rows and splits, so every arm's naive
        # per-sample PnL column is bit-identical; stacking one copy per arm would tilt
        # the CSCV rank denominator in a pass-friendly direction. PBO keeps exactly ONE
        # naive benchmark column — the control arm's.
        pbo_block = _pool_pbo([c for c in pool
                               if c["config"] != "naive" or c["arm"] == control_arm])
        solo_by = {c["identity_sha256"]:
                   _solo_pass(c, dsr_by[c["identity_sha256"]], _naive_for(c, pool), gate)
                   for c in pool}
        # Authorization condition (a): a non-naive CROSS-VENUE gate-rung candidate clears
        # the solo block. The Coinbase-only control cannot authorize the archive.
        cross_pass = [c for c in pool if c["arm"] != control_arm
                      and c["config"] in LGBM_RUNGS and solo_by[c["identity_sha256"]]]
        band = _noise_band_block(pool, gate=gate, control_arm=control_arm,
                                 combined_arm=combined_arm)
        pbo_ok = bool(pbo_block["pbo_available"] and pbo_block["pbo"] < gate["pbo_thresh"])
        h_pass = bool(cross_pass and band["beats_control"] and pbo_ok)
        horizons[tag] = {
            **pbo_block,
            "candidates": {c["identity_sha256"]:
                           _candidate_row(c, dsr_by[c["identity_sha256"]],
                                          solo_by[c["identity_sha256"]]) for c in pool},
            "solo_pass_cross_venue": [c["identity_sha256"] for c in cross_pass],
            "noise_band": band,
            "pass": h_pass,
            # No PBO verdict fails CLOSED: candidates may look strong, but without the
            # overfit probability the study is blocking/inconclusive, never a pass.
            "inconclusive_blocking": bool(cross_pass and band["beats_control"]
                                          and not pbo_block["pbo_available"]),
        }

    # Pin the matrix-level horizon verdicts (PBO, noise band, solo list, gate, pass)
    # into the SAME tamper-evident ledger as the trials, under the non-trial
    # g0xv-verdict protocol. The freeze re-verifies against these pinned verdicts, so
    # editing pass flags in a saved dev-result JSON cannot authorize the holdout for a
    # study that failed closed (PBO-unavailable or noise-band failures are not
    # recomputable from per-trial results alone).
    verdict_build = hash_obj({p["arm"]: p["manifest"]["build_id"] for p in preps})
    control_prep = next(p for p in preps if p["arm"] == control_arm)
    arms_out = {p["arm"]: _arm_echo(p) for p in preps}
    for tag, h in horizons.items():
        verdict_ident = trial_identity(
            protocol="g0xv-verdict", arm="unified",
            dataset_id=control_prep["manifest"]["dataset_id"], build_id=verdict_build,
            feature_cols=sorted(p["arm"] for p in preps),
            config="horizon_verdict", horizon=tag,
            variant_params={"gate_sha256": hash_obj(gate)})
        ledger.register(verdict_ident, {
            "pass": h["pass"],
            "inconclusive_blocking": h["inconclusive_blocking"],
            "pbo": h["pbo"], "pbo_available": h["pbo_available"],
            "pbo_n_rows": h["pbo_n_rows"],
            "pbo_candidates_sha256": hash_obj(list(h["pbo_candidates"])),
            "solo_pass_cross_venue_sha256": hash_obj(sorted(h["solo_pass_cross_venue"])),
            # The horizon's exact candidate pool: the freeze recomputes the winner's DSR
            # dispersion over THESE trials, so an append-only ledger reused across
            # builds cannot poison (or be poisoned by) another study's pool.
            "pool": sorted(h["candidates"]),
            "noise_band": dict(h["noise_band"]),
            "gate_sha256": hash_obj(gate),
            "matched_row_sha256": matched["row_content_sha256"],
            # Ledger-pinned audit of the carried search history: the freeze verifies
            # the dev result's reported import count against this, not the editable JSON.
            "n_imported_trials": n_imported,
            # Per-arm FULL content pins (reserved + feature values), ledger-pinned so an
            # edited dev-result JSON cannot substitute the hash the holdout refit
            # verifies against.
            "arm_matrix_hashes": {a: e["matrix_content_sha256"]
                                  for a, e in arms_out.items()},
        })

    winner = None
    passing_h = [t for t, h in horizons.items() if h["pass"]]
    if passing_h:
        pool = [c for c in candidates
                if c["horizon"] in passing_h and c["arm"] != control_arm
                and c["identity_sha256"] in horizons[c["horizon"]]["solo_pass_cross_venue"]]
        best = max(pool, key=lambda c: c["result"].net_pnl)
        winner = {"arm": best["arm"], "config": best["config"], "horizon": best["horizon"],
                  "variant": best["variant"], "identity_sha256": best["identity_sha256"],
                  "feature_cols": best["identity"]["feature_cols"],
                  "dataset_id": best["identity"]["dataset_id"],
                  "build_id": best["identity"]["build_id"],
                  "net_pnl": float(best["result"].net_pnl)}

    return {
        "protocol": "g0xv-development",
        "development_only": True,
        "g1_claim": False,
        "gate": gate,
        "arms": arms_out,
        "control_arm": control_arm,
        "combined_arm": combined_arm,
        "matched": matched,
        "partition_contract_sha256": contract_hash(contract),
        "horizons": horizons,
        "g0xv_dev_pass": bool(passing_h),
        "inconclusive_blocking": bool(not passing_h and any(
            h["inconclusive_blocking"] for h in horizons.values())),
        "winner": winner,
        "ledger": {"n_effective_trials": n_eff, "n_imported_trials": n_imported,
                   "ledger_sha256": ledger.ledger_hash()},
    }
