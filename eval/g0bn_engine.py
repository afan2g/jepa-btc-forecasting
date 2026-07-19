"""G0-BN development candidate engine (issue #88, slice 67-B).

Implements spec sections 3.4 and 4.1-4.2 of
docs/superpowers/specs/2026-07-13-g0bn-protocol.md for the DEVELOPMENT partition:

- runtime parameter re-resolution: every fitted candidate is re-instantiated and its
  complete `get_params(deep=False)` compared to the stored resolved object before
  every fit, and the installed package version must equal the pinned
  `package_version` — a drifted default, seed, or version fails closed instead of
  silently reidentifying a trial;
- the five fixed candidates per horizon: `persistence_zero` (empty-feature constant
  0.0 bps), `microprice_raw` (`1.0 * microprice_dev`), uniqueness-weighted Ridge,
  uniqueness-weighted LightGBM regression, and the 3-class LightGBM classifier whose
  forecast is `(P(+1)-P(-1)) * training_y_std_bps` with the exact unweighted binary64
  population scale (`ddof=0`, `+1e-9`) — uniqueness weights fit the classifier but
  cannot enter the scale;
- CPCV enumeration in lexicographic `itertools.combinations(range(6), 2)` order via
  `data.cv.cpcv_splits` (span purge on `t0=t_event`/`t1=t_barrier`,
  `embargo_ns == max_lookback_ns`; spec section 9) and the versioned
  `mean_repeated_test_forecasts_v1` collapse: exactly five finite float64 test
  forecasts per row, accumulated at the original row position in the fixed split
  order, then divided by exactly 5.0;
- fail-closed development-input verification binding the matrix, T8 manifest,
  protocol config, and logical development-data identity to each other before any
  trial runs; and
- `run_g0bn_development`: executes the 15 preregistered base trials in horizon-major
  ladder order against the separate append-only G0-BN ledger — completions must
  reproduce their result hash (idempotent reruns), infrastructure failures become
  aborted events that still count in effective N, and a conflicting completion
  propagates (non-determinism is never recorded as an abort).

This module is G0-BN-only: it never touches the legacy G0-CB/G0-XV evaluator or
ledger, and reuses only pure primitives (data.cv.cpcv_splits, eval.hashing, the
eval.manifest/eval.matrix/eval.writer reader contracts) whose contracts the spec
pins by name.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from dataclasses import dataclass

import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Ridge

from data.cv import cpcv_splits
from eval.g0bn_config import (
    DEV_END_NS,
    DEV_START_NS,
    FORECAST_COLLAPSE_VERSION,
    _exact,
    _fail,
    _validate_candidate,
    validate_protocol_config,
)
from eval.g0bn_identity import base_trial_identities, development_data_identity
from eval.g0bn_ledger import G0BNLedger
from eval.hashing import canonical_json, hash_obj
from eval.manifest import feature_list, manifest_sha256, validate_frame
from eval.matrix import validate_matrix
from eval.writer import (
    COST_ASSUMPTION_SOURCE,
    G0BN_DATA_SOURCES,
    PARTITION_BINDING,
    PROTOCOL_BINDING,
    classify_manifest,
    logical_row_sha256,
    ordered_manifest_columns,
)

# Fixed CV geometry (spec section 3.4): C(6,2) = 15 test-group pairs, so every row is
# tested in exactly C(5,1) = 5 splits. These are protocol constants, not knobs.
N_GROUPS = 6
K_TEST_GROUPS = 2
N_SPLITS = 15
TEST_MULTIPLICITY = 5

RESULT_SCHEMA = "g0bn-trial-result-v1"
FORECAST_SERIES_SCHEMA = "g0bn-forecast-series-v1"
# The development source-manifest identity: how G0-BN derives the config's
# `development_source_manifest_sha256` evidence pin from a T8 manifest's Binance
# source-object entries (ordered per-name hash lists), so a self-consistent
# manifest backed by different/uncertified source objects fails closed.
DEV_SOURCE_MANIFEST_SCHEMA = "g0bn-dev-source-manifest-v1"

_DAY_NS = 86_400_000_000_000

# The complete closed estimator registry: the G0-BN ladder admits exactly these three
# fitted estimator classes (spec section 4.1). Resolution never imports an arbitrary
# dotted path — an off-registry class fails closed.
ESTIMATOR_CLASSES = {
    "sklearn.linear_model.Ridge": Ridge,
    "lightgbm.LGBMRegressor": LGBMRegressor,
    "lightgbm.LGBMClassifier": LGBMClassifier,
}
PACKAGE_VERSIONS = {
    "scikit-learn": sklearn.__version__,
    "lightgbm": lightgbm.__version__,
}


# ------------------------------------------------------------ runtime re-resolution

def g0bn_candidate_code_sha256() -> str:
    """The runtime identity of the candidate implementation surface: SHA-256 over
    THIS module's source bytes plus data/cv.py — the CPCV split/purge/embargo
    machinery the candidates execute under is part of what `candidate_code_sha256`
    pins, so a modified splitter under an unchanged config is a DIFFERENT trial,
    never a silent reidentification (spec sections 3.4 and 4.1). Deliberately
    uncached: the files are re-read on every resolution so a mid-run source change
    cannot slip through."""
    import data.cv as _cv_module
    digest = hashlib.sha256()
    for source_path in (__file__, _cv_module.__file__):
        with open(source_path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def verify_runtime_software(config: dict) -> None:
    """Compare the config's pinned software versions to the RUNNING environment
    (spec section 3.2: the config repeats values and code compares runtime-resolved
    values to it). The repository commit/tree pins are deliberately NOT checked
    here — git state is the 67-E one-shot runner's pre-burn self-verification
    (spec section 6.3 step 1), where the operator supplies the checkout evidence."""
    installed = {
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__,
        "pyarrow_version": pyarrow.__version__,
    }
    for key, running in installed.items():
        if config["software"][key] != running:
            _fail(f"software.{key}",
                  f"config pins {config['software'][key]!r} but the running "
                  f"environment resolves {running!r}; a software change is a "
                  "different build and must not execute under stale provenance")


def resolve_runtime_candidate(defn: dict):
    """Re-validate a candidate definition and re-resolve its runtime instantiation.

    Static section-4.1 validation reuses the 67-A config validator, then for fitted
    candidates the installed package version must equal the pinned package_version
    and a fresh estimator's complete get_params(deep=False) must equal the stored
    resolved model_params exactly (type-strict; a bool is not an int). Returns the
    fresh estimator for fitted candidates, None for the two non-fitted candidates.
    Called before EVERY fit so a drift mid-run cannot slip through."""
    if not isinstance(defn, dict) or "candidate_id" not in defn:
        _fail("candidate definition", "must be a candidate definition dict")
    cid = defn["candidate_id"]
    _validate_candidate(f"candidates[{cid!r}]", defn, cid)
    running_code = g0bn_candidate_code_sha256()
    if defn["candidate_code_sha256"] != running_code:
        _fail(f"candidates[{cid!r}].candidate_code_sha256",
              f"pinned {defn['candidate_code_sha256']} but the running candidate "
              f"implementation hashes to {running_code}; a code change is a "
              "DIFFERENT trial and must never silently reidentify this one "
              "(spec section 4.1)")
    if not defn["fitted"]:
        return None
    installed = PACKAGE_VERSIONS.get(defn["package"])
    if installed != defn["package_version"]:
        _fail(f"candidates[{cid!r}].package_version",
              f"pinned {defn['package_version']!r} but the installed {defn['package']} "
              f"is {installed!r}; a package/version change creates a DIFFERENT trial "
              "and must never silently reidentify this one (spec section 4.1)")
    cls = ESTIMATOR_CLASSES.get(defn["estimator_class"])
    if cls is None:
        _fail(f"candidates[{cid!r}].estimator_class",
              f"{defn['estimator_class']!r} is not in the closed G0-BN estimator "
              f"registry {sorted(ESTIMATOR_CLASSES)}")
    estimator = cls(**copy.deepcopy(defn["model_params"]))
    resolved = estimator.get_params(deep=False)
    try:
        _exact(f"candidates[{cid!r}].model_params", resolved, defn["model_params"])
    except ValueError as exc:
        raise ValueError(
            f"candidates[{cid!r}]: runtime re-resolved get_params(deep=False) does "
            f"not match the stored resolved model_params ({exc}); the installed "
            "library exposes a different default — this is a different trial, not "
            "this one") from None
    return estimator


# ---------------------------------------------------------------- forecast collapse

def collapse_split_forecasts(n_rows: int, splits) -> np.ndarray:
    """The mandatory `mean_repeated_test_forecasts_v1` collapse (spec section 3.4).

    `splits` is the ordered 15-split sequence of `(test_row_positions, forecasts)`.
    Each split forecast is cast to float64, required to be finite, and added at its
    original row position in the fixed split order; after all 15 splits every row
    must have received exactly 5 test forecasts, and the sole CPCV-OOS forecast is
    `forecast_sum / 5.0`. No weighting, median, row duplication, or split choice."""
    splits = list(splits)
    if len(splits) != N_SPLITS:
        raise ValueError(f"CPCV must enumerate exactly {N_SPLITS} test-group pairs in "
                         f"lexicographic combinations(range(6), 2) order; got "
                         f"{len(splits)} splits")
    forecast_sum = np.zeros(n_rows, dtype=np.float64)
    count = np.zeros(n_rows, dtype=np.int64)
    for split_no, (test_idx, forecasts) in enumerate(splits):
        idx = np.asarray(test_idx)
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"split {split_no}: test row positions must be a 1-D "
                             f"integer array; got dtype {idx.dtype}")
        fc = np.asarray(forecasts)
        if fc.shape != idx.shape:
            raise ValueError(f"split {split_no}: forecast length {fc.shape} does not "
                             f"match its test rows {idx.shape}")
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_rows):
            raise ValueError(f"split {split_no}: test row index out of range for "
                             f"{n_rows} rows")
        if np.unique(idx).size != idx.size:
            raise ValueError(f"split {split_no}: duplicate test rows within one split "
                             "(a row is predicted at most once per split)")
        fc64 = fc.astype(np.float64)
        if not np.isfinite(fc64).all():
            raise ValueError(f"split {split_no}: non-finite forecast; every split "
                             "forecast must be finite float64 before it enters the "
                             "collapse")
        forecast_sum[idx] += fc64
        count[idx] += 1
    bad = np.where(count != TEST_MULTIPLICITY)[0]
    if bad.size:
        raise ValueError(
            f"every row must receive exactly {TEST_MULTIPLICITY} test forecasts "
            f"(C(5,1) with the fixed 6/2 geometry); rows {bad[:5].tolist()} received "
            f"{count[bad[:5]].tolist()}")
    return forecast_sum / 5.0


def forecast_series_sha256(t_event, horizon: str, forecasts) -> str:
    """Canonical hash of one collapsed CPCV-OOS forecast series: the (t_event,
    horizon) row keys in stable order plus the little-endian float64 forecast bytes.
    This is the trial result's determinism witness — an exact rerun must reproduce
    it bit for bit."""
    te = np.ascontiguousarray(np.asarray(t_event, dtype=np.int64), dtype="<i8")
    fc = np.ascontiguousarray(np.asarray(forecasts, dtype=np.float64), dtype="<f8")
    if te.ndim != 1 or te.shape != fc.shape:
        raise ValueError("forecast series requires matching 1-D t_event/forecast "
                         "arrays")
    if not np.isfinite(fc).all():
        raise ValueError("forecast series must be finite")
    header = canonical_json({"schema": FORECAST_SERIES_SCHEMA, "horizon": horizon,
                             "n_rows": int(te.size), "dtype": "<f8"})
    digest = hashlib.sha256()
    digest.update(header.encode())
    digest.update(b"\n")
    digest.update(te.tobytes())
    digest.update(fc.tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------- candidate execution

def classifier_training_scale(y_train_bps) -> float:
    """Exact spec-4.1 classifier scale over the fold's finite training y_fwd_bps:
    unweighted binary64 population standard deviation (`ddof=0`) plus 1e-9. There is
    deliberately no weight parameter — uniqueness weights cannot enter the scale."""
    y = np.asarray(y_train_bps, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("classifier training scale requires at least one finite "
                         "y_fwd_bps training value")
    return float(np.std(np.asarray(y, dtype=np.float64), dtype=np.float64,
                        ddof=0) + np.float64(1e-9))


def class_prob_spread(proba, classes, scale: float) -> np.ndarray:
    """`(P(+1) - P(-1)) * training_y_std_bps` with classes mapped through the fitted
    model's `classes_`; a class absent from the purged training fold has probability
    exactly 0.0."""
    proba = np.asarray(proba, dtype=np.float64)
    class_list = [int(c) for c in np.asarray(classes).tolist()]

    def _col(cls: int) -> np.ndarray:
        if cls in class_list:
            return proba[:, class_list.index(cls)]
        return np.zeros(len(proba), dtype=np.float64)

    return (_col(1) - _col(-1)) * np.float64(scale)


def cpcv_candidate_forecasts(defn: dict, rows: pd.DataFrame, *, embargo_ns: int):
    """Produce one collapsed CPCV-OOS forecast per row for one candidate at one
    horizon. Returns (float64 forecasts, ordered split-scale list or None). The
    split-scale list is the lgbm_clf realized-scale provenance (spec section 4.1);
    every other candidate returns None."""
    cid = defn.get("candidate_id") if isinstance(defn, dict) else None
    estimator_check = resolve_runtime_candidate(defn)  # static + runtime gate, once
    n = len(rows)
    t_event = rows["t_event"].to_numpy(np.int64)
    t_barrier = rows["t_barrier"].to_numpy(np.int64)
    if (np.diff(t_event) <= 0).any():
        raise ValueError(f"candidate {cid!r}: horizon rows must be strictly "
                         "t_event-ordered and unique (producer invariant)")
    feature_cols = list(defn["feature_cols"])
    X = (rows[feature_cols].to_numpy(np.float64) if feature_cols
         else np.zeros((n, 0), dtype=np.float64))
    y = rows["y_fwd_bps"].to_numpy(np.float64)
    label = rows["label"].to_numpy(np.int64)
    weights = rows["uniqueness"].to_numpy(np.float64)

    split_list = list(cpcv_splits(t_event, t_event, t_barrier, n_groups=N_GROUPS,
                                  k=K_TEST_GROUPS, embargo_ns=embargo_ns))
    outputs = []
    split_scales: list[float] = []
    for train_idx, test_idx in split_list:
        if cid == "persistence_zero":
            fc = np.zeros(len(test_idx), dtype=np.float64)
        elif cid == "microprice_raw":
            fc = np.float64(defn["model_params"]["multiplier"]) * X[test_idx, 0]
        elif cid in ("ofi_ridge", "lgbm_reg"):
            est = resolve_runtime_candidate(defn)   # re-resolved before every fit
            est.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
            fc = est.predict(X[test_idx])
        elif cid == "lgbm_clf":
            est = resolve_runtime_candidate(defn)
            scale = classifier_training_scale(y[train_idx])
            est.fit(X[train_idx], label[train_idx],
                    sample_weight=weights[train_idx])
            fc = class_prob_spread(est.predict_proba(X[test_idx]), est.classes_,
                                   scale)
            split_scales.append(scale)
        else:
            raise ValueError(f"unknown G0-BN candidate {cid!r}")
        outputs.append((test_idx, fc))
    del estimator_check
    forecasts = collapse_split_forecasts(n, outputs)
    return forecasts, (split_scales if cid == "lgbm_clf" else None)


# ------------------------------------------------------------- input verification

def _named_sources(manifest: dict) -> dict:
    named: dict = {}
    for s in manifest["sources"]:
        if isinstance(s, dict):
            named.setdefault(s["name"], []).append(s)
    return named


def verify_development_inputs(frame: pd.DataFrame, manifest: dict, config: dict,
                              data_identity: dict) -> dict:
    """Fail-closed binding of the development matrix, T8 manifest, protocol config,
    and logical data identity to each other (spec sections 2.2, 3.2, 4.2, 7). Runs
    before any trial; returns {horizon_tag: rows} with each horizon's rows stable-
    sorted by t_event (the producer's unique (t_event, horizon) invariant makes that
    order total)."""
    validate_protocol_config(config)
    verify_runtime_software(config)
    development_data_identity(data_identity)
    if data_identity["partition_plan_sha256"] != config["partition"]["sha256"]:
        _fail("partition_plan_sha256",
              "development data identity does not bind the config's partition plan")

    cls = classify_manifest(manifest)
    if not cls.is_g0bn or cls.holdout_bound or cls.partition != "development":
        _fail("manifest", f"G0-BN development requires a development-bound G0-BN "
                          f"manifest; got {cls}")
    if manifest_sha256(manifest) != data_identity["development_manifest_sha256"]:
        _fail("development_manifest_sha256",
              "manifest content does not match the pinned development manifest hash "
              "(stale or foreign manifest)")
    if manifest["build_id"] != data_identity["development_build_id"]:
        _fail("development_build_id",
              f"manifest build_id {manifest['build_id']} does not match the pinned "
              f"development build {data_identity['development_build_id']}")

    named = _named_sources(manifest)
    partition_binding = named[PARTITION_BINDING][0]
    if partition_binding["partition_plan_sha256"] != config["partition"]["sha256"]:
        _fail("partition_contract.partition_plan_sha256",
              "manifest partition binding does not pin the config's partition plan")
    protocol_binding = named[PROTOCOL_BINDING][0]
    if protocol_binding["protocol_config_sha256"] != config["sha256"]:
        _fail("g0bn_protocol.protocol_config_sha256",
              f"manifest pins protocol config "
              f"{protocol_binding['protocol_config_sha256']} but this config hashes "
              f"to {config['sha256']}")
    cert_sha = config["source_certification"]["certification_sha256"]
    if protocol_binding["source_certification_sha256"] != cert_sha:
        _fail("g0bn_protocol.source_certification_sha256",
              "manifest source-certification pin does not match the config")
    roles = {h["tag"]: h["role"] for h in config["horizons"]}
    if protocol_binding["horizon_roles_sha256"] != hash_obj(roles):
        _fail("g0bn_protocol.horizon_roles_sha256",
              "manifest horizon-role pin does not match the config's section-2.3 "
              "roles")
    # The manifest's ACTUAL evidence entries must reconcile with the certified
    # config — not just the copies inside the protocol binding. The T8 writer
    # validates these entries by allowlisted name and hex shape only; content
    # binding against the #64 evidence is this gate's job (spec section 2.1).
    cert_entry = named["source_certification"][0]
    if cert_entry["sha256"] != cert_sha:
        _fail("source_certification",
              f"manifest source_certification entry pins {cert_entry['sha256']}, "
              f"not the config's certified #64 evidence {cert_sha}")
    source_map = {name: sorted(entry["sha256"] for entry in named.get(name, []))
                  for name in G0BN_DATA_SOURCES}
    dev_source_manifest_sha256 = hash_obj({"schema": DEV_SOURCE_MANIFEST_SCHEMA,
                                           "sources": source_map})
    expected_sources = config["source_certification"][
        "development_source_manifest_sha256"]
    if dev_source_manifest_sha256 != expected_sources:
        _fail("sources",
              f"the manifest's Binance source-object hashes derive development "
              f"source-manifest {dev_source_manifest_sha256}, not the config's "
              f"certified evidence pin {expected_sources}; a manifest backed by "
              "different or uncertified source objects fails closed")
    manifest_cost = {k: v for k, v in named[COST_ASSUMPTION_SOURCE][0].items()
                     if k != "name"}
    if manifest_cost != config["costs"]["cost_assumption"]:
        _fail("cost_assumption",
              "manifest cost_assumption source does not equal the config's exact "
              "serialized CostAssumption (spec section 8.2)")
    if manifest["embargo_ns"] != config["cv"]["embargo_ns"]:
        _fail("embargo_ns", f"manifest {manifest['embargo_ns']} != config "
                            f"{config['cv']['embargo_ns']}")
    if manifest["max_lookback_ns"] != config["features"]["max_lookback_ns"]:
        _fail("max_lookback_ns", f"manifest {manifest['max_lookback_ns']} != config "
                                 f"{config['features']['max_lookback_ns']}")
    horizon_ns = {h["tag"]: h["ns"] for h in config["horizons"]}
    if manifest["horizons"] != horizon_ns:
        _fail("horizons", "manifest horizon map does not equal the config's")

    validate_frame(frame, manifest)
    validate_matrix(frame, feature_list(manifest))
    lrh = logical_row_sha256(frame, ordered_manifest_columns(manifest))
    if lrh != data_identity["development_logical_row_sha256"]:
        _fail("development_logical_row_sha256",
              f"the frame's canonical logical rows hash to {lrh}, not the pinned "
              f"{data_identity['development_logical_row_sha256']} — this is not the "
              "identified development matrix")

    t_event = frame["t_event"].to_numpy(np.int64)
    t_barrier = frame["t_barrier"].to_numpy(np.int64)
    row_horizon_ns = frame["horizon"].map(horizon_ns).to_numpy(np.int64)
    guard = config["partition"]["partition_guard_ns"]
    if (t_event < DEV_START_NS).any() or (t_event >= DEV_END_NS).any():
        _fail("t_event", "development support window violated: every row must have "
                         f"t_event in [{DEV_START_NS}, {DEV_END_NS})")
    if ((t_event + row_horizon_ns + guard) >= DEV_END_NS).any():
        _fail("t_event", "partition prefilter violated: require t_event + "
                         "horizons[horizon] + partition_guard_ns < partition_end_ns "
                         "(spec section 2.2); the producer must have dropped this row")
    if ((t_barrier + guard) >= DEV_END_NS).any():
        _fail("t_barrier", "partition guarded span violated: the actual span ending "
                           "at t_barrier plus the guard crosses the development end")
    days = pd.to_datetime(frame["t_event"], utc=True).dt.strftime("%Y-%m-%d")
    included = set(config["exclusions"]["included_days"])
    outside = sorted(set(days) - included)
    if outside:
        _fail("exclusions", f"rows on days outside the frozen included-day "
                            f"accounting: {outside[:5]}")

    horizon_rows = {}
    for h in config["horizons"]:
        sub = frame[frame["horizon"] == h["tag"]]
        if not len(sub):
            _fail("horizons", f"horizon {h['tag']!r} has no development rows; all "
                              "three ladder horizons run the complete fixed ladder "
                              "(spec section 2.3)")
        horizon_rows[h["tag"]] = (sub.sort_values("t_event", kind="mergesort")
                                  .reset_index(drop=True))
    return horizon_rows


# --------------------------------------------------------------------- run driver

@dataclass
class DevelopmentRun:
    """In-memory outcome of one development execution pass: the ledger holds the
    durable identity/event/result records; forecasts stay in memory for the
    selection stage, which re-verifies them against the ledger result hashes AND
    re-binds the carried rows to the pinned logical-row identity (the manifest is
    kept for that recomputation)."""
    config: dict
    data_identity: dict
    manifest: dict
    horizon_rows: dict
    identities: dict
    forecasts: dict
    split_scales: dict
    aborted: dict
    ledger: G0BNLedger


def run_g0bn_development(frame: pd.DataFrame, manifest: dict, config: dict,
                         data_identity: dict, ledger: G0BNLedger, *,
                         recorded_at: str | None = None) -> DevelopmentRun:
    """Execute the complete preregistered 15-trial base ladder (5 candidates x 3
    horizons, horizon-major ladder order) against the append-only G0-BN ledger.

    Per-trial infrastructure failures are recorded as aborted events (the unique
    identity still counts in effective N) and execution continues; a conflicting
    completion result propagates — a deterministic protocol must reproduce results
    exactly, so non-determinism is a run failure, never a silent abort.

    The ledger must be path-bound (durable): every start is persisted BEFORE its
    fit and every terminal event immediately after, so a process crash cannot
    erase attempted identities from effective N (spec section 4.2)."""
    if not ledger.is_durable():
        _fail("ledger", "run_g0bn_development requires a path-bound durable ledger "
                        "(G0BNLedger(path=...) or G0BNLedger.load(path)); an "
                        "in-memory ledger would lose attempted identities on a "
                        "crash and undercount effective N (spec section 4.2)")
    horizon_rows = verify_development_inputs(frame, manifest, config, data_identity)
    definitions = {d["candidate_id"]: d for d in config["candidates"]}
    embargo_ns = config["cv"]["embargo_ns"]
    identities: dict = {}
    forecasts: dict = {}
    split_scales: dict = {}
    aborted: dict = {}
    for identity in base_trial_identities(config, data_identity):
        tid = ledger.record_start(identity, recorded_at=recorded_at)
        identities[tid] = identity
        rows = horizon_rows[identity["horizon"]]
        try:
            fc, scales = cpcv_candidate_forecasts(
                definitions[identity["candidate_id"]], rows, embargo_ns=embargo_ns)
        except Exception as exc:  # infrastructure/fit failure -> aborted trial event
            message = f"{type(exc).__name__}: {exc}"
            ledger.record_abort(identity, error=message, recorded_at=recorded_at)
            aborted[tid] = message
            continue
        result = {
            "schema": RESULT_SCHEMA,
            "n_rows": int(len(rows)),
            "forecasts_sha256": forecast_series_sha256(
                rows["t_event"].to_numpy(np.int64), identity["horizon"], fc),
            "collapse_version": FORECAST_COLLAPSE_VERSION,
            "split_scales": scales,
        }
        # Outside the try-block: a conflicting result for an existing identity must
        # fail the run, not masquerade as an infrastructure abort.
        ledger.record_completion(identity, result, recorded_at=recorded_at)
        forecasts[tid] = fc
        split_scales[tid] = scales
    return DevelopmentRun(config=config, data_identity=data_identity,
                          manifest=manifest, horizon_rows=horizon_rows,
                          identities=identities, forecasts=forecasts,
                          split_scales=split_scales, aborted=aborted,
                          ledger=ledger)
