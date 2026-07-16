"""Strict G0-BN protocol config (issue #86, slice 67-A of issue #67).

Implements the canonical `g0bn-protocol-config-v1` schema from the binding
protocol (docs/superpowers/specs/2026-07-13-g0bn-protocol.md, sections 1-4).
G0-BN is NOT an arm or mode of the legacy G0-CB/G0-XV evaluator: this module
defines separate constants, types, and validators, and only reuses the pure
canonical-JSON hashing primitives from eval.hashing (spec section 1 reuse rule).

Validation is fail-closed and typed exactly as spec section 3.1 requires:
exact nested field sets (no unknown fields, no implicit defaults), integers
that are not booleans, finite numbers, non-empty strings that are not
TBD/UNRESOLVED placeholders, 64-lowercase-hex `*_sha256` fields, sorted unique
canonical YYYY-MM-DD day arrays, and no numeric-string coercion. Arrays are
ordered and decision-bearing. The embedded config `sha256` excludes exactly
itself and the top-level `generated_at`, matching eval.hashing.canonical_json.
"""
from __future__ import annotations

import datetime as _dt
import math
import re

from bars.cost import DRIFT_POLICY, CostAssumption, validate_cost_assumption
from bars.modes import VENUE_BINANCE
from eval.hashing import hash_obj

# --- protocol identity strings (spec sections 1, 3.2, 6.1, 7) -------------------------

CONFIG_SCHEMA = "g0bn-protocol-config-v1"
PROTOCOL_ID = "g0bn-v1"
PILOT_ID = "g0bn-2025-11_2026-01-v1"
PARTITION_PLAN_SCHEMA = "g0bn-partition-plan-v1"
TRIAL_SCHEMA = "g0bn-trial-v1"
UNIVERSE_SCHEMA = "g0bn-holdout-universe-v1"
ONE_SHOT_SCHEMA = "g0bn-one-shot-v1"
CONSUMPTION_SCHEMA = "g0bn-consumption-v1"
RAW_ACCESS_CLAIM_SCHEMA = "g0bn-raw-access-claim-v1"
MATRIX_ACCESS_CLAIM_SCHEMA = "g0bn-matrix-access-claim-v1"
ATTESTATION_SCHEMA = "g0bn-materialization-attestation-v1"
VERDICT_SCHEMA = "g0bn-verdict-v1"
REPORT_SCHEMA = "g0bn-report-v1"
BOOTSTRAP_DRAW_SCHEMA = "g0bn-circular-day-bootstrap-v1"

DEV_DATASET_ID = "binance_single_venue_g0bn_dev"
OOS_DATASET_ID = "binance_single_venue_g0bn_oos"

# The exact #64-certified Binance Futures BTC-USDT product identities the protocol
# admits (spec section 2.1: only the certified L2 snapshot/delta and trade products;
# Coinbase/CoinAPI, spot, other assets/perpetuals, and state feeds are rejected).
# These are part of the frozen v1 source declaration; if final #64 evidence pins
# different native IDs they update here like the other section-12 freeze inputs.
CERTIFIED_PROVIDER = "crypto-lake"
# The certified L2 source is snapshot + delta (both consumed by reconstruction), plus
# trades — three distinct products, matching the T8 writer's G0BN_DATA_SOURCES. #93
# reconciles these provider-path identities with writer.py's normalized source names.
CERTIFIED_L2_SNAPSHOT_PRODUCT = "binance-futures/book_snapshot_v2"
CERTIFIED_L2_DELTA_PRODUCT = "binance-futures/book_delta_v2"
CERTIFIED_TRADE_PRODUCT = "binance-futures/trades_v1"

# --- fixed instrument, windows, horizons, features, candidates (spec sections 2-4) ----

INSTRUMENT = {
    "exchange": "BINANCE_FUTURES",
    "native_symbol": "BTCUSDT",
    "symbol": "BTC-USDT-PERP",
    "contract_type": "linear_perpetual",
    "base_asset": "BTC",
    "quote_asset": "USDT",
    "settlement_asset": "USDT",
}

DEV_START_NS = 1_761_955_200_000_000_000    # 2025-11-01T00:00:00Z
DEV_END_NS = 1_767_225_600_000_000_000      # 2026-01-01T00:00:00Z
HOLDOUT_START_NS = 1_767_225_600_000_000_000
HOLDOUT_END_NS = 1_769_904_000_000_000_000  # 2026-02-01T00:00:00Z

# Distinct from the legacy eval.partition PREFILTER_RULE string on purpose: the
# G0-BN plan pins the spec section 2.2 wording and must not alias the g0xv contract.
PREFILTER_RULE = "t_event + horizons[horizon] + partition_guard_ns < partition_end_ns"

HORIZONS = (
    {"tag": "2s", "ns": 2_000_000_000, "role": "primary"},
    {"tag": "10s", "ns": 10_000_000_000, "role": "primary"},
    {"tag": "60s", "ns": 60_000_000_000, "role": "control-only"},
)
HORIZON_NS = {h["tag"]: h["ns"] for h in HORIZONS}
HORIZON_ROLES = {h["tag"]: h["role"] for h in HORIZONS}

FEATURE_REGISTRY = (
    "ofi_integrated",
    "microprice_dev",
    "queue_imb",
    "spread_tick",
    "cvd",
    "depth_imbalance",
    "book_slope",
    "vwap_minus_mid",
    "trade_count",
    "signed_vol",
    "aggressor_imb",
    "largest_print",
    "event_intensity",
    "rv_intrabar",
    "mae_intrabar",
    "elapsed_ns",
    "tod_sin",
    "tod_cos",
)

CANDIDATE_IDS = ("persistence_zero", "microprice_raw", "ofi_ridge", "lgbm_reg", "lgbm_clf")
SELECTABLE_CANDIDATE_IDS = CANDIDATE_IDS[1:]

RIDGE_FIXED_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True,
    "copy_X": True,
    "max_iter": None,
    "tol": 0.0001,
    "solver": "svd",
    "positive": False,
    "random_state": None,
}

LGBM_FIXED_COMMON_PARAMS = {
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample_for_bin": 200000,
    "min_split_gain": 0.0,
    "min_child_weight": 0.001,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 0,
    "n_jobs": 1,
    "importance_type": "split",
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
LGBM_REG_FIXED_PARAMS = dict(LGBM_FIXED_COMMON_PARAMS, objective="regression")
LGBM_CLF_FIXED_PARAMS = dict(LGBM_FIXED_COMMON_PARAMS, objective="multiclass", num_class=3)

PERSISTENCE_MODEL_PARAMS = {"forecast_bps": 0.0}
MICROPRICE_MODEL_PARAMS = {"input": "microprice_dev", "input_unit": "bps", "multiplier": 1.0}

# Exact preprocessing/stationarization contract per candidate (spec §4.1): fixed part
# of the eligible ladder, so an altered preprocessing is a different (non-eligible)
# trial. Fitted candidates share the producer's pinned causal stationarization with
# no candidate-local scaling (explicit for ofi_ridge); non-fitted candidates use none.
FITTED_PREPROCESSING = {"stationarization": "producer_pinned_causal_v1",
                        "candidate_local_scaling": False}
NONFITTED_PREPROCESSING = {"none": True}

CLASSIFIER_SCALE_RULE = "unweighted_population_float64_plus_1e-9_v1"
CLASS_ORDER = [-1, 0, 1]
FORECAST_COLLAPSE_VERSION = "mean_repeated_test_forecasts_v1"
DSR_ROUNDING_RULE = "nearest_ties_to_even_int64_v1"
PBO_IS_TIE_RULE = "first_max_v1"
PBO_OOS_RANK_RULE = "less_equal_count_v1"
BOOTSTRAP_KIND = "paired_utc_day_circular_moving_block"
CONTENTION_RESULT = "transaction_already_running"
SELECTION_RANKING_RULE = "trade_eligible_net_lift_then_predictive_lift_v1"
SELECTION_TIE_RULE = "earlier_ladder_order_v1"
SELECTION_ATTEMPT_ACCOUNTING = "unique_canonical_identity_v1"

# Fixed G0-BN label semantics (spec §3.2): log-mid bps return, trailing EWMA barriers,
# vertical-barrier fallback, per-horizon uniqueness. The EWMA half-life and TP/SL
# multipliers are operator-supplied (§12) and validated for type/presence, not pinned.
LABEL_RETURN_FORMULA = "log_mid_ratio_bps_v1"
LABEL_BARRIER_ESTIMATOR = "trailing_ewma_vol_v1"
LABEL_UNRESOLVED_BARRIER_POLICY = "vertical_barrier_return_v1"
LABEL_UNIQUENESS_POLICY = "per_horizon_concurrency_v1"

COMPONENT_COST_FIELDS = (
    "gross_bps", "fee_bps", "decision_cost_bps", "decision_total_cost_bps",
    "spread_bps", "base_slippage_bps", "latency_drift_bps", "slippage_bps",
    "cost_bps", "realized_total_cost_bps", "net_bps",
)

_PLACEHOLDERS = ("TBD", "UNRESOLVED")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_HEX_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def _dev_days() -> tuple:
    start = _dt.date(2025, 11, 1)
    return tuple((start + _dt.timedelta(days=i)).isoformat() for i in range(61))


DEV_DAYS = _dev_days()


# --- typed fail-closed primitives (spec section 3.1) -----------------------------------

def _fail(path: str, msg: str):
    raise ValueError(f"{path} {msg}")


def _dict(path: str, obj, required: tuple, optional: tuple = ()) -> dict:
    if not isinstance(obj, dict):
        _fail(path, f"must be a dict; got {type(obj).__name__}")
    missing = [k for k in required if k not in obj]
    if missing:
        _fail(path, f"missing required fields: {missing}")
    unknown = set(obj) - set(required) - set(optional)
    if unknown:
        _fail(path, f"unknown fields (misspelled?): {sorted(unknown)}")
    return obj


def _str(path: str, v) -> str:
    if not isinstance(v, str) or not v.strip():
        _fail(path, f"must be a non-empty string; got {v!r}")
    if v.strip().upper() in _PLACEHOLDERS:
        _fail(path, f"is an unresolved placeholder ({v!r}); a required protocol/operator "
                    "decision may not be TBD or UNRESOLVED")
    return v


def _bool(path: str, v) -> bool:
    if not isinstance(v, bool):
        _fail(path, f"must be a boolean; got {v!r}")
    return v


def _int(path: str, v, *, minimum=None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        _fail(path, f"must be an integer (booleans and numeric strings are not integers); "
                    f"got {v!r}")
    if minimum is not None and v < minimum:
        _fail(path, f"must be >= {minimum}; got {v}")
    return v


def _num(path: str, v, *, minimum=None, positive=False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _fail(path, f"must be a finite JSON number (numeric strings are rejected); got {v!r}")
    if not math.isfinite(v):
        _fail(path, f"must be finite; got {v!r}")
    # IEEE -0.0 == 0.0 and passes `< 0`, but canonical JSON serializes it differently,
    # so a signed zero would mint a distinct config/cost/trial identity for the same
    # value; reject it as the pinned-constant (_exact) path already does.
    if isinstance(v, float) and v == 0.0 and math.copysign(1.0, v) < 0:
        _fail(path, "must not be negative zero (-0.0 hashes differently from 0.0)")
    if minimum is not None and v < minimum:
        _fail(path, f"must be >= {minimum}; got {v}")
    if positive and v <= 0:
        _fail(path, f"must be strictly positive; got {v}")
    return v


def _sha256(path: str, v) -> str:
    if not isinstance(v, str) or not _SHA256_RE.match(v):
        _fail(path, f"must be 64 lowercase hexadecimal characters; got {v!r}")
    return v


def _git_hex(path: str, v) -> str:
    if not isinstance(v, str) or not _GIT_HEX_RE.match(v):
        _fail(path, f"must be a 40- or 64-char lowercase hex object id; got {v!r}")
    return v


def _exact(path: str, got, want):
    """Type-strict deep equality: bool is not int, int is not float, no coercion."""
    if want is None:
        if got is not None:
            _fail(path, f"must be JSON null; got {got!r}")
    elif isinstance(want, bool):
        if got is not want:
            _fail(path, f"must equal {want!r} (a JSON boolean); got {got!r}")
    elif isinstance(want, int):
        if isinstance(got, bool) or type(got) is not int or got != want:
            _fail(path, f"must equal {want!r} (a JSON integer); got {got!r}")
    elif isinstance(want, float):
        # IEEE -0.0 == 0.0, but canonical JSON serializes them differently, so a
        # signed-zero mismatch would silently drift every decision-bearing hash.
        if (type(got) is not float or got != want
                or (want == 0.0 and math.copysign(1.0, got) != math.copysign(1.0, want))):
            _fail(path, f"must equal {want!r} (a JSON number); got {got!r}")
    elif isinstance(want, str):
        if type(got) is not str or got != want:
            _fail(path, f"must equal {want!r}; got {got!r}")
    elif isinstance(want, (list, tuple)):
        if not isinstance(got, list):
            _fail(path, f"must be an ordered array; got {got!r}")
        if len(got) != len(want):
            _fail(path, f"must have exactly {len(want)} entries in the pinned order; "
                        f"got {len(got)}")
        for i, (g, w) in enumerate(zip(got, want)):
            _exact(f"{path}[{i}]", g, w)
    elif isinstance(want, dict):
        if not isinstance(got, dict):
            _fail(path, f"must be a dict; got {type(got).__name__}")
        missing = [k for k in want if k not in got]
        if missing:
            _fail(path, f"missing required fields: {missing}")
        unknown = set(got) - set(want)
        if unknown:
            _fail(path, f"unknown fields (misspelled?): {sorted(unknown)}")
        for k, w in want.items():
            _exact(f"{path}.{k}", got[k], w)
    else:  # pragma: no cover - constants above are JSON-typed
        _fail(path, f"unsupported pinned constant {want!r}")
    return got


def _scalar_tree(path: str, v):
    """JSON-safe opaque value (resolved model params, preprocessing declarations):
    finite numbers, booleans, strings, null, and nested dict/list of those."""
    if v is None or isinstance(v, bool) or isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            _fail(path, f"must be finite; got {v!r}")
        # -0.0 hashes differently from 0.0 under canonical JSON, so a signed zero
        # anywhere in an identity-bearing subtree would mint a phantom trial identity
        # for a logically identical value; reject it as _exact/_num already do.
        if v == 0.0 and math.copysign(1.0, v) < 0:
            _fail(path, "must not be negative zero (-0.0 hashes differently from 0.0)")
        return v
    if isinstance(v, str):
        if v.strip().upper() in _PLACEHOLDERS:
            _fail(path, f"is an unresolved placeholder ({v!r})")
        return v
    if isinstance(v, list):
        for i, item in enumerate(v):
            _scalar_tree(f"{path}[{i}]", item)
        return v
    if isinstance(v, dict):
        for k, item in v.items():
            if not isinstance(k, str) or not k:
                _fail(path, f"keys must be non-empty strings; got {k!r}")
            # Physical file hashes are audit-only and must never enter an
            # identity-bearing subtree (spec section 3.1); reject them at any depth
            # so Parquet layout/compression metadata cannot reidentify a trial.
            if k.endswith("_file_sha256"):
                _fail(f"{path}.{k}", "physical file hashes are audit-only and must "
                                     "not appear in an identity-bearing object")
            _scalar_tree(f"{path}.{k}", item)
        return v
    _fail(path, f"must be a JSON-encodable scalar/dict/list; got {type(v).__name__}")


def _day(path: str, v) -> str:
    if not isinstance(v, str) or not _DAY_RE.match(v):
        _fail(path, f"must be a canonical YYYY-MM-DD day string; got {v!r}")
    try:
        parsed = _dt.date.fromisoformat(v)
    except ValueError:
        parsed = None
    if parsed is None or parsed.isoformat() != v:
        _fail(path, f"must be a canonical YYYY-MM-DD day string; got {v!r}")
    return v


def _day_array(path: str, v) -> list:
    if not isinstance(v, list) or not v:
        _fail(path, f"must be a non-empty array of day strings; got {v!r}")
    for i, d in enumerate(v):
        _day(f"{path}[{i}]", d)
    if sorted(set(v)) != v:
        _fail(path, "must be sorted unique canonical YYYY-MM-DD day strings")
    return v


def _dev_day(path: str, v) -> str:
    _day(path, v)
    if not (DEV_DAYS[0] <= v <= DEV_DAYS[-1]):
        _fail(path, f"day {v} is outside the development window "
                    f"[{DEV_DAYS[0]}, {DEV_DAYS[-1]}]")
    return v


def _str_list(path: str, v, *, allow_empty=False) -> list:
    if not isinstance(v, list) or (not v and not allow_empty):
        _fail(path, f"must be a non-empty array of strings; got {v!r}")
    for i, s in enumerate(v):
        _str(f"{path}[{i}]", s)
    return v


# --- canonical hashing (spec section 3.1) ----------------------------------------------

def g0bn_artifact_sha256(obj: dict, self_field: str = "sha256") -> str:
    """Canonical G0-BN artifact hash: SHA-256 of the canonical JSON with exactly the
    artifact's own hash field and a top-level generated_at excluded. All other fields,
    including array order, remain hash-bearing."""
    return hash_obj(obj, exclude_keys=(self_field, "generated_at"))


def protocol_config_sha256(config: dict) -> str:
    return g0bn_artifact_sha256(config)


# --- section validators -----------------------------------------------------------------

def _validate_instrument(path: str, obj):
    _exact(path, obj, INSTRUMENT)


def _validate_source_certification(cert):
    path = "source_certification"
    _dict(path, cert, (
        "provider", "l2_snapshot_product", "l2_delta_product", "trade_product",
        "raw_schema_version", "normalized_schema_version", "timestamp_policy",
        "sequence_policy", "gap_policy", "certification_sha256", "custodian_seal_sha256",
        "coverage_sha256", "permission_policy_sha256", "development_source_manifest_sha256",
        "custodian_identity", "operator_identity",
    ))
    for k in ("provider", "l2_snapshot_product", "l2_delta_product", "trade_product",
              "raw_schema_version", "normalized_schema_version", "timestamp_policy",
              "sequence_policy", "gap_policy", "custodian_identity", "operator_identity"):
        _str(f"{path}.{k}", cert[k])
    # Pin the exact certified #64 provider and all three Binance products (spec §2.1: the
    # L2 snapshot/seed + L2 delta + trades). A self-consistent certification cannot
    # certify a CoinAPI/Coinbase provider fallback or a spot/other-asset product, leave
    # the reconstruction seed product unpinned, or collapse the distinct feeds.
    _exact(f"{path}.provider", cert["provider"], CERTIFIED_PROVIDER)
    _exact(f"{path}.l2_snapshot_product", cert["l2_snapshot_product"],
           CERTIFIED_L2_SNAPSHOT_PRODUCT)
    _exact(f"{path}.l2_delta_product", cert["l2_delta_product"], CERTIFIED_L2_DELTA_PRODUCT)
    _exact(f"{path}.trade_product", cert["trade_product"], CERTIFIED_TRADE_PRODUCT)
    for k in ("certification_sha256", "custodian_seal_sha256", "coverage_sha256",
              "permission_policy_sha256", "development_source_manifest_sha256"):
        _sha256(f"{path}.{k}", cert[k])
    if cert["custodian_identity"] == cert["operator_identity"]:
        _fail(path, "custodian_identity and operator_identity must be distinct "
                    "(separation of custody; spec section 5.1)")


def _validate_producer(prod):
    path = "producer"
    _dict(path, prod, (
        "entry_point", "repository_commit", "repository_tree", "transform_versions",
        "received_time_observability_rule", "staleness_cap_ns", "lookback_cap_ns",
        "partition_rule_version", "logical_row_hash_algorithm", "build_id_algorithm",
        "physical_schema_hash_algorithm",
    ))
    _str(f"{path}.entry_point", prod["entry_point"])
    _git_hex(f"{path}.repository_commit", prod["repository_commit"])
    _git_hex(f"{path}.repository_tree", prod["repository_tree"])
    _str_list(f"{path}.transform_versions", prod["transform_versions"])
    _str(f"{path}.received_time_observability_rule", prod["received_time_observability_rule"])
    _int(f"{path}.staleness_cap_ns", prod["staleness_cap_ns"], minimum=1)
    _int(f"{path}.lookback_cap_ns", prod["lookback_cap_ns"], minimum=1)
    for k in ("partition_rule_version", "logical_row_hash_algorithm",
              "build_id_algorithm", "physical_schema_hash_algorithm"):
        _str(f"{path}.{k}", prod[k])
    if len(prod["repository_commit"]) != len(prod["repository_tree"]):
        _fail(path, "repository_commit and repository_tree must use one declared "
                    "object format (both 40-hex or both 64-hex)")


def _validate_clock(clock, source_certification):
    path = "clock"
    _dict(path, clock, (
        "kind", "reference_stream", "development_schedule_sha256", "target_bars_per_day",
        "time_cap_ns", "warmup_bars", "coverage_normalization", "monotone_watermark",
        "adaptive_threshold_update_rule", "development_end_state_sha256",
    ))
    _exact(f"{path}.kind", clock["kind"], "dollar")
    _str(f"{path}.reference_stream", clock["reference_stream"])
    # Spec 2.1/3.2: the single certified Binance-perpetual trade product drives the
    # bar clock; any other reference stream is a source-isolation violation.
    if clock["reference_stream"] != source_certification.get("trade_product"):
        _fail(f"{path}.reference_stream",
              f"must equal source_certification.trade_product "
              f"({source_certification.get('trade_product')!r}); "
              f"got {clock['reference_stream']!r}")
    _sha256(f"{path}.development_schedule_sha256", clock["development_schedule_sha256"])
    _int(f"{path}.target_bars_per_day", clock["target_bars_per_day"], minimum=1)
    _int(f"{path}.time_cap_ns", clock["time_cap_ns"], minimum=1)
    _int(f"{path}.warmup_bars", clock["warmup_bars"], minimum=0)
    _str(f"{path}.coverage_normalization", clock["coverage_normalization"])
    _exact(f"{path}.monotone_watermark", clock["monotone_watermark"], True)
    _str(f"{path}.adaptive_threshold_update_rule", clock["adaptive_threshold_update_rule"])
    _sha256(f"{path}.development_end_state_sha256", clock["development_end_state_sha256"])


def _validate_features(features):
    path = "features"
    _dict(path, features, ("registry", "causal_update_rule", "max_lookback_ns"))
    registry = features["registry"]
    if not isinstance(registry, list) or len(registry) != len(FEATURE_REGISTRY):
        _fail(f"{path}.registry",
              f"must have exactly {len(FEATURE_REGISTRY)} entries in the pinned "
              f"section-3.3 order; got {len(registry) if isinstance(registry, list) else registry!r}")
    for i, entry in enumerate(registry):
        epath = f"{path}.registry[{i}]"
        _dict(epath, entry, ("name", "formula_version",
                             "development_end_normalizer_state_sha256"))
        _exact(f"{epath}.name", entry["name"], FEATURE_REGISTRY[i])
        _str(f"{epath}.formula_version", entry["formula_version"])
        _sha256(f"{epath}.development_end_normalizer_state_sha256",
                entry["development_end_normalizer_state_sha256"])
    _str(f"{path}.causal_update_rule", features["causal_update_rule"])
    _int(f"{path}.max_lookback_ns", features["max_lookback_ns"], minimum=1)


def _validate_labels(labels):
    path = "labels"
    _dict(path, labels, (
        "mid_anchor", "return_formula", "barrier_estimator", "ewma_half_life_ns",
        "tp_multiplier", "sl_multiplier", "unresolved_barrier_policy", "uniqueness_policy",
    ))
    _exact(f"{path}.mid_anchor", labels["mid_anchor"], "true_t_event_mid")
    _exact(f"{path}.return_formula", labels["return_formula"], LABEL_RETURN_FORMULA)
    _exact(f"{path}.barrier_estimator", labels["barrier_estimator"],
           LABEL_BARRIER_ESTIMATOR)
    _int(f"{path}.ewma_half_life_ns", labels["ewma_half_life_ns"], minimum=1)
    _num(f"{path}.tp_multiplier", labels["tp_multiplier"], positive=True)
    _num(f"{path}.sl_multiplier", labels["sl_multiplier"], positive=True)
    _exact(f"{path}.unresolved_barrier_policy", labels["unresolved_barrier_policy"],
           LABEL_UNRESOLVED_BARRIER_POLICY)
    _exact(f"{path}.uniqueness_policy", labels["uniqueness_policy"],
           LABEL_UNIQUENESS_POLICY)


def _validate_costs(costs):
    path = "costs"
    _dict(path, costs, (
        "cost_assumption", "fee_tier", "slippage_evidence_sha256", "fee_sides",
        "spread_crossings", "no_trade_margin_bps", "decision_cost_rule",
        "cost_column_dtype", "reconciliation_rel_tol", "reconciliation_abs_tol",
    ))
    ca = costs["cost_assumption"]
    capath = f"{path}.cost_assumption"
    _dict(capath, ca, ("venue", "product", "source", "version", "taker_fee_bps",
                       "base_slippage_bps", "drift_policy"))
    _exact(f"{capath}.venue", ca["venue"], VENUE_BINANCE)
    _exact(f"{capath}.product", ca["product"], INSTRUMENT["symbol"])
    _str(f"{capath}.source", ca["source"])
    _str(f"{capath}.version", ca["version"])
    _num(f"{capath}.taker_fee_bps", ca["taker_fee_bps"], minimum=0)
    _num(f"{capath}.base_slippage_bps", ca["base_slippage_bps"], minimum=0)
    _exact(f"{capath}.drift_policy", ca["drift_policy"], DRIFT_POLICY)
    validate_cost_assumption(CostAssumption(**ca))  # T7 identity contract (pure reuse)

    tier = costs["fee_tier"]
    tpath = f"{path}.fee_tier"
    _dict(tpath, tier, ("tier", "applicability_start_ns", "applicability_end_ns",
                        "evidence_sha256"))
    _str(f"{tpath}.tier", tier["tier"])
    start = _int(f"{tpath}.applicability_start_ns", tier["applicability_start_ns"])
    end = _int(f"{tpath}.applicability_end_ns", tier["applicability_end_ns"])
    _sha256(f"{tpath}.evidence_sha256", tier["evidence_sha256"])
    if not (start <= DEV_START_NS and end >= HOLDOUT_END_NS):
        _fail(tpath, "applicability interval must cover both the development and "
                     "holdout support spans")

    _sha256(f"{path}.slippage_evidence_sha256", costs["slippage_evidence_sha256"])
    _exact(f"{path}.fee_sides", costs["fee_sides"], 2)
    _exact(f"{path}.spread_crossings", costs["spread_crossings"], 2)
    _num(f"{path}.no_trade_margin_bps", costs["no_trade_margin_bps"], minimum=0)
    _exact(f"{path}.decision_cost_rule", costs["decision_cost_rule"],
           "decision_observable_realized_charged_v1")
    _exact(f"{path}.cost_column_dtype", costs["cost_column_dtype"], "float64")
    _exact(f"{path}.reconciliation_rel_tol", costs["reconciliation_rel_tol"], 1e-12)
    _exact(f"{path}.reconciliation_abs_tol", costs["reconciliation_abs_tol"], 1e-12)


def _validate_exclusions(exc):
    path = "exclusions"
    _dict(path, exc, (
        "rule_version", "included_days", "excluded_days", "staleness_rule", "gap_rule",
        "one_sided_book_rule", "lookback_rule", "drop_policy",
    ))
    _str(f"{path}.rule_version", exc["rule_version"])
    included = _day_array(f"{path}.included_days", exc["included_days"])
    for i, d in enumerate(included):
        _dev_day(f"{path}.included_days[{i}]", d)
    excluded = exc["excluded_days"]
    if not isinstance(excluded, dict):
        _fail(f"{path}.excluded_days", "must be a map of day -> {reason, evidence_sha256}")
    for d, entry in excluded.items():
        epath = f"{path}.excluded_days[{d!r}]"
        _dev_day(epath, d)
        _dict(epath, entry, ("reason", "evidence_sha256"))
        _str(f"{epath}.reason", entry["reason"])
        _sha256(f"{epath}.evidence_sha256", entry["evidence_sha256"])
    both = sorted(set(included) & set(excluded))
    if both:
        _fail(path, f"days in both included_days and excluded_days: {both}")
    unaccounted = sorted(set(DEV_DAYS) - set(included) - set(excluded))
    if unaccounted:
        _fail(path, f"development days in neither included_days nor excluded_days "
                    f"(the outcome-blind accounting must cover the full window): {unaccounted}")
    for k in ("staleness_rule", "gap_rule", "one_sided_book_rule", "lookback_rule",
              "drop_policy"):
        _str(f"{path}.{k}", exc[k])


def _validate_partition(part, horizons):
    path = "partition"
    _dict(path, part, (
        "schema", "development_start_ns", "development_end_ns", "holdout_start_ns",
        "holdout_end_ns", "horizons", "partition_guard_ns", "prefilter_rule",
        "development_drop_counts", "holdout_drop_count_categories",
        "sufficiency_thresholds", "sha256",
    ))
    _exact(f"{path}.schema", part["schema"], PARTITION_PLAN_SCHEMA)
    _exact(f"{path}.development_start_ns", part["development_start_ns"], DEV_START_NS)
    _exact(f"{path}.development_end_ns", part["development_end_ns"], DEV_END_NS)
    _exact(f"{path}.holdout_start_ns", part["holdout_start_ns"], HOLDOUT_START_NS)
    _exact(f"{path}.holdout_end_ns", part["holdout_end_ns"], HOLDOUT_END_NS)
    _exact(f"{path}.horizons", part["horizons"],
           {h["tag"]: h["ns"] for h in horizons})
    _int(f"{path}.partition_guard_ns", part["partition_guard_ns"], minimum=0)
    _exact(f"{path}.prefilter_rule", part["prefilter_rule"], PREFILTER_RULE)
    counts = part["development_drop_counts"]
    cpath = f"{path}.development_drop_counts"
    _dict(cpath, counts, tuple(h["tag"] for h in horizons))
    for tag, n in counts.items():
        _int(f"{cpath}[{tag!r}]", n, minimum=0)
    _str_list(f"{path}.holdout_drop_count_categories", part["holdout_drop_count_categories"])
    # The plan freezes the holdout COUNT SCHEMA and sufficiency rules only; realized
    # January counts must not exist before the raw-access burn (spec section 2.2), so
    # any holdout_drop_counts-style field is rejected by the exact-field-set check above.
    _exact(f"{path}.sufficiency_thresholds", part["sufficiency_thresholds"],
           {"min_valid_days": 20, "min_uniqueness_sum": 100})
    embedded = _sha256(f"{path}.sha256", part["sha256"])
    if hash_obj(part, exclude_keys=("sha256",)) != embedded:
        _fail(f"{path}.sha256", "embedded partition-plan sha256 does not match the "
                                "plan content (tampered or stale)")


def _validate_cv(cv, features):
    path = "cv"
    _dict(path, cv, (
        "n_groups", "k", "split_version", "grouping_version", "forecast_collapse_version",
        "combination_order", "forecast_dtype", "expected_test_multiplicity", "embargo_ns",
        "embargo_rule", "dsr", "pbo",
    ))
    _exact(f"{path}.n_groups", cv["n_groups"], 6)
    _exact(f"{path}.k", cv["k"], 2)
    _str(f"{path}.split_version", cv["split_version"])
    _str(f"{path}.grouping_version", cv["grouping_version"])
    _exact(f"{path}.forecast_collapse_version", cv["forecast_collapse_version"],
           FORECAST_COLLAPSE_VERSION)
    _exact(f"{path}.combination_order", cv["combination_order"],
           "lexicographic_itertools_combinations_range_6_2_v1")
    _exact(f"{path}.forecast_dtype", cv["forecast_dtype"], "float64")
    _exact(f"{path}.expected_test_multiplicity", cv["expected_test_multiplicity"], 5)
    _int(f"{path}.embargo_ns", cv["embargo_ns"], minimum=1)
    _exact(f"{path}.embargo_rule", cv["embargo_rule"], "embargo_equals_max_lookback_v1")
    if cv["embargo_ns"] != features["max_lookback_ns"]:
        _fail(f"{path}.embargo_ns",
              f"must equal features.max_lookback_ns exactly (spec section 9; "
              f"got {cv['embargo_ns']} vs {features['max_lookback_ns']})")
    dsr = cv["dsr"]
    dpath = f"{path}.dsr"
    _dict(dpath, dsr, ("epsilon", "rounding_rule", "n_trials_source", "threshold",
                       "code_sha256"))
    _exact(f"{dpath}.epsilon", dsr["epsilon"], 1e-9)
    _exact(f"{dpath}.rounding_rule", dsr["rounding_rule"], DSR_ROUNDING_RULE)
    _exact(f"{dpath}.n_trials_source", dsr["n_trials_source"],
           "g0bn_ledger_unique_identity_count_v1")
    _exact(f"{dpath}.threshold", dsr["threshold"], 0.95)
    _sha256(f"{dpath}.code_sha256", dsr["code_sha256"])
    pbo = cv["pbo"]
    ppath = f"{path}.pbo"
    _dict(ppath, pbo, ("n_blocks", "block_split", "is_tie_rule", "oos_rank_rule",
                       "column_order", "min_columns", "min_rows", "threshold",
                       "code_sha256"))
    _exact(f"{ppath}.n_blocks", pbo["n_blocks"], 8)
    _exact(f"{ppath}.block_split", pbo["block_split"], "numpy_array_split_contiguous_v1")
    _exact(f"{ppath}.is_tie_rule", pbo["is_tie_rule"], PBO_IS_TIE_RULE)
    _exact(f"{ppath}.oos_rank_rule", pbo["oos_rank_rule"], PBO_OOS_RANK_RULE)
    _exact(f"{ppath}.column_order", pbo["column_order"],
           "base_ladder_then_ascending_trial_id_v1")
    _exact(f"{ppath}.min_columns", pbo["min_columns"], 2)
    _exact(f"{ppath}.min_rows", pbo["min_rows"], 32)
    _exact(f"{ppath}.threshold", pbo["threshold"], 0.5)
    _sha256(f"{ppath}.code_sha256", pbo["code_sha256"])


def _validate_horizons(horizons):
    path = "horizons"
    if not isinstance(horizons, list) or len(horizons) != len(HORIZONS):
        _fail(path, f"must be exactly the {len(HORIZONS)} ordered section-2.3 horizon "
                    f"objects; got {horizons!r}")
    for i, h in enumerate(horizons):
        _exact(f"{path}[{i}]", h, HORIZONS[i])


_NON_FITTED_FIELDS = (
    "candidate_id", "fitted", "software_version", "prediction_transform", "preprocessing",
    "preprocessing_sha256", "feature_cols", "feature_formula_sha256s",
    "candidate_code_sha256", "model_params", "model_params_sha256",
)
_FITTED_FIELDS = (
    "candidate_id", "fitted", "package", "package_version", "estimator_class", "target",
    "loss", "sample_weight_semantics", "seed_and_thread_settings", "prediction_transform",
    "preprocessing", "preprocessing_sha256", "feature_cols", "feature_formula_sha256s",
    "candidate_code_sha256", "model_params", "model_params_sha256",
)

# Per-candidate pinned values (spec section 4.1). Fitted candidates must contain the
# fixed overrides exactly; additional keys are the installed pinned library's resolved
# defaults, hash-bearing via model_params_sha256. Non-fitted params are exact objects.
_CANDIDATE_PINS = {
    "persistence_zero": {
        "fitted": False,
        "prediction_transform": "constant_zero_bps_v1",
        "preprocessing": NONFITTED_PREPROCESSING,
        "feature_cols": [],
        "model_params_exact": PERSISTENCE_MODEL_PARAMS,
    },
    "microprice_raw": {
        "fitted": False,
        "prediction_transform": "multiplier_times_input_bps_v1",
        "preprocessing": NONFITTED_PREPROCESSING,
        "feature_cols": ["microprice_dev"],
        "model_params_exact": MICROPRICE_MODEL_PARAMS,
    },
    "ofi_ridge": {
        "fitted": True,
        "package": "scikit-learn",
        "estimator_class": "sklearn.linear_model.Ridge",
        "target": "y_fwd_bps",
        "prediction_transform": "identity_bps_v1",
        "preprocessing": FITTED_PREPROCESSING,
        "seed_and_thread_settings": {"random_state": None},
        "feature_cols": ["ofi_integrated"],
        "fixed_params": RIDGE_FIXED_PARAMS,
    },
    "lgbm_reg": {
        "fitted": True,
        "package": "lightgbm",
        "estimator_class": "lightgbm.LGBMRegressor",
        "target": "y_fwd_bps",
        "prediction_transform": "identity_bps_v1",
        "preprocessing": FITTED_PREPROCESSING,
        "seed_and_thread_settings": {"random_state": 0, "n_jobs": 1,
                                     "deterministic": True, "force_col_wise": True},
        "feature_cols": list(FEATURE_REGISTRY),
        "fixed_params": LGBM_REG_FIXED_PARAMS,
    },
    "lgbm_clf": {
        "fitted": True,
        "package": "lightgbm",
        "estimator_class": "lightgbm.LGBMClassifier",
        "target": "label",
        "prediction_transform": "class_prob_spread_times_training_y_std_bps_v1",
        "preprocessing": FITTED_PREPROCESSING,
        "seed_and_thread_settings": {"random_state": 0, "n_jobs": 1,
                                     "deterministic": True, "force_col_wise": True},
        "feature_cols": list(FEATURE_REGISTRY),
        "fixed_params": LGBM_CLF_FIXED_PARAMS,
    },
}


def _validate_candidate(path, defn, candidate_id):
    pins = _CANDIDATE_PINS[candidate_id]
    fitted = pins["fitted"]
    extra = ("training_y_std_rule", "class_order") if candidate_id == "lgbm_clf" else ()
    _dict(path, defn, (_FITTED_FIELDS if fitted else _NON_FITTED_FIELDS) + extra)
    _exact(f"{path}.candidate_id", defn["candidate_id"], candidate_id)
    _exact(f"{path}.fitted", defn["fitted"], fitted)
    _exact(f"{path}.prediction_transform", defn["prediction_transform"],
           pins["prediction_transform"])
    _exact(f"{path}.feature_cols", defn["feature_cols"], pins["feature_cols"])

    preprocessing = defn["preprocessing"]
    _exact(f"{path}.preprocessing", preprocessing, pins["preprocessing"])
    _sha256(f"{path}.preprocessing_sha256", defn["preprocessing_sha256"])
    if hash_obj(preprocessing) != defn["preprocessing_sha256"]:
        _fail(f"{path}.preprocessing_sha256", "does not match the preprocessing object")

    formulas = defn["feature_formula_sha256s"]
    fpath = f"{path}.feature_formula_sha256s"
    if not isinstance(formulas, dict) or set(formulas) != set(defn["feature_cols"]):
        _fail(fpath, "must map exactly the candidate's ordered inputs to formula hashes")
    for col, sha in formulas.items():
        _sha256(f"{fpath}[{col!r}]", sha)
    _sha256(f"{path}.candidate_code_sha256", defn["candidate_code_sha256"])

    params = defn["model_params"]
    mpath = f"{path}.model_params"
    if not isinstance(params, dict):
        _fail(mpath, "must be the complete resolved parameter object")
    if fitted:
        for k, want in pins["fixed_params"].items():
            if k not in params:
                _fail(mpath, f"missing pinned parameter {k!r}")
            _exact(f"{mpath}.{k}", params[k], want)
        _scalar_tree(mpath, params)
        _exact(f"{path}.package", defn["package"], pins["package"])
        _str(f"{path}.package_version", defn["package_version"])
        _exact(f"{path}.estimator_class", defn["estimator_class"], pins["estimator_class"])
        _exact(f"{path}.target", defn["target"], pins["target"])
        _str(f"{path}.loss", defn["loss"])
        _exact(f"{path}.sample_weight_semantics", defn["sample_weight_semantics"],
               "uniqueness_weight_fit_v1")
        _exact(f"{path}.seed_and_thread_settings", defn["seed_and_thread_settings"],
               pins["seed_and_thread_settings"])
    else:
        _exact(mpath, params, pins["model_params_exact"])
        _str(f"{path}.software_version", defn["software_version"])
    _sha256(f"{path}.model_params_sha256", defn["model_params_sha256"])
    if hash_obj(params) != defn["model_params_sha256"]:
        _fail(f"{path}.model_params_sha256", "does not match the model_params object")

    if candidate_id == "lgbm_clf":
        _exact(f"{path}.training_y_std_rule", defn["training_y_std_rule"],
               CLASSIFIER_SCALE_RULE)
        _exact(f"{path}.class_order", defn["class_order"], CLASS_ORDER)


def _validate_candidates(candidates):
    path = "candidates"
    if not isinstance(candidates, list) or len(candidates) != len(CANDIDATE_IDS):
        _fail(path, f"must be exactly the {len(CANDIDATE_IDS)} ordered section-4.1 "
                    f"candidate definitions")
    for i, defn in enumerate(candidates):
        cpath = f"{path}[{i}]"
        if not isinstance(defn, dict) or defn.get("candidate_id") != CANDIDATE_IDS[i]:
            _fail(f"{cpath}.candidate_id",
                  f"must equal {CANDIDATE_IDS[i]!r} (the pinned ladder order); "
                  f"got {defn.get('candidate_id') if isinstance(defn, dict) else defn!r}")
        _validate_candidate(cpath, defn, CANDIDATE_IDS[i])


def _validate_selection(sel):
    path = "selection"
    _dict(path, sel, ("ranking_rule", "tie_rule", "eligible_candidate_ids",
                      "attempt_accounting_policy"))
    _exact(f"{path}.ranking_rule", sel["ranking_rule"], SELECTION_RANKING_RULE)
    _exact(f"{path}.tie_rule", sel["tie_rule"], SELECTION_TIE_RULE)
    _exact(f"{path}.eligible_candidate_ids", sel["eligible_candidate_ids"],
           list(SELECTABLE_CANDIDATE_IDS))
    _exact(f"{path}.attempt_accounting_policy", sel["attempt_accounting_policy"],
           SELECTION_ATTEMPT_ACCOUNTING)


def _validate_verdict_thresholds(vt):
    path = "verdict_thresholds"
    _dict(path, vt, (
        "min_valid_days", "min_uniqueness_sum", "min_trades", "min_effective_trades",
        "dsr_threshold", "pbo_threshold", "bootstrap", "alpha_dev", "alpha_oos",
        "lift_rule", "net_rule", "truth_table_version",
    ))
    _exact(f"{path}.min_valid_days", vt["min_valid_days"], 20)
    _exact(f"{path}.min_uniqueness_sum", vt["min_uniqueness_sum"], 100)
    _exact(f"{path}.min_trades", vt["min_trades"], 30)
    _exact(f"{path}.min_effective_trades", vt["min_effective_trades"], 10)
    _exact(f"{path}.dsr_threshold", vt["dsr_threshold"], 0.95)
    _exact(f"{path}.pbo_threshold", vt["pbo_threshold"], 0.5)
    _exact(f"{path}.bootstrap", vt["bootstrap"], {
        "kind": BOOTSTRAP_KIND,
        "block_length_days": 2,
        "n_boot": 10000,
        "seed": 0,
        "bit_generator": "PCG64",
        "percentile_method": "linear",
        "draw_hash_schema": BOOTSTRAP_DRAW_SCHEMA,
    })
    _exact(f"{path}.alpha_dev", vt["alpha_dev"], 0.00625)
    _exact(f"{path}.alpha_oos", vt["alpha_oos"], 0.025)
    _exact(f"{path}.lift_rule", vt["lift_rule"],
           "one_sided_lower_bound_strictly_positive_v1")
    _exact(f"{path}.net_rule", vt["net_rule"],
           "one_sided_lower_bound_strictly_positive_v1")
    _exact(f"{path}.truth_table_version", vt["truth_table_version"], VERDICT_SCHEMA)


def _validate_reporting(rep):
    path = "reporting"
    _dict(path, rep, ("metrics_version", "report_schema", "verdict_schema",
                      "spread_regime", "volatility_regime", "component_cost_fields"))
    _str(f"{path}.metrics_version", rep["metrics_version"])
    _exact(f"{path}.report_schema", rep["report_schema"], REPORT_SCHEMA)
    _exact(f"{path}.verdict_schema", rep["verdict_schema"], VERDICT_SCHEMA)
    spread = rep["spread_regime"]
    _dict(f"{path}.spread_regime", spread, ("rule", "boundary_spread_tick"))
    _str(f"{path}.spread_regime.rule", spread["rule"])
    _num(f"{path}.spread_regime.boundary_spread_tick", spread["boundary_spread_tick"],
         positive=True)
    vol = rep["volatility_regime"]
    _dict(f"{path}.volatility_regime", vol, ("statistic", "bin_edges"))
    _str(f"{path}.volatility_regime.statistic", vol["statistic"])
    edges = vol["bin_edges"]
    if not isinstance(edges, list) or not edges:
        _fail(f"{path}.volatility_regime.bin_edges", "must be a non-empty ordered array")
    for i, e in enumerate(edges):
        _num(f"{path}.volatility_regime.bin_edges[{i}]", e)
    if any(b <= a for a, b in zip(edges, edges[1:])):
        _fail(f"{path}.volatility_regime.bin_edges", "must be strictly increasing")
    _exact(f"{path}.component_cost_fields", rep["component_cost_fields"],
           list(COMPONENT_COST_FIELDS))


def _validate_oos(oos):
    path = "oos"
    _dict(path, oos, (
        "dataset_id", "holdout_start_ns", "holdout_end_ns", "holdout_universe_schema",
        "transaction_schema", "lock", "raw_access_claim_schema",
        "matrix_access_claim_schema", "consumption_schema", "attestation_schema",
        "raw_access_boundary", "matrix_access_boundary", "materializer_version",
        "validator_version", "scorer_version", "report_version", "output_allowlist",
        "terminal_failure_policy",
    ))
    _exact(f"{path}.dataset_id", oos["dataset_id"], OOS_DATASET_ID)
    _exact(f"{path}.holdout_start_ns", oos["holdout_start_ns"], HOLDOUT_START_NS)
    _exact(f"{path}.holdout_end_ns", oos["holdout_end_ns"], HOLDOUT_END_NS)
    _exact(f"{path}.holdout_universe_schema", oos["holdout_universe_schema"],
           UNIVERSE_SCHEMA)
    _exact(f"{path}.transaction_schema", oos["transaction_schema"], ONE_SHOT_SCHEMA)
    lock = oos["lock"]
    lpath = f"{path}.lock"
    _dict(lpath, lock, ("path_template", "algorithm", "lifetime", "contention_result"))
    _str(f"{lpath}.path_template", lock["path_template"])
    _exact(f"{lpath}.algorithm", lock["algorithm"], "fcntl_flock_LOCK_EX_LOCK_NB_v1")
    _exact(f"{lpath}.lifetime", lock["lifetime"], "held_until_terminal_journal_fsync_v1")
    _exact(f"{lpath}.contention_result", lock["contention_result"], CONTENTION_RESULT)
    _exact(f"{path}.raw_access_claim_schema", oos["raw_access_claim_schema"],
           RAW_ACCESS_CLAIM_SCHEMA)
    _exact(f"{path}.matrix_access_claim_schema", oos["matrix_access_claim_schema"],
           MATRIX_ACCESS_CLAIM_SCHEMA)
    _exact(f"{path}.consumption_schema", oos["consumption_schema"], CONSUMPTION_SCHEMA)
    _exact(f"{path}.attestation_schema", oos["attestation_schema"], ATTESTATION_SCHEMA)
    _exact(f"{path}.raw_access_boundary", oos["raw_access_boundary"],
           "before_first_january_source_or_footer_read_v1")
    _exact(f"{path}.matrix_access_boundary", oos["matrix_access_boundary"],
           "after_attestation_before_first_derived_matrix_read_v1")
    for k in ("materializer_version", "validator_version", "scorer_version",
              "report_version"):
        _str(f"{path}.{k}", oos[k])
    _str_list(f"{path}.output_allowlist", oos["output_allowlist"])
    _exact(f"{path}.terminal_failure_policy", oos["terminal_failure_policy"],
           "inconclusive_after_raw_burn_v1")


def _validate_software(software, producer):
    path = "software"
    _dict(path, software, (
        "python_version", "numpy_version", "pandas_version", "scikit_learn_version",
        "lightgbm_version", "pyarrow_version", "repository_commit",
        "deterministic_settings",
    ))
    for k in ("python_version", "numpy_version", "pandas_version", "scikit_learn_version",
              "lightgbm_version", "pyarrow_version"):
        _str(f"{path}.{k}", software[k])
    _git_hex(f"{path}.repository_commit", software["repository_commit"])
    if software["repository_commit"] != producer["repository_commit"]:
        _fail(f"{path}.repository_commit",
              "must equal producer.repository_commit (one pinned build)")
    det = software["deterministic_settings"]
    dpath = f"{path}.deterministic_settings"
    _dict(dpath, det, ("random_seed", "n_threads"))
    _int(f"{dpath}.random_seed", det["random_seed"], minimum=0)
    _int(f"{dpath}.n_threads", det["n_threads"], minimum=1)


def _validate_generated_at(v):
    dt = None
    if isinstance(v, str):
        try:
            dt = _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            dt = None
    if dt is None or dt.tzinfo is None:
        _fail("generated_at", f"must be an ISO-8601 timestamp with an explicit timezone; "
                              f"got {v!r}")


_TOP_LEVEL_FIELDS = (
    "schema", "protocol_id", "pilot_id", "instrument", "source_certification", "producer",
    "clock", "features", "labels", "costs", "exclusions", "partition", "cv", "horizons",
    "candidates", "selection", "verdict_thresholds", "reporting", "oos", "software",
    "generated_at", "sha256",
)


def validate_protocol_config(config: dict) -> dict:
    """Strict, fail-closed validation of a canonical g0bn-protocol-config-v1 object.

    Field/type/cross-checks run first so a targeted error names the offending field;
    the embedded self-hash is verified last (tamper detection). Returns the config
    unchanged for chaining."""
    _dict("g0bn protocol config", config, _TOP_LEVEL_FIELDS)
    _exact("schema", config["schema"], CONFIG_SCHEMA)
    _exact("protocol_id", config["protocol_id"], PROTOCOL_ID)
    _exact("pilot_id", config["pilot_id"], PILOT_ID)
    _validate_instrument("instrument", config["instrument"])
    _validate_source_certification(config["source_certification"])
    _validate_producer(config["producer"])
    _validate_clock(config["clock"], config["source_certification"])
    _validate_features(config["features"])
    _validate_labels(config["labels"])
    _validate_costs(config["costs"])
    _validate_exclusions(config["exclusions"])
    _validate_horizons(config["horizons"])
    _validate_partition(config["partition"], config["horizons"])
    _validate_cv(config["cv"], config["features"])
    _validate_candidates(config["candidates"])
    _validate_selection(config["selection"])
    _validate_verdict_thresholds(config["verdict_thresholds"])
    _validate_reporting(config["reporting"])
    _validate_oos(config["oos"])
    _validate_software(config["software"], config["producer"])
    _validate_generated_at(config["generated_at"])
    embedded = _sha256("sha256", config["sha256"])
    recomputed = protocol_config_sha256(config)
    if embedded != recomputed:
        _fail("sha256", f"embedded config sha256 does not match the canonical content "
                        f"(tampered or stale): {embedded} != {recomputed}")
    return config
