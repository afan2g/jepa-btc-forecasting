"""Synthetic G0-BN protocol-config / identity fixtures (issue #86, slice 67-A).

Everything here is spec-literal: the instrument, horizons, feature registry,
candidate ladder, and fixed model parameters are hard-coded from
docs/superpowers/specs/2026-07-13-g0bn-protocol.md (NOT imported from the
modules under test) so the tests fail if the implementation constants drift
from the binding contract. Hashes that the protocol treats as operator/evidence
inputs are deterministic synthetic 64-hex digests; no vendor data, no January
values, no real market data anywhere.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json

from eval.hashing import hash_obj

# --- spec literals (2026-07-13-g0bn-protocol.md) -------------------------------------

CONFIG_SCHEMA = "g0bn-protocol-config-v1"
PROTOCOL_ID = "g0bn-v1"
PILOT_ID = "g0bn-2025-11_2026-01-v1"
PARTITION_PLAN_SCHEMA = "g0bn-partition-plan-v1"
TRIAL_SCHEMA = "g0bn-trial-v1"
UNIVERSE_SCHEMA = "g0bn-holdout-universe-v1"
ONE_SHOT_SCHEMA = "g0bn-one-shot-v1"

DEV_DATASET_ID = "binance_single_venue_g0bn_dev"
OOS_DATASET_ID = "binance_single_venue_g0bn_oos"

INSTRUMENT = {
    "exchange": "BINANCE_FUTURES",
    "native_symbol": "BTCUSDT",
    "symbol": "BTC-USDT-PERP",
    "contract_type": "linear_perpetual",
    "base_asset": "BTC",
    "quote_asset": "USDT",
    "settlement_asset": "USDT",
}

DEV_START_NS = 1_761_955_200_000_000_000   # 2025-11-01T00:00:00Z
DEV_END_NS = 1_767_225_600_000_000_000     # 2026-01-01T00:00:00Z
HOLDOUT_START_NS = 1_767_225_600_000_000_000
HOLDOUT_END_NS = 1_769_904_000_000_000_000  # 2026-02-01T00:00:00Z

HORIZONS = (
    {"tag": "2s", "ns": 2_000_000_000, "role": "primary"},
    {"tag": "10s", "ns": 10_000_000_000, "role": "primary"},
    {"tag": "60s", "ns": 60_000_000_000, "role": "control-only"},
)
HORIZON_NS = {h["tag"]: h["ns"] for h in HORIZONS}

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

RIDGE_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True,
    "copy_X": True,
    "max_iter": None,
    "tol": 0.0001,
    "solver": "svd",
    "positive": False,
    "random_state": None,
}

LGBM_COMMON_PARAMS = {
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
    "class_weight": None,  # non-overridden get_params default, pinned for completeness
}
LGBM_REG_PARAMS = dict(LGBM_COMMON_PARAMS, objective="regression")
LGBM_CLF_PARAMS = dict(LGBM_COMMON_PARAMS, objective="multiclass", num_class=3)

PERSISTENCE_PARAMS = {"forecast_bps": 0.0}
MICROPRICE_PARAMS = {"input": "microprice_dev", "input_unit": "bps", "multiplier": 1.0}

CLASSIFIER_SCALE_RULE = "unweighted_population_float64_plus_1e-9_v1"
DRIFT_POLICY = "abs_true_over_observable_mid_v1"

# Crypto Lake native vendor product IDs (source_certification.*_product).
L2_SNAPSHOT_PRODUCT = "book"
L2_DELTA_PRODUCT = "book_delta_v2"
TRADE_PRODUCT = "trades"
# Normalized internal producer stream identity (clock.reference_stream) — a different
# layer from the native product ID above; matches the T8 writer's manifest bar_clock.
NORMALIZED_TRADE_STREAM = "binance_futures_trades"

GENERATED_AT = "2026-07-15T00:00:00Z"


def sha_hex(tag: str) -> str:
    """Deterministic synthetic 64-lowercase-hex evidence hash."""
    return hashlib.sha256(f"g0bn-fixture:{tag}".encode()).hexdigest()


def dev_days() -> list:
    """All 61 UTC development days, 2025-11-01 .. 2025-12-31."""
    start = _dt.date(2025, 11, 1)
    return [(start + _dt.timedelta(days=i)).isoformat() for i in range(61)]


def spec_universe_object() -> dict:
    """The exact spec 6.1 identity object, hand-built (independent of eval.g0bn_*)."""
    return {
        "schema": UNIVERSE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "instrument": dict(INSTRUMENT),
        "holdout_start_ns": HOLDOUT_START_NS,
        "holdout_end_ns": HOLDOUT_END_NS,
    }


def spec_universe_id() -> str:
    """Hand-computed universe ID straight from the spec's canonical encoding."""
    canonical = json.dumps(
        spec_universe_object(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def spec_transaction_id() -> str:
    canonical = json.dumps(
        {"schema": ONE_SHOT_SCHEMA, "holdout_universe_id": spec_universe_id()},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# --- candidate definitions ------------------------------------------------------------

def _preprocessing_fitted() -> dict:
    return {"stationarization": "producer_pinned_causal_v1", "candidate_local_scaling": False}


def _candidate(candidate_id: str, **fields) -> dict:
    d = {"candidate_id": candidate_id}
    d.update(fields)
    d["preprocessing_sha256"] = hash_obj(d["preprocessing"])
    d["model_params_sha256"] = hash_obj(d["model_params"])
    d["feature_formula_sha256s"] = {c: sha_hex(f"formula-{c}") for c in d["feature_cols"]}
    d["candidate_code_sha256"] = sha_hex(f"code-{candidate_id}")
    return d


def make_candidates() -> list:
    return [
        _candidate(
            "persistence_zero",
            fitted=False,
            software_version="eval.g0bn/1",
            prediction_transform="constant_zero_bps_v1",
            preprocessing={"none": True},
            feature_cols=[],
            model_params=dict(PERSISTENCE_PARAMS),
        ),
        _candidate(
            "microprice_raw",
            fitted=False,
            software_version="eval.g0bn/1",
            prediction_transform="multiplier_times_input_bps_v1",
            preprocessing={"none": True},
            feature_cols=["microprice_dev"],
            model_params=dict(MICROPRICE_PARAMS),
        ),
        _candidate(
            "ofi_ridge",
            fitted=True,
            package="scikit-learn",
            package_version="1.7.1",
            estimator_class="sklearn.linear_model.Ridge",
            target="y_fwd_bps",
            loss="squared_error",
            sample_weight_semantics="uniqueness_weight_fit_v1",
            seed_and_thread_settings={"random_state": None},
            prediction_transform="identity_bps_v1",
            preprocessing=_preprocessing_fitted(),
            feature_cols=["ofi_integrated"],
            model_params=dict(RIDGE_PARAMS),
        ),
        _candidate(
            "lgbm_reg",
            fitted=True,
            package="lightgbm",
            package_version="4.6.0",
            estimator_class="lightgbm.LGBMRegressor",
            target="y_fwd_bps",
            loss="l2",
            sample_weight_semantics="uniqueness_weight_fit_v1",
            seed_and_thread_settings={
                "random_state": 0, "n_jobs": 1, "deterministic": True, "force_col_wise": True,
            },
            prediction_transform="identity_bps_v1",
            preprocessing=_preprocessing_fitted(),
            feature_cols=list(FEATURE_REGISTRY),
            model_params=dict(LGBM_REG_PARAMS),
        ),
        _candidate(
            "lgbm_clf",
            fitted=True,
            package="lightgbm",
            package_version="4.6.0",
            estimator_class="lightgbm.LGBMClassifier",
            target="label",
            loss="multiclass_logloss",
            sample_weight_semantics="uniqueness_weight_fit_v1",
            seed_and_thread_settings={
                "random_state": 0, "n_jobs": 1, "deterministic": True, "force_col_wise": True,
            },
            prediction_transform="class_prob_spread_times_training_y_std_bps_v1",
            preprocessing=_preprocessing_fitted(),
            feature_cols=list(FEATURE_REGISTRY),
            model_params=dict(LGBM_CLF_PARAMS),
            training_y_std_rule=CLASSIFIER_SCALE_RULE,
            class_order=[-1, 0, 1],
        ),
    ]


# --- protocol-config sections ---------------------------------------------------------

def make_source_certification(**over) -> dict:
    d = {
        "provider": "crypto-lake",
        "l2_snapshot_product": L2_SNAPSHOT_PRODUCT,
        "l2_delta_product": L2_DELTA_PRODUCT,
        "trade_product": TRADE_PRODUCT,
        "raw_schema_version": "lake_raw_v2",
        "normalized_schema_version": "g0bn_normalized_v1",
        "timestamp_policy": "received_time_primary_origin_diag_v1",
        "sequence_policy": "strict_monotone_seq_gap_fail_v1",
        "gap_policy": "day_drop_on_gap_v1",
        "certification_sha256": sha_hex("certification"),
        "custodian_seal_sha256": sha_hex("custodian-seal"),
        "coverage_sha256": sha_hex("coverage"),
        "permission_policy_sha256": sha_hex("permission-policy"),
        "development_source_manifest_sha256": sha_hex("dev-source-manifest"),
        "custodian_identity": "g0bn-custodian-svc",
        "operator_identity": "g0bn-operator-dev",
    }
    d.update(over)
    return d


def make_producer(**over) -> dict:
    d = {
        "entry_point": "bars.pipeline:build_g0bn_matrix",
        "repository_commit": sha_hex("repo-commit")[:40],
        "repository_tree": sha_hex("repo-tree")[:40],
        "transform_versions": ["t1_clock_v1", "t2_snapshot_v1", "t3_features_v1",
                               "t5_labels_v1", "t6_uniqueness_v1", "t7_cost_v1"],
        "received_time_observability_rule": "received_le_read_ts_v1",
        "staleness_cap_ns": 5_000_000_000,
        "lookback_cap_ns": 900_000_000_000,
        "partition_rule_version": "g0bn_partition_rule_v1",
        "logical_row_hash_algorithm": "eval.hashing.matrix_content_hash_v1",
        "build_id_algorithm": "g0bn_logical_build_id_v1",
        "physical_schema_hash_algorithm": "arrow_schema_sha256_v1",
    }
    d.update(over)
    return d


def make_clock(**over) -> dict:
    d = {
        "kind": "dollar",
        # Normalized internal producer stream identity (NOT the native vendor product
        # ID TRADE_PRODUCT="trades") — matches the T8 writer's manifest bar_clock.
        "reference_stream": NORMALIZED_TRADE_STREAM,
        "development_schedule_sha256": sha_hex("dev-schedule"),
        "target_bars_per_day": 5000,
        "time_cap_ns": 60_000_000_000,
        "warmup_bars": 50,
        # The rule identities the T9 producer actually implements and reconciles
        # (bars/produce.py _ADAPTIVE_THRESHOLD_RULE/_COVERAGE_NORMALIZATION_RULE);
        # a foreign spelling here fails _validate_runtime, never a silent re-pin.
        "coverage_normalization": "full_day_coverage_v1",
        "monotone_watermark": True,
        "adaptive_threshold_update_rule": "trailing_window_mean_threshold_v1",
        "development_end_state_sha256": sha_hex("clock-dev-end-state"),
    }
    d.update(over)
    return d


def make_features(**over) -> dict:
    d = {
        "registry": [
            {
                "name": name,
                "formula_version": f"{name}_v1",
                "development_end_normalizer_state_sha256": sha_hex(f"norm-{name}"),
            }
            for name in FEATURE_REGISTRY
        ],
        "causal_update_rule": "causal_asof_normalizer_update_v1",
        "max_lookback_ns": 900_000_000_000,
    }
    d.update(over)
    return d


def make_labels(**over) -> dict:
    d = {
        "mid_anchor": "true_t_event_mid",
        "return_formula": "log_mid_ratio_bps_v1",
        "barrier_estimator": "trailing_ewma_vol_v1",
        "ewma_half_life_ns": 600_000_000_000,
        "tp_multiplier": 1.0,
        "sl_multiplier": 1.0,
        "unresolved_barrier_policy": "vertical_barrier_return_v1",
        "uniqueness_policy": "per_horizon_concurrency_v1",
    }
    d.update(over)
    return d


def make_costs(**over) -> dict:
    d = {
        "cost_assumption": {
            "venue": "binance",
            "product": "BTC-USDT-PERP",
            "source": "binance-futures/normalized_v1",
            "version": "g0bn-cost-v1",
            "taker_fee_bps": 4.5,
            "base_slippage_bps": 0.5,
            "drift_policy": DRIFT_POLICY,
        },
        "fee_tier": {
            "tier": "regular-taker",
            "applicability_start_ns": DEV_START_NS,
            "applicability_end_ns": HOLDOUT_END_NS,
            "evidence_sha256": sha_hex("fee-evidence"),
        },
        "slippage_evidence_sha256": sha_hex("slippage-evidence"),
        "fee_sides": 2,
        "spread_crossings": 2,
        "no_trade_margin_bps": 0.25,
        "decision_cost_rule": "decision_observable_realized_charged_v1",
        "cost_column_dtype": "float64",
        "reconciliation_rel_tol": 1e-12,
        "reconciliation_abs_tol": 1e-12,
    }
    d.update(over)
    return d


def make_exclusions(**over) -> dict:
    days = dev_days()
    excluded = {
        "2025-11-14": {"reason": "source_gap", "evidence_sha256": sha_hex("gap-2025-11-14")},
        "2025-12-25": {"reason": "one_sided_book", "evidence_sha256": sha_hex("book-2025-12-25")},
    }
    d = {
        "rule_version": "outcome_blind_exclusion_v1",
        "included_days": [day for day in days if day not in excluded],
        "excluded_days": excluded,
        "staleness_rule": "staleness_cap_drop_v1",
        "gap_rule": "seq_gap_day_drop_v1",
        "one_sided_book_rule": "one_sided_book_drop_v1",
        "lookback_rule": "lookback_cap_drop_not_clip_v1",
        "drop_policy": "drop_not_clip_v1",
    }
    d.update(over)
    return d


def make_partition(**over) -> dict:
    d = {
        "schema": PARTITION_PLAN_SCHEMA,
        "development_start_ns": DEV_START_NS,
        "development_end_ns": DEV_END_NS,
        "holdout_start_ns": HOLDOUT_START_NS,
        "holdout_end_ns": HOLDOUT_END_NS,
        "horizons": dict(HORIZON_NS),
        "partition_guard_ns": 120_000_000_000,
        "prefilter_rule": "t_event + horizons[horizon] + partition_guard_ns < partition_end_ns",
        "development_drop_counts": {"2s": 180, "10s": 240, "60s": 900},
        # The T9 producer's pinned drop taxonomy (bars/produce.py
        # DROP_COUNT_CATEGORIES), spelled literally so a drift in the
        # implementation fails these fixtures instead of silently re-pinning.
        "holdout_drop_count_categories": [
            "warmup", "day_end_truncation", "book_rejection", "staleness",
            "feature_rejection", "before_start", "lookback_cap", "prefilter",
            "coverage_gap", "label_rejection", "actual_span",
        ],
        "sufficiency_thresholds": {"min_valid_days": 20, "min_uniqueness_sum": 100},
    }
    d.update(over)
    if "sha256" not in over:
        d["sha256"] = hash_obj(d, exclude_keys=("sha256",))
    return d


def make_cv(**over) -> dict:
    d = {
        "n_groups": 6,
        "k": 2,
        "split_version": "cpcv_span_purge_embargo_v1",
        "grouping_version": "contiguous_time_groups_v1",
        "forecast_collapse_version": "mean_repeated_test_forecasts_v1",
        "combination_order": "lexicographic_itertools_combinations_range_6_2_v1",
        "forecast_dtype": "float64",
        "expected_test_multiplicity": 5,
        "embargo_ns": 900_000_000_000,
        "embargo_rule": "embargo_equals_max_lookback_v1",
        "dsr": {
            "epsilon": 1e-9,
            "rounding_rule": "nearest_ties_to_even_int64_v1",
            "n_trials_source": "g0bn_ledger_unique_identity_count_v1",
            "threshold": 0.95,
            "code_sha256": sha_hex("dsr-code"),
        },
        "pbo": {
            "n_blocks": 8,
            "block_split": "numpy_array_split_contiguous_v1",
            "is_tie_rule": "first_max_v1",
            "oos_rank_rule": "less_equal_count_v1",
            "column_order": "base_ladder_then_ascending_trial_id_v1",
            "min_columns": 2,
            "min_rows": 32,
            "threshold": 0.5,
            "code_sha256": sha_hex("pbo-code"),
        },
    }
    d.update(over)
    return d


def make_selection(**over) -> dict:
    d = {
        "ranking_rule": "trade_eligible_net_lift_then_predictive_lift_v1",
        "tie_rule": "earlier_ladder_order_v1",
        "eligible_candidate_ids": ["microprice_raw", "ofi_ridge", "lgbm_reg", "lgbm_clf"],
        "attempt_accounting_policy": "unique_canonical_identity_v1",
    }
    d.update(over)
    return d


def make_verdict_thresholds(**over) -> dict:
    d = {
        "min_valid_days": 20,
        "min_uniqueness_sum": 100,
        "min_trades": 30,
        "min_effective_trades": 10,
        "dsr_threshold": 0.95,
        "pbo_threshold": 0.5,
        "bootstrap": {
            "kind": "paired_utc_day_circular_moving_block",
            "block_length_days": 2,
            "n_boot": 10000,
            "seed": 0,
            "bit_generator": "PCG64",
            "percentile_method": "linear",
            "draw_hash_schema": "g0bn-circular-day-bootstrap-v1",
        },
        "alpha_dev": 0.00625,
        "alpha_oos": 0.025,
        "lift_rule": "one_sided_lower_bound_strictly_positive_v1",
        "net_rule": "one_sided_lower_bound_strictly_positive_v1",
        "truth_table_version": "g0bn-verdict-v1",
    }
    d.update(over)
    return d


def make_reporting(**over) -> dict:
    d = {
        "metrics_version": "g0bn_metrics_v1",
        "report_schema": "g0bn-report-v1",
        "verdict_schema": "g0bn-verdict-v1",
        "spread_regime": {
            "rule": "tight_le_boundary_wide_gt_boundary_v1",
            "boundary_spread_tick": 1.5,
        },
        "volatility_regime": {
            "statistic": "development_rv_intrabar_v1",
            "bin_edges": [0.5, 2.0, 8.0],
        },
        "component_cost_fields": [
            "gross_bps", "fee_bps", "decision_cost_bps", "decision_total_cost_bps",
            "spread_bps", "base_slippage_bps", "latency_drift_bps", "slippage_bps",
            "cost_bps", "realized_total_cost_bps", "net_bps",
        ],
    }
    d.update(over)
    return d


def make_oos(**over) -> dict:
    d = {
        "dataset_id": OOS_DATASET_ID,
        "holdout_start_ns": HOLDOUT_START_NS,
        "holdout_end_ns": HOLDOUT_END_NS,
        "holdout_universe_schema": UNIVERSE_SCHEMA,
        "transaction_schema": ONE_SHOT_SCHEMA,
        "lock": {
            "path_template": "data/g0bn/consumption/{transaction_id}/owner.lock",
            "algorithm": "fcntl_flock_LOCK_EX_LOCK_NB_v1",
            "lifetime": "held_until_terminal_journal_fsync_v1",
            "contention_result": "transaction_already_running",
        },
        "raw_access_claim_schema": "g0bn-raw-access-claim-v1",
        "matrix_access_claim_schema": "g0bn-matrix-access-claim-v1",
        "consumption_schema": "g0bn-consumption-v1",
        "attestation_schema": "g0bn-materialization-attestation-v1",
        "raw_access_boundary": "before_first_january_source_or_footer_read_v1",
        "matrix_access_boundary": "after_attestation_before_first_derived_matrix_read_v1",
        "materializer_version": "g0bn_materializer_v1",
        "validator_version": "g0bn_oos_validator_v1",
        "scorer_version": "g0bn_scorer_v1",
        "report_version": "g0bn_report_v1",
        "output_allowlist": [
            "data/g0bn/oos/matrix.parquet",
            "data/g0bn/oos/manifest.json",
            "data/g0bn/consumption/{transaction_id}/journal.json",
        ],
        "terminal_failure_policy": "inconclusive_after_raw_burn_v1",
    }
    d.update(over)
    return d


def make_software(**over) -> dict:
    d = {
        "python_version": "3.12.3",
        "numpy_version": "2.3.1",
        "pandas_version": "2.3.1",
        "scikit_learn_version": "1.7.1",
        "lightgbm_version": "4.6.0",
        "pyarrow_version": "20.0.0",
        "repository_commit": sha_hex("repo-commit")[:40],
        "deterministic_settings": {"random_seed": 0, "n_threads": 1},
    }
    d.update(over)
    return d


def make_config(sha256=None, **over) -> dict:
    cfg = {
        "schema": CONFIG_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "pilot_id": PILOT_ID,
        "instrument": dict(INSTRUMENT),
        "source_certification": make_source_certification(),
        "producer": make_producer(),
        "clock": make_clock(),
        "features": make_features(),
        "labels": make_labels(),
        "costs": make_costs(),
        "exclusions": make_exclusions(),
        "partition": make_partition(),
        "cv": make_cv(),
        "horizons": [dict(h) for h in HORIZONS],
        "candidates": make_candidates(),
        "selection": make_selection(),
        "verdict_thresholds": make_verdict_thresholds(),
        "reporting": make_reporting(),
        "oos": make_oos(),
        "software": make_software(),
        "generated_at": GENERATED_AT,
    }
    cfg.update(over)
    cfg["sha256"] = sha256 if sha256 is not None else hash_obj(
        cfg, exclude_keys=("sha256", "generated_at"))
    return cfg


def with_sha(cfg: dict) -> dict:
    """Recompute the self-hash after a nested mutation (keeps tamper tests targeted)."""
    out = copy.deepcopy(cfg)
    out.pop("sha256", None)
    out["sha256"] = hash_obj(out, exclude_keys=("sha256", "generated_at"))
    return out


# --- data / trial identities ----------------------------------------------------------

def make_data_identity(config=None, **over) -> dict:
    plan_sha = (config or {"partition": make_partition()})["partition"]["sha256"]
    d = {
        "development_dataset_id": DEV_DATASET_ID,
        "development_build_id": sha_hex("dev-build-0001"),
        "development_manifest_sha256": sha_hex("dev-manifest"),
        "development_logical_row_sha256": sha_hex("dev-logical-rows"),
        "partition_plan_sha256": plan_sha,
    }
    d.update(over)
    return d


def make_trial_identity(**over) -> dict:
    candidates = make_candidates()
    ridge = candidates[2]
    identity = {
        "schema": TRIAL_SCHEMA,
        "protocol_config_sha256": sha_hex("protocol-config"),
        "source_certification_sha256": hash_obj(make_source_certification()),
        "development_dataset_id": DEV_DATASET_ID,
        "development_build_id": sha_hex("dev-build-0001"),
        "development_manifest_sha256": sha_hex("dev-manifest"),
        "development_logical_row_sha256": sha_hex("dev-logical-rows"),
        "partition_plan_sha256": make_partition()["sha256"],
        "candidate_id": "ofi_ridge",
        "candidate_definition_sha256": hash_obj(ridge),
        "candidate_code_sha256": ridge["candidate_code_sha256"],
        "preprocessing": copy.deepcopy(ridge["preprocessing"]),
        "preprocessing_sha256": ridge["preprocessing_sha256"],
        "model_params": copy.deepcopy(ridge["model_params"]),
        "model_params_sha256": ridge["model_params_sha256"],
        "feature_cols": list(ridge["feature_cols"]),
        "seed_and_thread_settings": copy.deepcopy(ridge["seed_and_thread_settings"]),
        "software_versions_sha256": hash_obj(make_software()),
        "horizon": "2s",
        "horizon_role": "primary",
        "cv_sha256": hash_obj(make_cv()),
        "label_sha256": hash_obj(make_labels()),
        "cost_sha256": hash_obj(make_costs()),
        "thresholds_sha256": hash_obj(make_verdict_thresholds()),
        "variant": "base",
        "variant_params": {},
    }
    identity.update(over)
    return identity
