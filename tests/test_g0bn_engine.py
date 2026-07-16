"""G0-BN candidate engine tests (issue #88, slice 67-B; spec sections 3.4 and 4.1-4.2).

Covers spec section 11 items 3 and 5 (engine half): the exact 15 base trial
executions, runtime parameter re-resolution before fitting, empty-feature
persistence and raw microprice behavior, exact unweighted float64 population
classifier scaling, lexicographic 15-split enumeration, and the ordered float64
arithmetic-mean collapse with missing/extra/duplicate/non-finite rejection.
Synthetic bundles only (tests/g0bn_dev_fixtures.py).
"""
from __future__ import annotations

import copy
from itertools import combinations

import numpy as np
import pytest

import eval.g0bn_engine as eng
from eval.g0bn_engine import (
    N_SPLITS,
    TEST_MULTIPLICITY,
    classifier_training_scale,
    class_prob_spread,
    collapse_split_forecasts,
    cpcv_candidate_forecasts,
    forecast_series_sha256,
    resolve_runtime_candidate,
    run_g0bn_development,
    verify_development_inputs,
)
from eval.g0bn_identity import base_trial_identities
from eval.g0bn_ledger import G0BNLedger
from eval.hashing import hash_obj
from tests.g0bn_dev_fixtures import (
    dev_bundle,
    dev_data_identity,
    dev_manifest,
    durable_ledger,
    horizon_roles_sha256,
    runtime_candidates,
)


def _splits_for_six_rows():
    """15 CPCV-shaped test sets over 6 rows (row i == group i): each row appears in
    exactly C(5,1)=5 of the C(6,2)=15 lexicographic combinations."""
    return [np.array(combo, dtype=np.int64) for combo in combinations(range(6), 2)]


# ----------------------------------------------------------------- forecast collapse

def test_collapse_is_mean_of_exactly_five_forecasts():
    test_sets = _splits_for_six_rows()
    splits = [(idx, np.full(len(idx), float(s), dtype=np.float64))
              for s, idx in enumerate(test_sets)]
    f = collapse_split_forecasts(6, splits)
    assert f.dtype == np.float64
    for row in range(6):
        vals = [float(s) for s, idx in enumerate(test_sets) if row in idx]
        assert len(vals) == TEST_MULTIPLICITY
        assert f[row] == sum(vals) / 5.0


def test_collapse_accumulates_in_fixed_split_order():
    # Float64 addition is not associative: the pinned order (original row position,
    # fixed split enumeration order) is decision-bearing. Row 0 receives, in split
    # order, [1e16, 1.0, 1.0, -1e16, 1.0]: ordered accumulation gives 1.0/5, any
    # sorted/pairwise regrouping gives a different sum.
    test_sets = _splits_for_six_rows()
    row0_splits = [s for s, idx in enumerate(test_sets) if 0 in idx]
    planted = {row0_splits[0]: 1e16, row0_splits[1]: 1.0, row0_splits[2]: 1.0,
               row0_splits[3]: -1e16, row0_splits[4]: 1.0}
    splits = [(idx, np.array([planted.get(s, 0.0) if r == 0 else 0.0 for r in idx]))
              for s, idx in enumerate(test_sets)]
    f = collapse_split_forecasts(6, splits)
    expected = np.float64(0.0)
    for s in row0_splits:
        expected = expected + np.float64(planted[s])
    assert f[0] == expected / 5.0
    assert f[0] == pytest.approx(0.2)


def test_collapse_rejects_wrong_split_count():
    splits = [(idx, np.zeros(len(idx))) for idx in _splits_for_six_rows()]
    with pytest.raises(ValueError, match="15"):
        collapse_split_forecasts(6, splits[:-1])
    with pytest.raises(ValueError, match="15"):
        collapse_split_forecasts(6, splits + [splits[0]])


def test_collapse_rejects_missing_coverage():
    splits = [(idx, np.zeros(len(idx))) for idx in _splits_for_six_rows()]
    idx, fc = splits[0]
    splits[0] = (idx[:1], fc[:1])          # row dropped from one test set -> count 4
    with pytest.raises(ValueError, match="exactly 5"):
        collapse_split_forecasts(6, splits)


def test_collapse_rejects_extra_coverage():
    splits = [(idx, np.zeros(len(idx))) for idx in _splits_for_six_rows()]
    idx, fc = splits[0]
    extra = np.append(idx, 5)              # row 5 smuggled into a foreign test set
    splits[0] = (extra, np.zeros(len(extra)))
    with pytest.raises(ValueError, match="exactly 5"):
        collapse_split_forecasts(6, splits)


def test_collapse_rejects_duplicate_rows_within_a_split():
    splits = [(idx, np.zeros(len(idx))) for idx in _splits_for_six_rows()]
    splits[0] = (np.array([0, 0]), np.zeros(2))
    with pytest.raises(ValueError, match="duplicate"):
        collapse_split_forecasts(6, splits)


def test_collapse_rejects_non_finite_forecasts():
    for bad in (np.nan, np.inf, -np.inf):
        splits = [(idx, np.zeros(len(idx))) for idx in _splits_for_six_rows()]
        idx, fc = splits[3]
        fc = fc.copy()
        fc[0] = bad
        splits[3] = (idx, fc)
        with pytest.raises(ValueError, match="finite"):
            collapse_split_forecasts(6, splits)


def test_collapse_rejects_out_of_range_and_mismatched_lengths():
    splits = [(idx, np.zeros(len(idx))) for idx in _splits_for_six_rows()]
    splits[0] = (np.array([0, 6]), np.zeros(2))
    with pytest.raises(ValueError, match="row index"):
        collapse_split_forecasts(6, splits)
    splits = [(idx, np.zeros(len(idx))) for idx in _splits_for_six_rows()]
    splits[0] = (splits[0][0], np.zeros(3))
    with pytest.raises(ValueError, match="length"):
        collapse_split_forecasts(6, splits)


# --------------------------------------------------------- runtime re-resolution

def test_runtime_resolution_accepts_all_five_pinned_candidates():
    for defn in runtime_candidates():
        resolve_runtime_candidate(defn)     # must not raise


def test_runtime_resolution_rejects_package_version_drift():
    defn = copy.deepcopy(runtime_candidates()[2])   # ofi_ridge
    defn["package_version"] = "0.0.0"
    with pytest.raises(ValueError, match="package_version"):
        resolve_runtime_candidate(defn)


def test_runtime_resolution_rejects_resolved_parameter_drift(monkeypatch):
    class _DriftingRidge:
        def __init__(self, **kw):
            self._kw = dict(kw)

        def get_params(self, deep=False):
            return dict(self._kw, tol=0.001)   # a library default silently changed

    monkeypatch.setitem(eng.ESTIMATOR_CLASSES, "sklearn.linear_model.Ridge",
                        _DriftingRidge)
    with pytest.raises(ValueError, match="re-resolved"):
        resolve_runtime_candidate(runtime_candidates()[2])


def test_runtime_resolution_rejects_unknown_estimator_class():
    defn = copy.deepcopy(runtime_candidates()[2])
    defn["estimator_class"] = "sklearn.linear_model.Lasso"
    with pytest.raises(ValueError, match="estimator_class"):
        resolve_runtime_candidate(defn)


def test_runtime_resolution_rejects_tampered_nonfitted_params():
    defn = copy.deepcopy(runtime_candidates()[1])   # microprice_raw
    defn["model_params"]["multiplier"] = 2.0
    defn["model_params_sha256"] = hash_obj(defn["model_params"])
    with pytest.raises(ValueError):
        resolve_runtime_candidate(defn)


# ------------------------------------------------------------- classifier scaling

def test_classifier_training_scale_exact_formula():
    y = [3.0, -1.0, 4.0, 1.0, -5.0]
    expected = np.std(np.asarray(y, dtype=np.float64), dtype=np.float64,
                      ddof=0) + np.float64(1e-9)
    got = classifier_training_scale(np.array(y))
    assert isinstance(got, float)
    assert got == float(expected)


def test_classifier_training_scale_uses_only_finite_values():
    y = np.array([3.0, np.nan, -1.0, np.inf, 4.0, 1.0, -np.inf, -5.0])
    assert classifier_training_scale(y) == classifier_training_scale(
        np.array([3.0, -1.0, 4.0, 1.0, -5.0]))


def test_classifier_training_scale_takes_no_weights():
    # Uniqueness weights fit the classifier but must be unable to enter the scale:
    # the rule is enforced structurally (no weight parameter exists at all).
    import inspect
    assert "weight" not in " ".join(
        inspect.signature(classifier_training_scale).parameters)


def test_class_prob_spread_maps_classes_and_fills_missing_with_zero():
    proba = np.array([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])
    f = class_prob_spread(proba, np.array([-1, 0, 1]), 10.0)
    assert np.allclose(f, [(0.3 - 0.2) * 10.0, (0.1 - 0.6) * 10.0])
    # Defensive narrow-proba handling: a class absent from classes_ has
    # probability exactly 0.0 regardless of the physical proba width.
    proba2 = np.array([[0.7, 0.3]])
    f2 = class_prob_spread(proba2, np.array([-1, 0]), 10.0)
    assert np.allclose(f2, (0.0 - 0.7) * 10.0)


def test_real_classifier_two_class_fold_column_alignment():
    # Regression lock for the installed LightGBM: with the pinned num_class=3, a
    # purged training fold observing only {-1, +1} still yields a 3-column
    # predict_proba, with the OBSERVED classes in classes_ order occupying the
    # leading columns and a near-zero phantom third column. class_prob_spread must
    # read P(+1)/P(-1) through classes_, never by physical position 2.
    defn = runtime_candidates()[4]
    cls = eng.ESTIMATOR_CLASSES[defn["estimator_class"]]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((160, 4))
    y = np.where(X[:, 0] > 0, 1, -1)
    model = cls(**defn["model_params"])
    model.fit(X, y, sample_weight=np.ones(len(y)))
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 3)
    assert list(model.classes_) == [-1, 1]
    assert proba[:, 2].max() < 1e-6          # phantom class stays ~0
    f = class_prob_spread(proba, model.classes_, 2.5)
    assert np.array_equal(f, (proba[:, 1] - proba[:, 0]) * np.float64(2.5))


# ------------------------------------------------------ candidate CPCV forecasts

@pytest.fixture(scope="module")
def bundle():
    return dev_bundle()


@pytest.fixture(scope="module")
def horizon_rows(bundle):
    frame, manifest, config, identity = bundle
    return verify_development_inputs(frame, manifest, config, identity)


def test_persistence_zero_forecasts_are_exactly_zero(bundle, horizon_rows):
    _, _, config, _ = bundle
    rows = horizon_rows["2s"]
    f, scales = cpcv_candidate_forecasts(config["candidates"][0], rows,
                                         embargo_ns=config["cv"]["embargo_ns"])
    assert scales is None
    assert f.dtype == np.float64
    assert (f == 0.0).all()


def test_microprice_raw_forecast_is_exactly_the_input_column(bundle, horizon_rows):
    _, _, config, _ = bundle
    rows = horizon_rows["10s"]
    f, scales = cpcv_candidate_forecasts(config["candidates"][1], rows,
                                         embargo_ns=config["cv"]["embargo_ns"])
    assert scales is None
    x = rows["microprice_dev"].to_numpy(np.float64)
    # The raw multiplier forecast still flows through the mandatory collapse: five
    # identical float64 values accumulated in split order, divided by 5.0.
    assert np.array_equal(f, (x + x + x + x + x) / 5.0)
    assert np.allclose(f, x, rtol=1e-15, atol=0.0)


def test_fitted_candidates_produce_finite_collapsed_forecasts(bundle, horizon_rows):
    _, _, config, _ = bundle
    rows = horizon_rows["2s"]
    for defn in config["candidates"][2:]:
        f, scales = cpcv_candidate_forecasts(defn, rows,
                                             embargo_ns=config["cv"]["embargo_ns"])
        assert f.shape == (len(rows),)
        assert np.isfinite(f).all()
        if defn["candidate_id"] == "lgbm_clf":
            assert scales is not None and len(scales) == N_SPLITS
            assert all(s > 0 for s in scales)
        else:
            assert scales is None


def test_classifier_split_scales_ignore_uniqueness_weights(bundle, horizon_rows):
    _, _, config, _ = bundle
    rows = horizon_rows["2s"]
    reweighted = rows.copy()
    reweighted["uniqueness"] = np.linspace(0.05, 1.0, len(rows))
    _, scales_a = cpcv_candidate_forecasts(config["candidates"][4], rows,
                                           embargo_ns=config["cv"]["embargo_ns"])
    _, scales_b = cpcv_candidate_forecasts(config["candidates"][4], reweighted,
                                           embargo_ns=config["cv"]["embargo_ns"])
    assert scales_a == scales_b


def test_forecast_series_sha256_is_key_and_value_sensitive():
    t = np.array([1, 2, 3], dtype=np.int64)
    f = np.array([0.5, -0.25, 0.0])
    base = forecast_series_sha256(t, "2s", f)
    assert base != forecast_series_sha256(t, "10s", f)
    assert base != forecast_series_sha256(t + 1, "2s", f)
    f2 = f.copy()
    f2[0] = 0.5 + 1e-12
    assert base != forecast_series_sha256(t, "2s", f2)
    assert base == forecast_series_sha256(t.copy(), "2s", f.copy())


# ------------------------------------------------------------- input verification

def test_verify_development_inputs_returns_sorted_horizon_rows(bundle):
    frame, manifest, config, identity = bundle
    rows = verify_development_inputs(frame, manifest, config, identity)
    assert set(rows) == {"2s", "10s", "60s"}
    for tag, sub in rows.items():
        te = sub["t_event"].to_numpy()
        assert (np.diff(te) > 0).all()
        assert (sub["horizon"] == tag).all()


def test_verify_rejects_holdout_manifest(bundle):
    frame, manifest, config, identity = bundle
    man = copy.deepcopy(manifest)
    for s in man["sources"]:
        if s.get("name") == "partition_contract":
            s["partition"] = "holdout"
    with pytest.raises(ValueError):
        verify_development_inputs(frame, man, config, identity)


def test_verify_rejects_tampered_matrix_values(bundle):
    frame, manifest, config, identity = bundle
    tampered = frame.copy()
    tampered.loc[0, "y_fwd_bps"] = tampered.loc[0, "y_fwd_bps"] + 1e-9
    with pytest.raises(ValueError, match="logical"):
        verify_development_inputs(tampered, manifest, config, identity)


def test_verify_rejects_foreign_protocol_config(bundle):
    frame, manifest, config, identity = bundle
    other = copy.deepcopy(config)
    other["sha256"] = "f" * 64          # self-hash no longer matches the content
    with pytest.raises(ValueError):
        verify_development_inputs(frame, manifest, other, identity)


def test_verify_rejects_binding_mismatches(bundle):
    # Re-derive the data identity for each mutated manifest so the manifest-hash
    # gate passes and the SPECIFIC config-binding gate is what fires.
    frame, manifest, config, _ = bundle
    man = copy.deepcopy(manifest)
    for s in man["sources"]:
        if s.get("name") == "partition_contract":
            s["partition_plan_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="partition"):
        verify_development_inputs(frame, man, config,
                                  dev_data_identity(config, man, frame))
    man = copy.deepcopy(manifest)
    for s in man["sources"]:
        if s.get("name") == "g0bn_protocol":
            s["protocol_config_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="protocol_config_sha256"):
        verify_development_inputs(frame, man, config,
                                  dev_data_identity(config, man, frame))


def test_verify_rejects_stale_manifest_content(bundle):
    # A manifest edit WITHOUT a re-derived identity trips the manifest-hash binding.
    frame, manifest, config, identity = bundle
    man = copy.deepcopy(manifest)
    for s in man["sources"]:
        if s.get("name") == "g0bn_protocol":
            s["protocol_config_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="manifest"):
        verify_development_inputs(frame, man, config, identity)


def test_verify_rejects_stale_data_identity(bundle):
    frame, manifest, config, identity = bundle
    stale = dict(identity, development_build_id="a" * 64)
    with pytest.raises(ValueError, match="build"):
        verify_development_inputs(frame, manifest, config, stale)
    stale = dict(identity, development_manifest_sha256="b" * 64)
    with pytest.raises(ValueError, match="manifest"):
        verify_development_inputs(frame, manifest, config, stale)


def test_verify_rejects_cost_assumption_mismatch(bundle):
    frame, manifest, config, _ = bundle
    man = copy.deepcopy(manifest)
    for s in man["sources"]:
        if s.get("name") == "cost_assumption":
            s["taker_fee_bps"] = s["taker_fee_bps"] + 0.1
    with pytest.raises(ValueError, match="cost_assumption"):
        verify_development_inputs(frame, man, config,
                                  dev_data_identity(config, man, frame))


def test_verify_rejects_rows_outside_included_days(bundle):
    from tests.g0bn_dev_fixtures import dev_config, dev_matrix
    config = dev_config()
    frame = dev_matrix(config, days=["2025-11-25"])   # an excluded (out-of-scope) day
    manifest = dev_manifest(config, frame)
    identity = dev_data_identity(config, manifest, frame)
    with pytest.raises(ValueError, match="included"):
        verify_development_inputs(frame, manifest, config, identity)


def test_verify_rejects_partition_prefilter_violation(bundle):
    # A row whose guarded span crosses the development end must have been dropped by
    # the producer; the engine fails closed if one survives. Plant a 60s row 30s
    # before the New Year boundary: t_event + 60s horizon + 120s guard >= dev end.
    import tests.g0bn_dev_fixtures as fx
    config = fx.dev_config(exclusions=fx.dev_exclusions(61))   # include every dev day
    frame = fx.dev_matrix(config, days=["2025-12-30"])
    end_ns = 1_767_225_600_000_000_000
    late = frame.index[frame["horizon"] == "60s"][-1]
    frame.loc[late, "t_event"] = end_ns - 30_000_000_000
    frame.loc[late, "t_barrier"] = end_ns - 30_000_000_000 + 60_000_000_000
    frame.loc[late, "t_feature_start"] = end_ns - 90_000_000_000
    frame.loc[late, "t_available"] = end_ns - 30_000_000_000
    manifest = fx.dev_manifest(config, frame)
    identity = fx.dev_data_identity(config, manifest, frame)
    with pytest.raises(ValueError, match="prefilter"):
        verify_development_inputs(frame, manifest, config, identity)


def test_verify_rejects_uncertified_source_certification_entry(bundle):
    # The manifest's actual source_certification ENTRY must equal the config's
    # certified evidence hash — not just the copy inside the protocol binding.
    frame, manifest, config, _ = bundle
    man = copy.deepcopy(manifest)
    for s in man["sources"]:
        if s.get("name") == "source_certification":
            s["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="source_certification"):
        verify_development_inputs(frame, man, config,
                                  dev_data_identity(config, man, frame))


def test_verify_rejects_uncertified_source_object_hashes(bundle):
    # Every Binance source-object hash in the manifest must reconcile with the
    # config's development source-manifest evidence pin.
    frame, manifest, config, _ = bundle
    man = copy.deepcopy(manifest)
    for s in man["sources"]:
        if s.get("name") == "binance_futures_l2_delta":
            s["sha256"] = "8" * 64
    with pytest.raises(ValueError, match="source"):
        verify_development_inputs(frame, man, config,
                                  dev_data_identity(config, man, frame))


def test_verify_rejects_installed_software_drift(bundle):
    from tests.g0bn_dev_fixtures import dev_config, runtime_software
    frame, _, _, _ = bundle
    config = dev_config(software=runtime_software(numpy_version="0.0.0"))
    manifest = dev_manifest(config, frame)
    identity = dev_data_identity(config, manifest, frame)
    with pytest.raises(ValueError, match="numpy"):
        verify_development_inputs(frame, manifest, config, identity)


def test_runtime_resolution_rejects_candidate_code_hash_drift():
    from tests.g0bn_protocol_fixtures import make_candidates
    # The 67-A fixture pins a synthetic code hash; the runtime gate must compare
    # against the RUNNING candidate implementation and refuse the mismatch.
    defn = make_candidates()[2]
    with pytest.raises(ValueError, match="candidate_code_sha256"):
        resolve_runtime_candidate(defn)


def test_run_requires_a_durable_ledger(bundle):
    frame, manifest, config, identity = bundle
    with pytest.raises(ValueError, match="durable"):
        run_g0bn_development(frame, manifest, config, identity, G0BNLedger())


def test_verify_requires_rows_for_every_ladder_horizon(bundle):
    frame, _, config, _ = bundle
    sub = frame[frame["horizon"] != "60s"].reset_index(drop=True)
    manifest = dev_manifest(config, sub)
    identity = dev_data_identity(config, manifest, sub)
    with pytest.raises(ValueError, match="60s"):
        verify_development_inputs(sub, manifest, config, identity)


# ----------------------------------------------------------------- run driver

@pytest.fixture(scope="module")
def dev_run(bundle):
    frame, manifest, config, identity = bundle
    ledger = durable_ledger()
    run = run_g0bn_development(frame, manifest, config, identity, ledger)
    return run, ledger


def test_run_registers_exactly_fifteen_base_trials(dev_run, bundle):
    run, ledger = dev_run
    frame, manifest, config, identity = bundle
    assert ledger.n_effective_trials() == 15
    assert len(ledger.scored_trial_ids()) == 15
    expected = [hash_obj(i) for i in base_trial_identities(config, identity)]
    assert ledger.trial_ids() == expected
    assert set(run.forecasts) == set(expected)
    assert run.aborted == {}


def test_run_results_pin_forecast_hashes(dev_run):
    run, ledger = dev_run
    for tid, f in run.forecasts.items():
        ident = ledger.identity_for(tid)
        rows = run.horizon_rows[ident["horizon"]]
        result = ledger.result_for(tid)
        assert result["schema"] == "g0bn-trial-result-v1"
        assert result["n_rows"] == len(rows)
        assert result["forecasts_sha256"] == forecast_series_sha256(
            rows["t_event"].to_numpy(np.int64), ident["horizon"], f)
        if ident["candidate_id"] == "lgbm_clf":
            assert len(result["split_scales"]) == N_SPLITS
        else:
            assert result["split_scales"] is None


def test_rerun_is_idempotent_and_deterministic(dev_run, bundle):
    run, ledger = dev_run
    frame, manifest, config, identity = bundle
    ledger2 = durable_ledger()
    run2 = run_g0bn_development(frame, manifest, config, identity, ledger2)
    assert ledger2.identity_set_sha256() == ledger.identity_set_sha256()
    for tid in run.forecasts:
        assert np.array_equal(run.forecasts[tid], run2.forecasts[tid])
    # Re-running against the SAME ledger appends execution events but neither
    # increases effective N nor changes any result.
    before = ledger.identity_set_sha256()
    run_g0bn_development(frame, manifest, config, identity, ledger)
    assert ledger.n_effective_trials() == 15
    assert ledger.identity_set_sha256() == before


def test_canonical_ordering_row_shuffle_does_not_change_results(dev_run, bundle):
    run, ledger = dev_run
    frame, manifest, config, identity = bundle
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    ledger3 = durable_ledger()
    run3 = run_g0bn_development(shuffled, manifest, config, identity, ledger3)
    assert ledger3.identity_set_sha256() == ledger.identity_set_sha256()


def test_cpcv_geometry_purges_test_spans_and_embargo(bundle, horizon_rows):
    # The engine pins t0=t_event, t1=t_barrier (spec section 9): no training row's
    # realized label span may overlap any test span, and no training row may start
    # inside the embargo window after a test barrier. Checked pairwise, which the
    # merged-interval purge in data.cv implies.
    from data.cv import cpcv_splits
    _, _, config, _ = bundle
    rows = horizon_rows["60s"]
    t_event = rows["t_event"].to_numpy(np.int64)
    t_barrier = rows["t_barrier"].to_numpy(np.int64)
    embargo = config["cv"]["embargo_ns"]
    splits = list(cpcv_splits(t_event, t_event, t_barrier, n_groups=6, k=2,
                              embargo_ns=embargo))
    assert len(splits) == N_SPLITS
    for train_idx, test_idx in splits:
        t0_tr, t1_tr = t_event[train_idx][:, None], t_barrier[train_idx][:, None]
        t0_te, t1_te = t_event[test_idx][None, :], t_barrier[test_idx][None, :]
        overlap = (t0_tr <= t1_te) & (t1_tr >= t0_te)
        embargoed = (t0_tr > t1_te) & (t0_tr <= t1_te + embargo)
        assert not overlap.any()
        assert not embargoed.any()


def test_conflicting_prior_completion_fails_the_run_not_an_abort(bundle):
    # A pre-existing completed result that the deterministic rerun cannot
    # reproduce is non-determinism/tampering: the run must fail closed, never
    # downgrade the conflict to an aborted event.
    frame, manifest, config, identity = bundle
    ledger = durable_ledger()
    poisoned = base_trial_identities(config, identity)[0]
    ledger.record_completion(poisoned, {
        "schema": "g0bn-trial-result-v1", "n_rows": 1,
        "forecasts_sha256": "0" * 64,
        "collapse_version": "mean_repeated_test_forecasts_v1",
        "split_scales": None})
    with pytest.raises(ValueError, match="DIFFERENT result"):
        run_g0bn_development(frame, manifest, config, identity, ledger)
    assert ledger.n_effective_trials() == 1     # nothing was silently replaced


def test_infrastructure_abort_is_recorded_and_counted(bundle, monkeypatch):
    frame, manifest, config, identity = bundle

    class _Dies:
        def __init__(self, **kw):
            self._kw = dict(kw)

        def get_params(self, deep=False):
            return dict(self._kw)

        def fit(self, *a, **kw):
            raise MemoryError("synthetic infrastructure failure")

    monkeypatch.setitem(eng.ESTIMATOR_CLASSES, "lightgbm.LGBMRegressor", _Dies)
    ledger = durable_ledger()
    run = run_g0bn_development(frame, manifest, config, identity, ledger)
    assert ledger.n_effective_trials() == 15          # aborted identities still count
    assert len(ledger.scored_trial_ids()) == 12       # lgbm_reg aborted at 3 horizons
    assert len(run.aborted) == 3
    for tid, err in run.aborted.items():
        assert ledger.identity_for(tid)["candidate_id"] == "lgbm_reg"
        assert "MemoryError" in err
        assert ledger.result_for(tid) is None
    # A later intact rerun may complete the aborted identities (retry after abort).
    monkeypatch.undo()
    run2 = run_g0bn_development(frame, manifest, config, identity, ledger)
    assert run2.aborted == {}
    assert ledger.n_effective_trials() == 15
    assert len(ledger.scored_trial_ids()) == 15
