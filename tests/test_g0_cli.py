"""G0 orchestrator CLI: the G0-CB command cannot name or open a holdout input (fails
before any matrix read), G0-XV development rejects holdout-bound arms pre-load and
requires the G0-CB trial history, the freeze -> open -> validate -> score flow enforces
the one-time transaction end to end, and fail-open report coercion / mid-run ledger loss
are pinned by regression tests."""
import json

import pytest

import eval.g0 as g0mod
import scripts.run_g0 as rg
from eval.consumption import load_record, record_path_for
from eval.ledger import TrialLedger

GATE = {"n_groups": 4, "k": 2, "min_trades": 5, "min_eff_trades": 3.0}
XV_GATE = {**GATE, "noise_band_n_boot": 500}


class Store:
    """In-memory matrix 'files': the CLI's read_matrix seam, with call accounting so
    tests can prove a rejection happened BEFORE any data was loaded."""

    def __init__(self):
        self.frames = {}
        self.calls = []

    def put(self, name, df):
        self.frames[name] = df
        return name

    def __call__(self, path):
        self.calls.append(str(path))
        return self.frames[str(path)].copy()


def _dump(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _setup(tmp_path, world):
    store = Store()
    files = {"contract": _dump(tmp_path, "contract.json", world["contract"]),
             "gate": _dump(tmp_path, "gate.json", GATE),
             "xv_gate": _dump(tmp_path, "xv_gate.json", XV_GATE)}
    for part in ("dev", "holdout"):
        for arm, a in world[part]["arms"].items():
            files[f"{part}_{arm}_man"] = _dump(tmp_path, f"{part}_{arm}_man.json",
                                               a["manifest"])
            files[f"{part}_{arm}_mat"] = store.put(f"{part}_{arm}.parquet", a["matrix"])
    return store, files


def test_g0cb_cli_happy_path(tmp_path, g0_world):
    store, f = _setup(tmp_path, g0_world)
    out, ledger = str(tmp_path / "g0cb.json"), str(tmp_path / "g0cb_ledger.json")
    rc = rg.main(["g0cb", "--matrix", f["dev_coinbase_only_mat"],
                  "--manifest", f["dev_coinbase_only_man"], "--contract", f["contract"],
                  "--ledger", ledger, "--gate-json", f["gate"], "--out", out],
                 read_matrix=store)
    assert rc == 0
    res = json.loads(open(out).read())
    assert res["protocol"] == "g0cb-development" and res["g0cb_pass"] is True
    assert TrialLedger.load(ledger).n_effective_trials() == 4


def test_g0cb_cli_rejects_holdout_input_before_data_loading(tmp_path, g0_world):
    store, f = _setup(tmp_path, g0_world)
    rc = rg.main(["g0cb", "--matrix", f["holdout_coinbase_only_mat"],
                  "--manifest", f["holdout_coinbase_only_man"],
                  "--contract", f["contract"], "--ledger", str(tmp_path / "l.json"),
                  "--out", str(tmp_path / "o.json")], read_matrix=store)
    assert rc == 2
    assert store.calls == []              # rejected before the matrix was ever opened


def test_g0cb_cli_has_no_holdout_arguments(tmp_path, g0_world):
    store, f = _setup(tmp_path, g0_world)
    with pytest.raises(SystemExit):
        rg.main(["g0cb", "--matrix", f["dev_coinbase_only_mat"],
                 "--manifest", f["dev_coinbase_only_man"], "--contract", f["contract"],
                 "--ledger", str(tmp_path / "l.json"), "--out", str(tmp_path / "o.json"),
                 "--holdout-matrix", f["holdout_coinbase_only_mat"]], read_matrix=store)
    assert store.calls == []


def test_g0cb_cli_persists_attempted_trials_when_study_aborts(tmp_path, g0_world,
                                                              monkeypatch):
    """Every attempted candidate is trial history even when a later candidate aborts
    the run — the ledger must not lose attempts to an exception (DSR undercount)."""
    store, f = _setup(tmp_path, g0_world)
    real = g0mod.evaluate_config
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 3:
            raise ValueError("synthetic mid-run failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(g0mod, "evaluate_config", flaky)
    ledger = str(tmp_path / "ledger.json")
    rc = rg.main(["g0cb", "--matrix", f["dev_coinbase_only_mat"],
                  "--manifest", f["dev_coinbase_only_man"], "--contract", f["contract"],
                  "--ledger", ledger, "--gate-json", f["gate"],
                  "--out", str(tmp_path / "o.json")], read_matrix=store)
    assert rc == 2
    assert TrialLedger.load(ledger).n_effective_trials() == 3   # attempts survived


def test_g0xv_cli_requires_prior_history(tmp_path, g0_world):
    store, f = _setup(tmp_path, g0_world)
    arm_args = []
    for arm in ("coinbase_only", "binance_only", "combined"):
        arm_args += ["--arm", arm, f[f"dev_{arm}_man"], f[f"dev_{arm}_mat"]]
    rc = rg.main(["g0xv-dev", *arm_args, "--contract", f["contract"],
                  "--ledger", str(tmp_path / "l.json"),
                  "--out", str(tmp_path / "o.json")], read_matrix=store)
    assert rc == 2                        # no --prior-ledger and no explicit opt-out
    assert store.calls == []


def test_g0xv_cli_rejects_holdout_bound_arm_before_data_loading(tmp_path, g0_world):
    store, f = _setup(tmp_path, g0_world)
    rc = rg.main(["g0xv-dev",
                  "--arm", "coinbase_only", f["dev_coinbase_only_man"],
                  f["dev_coinbase_only_mat"],
                  "--arm", "binance_only", f["dev_binance_only_man"],
                  f["dev_binance_only_mat"],
                  "--arm", "combined", f["holdout_combined_man"],
                  f["holdout_combined_mat"],
                  "--no-prior-history",
                  "--contract", f["contract"], "--ledger", str(tmp_path / "l.json"),
                  "--out", str(tmp_path / "o.json")], read_matrix=store)
    assert rc == 2
    assert store.calls == []


def test_full_cli_flow_one_time_holdout(tmp_path, g0_world):
    store, f = _setup(tmp_path, g0_world)
    cb_ledger = str(tmp_path / "cb_ledger.json")
    assert rg.main(["g0cb", "--matrix", f["dev_coinbase_only_mat"],
                    "--manifest", f["dev_coinbase_only_man"], "--contract", f["contract"],
                    "--ledger", cb_ledger, "--gate-json", f["gate"],
                    "--out", str(tmp_path / "cb.json")], read_matrix=store) == 0

    xv_ledger, dev_out = str(tmp_path / "xv_ledger.json"), str(tmp_path / "xv.json")
    arm_args = []
    for arm in ("coinbase_only", "binance_only", "combined"):
        arm_args += ["--arm", arm, f[f"dev_{arm}_man"], f[f"dev_{arm}_mat"]]
    assert rg.main(["g0xv-dev", *arm_args, "--contract", f["contract"],
                    "--ledger", xv_ledger, "--prior-ledger", cb_ledger,
                    "--gate-json", f["xv_gate"], "--out", dev_out],
                   read_matrix=store) == 0
    dev_res = json.loads(open(dev_out).read())
    assert dev_res["g0xv_dev_pass"] is True
    assert dev_res["ledger"]["n_effective_trials"] == 16
    arm = dev_res["winner"]["arm"]

    scope = {"days": g0_world["holdout_days"], "venues": ["coinbase"],
             "dataset_id": "synthetic-xv-pilot", "build_id": f"holdout-seeded-{arm}"}
    freeze_path = str(tmp_path / "freeze.json")
    assert rg.main(["freeze", "--dev-result", dev_out, "--contract", f["contract"],
                    "--ledger", xv_ledger,
                    "--thresholds-json", _dump(tmp_path, "thr.json", {"min_rows": 10}),
                    "--scope-json", _dump(tmp_path, "scope.json", scope),
                    "--generated-at", "2026-07-10T12:00:00+00:00",
                    "--out", freeze_path], read_matrix=store) == 0

    records = tmp_path / "records"
    records.mkdir()
    assert rg.main(["holdout-open", "--freeze", freeze_path,
                    "--records-dir", str(records)], read_matrix=store) == 0
    # scoring before validation must refuse BEFORE any holdout read
    pre_calls = list(store.calls)
    score_args = ["holdout-score", "--freeze", freeze_path,
                  "--records-dir", str(records), "--contract", f["contract"],
                  "--dev-matrix", f[f"dev_{arm}_mat"],
                  "--dev-manifest", f[f"dev_{arm}_man"],
                  "--holdout-matrix", f[f"holdout_{arm}_mat"],
                  "--holdout-manifest", f[f"holdout_{arm}_man"],
                  "--out", str(tmp_path / "score.json")]
    assert rg.main(score_args, read_matrix=store) == 2
    assert store.calls == pre_calls

    report = {"scope_days": scope["days"], "scope_venues": ["coinbase"], "passed": True}
    assert rg.main(["holdout-validate", "--freeze", freeze_path,
                    "--records-dir", str(records),
                    "--report-json", _dump(tmp_path, "report.json", report)],
                   read_matrix=store) == 0

    # a VALIDATED transaction with a wrong holdout build must refuse BEFORE any matrix
    # is opened — repeated mismatched invocations cannot re-open the holdout
    other = "binance_only" if arm != "binance_only" else "combined"
    wrong_build = [a for a in score_args]
    wrong_build[wrong_build.index(f[f"holdout_{arm}_mat"])] = f[f"holdout_{other}_mat"]
    wrong_build[wrong_build.index(f[f"holdout_{arm}_man"])] = f[f"holdout_{other}_man"]
    pre_calls = list(store.calls)
    assert rg.main(wrong_build, read_matrix=store) == 2
    assert store.calls == pre_calls

    # an unwritable --out must fail BEFORE the transaction is consumed — both a missing
    # parent dir and an existing DIRECTORY at the leaf (a writable parent is not enough)
    bad_out = [a if not a.endswith("score.json") else str(tmp_path / "nodir" / "s.json")
               for a in score_args]
    assert rg.main(bad_out, read_matrix=store) == 2
    dir_out = [a if not a.endswith("score.json") else str(tmp_path)
               for a in score_args]
    assert rg.main(dir_out, read_matrix=store) == 2
    rec_path = record_path_for(str(records), json.loads(open(freeze_path).read()))
    assert load_record(rec_path)["state"] == "validated"       # nothing consumed

    assert rg.main(score_args, read_matrix=store) == 0
    score = json.loads(open(tmp_path / "score.json").read())
    assert score["consumed"] is True and score["protocol"] == "g0xv-holdout"
    assert rg.main(score_args, read_matrix=store) == 2            # one-time consumption
    assert rg.main(score_args + ["--verify-only"], read_matrix=store) == 0
    # a second validation attempt is also rejected
    assert rg.main(["holdout-validate", "--freeze", freeze_path,
                    "--records-dir", str(records),
                    "--report-json", _dump(tmp_path, "report2.json", report)],
                   read_matrix=store) == 2


def test_cli_validate_rejects_non_boolean_passed(tmp_path, g0_pipeline):
    """bool('false') is True: the CLI must never coerce the report verdict — a truthy
    string from external tooling would permanently record a PASS on the single gate
    protecting holdout consumption."""
    from eval.freeze import write_freeze
    store = Store()
    freeze_path = str(tmp_path / "freeze.json")
    write_freeze(g0_pipeline["freeze"], freeze_path)
    records = tmp_path / "records"
    records.mkdir()
    assert rg.main(["holdout-open", "--freeze", freeze_path,
                    "--records-dir", str(records)], read_matrix=store) == 0
    report = {"scope_days": g0_pipeline["scope"]["days"], "scope_venues": ["coinbase"],
              "passed": "false"}
    rc = rg.main(["holdout-validate", "--freeze", freeze_path,
                  "--records-dir", str(records),
                  "--report-json", _dump(tmp_path, "report.json", report)],
                 read_matrix=store)
    assert rc == 2
    rec_path = record_path_for(str(records), g0_pipeline["freeze"])
    assert load_record(rec_path)["state"] == "frozen"    # verdict NOT recorded


def test_cli_validation_failure_blocks_scoring_without_loading(tmp_path, g0_pipeline):
    from eval.freeze import write_freeze
    world = g0_pipeline["world"]
    store, f = _setup(tmp_path, world)
    freeze_path = str(tmp_path / "freeze.json")
    write_freeze(g0_pipeline["freeze"], freeze_path)
    records = tmp_path / "records"
    records.mkdir()
    assert rg.main(["holdout-open", "--freeze", freeze_path,
                    "--records-dir", str(records)], read_matrix=store) == 0
    report = {"scope_days": g0_pipeline["scope"]["days"], "scope_venues": ["coinbase"],
              "passed": False}
    assert rg.main(["holdout-validate", "--freeze", freeze_path,
                    "--records-dir", str(records),
                    "--report-json", _dump(tmp_path, "report.json", report)],
                   read_matrix=store) == 3
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    rc = rg.main(["holdout-score", "--freeze", freeze_path,
                  "--records-dir", str(records),
                  "--contract", f["contract"], "--dev-matrix", f[f"dev_{arm}_mat"],
                  "--dev-manifest", f[f"dev_{arm}_man"],
                  "--holdout-matrix", f[f"holdout_{arm}_mat"],
                  "--holdout-manifest", f[f"holdout_{arm}_man"],
                  "--out", str(tmp_path / "score.json")], read_matrix=store)
    assert rc == 2
    assert store.calls == []              # blocked before any matrix was read
