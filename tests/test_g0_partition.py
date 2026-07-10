"""Partition contract: schema, hash pinning, manifest binding reconciliation, and the
fail-closed support-span rules (issue #52 acceptance: boundary rows, early-barrier bypass,
April->May label support, drop-count reconciliation)."""
import copy
import json

import pandas as pd
import pytest

from eval.partition import (contract_binding, contract_hash, load_partition_contract,
                            require_binding, validate_development_span,
                            validate_holdout_span, validate_partition_contract)
from eval.synthetic import (G0_CB_FEATURES, _iso_ns, g0_binding, make_g0_contract,
                            make_g0_manifest, make_g0_world)

H10 = 10_000_000_000
GUARD = 60_000_000_000
DEV_START = _iso_ns("2025-11-01T00:00:00+00:00")
APR = _iso_ns("2026-04-01T00:00:00+00:00")
MAY = _iso_ns("2026-05-01T00:00:00+00:00")


def _contract(**over):
    c = make_g0_contract(horizons={"10s": H10}, guard_ns=GUARD,
                         drop_counts={"development": {"10s": 0}, "holdout": {"10s": 0}})
    c.update(over)
    return c


def _rows(t_event, t_barrier, horizon="10s"):
    return pd.DataFrame({"t_event": pd.array(t_event, dtype="int64"),
                         "t_barrier": pd.array(t_barrier, dtype="int64"),
                         "horizon": horizon})


# ------------------------------------------------------------------------------ schema
def test_contract_validates():
    assert validate_partition_contract(_contract())


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c.pop("guard_ns"), "missing required"),
    (lambda c: c.update(guard_nz=1), "unknown partition contract keys"),
    (lambda c: c.update(partition_contract_version=2), "unsupported"),
    (lambda c: c.update(prefilter_rule="t_event < boundary"), "unsupported prefilter_rule"),
    (lambda c: c.update(dev_start_ns=c["holdout_start_ns"] + 1),
     "dev_start_ns < holdout_start_ns"),
    (lambda c: c.update(guard_ns=-1), "guard_ns"),
    (lambda c: c.update(horizons={"10s": 0}), "positive int nanoseconds"),
    (lambda c: c.update(boundary_drop_counts={"development": {"10s": 0}}),
     "exactly.*partitions"),
    (lambda c: c["boundary_drop_counts"]["development"].update({"60s": 1}),
     "exactly match declared horizons"),
    (lambda c: c["boundary_drop_counts"]["development"].update({"10s": -1}), "int >= 0"),
])
def test_contract_schema_fails_closed(mutate, match):
    c = _contract()
    mutate(c)
    with pytest.raises(ValueError, match=match):
        validate_partition_contract(c)


def test_contract_hash_excludes_generated_at_only():
    a, b = _contract(), _contract(generated_at="1999-01-01T00:00:00+00:00")
    assert contract_hash(a) == contract_hash(b)
    assert contract_hash(a) != contract_hash(_contract(guard_ns=GUARD + 1))


def test_load_partition_contract_roundtrip(tmp_path):
    p = tmp_path / "contract.json"
    p.write_text(json.dumps(_contract()))
    assert load_partition_contract(p)["guard_ns"] == GUARD
    p.write_text(json.dumps({**_contract(), "extra": 1}))
    with pytest.raises(ValueError, match="unknown partition contract keys"):
        load_partition_contract(p)


# ----------------------------------------------------------------------------- binding
def _manifest(contract, partition="development", **over):
    return make_g0_manifest("coinbase_only", G0_CB_FEATURES, contract=contract,
                            partition=partition, dataset_id="d", build_id="b", **over)


def test_binding_happy_path_reconciles():
    c = _contract()
    man = _manifest(c)
    assert require_binding(man, c, "development")["partition"] == "development"


def test_binding_must_be_unique_and_complete():
    c = _contract()
    man = _manifest(c)
    man["sources"] = ["eval/synthetic.py"]                      # no binding at all
    with pytest.raises(ValueError, match="exactly one"):
        contract_binding(man)
    man["sources"] = ["x", g0_binding(c, "development"), g0_binding(c, "development")]
    with pytest.raises(ValueError, match="exactly one"):
        contract_binding(man)
    bad = g0_binding(c, "development")
    del bad["boundary_drop_counts"]
    with pytest.raises(ValueError, match="boundary_drop_counts"):
        contract_binding({"sources": ["x", bad]})
    bad = g0_binding(c, "development")
    bad["partition"] = "test"
    with pytest.raises(ValueError, match="partition"):
        contract_binding({"sources": [bad]})


def test_binding_wrong_partition_rejected():
    c = _contract()
    man = _manifest(c, partition="holdout")
    with pytest.raises(ValueError, match="accepts only 'development'"):
        require_binding(man, c, "development")


def test_binding_stale_contract_hash_rejected():
    c = _contract()
    man = _manifest(c)
    with pytest.raises(ValueError, match="stale/substituted"):
        require_binding(man, _contract(guard_ns=GUARD + 1), "development")


def test_binding_drop_count_mismatch_rejected():
    c = _contract()
    man = _manifest(c)
    b = contract_binding(man)
    b["boundary_drop_counts"] = {"10s": 7}                      # does not reconcile
    with pytest.raises(ValueError, match="reconcile"):
        require_binding(man, c, "development")


def test_binding_requires_exact_horizon_map():
    """Both directions: an extra rung is undeclared in the contract, and a DROPPED rung
    would shrink the registered trial/PBO scope while echoing the full drop counts."""
    two = make_g0_contract(horizons={"10s": H10, "60s": 60_000_000_000}, guard_ns=GUARD,
                           drop_counts={"development": {"10s": 0, "60s": 0},
                                        "holdout": {"10s": 0, "60s": 0}})
    man = _manifest(_contract())
    man["horizons"] = {"10s": H10, "60s": 60_000_000_000}       # superset of contract
    with pytest.raises(ValueError, match="exactly match"):
        require_binding(man, _contract(), "development")
    man2 = make_g0_manifest("coinbase_only", G0_CB_FEATURES, contract=two,
                            partition="development", dataset_id="d", build_id="b")
    man2["horizons"] = {"10s": H10}                             # dropped the 60s rung
    with pytest.raises(ValueError, match="exactly match"):
        require_binding(man2, two, "development")


def test_world_drop_counts_reconcile_to_generator():
    """The synthetic producer's ACTUAL per-horizon prefilter drops are what the contract
    and every manifest binding carry — the acceptance-criterion reconciliation."""
    two_days = 2 * 24 * 3600 * 1_000_000_000
    w = make_g0_world(n_dev_bars=80, n_holdout_bars=40,
                      horizons={"10s": H10, "2d": two_days})
    c = w["contract"]
    assert c["boundary_drop_counts"]["development"] == w["dev"]["drop_counts"]
    assert c["boundary_drop_counts"]["holdout"] == w["holdout"]["drop_counts"]
    # the longer horizon reaches the boundary and must actually drop bars on both sides
    assert w["dev"]["drop_counts"]["2d"] > w["dev"]["drop_counts"]["10s"]
    assert w["holdout"]["drop_counts"]["2d"] > w["holdout"]["drop_counts"]["10s"]
    for part, arms in (("development", w["dev"]["arms"]), ("holdout", w["holdout"]["arms"])):
        for arm in arms.values():
            require_binding(arm["manifest"], c, part)


# ---------------------------------------------------------------------- span validation
def test_dev_prefilter_rejects_forward_support_reaching_holdout():
    """A March row whose forward support reaches April fails BEFORE fit, and an early
    barrier (label changed to resolve early) cannot bypass the conservative prefilter.
    The fixture is built so ONLY the prefilter clause is violated: t_barrier resolves so
    early that the actual guarded span stays clear of April (tb + guard < April 1), so a
    mutant that dropped the prefilter clause would pass this row."""
    c = _contract()
    te = APR - H10 - GUARD + 5            # te + h + guard = April 1 + 5ns  -> violates
    early_barrier = te + 1                # tb + guard = April 1 - h + 6ns  -> clean
    assert early_barrier + GUARD < APR
    with pytest.raises(ValueError,
                       match=r"'prefilter': 1, 'actual_span': 0.*regardless of t_barrier"):
        validate_development_span(_rows([te], [early_barrier]), c)


def test_span_arithmetic_cannot_overflow_int64():
    """numpy int64 addition wraps silently: a schema-valid contract with a huge guard
    (or a garbage t_event near the int64 max) must still REJECT holdout rows, not wrap
    negative and admit them. The thresholds are computed in Python ints, so these rows
    are counted as violations rather than slipping through."""
    huge_guard = _contract(guard_ns=8_000_000_000_000_000_000)
    apr_row = _rows([APR + 14 * 86_400_000_000_000], [APR + 14 * 86_400_000_000_000 + H10])
    with pytest.raises(ValueError, match="span-safe"):
        validate_development_span(apr_row, huge_guard)
    near_max = 2**63 - H10 - 1
    with pytest.raises(ValueError, match="span-safe"):
        validate_development_span(_rows([near_max], [near_max + H10]), _contract())


def test_dev_boundary_is_strict():
    c = _contract()
    ok = APR - H10 - GUARD - 1            # t_event + h + guard == boundary - 1 -> OK
    assert validate_development_span(_rows([ok], [ok + H10]), c) is None
    with pytest.raises(ValueError, match="span-safe"):                   # == boundary
        validate_development_span(_rows([ok + 1], [ok + 1 + H10]), c)


def test_dev_rejects_rows_before_partition_start():
    c = _contract()
    with pytest.raises(ValueError, match="span-safe"):
        validate_development_span(_rows([DEV_START - 1], [DEV_START - 1 + H10]), c)


def test_dev_actual_span_checked_independently():
    # A malformed row whose t_barrier runs past its declared horizon must still be caught
    # by the actual-guarded-span containment even where the prefilter passes.
    c = _contract()
    te = DEV_START + H10
    with pytest.raises(ValueError, match="actual_span"):
        validate_development_span(_rows([te], [APR - GUARD]), c)


def test_dev_undeclared_horizon_tag_rejected():
    c = _contract()
    with pytest.raises(ValueError, match="not declared in the partition contract"):
        validate_development_span(_rows([DEV_START + 1], [DEV_START + 1 + H10], "60s"), c)


def test_holdout_row_reaching_may_rejected():
    """An April holdout row whose label support reaches May is rejected (symmetric rule)."""
    c = _contract()
    te = MAY - H10                        # support crosses the May boundary
    with pytest.raises(ValueError, match="span-safe"):
        validate_holdout_span(_rows([te], [te + H10]), c)
    ok = MAY - H10 - GUARD - 1
    assert validate_holdout_span(_rows([ok], [ok + H10]), c) is None


def test_holdout_rejects_pre_april_rows():
    c = _contract()
    with pytest.raises(ValueError, match="span-safe"):
        validate_holdout_span(_rows([APR - 1], [APR - 1 + H10]), c)
