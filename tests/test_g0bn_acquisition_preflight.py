"""Issue #102 (G0-BN 68-A): deterministic, no-I/O acquisition and custody preflight.

Everything here is synthetic and outcome-blind. The tests prove:
  * exact 92-day / three-product scope with deterministic ordering and hashes;
  * disjoint development vs sealed-holdout raw/normalized destinations;
  * reconciliation against the real downloader planner/estimator constants;
  * an inert approval packet whose exact commands can never silently drift;
  * a custody handoff whose emitted inventory satisfies the CURRENT
    eval.g0bn_freeze.validate_custody_inventory consumer contract, including the
    g0bn-custodian-seal-content-v1 commitment and prohibited-field rejection;
  * fail-closed rejection of extra dates/products/providers, fallback sources,
    identity drift, malformed seals, unsafe destinations, and activity proxies;
  * network isolation: the whole preflight runs with sockets disabled and the
    module never imports a vendor SDK or network client.

No vendor I/O anywhere: no Lake/S3/HTTP call, no credentials, no raw data.
"""
from __future__ import annotations

import copy
import json
import math
import os
import socket
import subprocess
import sys

import pytest

from eval.g0bn_config import DEV_DAYS, INSTRUMENT, PILOT_ID, PROTOCOL_ID, g0bn_artifact_sha256
from eval.g0bn_freeze import (
    HOLDOUT_DAYS,
    SEAL_CONTENT_SCHEMA,
    custodian_seal_content_sha256,
    validate_custody_inventory,
)
from eval.g0bn_identity import G0BN_HOLDOUT_UNIVERSE_ID, G0BN_TRANSACTION_ID
from eval.hashing import hash_obj
from eval.writer import G0BN_DATA_SOURCES
from ingest import download_lake_binance as dl
from ingest import g0bn_acquisition_preflight as pf
from ingest import lake_binance as lb
from tests.g0bn_dev_fixtures import dev_config, dev_source_manifest_sha256
from tests.g0bn_holdout_fixtures import make_objects
from tests.g0bn_protocol_fixtures import make_source_certification, sha_hex

CUSTODIAN = "g0bn-custodian-svc"
OPERATOR = "g0bn-operator-dev"

# Keys that would be January activity proxies if they ever appeared in a public
# preflight artifact (spec section 5.1). Estimated GB values derived from pinned
# April constants x day counts are NOT proxies; measured counts/sizes are.
ACTIVITY_PROXY_KEYS = {
    "rows", "row_count", "row_counts", "n_rows", "record_count", "record_counts",
    "byte_size", "byte_sizes", "src_bytes", "out_bytes", "measured_bytes",
    # Outcome-bearing keys from the spec-5.1 prohibited list. (The packet's
    # vendor_cost section is the subscription dollar model, not a January
    # outcome cost, hence the distinct spelling.)
    "price", "prices", "side", "sides", "return_bps", "returns", "spread",
    "spreads", "label", "labels", "feature", "features", "cost", "costs",
    "cost_bps", "forecast", "forecasts", "metric", "metrics", "pnl",
}


def build_plan(**over):
    over.setdefault("custodian_identity", CUSTODIAN)
    over.setdefault("operator_identity", OPERATOR)
    return pf.build_acquisition_plan(**over)


def resha(artifact: dict) -> dict:
    """Recompute the embedded self-hash after a targeted mutation, so tests hit
    the content validator rather than the tamper check."""
    out = copy.deepcopy(artifact)
    out["sha256"] = g0bn_artifact_sha256(out)
    return out


def make_evidence(**over) -> dict:
    d = {
        "schema": "g0bn-permission-policy-evidence-v1",
        "mechanism": "aws_s3_bucket_policy",
        "custodian_identity": CUSTODIAN,
        "operator_identity": OPERATOR,
        "resource": "s3://g0bn-custody/holdout",
        "covered_paths": [pf.HOLDOUT_RAW_ROOT, pf.HOLDOUT_NORMALIZED_ROOT,
                          pf.HOLDOUT_REPORT_ROOT],
        "custodian_owns_objects": True,
        "operator_payload_read_denied": True,
        "operator_footer_read_denied": True,
        "operator_storage_listing_denied": True,
        "operator_policy_write_denied": True,
        "effective_policy_capture": {
            "method": "aws_s3api_get_bucket_policy",
            "command": "aws s3api get-bucket-policy --bucket g0bn-custody",
            "policy_document_sha256": sha_hex("bucket-policy-doc"),
        },
        "captured_at": "2026-07-19T00:00:00Z",
    }
    d.update(over)
    return d


DEFAULT_EXCLUDED = {
    "2026-01-14": {"reason": "custody_source_gap",
                   "evidence_sha256": sha_hex("jan-gap-2026-01-14")},
    "2026-01-25": {"reason": "custody_one_sided_book",
                   "evidence_sha256": sha_hex("jan-book-2026-01-25")},
}


def build_inventory(**over):
    excluded = over.pop("excluded_days", copy.deepcopy(DEFAULT_EXCLUDED))
    included = [d for d in HOLDOUT_DAYS if d not in excluded]
    kwargs = {
        "custodian_identity": CUSTODIAN,
        "operator_identity": OPERATOR,
        "coverage_sha256": sha_hex("coverage"),
        "evidence": make_evidence(),
        "excluded_days": excluded,
        "objects": make_objects(included),
    }
    kwargs.update(over)
    return pf.build_custody_inventory(**kwargs)


def sealed_config(inventory: dict, evidence: dict) -> dict:
    return dev_config(source_certification=make_source_certification(
        custodian_seal_sha256=inventory["custodian_seal_sha256"],
        permission_policy_sha256=pf.permission_policy_evidence_sha256(evidence),
        development_source_manifest_sha256=dev_source_manifest_sha256(),
    ))


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_keys(item)


# --------------------------------------------------------------- scope and determinism


def test_plan_scope_is_exactly_92_days_and_three_products():
    plan = build_plan()
    assert plan["schema"] == "g0bn-acquisition-plan-v2"
    assert plan["provider"] == "crypto-lake"
    assert plan["fallback_policy"] == "none"
    assert plan["instrument"] == INSTRUMENT
    assert plan["instrument_key"] == "binance-perp"
    assert plan["products"] == ["book", "book_delta_v2", "trades"]
    assert plan["window"] == {"start_day": "2025-11-01", "end_day": "2026-01-31",
                              "n_days": 92}
    assert plan["development"]["days"] == list(DEV_DAYS)
    assert plan["development"]["n_days"] == 61
    assert plan["holdout"]["days"] == list(HOLDOUT_DAYS)
    assert plan["holdout"]["n_days"] == 31
    assert not set(plan["development"]["days"]) & set(plan["holdout"]["days"])
    assert plan["protocol_id"] == PROTOCOL_ID
    assert plan["pilot_id"] == PILOT_ID
    assert plan["holdout_universe_id"] == G0BN_HOLDOUT_UNIVERSE_ID
    assert plan["transaction_id"] == G0BN_TRANSACTION_ID


def test_plan_units_have_deterministic_day_product_ordering():
    plan = build_plan()
    assert plan["n_units"] == 276
    assert plan["development"]["n_units"] == 61 * 3
    assert plan["holdout"]["n_units"] == 31 * 3
    keys = [(u["day"], u["product"]) for u in plan["units"]]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    for unit in plan["units"]:
        root = (plan["development"]["raw_root"] if unit["partition"] == "development"
                else plan["holdout"]["raw_root"])
        assert unit["raw_path"].startswith(root + "/")
        assert unit["raw_path"] == lb.raw_parquet_path(
            root, unit["product"], "BINANCE_FUTURES", "BTC-USDT-PERP", unit["day"])


def test_plan_is_deterministic_and_hash_stable():
    a = build_plan()
    b = build_plan()
    assert a == b
    assert a["sha256"] == g0bn_artifact_sha256(a)
    assert pf.validate_acquisition_plan(a) is a


def test_plan_units_reconcile_with_downloader_planner():
    plan = build_plan()
    all_days = plan["development"]["days"] + plan["holdout"]["days"]
    expected = dl.plan_units(["binance-perp"], "book_delta_v2,trades", all_days)
    assert {(u.feed, u.day) for u in expected} == \
        {(u["product"], u["day"]) for u in plan["units"]}
    assert len(expected) == plan["n_units"]
    for u in expected:
        assert (u.exchange, u.symbol) == ("BINANCE_FUTURES", "BTC-USDT-PERP")


def test_plan_estimates_reconcile_with_downloader_estimator():
    plan = build_plan()
    est = plan["estimates"]
    per = lb.LAKE_GB_PER_DAY[("BINANCE_FUTURES", "BTC-USDT-PERP")]
    assert est["gb_per_day_by_product"] == {p: per[p] for p in plan["products"]}
    assert est["gb_per_day"] == lb.estimate_gb("binance-perp",
                                               ["book_delta_v2", "trades"], 1)
    assert est["total_gb"] == lb.estimate_gb("binance-perp",
                                             ["book_delta_v2", "trades"], 92)
    assert plan["development"]["estimated_gb"] == lb.estimate_gb(
        "binance-perp", ["book_delta_v2", "trades"], 61)
    assert plan["holdout"]["estimated_gb"] == lb.estimate_gb(
        "binance-perp", ["book_delta_v2", "trades"], 31)


def test_plan_execution_pins_match_downloader_constants():
    ex = build_plan()["execution"]
    assert ex["retries"] == dl.DEFAULT_RETRIES
    assert ex["backoff_base_s"] == dl.DEFAULT_BACKOFF_BASE_S
    assert ex["backoff_cap_s"] == dl.DEFAULT_BACKOFF_CAP_S
    assert ex["expensive_compute_lock"] == "/tmp/jepa-expensive-compute.lock"
    assert ex["download_jobs"] == ex["download_jobs_max"] == dl.MAX_DOWNLOAD_JOBS == 4
    assert ex["recon_jobs"] == ex["recon_jobs_max"] == 1
    assert "row-group pre-buffer" in ex["concurrency_rationale"]
    assert "memory-heavy" in ex["concurrency_rationale"]
    assert "guaranteed" in ex["concurrency_rationale"]


@pytest.mark.parametrize(
    ("field", "value", "drop"),
    [
        ("download_jobs", True, False),       # bool is an int subclass in Python
        ("download_jobs", None, True),        # missing immutable input
        ("download_jobs", 5, False),          # above the Stage-1 ceiling
        ("download_jobs", 3, False),          # within bounds but drifted from approval
        ("download_jobs_max", True, False),
        ("download_jobs_max", None, True),
        ("download_jobs_max", 5, False),
        ("download_jobs_max", 3, False),
        ("recon_jobs", True, False),
        ("recon_jobs", None, True),
        ("recon_jobs", 2, False),             # above the memory-safe Stage-2 ceiling
        ("recon_jobs_max", True, False),
        ("recon_jobs_max", None, True),
        ("recon_jobs_max", 2, False),
    ],
)
def test_plan_validator_rejects_invalid_or_drifted_stage_concurrency(field, value, drop):
    plan = copy.deepcopy(build_plan())
    if drop:
        plan["execution"].pop(field)
    else:
        plan["execution"][field] = value
    with pytest.raises(ValueError, match=field):
        pf.validate_acquisition_plan(resha(plan))


def test_plan_quota_fits_a_single_monthly_window():
    q = build_plan()["quota"]
    assert q["quota_gb_per_month"] == lb.QUOTA_GB
    assert q["headroom_gb"] == lb.DEFAULT_HEADROOM_GB
    assert q["fits_single_window"] is True
    assert build_plan()["estimates"]["total_gb"] <= lb.QUOTA_GB - lb.DEFAULT_HEADROOM_GB


# --------------------------------------------------------------------- destinations


def test_plan_destinations_are_disjoint_and_ignored():
    plan = build_plan()
    roots = [plan["development"]["raw_root"], plan["holdout"]["raw_root"],
             plan["development"]["normalized_root"], plan["holdout"]["normalized_root"]]
    assert len(set(roots)) == 4
    for a in roots:
        for b in roots:
            if a != b:
                assert not a.startswith(b + "/")
    assert plan["development"]["raw_root"].startswith("data/raw/")
    assert plan["holdout"]["raw_root"].startswith("data/raw/")
    assert plan["development"]["normalized_root"].startswith("data/processed/")
    assert plan["holdout"]["normalized_root"].startswith("data/processed/")


def test_build_rejects_shared_nested_or_unsafe_roots():
    with pytest.raises(ValueError, match="disjoint"):
        build_plan(holdout_raw_root=pf.DEV_RAW_ROOT)
    with pytest.raises(ValueError, match="disjoint"):
        build_plan(holdout_raw_root=pf.DEV_RAW_ROOT + "/holdout")
    with pytest.raises(ValueError, match="data/raw"):
        build_plan(dev_raw_root="/abs/elsewhere")
    with pytest.raises(ValueError, match="data/raw"):
        build_plan(dev_raw_root="data/raw/../secrets")
    with pytest.raises(ValueError, match="data/processed"):
        build_plan(dev_normalized_root="data/raw/normalized_in_raw")
    # Roots land unquoted in approved shell commands: shell metacharacters and
    # whitespace must fail the segment-token allowlist (Codex P2 on PR #103).
    for evil in ("data/raw/lake;touch${IFS}x", "data/raw/lake`id`",
                 "data/raw/lake dev", "data/raw/$(whoami)"):
        with pytest.raises(ValueError, match="shell-inert"):
            build_plan(dev_raw_root=evil)


# ------------------------------------------------------------- fail-closed validation


def test_validator_rejects_extra_or_missing_days():
    plan = build_plan()
    extra = copy.deepcopy(plan)
    extra["holdout"]["days"] = extra["holdout"]["days"] + ["2026-02-01"]
    with pytest.raises(ValueError):
        pf.validate_acquisition_plan(resha(extra))
    missing = copy.deepcopy(plan)
    missing["development"]["days"] = missing["development"]["days"][:-1]
    with pytest.raises(ValueError):
        pf.validate_acquisition_plan(resha(missing))


def test_validator_rejects_extra_products_and_foreign_provider():
    plan = build_plan()
    prod = copy.deepcopy(plan)
    prod["products"] = prod["products"] + ["funding"]
    with pytest.raises(ValueError, match="products"):
        pf.validate_acquisition_plan(resha(prod))
    prov = copy.deepcopy(plan)
    prov["provider"] = "coinapi"
    with pytest.raises(ValueError, match="crypto-lake"):
        pf.validate_acquisition_plan(resha(prov))
    fb = copy.deepcopy(plan)
    fb["fallback_policy"] = "coinapi_rest"
    with pytest.raises(ValueError, match="none"):
        pf.validate_acquisition_plan(resha(fb))


def test_validator_rejects_unknown_fields_and_activity_proxies():
    plan = build_plan()
    top = copy.deepcopy(plan)
    top["holdout_row_count"] = 5
    with pytest.raises(ValueError, match="unknown"):
        pf.validate_acquisition_plan(resha(top))
    nested = copy.deepcopy(plan)
    nested["estimates"]["measured_january_gb"] = 1.0
    with pytest.raises(ValueError, match="unknown"):
        pf.validate_acquisition_plan(resha(nested))


def test_validator_rejects_tampered_hash_and_estimate_drift():
    plan = build_plan()
    # Content checks run first (targeted errors), the self-hash is verified
    # LAST — so a stale hash on otherwise-valid content fails on "sha256"...
    stale = dict(copy.deepcopy(plan), sha256=sha_hex("stale"))
    with pytest.raises(ValueError, match="sha256"):
        pf.validate_acquisition_plan(stale)
    # ...and a rehashed measured-value edit fails on the exact recomputation.
    drift = copy.deepcopy(plan)
    drift["estimates"]["total_gb"] += 1.0
    with pytest.raises(ValueError, match="total_gb"):
        pf.validate_acquisition_plan(resha(drift))


def test_validator_rejects_units_tamper():
    plan = build_plan()
    cut = copy.deepcopy(plan)
    cut["units"] = cut["units"][:-1]
    cut["n_units"] = len(cut["units"])
    cut["holdout"]["n_units"] -= 1
    with pytest.raises(ValueError):
        pf.validate_acquisition_plan(resha(cut))
    # Count-preserving single-unit tampers are rejected ONLY by the exact
    # units rebuild — pin each escape route separately.
    redirected = copy.deepcopy(plan)
    redirected["units"][5] = dict(redirected["units"][5],
                                  raw_path="data/raw/elsewhere/data.parquet")
    with pytest.raises(ValueError, match="raw_path"):
        pf.validate_acquisition_plan(resha(redirected))
    flipped = copy.deepcopy(plan)
    flipped["units"][0] = dict(flipped["units"][0], partition="holdout")
    with pytest.raises(ValueError, match="partition"):
        pf.validate_acquisition_plan(resha(flipped))
    swapped = copy.deepcopy(plan)
    swapped["units"][0], swapped["units"][1] = \
        swapped["units"][1], swapped["units"][0]
    with pytest.raises(ValueError):
        pf.validate_acquisition_plan(resha(swapped))


def test_validator_rejects_identity_drift():
    plan = build_plan()
    inst = copy.deepcopy(plan)
    inst["instrument"] = dict(inst["instrument"], symbol="BTC-USD")
    with pytest.raises(ValueError):
        pf.validate_acquisition_plan(resha(inst))
    uid = copy.deepcopy(plan)
    uid["holdout_universe_id"] = sha_hex("foreign-universe")
    with pytest.raises(ValueError):
        pf.validate_acquisition_plan(resha(uid))


def test_build_rejects_equal_or_placeholder_identities():
    with pytest.raises(ValueError, match="distinct"):
        build_plan(operator_identity=CUSTODIAN)
    with pytest.raises(ValueError, match="placeholder"):
        build_plan(custodian_identity="TBD")
    with pytest.raises(ValueError, match="shell-inert"):
        build_plan(custodian_identity="ops team")


def test_validator_pins_days_files_and_report_roots():
    plan = build_plan()
    # The days-file CONTENT is what a later execution would actually transfer,
    # so a rehashed plan redirecting it (or aliasing dev/holdout onto one file)
    # must fail the machine gate, not just the human regenerate-and-compare.
    redirect = copy.deepcopy(plan)
    redirect["development"]["days_file"] = "data/reports/evil/days.txt"
    with pytest.raises(ValueError, match="days_file"):
        pf.validate_acquisition_plan(resha(redirect))
    shared = copy.deepcopy(plan)
    shared["development"]["days_file"] = shared["holdout"]["days_file"]
    with pytest.raises(ValueError, match="days_file"):
        pf.validate_acquisition_plan(resha(shared))
    report = copy.deepcopy(plan)
    report["holdout"]["report_root"] = "data/reports/elsewhere"
    with pytest.raises(ValueError, match="report_root"):
        pf.validate_acquisition_plan(resha(report))


def test_days_file_content_commitment_matches_written_files(tmp_path,
                                                            monkeypatch):
    plan = build_plan()
    monkeypatch.setattr(pf, "_free_gb", lambda path: 100000.0)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    assert run_cli(["plan", "--custodian-identity", CUSTODIAN,
                    "--operator-identity", OPERATOR,
                    "--out-dir", str(out)]) == 0
    import hashlib
    for name, block in (("dev_days.txt", plan["development"]),
                        ("holdout_days.txt", plan["holdout"])):
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert digest == block["days_file_sha256"]
        # A custom --out-dir run must ALSO refresh the plan-pinned paths the
        # packet commands reference, so they can never be missing or stale
        # (Codex P2 on PR #103).
        pinned = tmp_path / block["days_file"]
        assert hashlib.sha256(pinned.read_bytes()).hexdigest() == \
            block["days_file_sha256"]
    tampered = copy.deepcopy(plan)
    tampered["holdout"]["days_file_sha256"] = sha_hex("tampered-days")
    with pytest.raises(ValueError, match="days_file_sha256"):
        pf.validate_acquisition_plan(resha(tampered))


def test_custody_scope_covers_all_holdout_destinations():
    plan = build_plan()
    scope = plan["custody"]["holdout_custody_scope"]
    assert scope == [plan["holdout"]["raw_root"],
                     plan["holdout"]["normalized_root"],
                     plan["holdout"]["report_root"]]
    gutted = copy.deepcopy(plan)
    gutted["custody"]["holdout_custody_scope"] = scope[:2]
    with pytest.raises(ValueError, match="holdout_custody_scope"):
        pf.validate_acquisition_plan(resha(gutted))


def test_validator_pins_custody_disclosures():
    plan = build_plan()
    gutted = copy.deepcopy(plan)
    gutted["custody"]["unresolved_assumptions"] = ["nothing to see here"]
    with pytest.raises(ValueError, match="unresolved_assumptions"):
        pf.validate_acquisition_plan(resha(gutted))
    basis = copy.deepcopy(plan)
    basis["estimates"]["basis"] = "measured January bytes"
    with pytest.raises(ValueError, match="basis"):
        pf.validate_acquisition_plan(resha(basis))


# ---------------------------------------------------------------- approval packet


def test_packet_builds_and_validates_against_plan():
    plan = build_plan()
    packet = pf.build_approval_packet(plan)
    assert packet["schema"] == "g0bn-acquisition-approval-packet-v2"
    assert packet["inert"] is True
    assert packet["plan_sha256"] == plan["sha256"]
    assert packet["no_vendor_io_statement"] == pf.NO_VENDOR_IO_STATEMENT
    assert packet["sha256"] == g0bn_artifact_sha256(packet)
    assert pf.validate_approval_packet(packet, plan) is packet


def test_packet_commands_are_exact_inert_and_lock_serialized():
    plan = build_plan()
    packet = pf.build_approval_packet(plan)
    commands = packet["commands"]
    by_step = {c["step"]: c for c in commands}
    assert list(by_step) == ["regenerate-preflight", "development-download",
                             "development-recon", "holdout-download",
                             "holdout-recon", "custody-seal",
                             "custody-seal-verify"]
    # The seal is minted WITHOUT --config (the config cannot pre-exist its own
    # seal pin), then the mandatory verify step reruns WITH --config.
    assert "--config" not in by_step["custody-seal"]["command"]
    assert "--config" in by_step["custody-seal-verify"]["command"]
    for c in commands:
        assert c["approval_required"] is True
    for step in ("development-download", "development-recon",
                 "holdout-download", "holdout-recon"):
        cmd = by_step[step]["command"]
        assert cmd.startswith("flock -w 14400 /tmp/jepa-expensive-compute.lock ")
        # run_as is not mere metadata: every download/recon command chains the
        # offline verify-execution gate (plan + days-file commitment +
        # holdout custodian identity) before the real work (Codex deep P1s).
        partition = "holdout" if step.startswith("holdout") else "development"
        assert f"verify-execution --partition {partition}" in cmd
        stage = "download" if step.endswith("download") else "recon"
        jobs = 4 if stage == "download" else 1
        prefix = "flock -w 14400 /tmp/jepa-expensive-compute.lock sh -c \""
        verify, work = cmd[len(prefix):-1].split(" && ")
        assert verify == (
            ".venv/bin/python -m ingest.g0bn_acquisition_preflight "
            f"verify-execution --partition {partition} --stage {stage} --jobs {jobs} "
            "--plan data/reports/g0bn_acquisition_preflight/g0bn_acquisition_plan.json"
        )
        assert work.endswith(f"--jobs {jobs} --resume")
        assert cmd.count(f"--jobs {jobs}") == 2
        assert cmd.index("verify-execution") < cmd.index("--days-file")
        assert " && " in cmd
    dev_dl = by_step["development-download"]["command"]
    assert "--instrument binance-perp" in dev_dl
    assert "--feeds book_delta_v2,trades" in dev_dl
    assert f"--days-file {plan['development']['days_file']}" in dev_dl
    assert f"--out {plan['development']['raw_root']}" in dev_dl
    assert "--max-gb" in dev_dl
    assert dev_dl.count("--jobs 4") == 2 and "--stage download" in dev_dl
    hold_dl = by_step["holdout-download"]["command"]
    assert f"--out {plan['holdout']['raw_root']}" in hold_dl
    assert by_step["holdout-download"]["run_as"] == "custodian"
    assert by_step["holdout-recon"]["run_as"] == "custodian"
    assert by_step["development-download"]["run_as"] == "operator"
    # Executable command strings only (notes may legitimately NAME a forbidden
    # flag while explaining its absence). --allow-broad would bypass the
    # est_gb > --max-gb refusal in lake_binance.check_broad_gate, so no packet
    # command may carry it: the approved cap is expressed by raising --max-gb
    # (Codex P1 on PR #103).
    cmd_blob = " ".join(c["command"] for c in commands)
    for forbidden in ("funding", "open_interest", "liquidations", "binance-spot",
                      "--start", "--end", "--overwrite", "--allow-broad"):
        assert forbidden not in cmd_blob


def test_packet_caps_cover_estimates_within_quota():
    plan = build_plan()
    caps = pf.build_approval_packet(plan)["caps"]
    assert caps["development_max_gb"] == float(
        math.ceil(plan["development"]["estimated_gb"] * 1.1))
    assert caps["holdout_max_gb"] == float(
        math.ceil(plan["holdout"]["estimated_gb"] * 1.1))
    assert caps["development_max_gb"] >= plan["development"]["estimated_gb"]
    assert caps["holdout_max_gb"] >= plan["holdout"]["estimated_gb"]
    assert caps["development_max_gb"] + caps["holdout_max_gb"] <= \
        lb.QUOTA_GB - lb.DEFAULT_HEADROOM_GB
    cost = pf.build_approval_packet(plan)["vendor_cost"]
    assert caps["download_jobs"] == caps["download_jobs_max"] == 4
    assert caps["recon_jobs"] == caps["recon_jobs_max"] == 1
    assert caps["concurrency_rationale"] == plan["execution"]["concurrency_rationale"]
    assert cost["incremental_usd_within_quota"] == 0.0
    assert cost["quota_consumed_gb_estimate"] == plan["estimates"]["total_gb"]


def test_packet_rejects_tampering_and_unknown_fields():
    plan = build_plan()
    packet = pf.build_approval_packet(plan)
    bent = copy.deepcopy(packet)
    bent["commands"][1]["command"] = bent["commands"][1]["command"].replace(
        "binance-perp", "binance-spot")
    with pytest.raises(ValueError):
        pf.validate_approval_packet(resha(bent), plan)
    extra = copy.deepcopy(packet)
    extra["january_rows"] = 1
    with pytest.raises(ValueError):
        pf.validate_approval_packet(resha(extra), plan)
    with pytest.raises(ValueError, match="sha256"):
        pf.validate_approval_packet(dict(packet, inert=True, sha256=sha_hex("x")),
                                    plan)


@pytest.mark.parametrize(
    ("field", "value", "drop"),
    [
        ("download_jobs", True, False),
        ("download_jobs", None, True),
        ("download_jobs", 5, False),
        ("download_jobs", 3, False),
        ("download_jobs_max", True, False),
        ("download_jobs_max", None, True),
        ("download_jobs_max", 5, False),
        ("download_jobs_max", 3, False),
        ("recon_jobs", True, False),
        ("recon_jobs", None, True),
        ("recon_jobs", 2, False),
        ("recon_jobs_max", True, False),
        ("recon_jobs_max", None, True),
        ("recon_jobs_max", 2, False),
    ],
)
def test_packet_rejects_invalid_or_drifted_stage_concurrency(field, value, drop):
    plan = build_plan()
    packet = copy.deepcopy(pf.build_approval_packet(plan))
    if drop:
        packet["caps"].pop(field)
    else:
        packet["caps"][field] = value
    with pytest.raises(ValueError, match=field):
        pf.validate_approval_packet(resha(packet), plan)


def test_packet_pins_serial_runtime_assumptions_and_supersedes_v1_approval():
    packet = pf.build_approval_packet(build_plan())
    runtime = packet["runtime_assumptions"]
    assert runtime["certified_serial_day"] == "2026-04-01"
    assert runtime["certified_serial_seconds_by_product"] == {
        "book": 242.671,
        "book_delta_v2": 1441.344,
        "trades": 24.936,
    }
    assert runtime["serial_minutes_per_day_approx"] == 28.5
    assert runtime["development_days"] == 61
    assert runtime["development_serial_hours_approx"] == 29.0
    assert runtime["idealized_development_runtime_range_hours"] == [7.2, 29.0]
    assert "threaded lakeapi" in runtime["comparison"]
    assert "persistent normalized raw store" in runtime["comparison"]
    assert "no 4x live scaling is guaranteed" in runtime["caveat"]

    prior = packet["superseded_operational_evidence"]
    assert prior["prior_plan_sha256"] == pf.PRIOR_PLAN_SHA256
    assert prior["prior_packet_sha256"] == pf.PRIOR_PACKET_SHA256
    assert prior["status"] == "superseded_operational_evidence_only"
    assert "regenerate" in prior["required_action"]
    assert "fresh explicit human approval" in prior["required_action"]
    assert "must not be reused" in prior["required_action"]


def test_packet_stop_conditions_and_custody_section():
    plan = build_plan()
    packet = pf.build_approval_packet(plan)
    stops = "\n".join(packet["stop_conditions"])
    for needle in ("exit 2", "exit 3", "exit 4", "quota", "disk",
                   "schema drift", "fallback", "superseded", "fresh explicit"):
        assert needle in stops
    custody = packet["custody"]
    assert custody["custodian_identity"] == CUSTODIAN
    assert custody["operator_identity"] == OPERATOR
    assert custody["seal_content_schema"] == SEAL_CONTENT_SCHEMA
    assert custody["permission_evidence_schema"] == pf.PERMISSION_EVIDENCE_SCHEMA
    assert custody["unresolved_assumptions"]
    assert "#93" in "\n".join(custody["unresolved_assumptions"])


def test_packet_markdown_renders_every_command():
    plan = build_plan()
    packet = pf.build_approval_packet(plan)
    md = pf.render_approval_packet(packet)
    assert "INERT" in md
    assert plan["sha256"] in md
    for c in packet["commands"]:
        assert c["command"] in md
    assert pf.NO_VENDOR_IO_STATEMENT in md
    assert "FRESH APPROVAL REQUIRED" in md
    assert "1441.344s" in md and "242.671s" in md and "24.936s" in md
    assert "28.5 minutes/day" in md and "29.0 serial hours" in md
    assert "threaded lakeapi" in md and "persistent normalized raw store" in md
    assert "no 4x live scaling is guaranteed" in md
    assert pf.PRIOR_PLAN_SHA256 in md and pf.PRIOR_PACKET_SHA256 in md


# ------------------------------------------------------- permission-policy evidence


def test_evidence_valid_and_hash_deterministic():
    ev = make_evidence()
    assert pf.validate_permission_policy_evidence(ev) is ev
    assert pf.permission_policy_evidence_sha256(ev) == hash_obj(ev)


def test_evidence_rejects_local_chmod_style_mechanisms():
    for mechanism in ("chmod", "posix_mode_bits", "umask", "developer_copy"):
        with pytest.raises(ValueError, match="custody"):
            pf.validate_permission_policy_evidence(make_evidence(mechanism=mechanism))
    with pytest.raises(ValueError, match="mechanism"):
        pf.validate_permission_policy_evidence(make_evidence(mechanism="handshake"))


def test_evidence_rejects_identity_and_denial_failures():
    with pytest.raises(ValueError, match="distinct"):
        pf.validate_permission_policy_evidence(
            make_evidence(operator_identity=CUSTODIAN))
    with pytest.raises(ValueError, match="placeholder"):
        pf.validate_permission_policy_evidence(
            make_evidence(custodian_identity="UNRESOLVED"))
    for flag in ("custodian_owns_objects", "operator_payload_read_denied",
                 "operator_footer_read_denied", "operator_storage_listing_denied",
                 "operator_policy_write_denied"):
        with pytest.raises(ValueError):
            pf.validate_permission_policy_evidence(make_evidence(**{flag: False}))
    # A policy the operator can rewrite cannot bind effective access.
    with pytest.raises(ValueError, match="regrant"):
        pf.validate_permission_policy_evidence(
            make_evidence(operator_policy_write_denied=False))
    # Storage listing exposes per-object January byte sizes (activity proxies),
    # so the denial is required with a spec-citing message.
    with pytest.raises(ValueError, match="activity prox"):
        pf.validate_permission_policy_evidence(
            make_evidence(operator_storage_listing_denied=False))
    # Identities land verbatim in approved shell commands: shell-unsafe tokens
    # are rejected at the evidence layer too.
    with pytest.raises(ValueError, match="shell-inert"):
        pf.validate_permission_policy_evidence(
            make_evidence(custodian_identity="ops team; rm -rf"))


def test_evidence_binds_capture_to_mechanism():
    # A chmod-flavored capture cannot hide under an approved mechanism name
    # (Codex P2 on PR #103): methods are pinned per mechanism...
    bad_method = make_evidence()
    bad_method["effective_policy_capture"] = dict(
        bad_method["effective_policy_capture"], method="chmod")
    with pytest.raises(ValueError, match="capture method"):
        pf.validate_permission_policy_evidence(bad_method)
    mismatched = make_evidence()
    mismatched["effective_policy_capture"] = dict(
        mismatched["effective_policy_capture"], method="getfacl")
    with pytest.raises(ValueError, match="capture method"):
        pf.validate_permission_policy_evidence(mismatched)
    # ...and the captured command may not contain local mutation spellings.
    bad_command = make_evidence()
    bad_command["effective_policy_capture"] = dict(
        bad_command["effective_policy_capture"],
        command="chmod 600 /data && aws s3api get-bucket-policy")
    with pytest.raises(ValueError, match="mutation"):
        pf.validate_permission_policy_evidence(bad_command)


def test_evidence_rejects_malformed_shape():
    with pytest.raises(ValueError, match="unknown"):
        pf.validate_permission_policy_evidence(make_evidence(surprise=1))
    incomplete = make_evidence()
    del incomplete["effective_policy_capture"]
    with pytest.raises(ValueError, match="missing"):
        pf.validate_permission_policy_evidence(incomplete)
    with pytest.raises(ValueError):
        pf.validate_permission_policy_evidence(make_evidence(captured_at="whenever"))


# ------------------------------------------------------------- custody inventory


def test_built_inventory_matches_freeze_consumer_contract_exactly():
    ev = make_evidence()
    inv = build_inventory(evidence=ev)
    assert set(inv) == {"custodian_seal_sha256", "coverage_sha256",
                        "permission_policy_sha256", "custodian_identity",
                        "operator_identity", "included_days", "excluded_days",
                        "objects"}
    config = sealed_config(inv, ev)
    assert validate_custody_inventory(inv, config) is inv
    assert pf.validate_custody_handoff(inv, ev, config) is inv


def test_built_inventory_seal_follows_the_v1_recipe():
    inv = build_inventory()
    assert inv["custodian_seal_sha256"] == custodian_seal_content_sha256(inv)
    hand_computed = hash_obj({
        "schema": "g0bn-custodian-seal-content-v1",
        "custodian_identity": inv["custodian_identity"],
        "operator_identity": inv["operator_identity"],
        "coverage_sha256": inv["coverage_sha256"],
        "permission_policy_sha256": inv["permission_policy_sha256"],
        "included_days": list(inv["included_days"]),
        "excluded_days": inv["excluded_days"],
        "objects": sorted(inv["objects"],
                          key=lambda o: (o["day"], o["layer"], o["product"],
                                         o["object_id"])),
    })
    assert inv["custodian_seal_sha256"] == hand_computed


def test_built_inventory_canonicalizes_object_order():
    excluded = copy.deepcopy(DEFAULT_EXCLUDED)
    included = [d for d in HOLDOUT_DAYS if d not in excluded]
    objects = make_objects(included)
    shuffled = list(reversed(objects))
    inv = build_inventory(objects=shuffled)
    keys = [(o["day"], o["layer"], o["product"], o["object_id"])
            for o in inv["objects"]]
    assert keys == sorted(keys)


def test_builder_rejects_activity_proxies_and_mismatches():
    excluded = copy.deepcopy(DEFAULT_EXCLUDED)
    included = [d for d in HOLDOUT_DAYS if d not in excluded]
    objects = make_objects(included)
    objects[0] = dict(objects[0], byte_size=123)
    with pytest.raises(ValueError, match="unknown"):
        build_inventory(objects=objects)
    with pytest.raises(ValueError, match="identity"):
        build_inventory(evidence=make_evidence(
            custodian_identity="someone-else",))
    with pytest.raises(ValueError, match="holdout"):
        build_inventory(excluded_days={
            "2026-02-01": {"reason": "custody_source_gap",
                           "evidence_sha256": sha_hex("x")}})
    # Free-text exclusion reasons could publish outcome prose in the sealed
    # inventory; only allowlisted outcome-blind codes pass (Codex P2, PR #103).
    with pytest.raises(ValueError, match="outcome-blind"):
        build_inventory(excluded_days={
            "2026-01-14": {"reason": "price crashed 5%",
                           "evidence_sha256": sha_hex("x")}})
    # Malformed object VALUES fail closed before the canonical sort — a
    # config-less mint must never seal a garbage body (typed like the consumer).
    mixed = make_objects(included)
    mixed[0] = dict(mixed[0], day=1)
    with pytest.raises(ValueError, match="day"):
        build_inventory(objects=mixed)
    badsha = make_objects(included)
    badsha[0] = dict(badsha[0], sha256="not-hex")
    with pytest.raises(ValueError, match="hex"):
        build_inventory(objects=badsha)
    badlayer = make_objects(included)
    badlayer[0] = dict(badlayer[0], layer="sideways")
    with pytest.raises(ValueError, match="layer"):
        build_inventory(objects=badlayer)
    # object_id is a pure function of (layer, product, day): verbose
    # custodian-chosen store keys could smuggle activity metadata into the
    # public inventory (Codex deep P2).
    verbose = make_objects(included)
    verbose[0] = dict(verbose[0],
                      object_id="s3://bucket/book/2026-01-01/rows=12345")
    with pytest.raises(ValueError, match="canonical derived id"):
        build_inventory(objects=verbose)


def test_handoff_rejects_foreign_evidence_and_forged_seal():
    ev = make_evidence()
    inv = build_inventory(evidence=ev)
    config = sealed_config(inv, ev)
    other = make_evidence(resource="s3://other-bucket/holdout")
    with pytest.raises(ValueError, match="permission_policy"):
        pf.validate_custody_handoff(inv, other, config)
    forged = copy.deepcopy(inv)
    forged["objects"][0]["sha256"] = sha_hex("forged-object")
    with pytest.raises(ValueError, match="custodian seal|checksum"):
        pf.validate_custody_handoff(forged, ev, config)


def test_handoff_binds_evidence_coverage_to_plan_custody_scope():
    plan = build_plan()
    ev = make_evidence()
    inv = build_inventory(evidence=ev)
    config = sealed_config(inv, ev)
    # Full coverage of custody.holdout_custody_scope passes with the plan...
    assert pf.validate_custody_handoff(inv, ev, config, plan=plan) is inv
    # ...but evidence captured for only a subset of the planned destinations
    # (e.g. missing the report root, whose run records carry January
    # rows/bytes) cannot vouch (Codex P2 on PR #103).
    partial = make_evidence(covered_paths=[pf.HOLDOUT_RAW_ROOT,
                                           pf.HOLDOUT_NORMALIZED_ROOT])
    inv_partial = build_inventory(evidence=partial)
    config_partial = sealed_config(inv_partial, partial)
    with pytest.raises(ValueError, match="holdout custody scope"):
        pf.validate_custody_handoff(inv_partial, partial, config_partial,
                                    plan=plan)
    # Evidence must always carry a coverage attestation at all.
    incomplete = make_evidence()
    del incomplete["covered_paths"]
    with pytest.raises(ValueError, match="missing"):
        pf.validate_permission_policy_evidence(incomplete)


def test_consumer_rejects_insufficient_included_days():
    excluded = {
        day: {"reason": "custody_source_gap", "evidence_sha256": sha_hex(day)}
        for day in HOLDOUT_DAYS[:15]
    }
    ev = make_evidence()
    inv = build_inventory(evidence=ev, excluded_days=excluded)
    config = sealed_config(inv, ev)
    with pytest.raises(ValueError, match="min_valid_days"):
        pf.validate_custody_handoff(inv, ev, config)


def test_consumer_rejects_proxy_fields_via_exact_object_shape():
    ev = make_evidence()
    inv = build_inventory(evidence=ev)
    config = sealed_config(inv, ev)
    poisoned = copy.deepcopy(inv)
    poisoned["objects"][0]["record_count"] = 10
    with pytest.raises(ValueError, match="unknown"):
        validate_custody_inventory(poisoned, config)


def test_public_artifacts_carry_no_activity_proxy_keys():
    plan = build_plan()
    packet = pf.build_approval_packet(plan)
    inv = build_inventory()
    for artifact in (plan, packet, inv):
        assert not (set(walk_keys(artifact)) & ACTIVITY_PROXY_KEYS)


# ----------------------------------------------------------------------------- CLI


def run_cli(args):
    return pf.main(args)


def test_cli_plan_writes_deterministic_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_free_gb", lambda path: 100000.0)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        assert run_cli(["plan", "--custodian-identity", CUSTODIAN,
                        "--operator-identity", OPERATOR,
                        "--out-dir", str(out)]) == 0
    names = ["g0bn_acquisition_plan.json", "g0bn_approval_packet.json",
             "g0bn_approval_packet.md", "dev_days.txt", "holdout_days.txt"]
    for name in names:
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    plan = json.loads((out_a / "g0bn_acquisition_plan.json").read_text())
    pf.validate_acquisition_plan(plan)
    packet = json.loads((out_a / "g0bn_approval_packet.json").read_text())
    pf.validate_approval_packet(packet, plan)
    assert dl.load_days_file(str(out_a / "dev_days.txt")) == list(DEV_DAYS)
    assert dl.load_days_file(str(out_a / "holdout_days.txt")) == list(HOLDOUT_DAYS)


def test_cli_plan_rejects_bad_identities(tmp_path, capsys):
    rc = run_cli(["plan", "--custodian-identity", CUSTODIAN,
                  "--operator-identity", CUSTODIAN,
                  "--out-dir", str(tmp_path / "out")])
    assert rc == 2
    assert "distinct" in capsys.readouterr().err


def test_cli_refuses_unsafe_in_repo_out_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pf, "_free_gb", lambda path: 100000.0)
    # Simulate invoking from a checkout root of THIS project (.git + AGENTS.md
    # are the guard's project markers).
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("marker\n")
    monkeypatch.chdir(tmp_path)
    rc = run_cli(["plan", "--custodian-identity", CUSTODIAN,
                  "--operator-identity", OPERATOR,
                  "--out-dir", "somewhere/in/repo"])
    assert rc == 2
    assert "--out-dir" in capsys.readouterr().err
    assert run_cli(["plan", "--custodian-identity", CUSTODIAN,
                    "--operator-identity", OPERATOR,
                    "--out-dir", "data/reports/g0bn_acquisition_preflight"]) == 0
    # The guard is anchored on the module's repo root too, so pointing an
    # absolute out-dir at a tracked path of the real checkout from an
    # unrelated cwd is also refused.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(pf.__file__)))
    rc = run_cli(["plan", "--custodian-identity", CUSTODIAN,
                  "--operator-identity", OPERATOR,
                  "--out-dir", os.path.join(repo_root, "eval", "generated")])
    assert rc == 2


def test_cli_refuses_sibling_checkout_out_dir(tmp_path, monkeypatch):
    # A DIFFERENT checkout/worktree of the repo (neither the module's own root
    # nor the cwd) is detected via its .git entry and protected the same way.
    monkeypatch.setattr(pf, "_free_gb", lambda path: 100000.0)
    sibling = tmp_path / "sibling-checkout"
    (sibling / ".git").mkdir(parents=True)
    (sibling / "AGENTS.md").write_text("marker\n")
    assert run_cli(["plan", "--custodian-identity", CUSTODIAN,
                    "--operator-identity", OPERATOR,
                    "--out-dir", str(sibling / "eval" / "generated")]) == 2
    assert run_cli(["plan", "--custodian-identity", CUSTODIAN,
                    "--operator-identity", OPERATOR,
                    "--out-dir", str(sibling / "data" / "reports" / "x")]) == 0


def test_cli_disk_gate_refuses_insufficient_free_space(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.setattr(pf, "_free_gb", lambda path: 1.0)
    rc = run_cli(["plan", "--custodian-identity", CUSTODIAN,
                  "--operator-identity", OPERATOR,
                  "--out-dir", str(tmp_path / "out")])
    assert rc == 4
    assert "PREFLIGHT GATE" in capsys.readouterr().err
    # Artifacts are still written (the report is evidence of the refusal).
    assert (tmp_path / "out" / "g0bn_acquisition_plan.json").exists()


def test_cli_build_inventory_roundtrip(tmp_path):
    ev = make_evidence()
    excluded = copy.deepcopy(DEFAULT_EXCLUDED)
    included = [d for d in HOLDOUT_DAYS if d not in excluded]
    objects = make_objects(included)
    inv = pf.build_custody_inventory(
        custodian_identity=CUSTODIAN, operator_identity=OPERATOR,
        coverage_sha256=sha_hex("coverage"), evidence=ev,
        excluded_days=excluded, objects=objects)
    config = sealed_config(inv, ev)
    paths = {}
    for name, payload in [("evidence.json", ev), ("excluded.json", excluded),
                          ("objects.json", objects), ("config.json", config)]:
        p = tmp_path / name
        p.write_text(json.dumps(payload))
        paths[name] = str(p)
    out = tmp_path / "inventory.json"
    rc = run_cli(["build-inventory",
                  "--custodian-identity", CUSTODIAN,
                  "--operator-identity", OPERATOR,
                  "--coverage-sha256", sha_hex("coverage"),
                  "--evidence", paths["evidence.json"],
                  "--excluded-days", paths["excluded.json"],
                  "--objects", paths["objects.json"],
                  "--config", paths["config.json"],
                  "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text()) == inv


def test_cli_build_inventory_fails_closed(tmp_path, capsys):
    ev = make_evidence(mechanism="chmod")
    excluded = copy.deepcopy(DEFAULT_EXCLUDED)
    included = [d for d in HOLDOUT_DAYS if d not in excluded]
    for name, payload in [("evidence.json", ev), ("excluded.json", excluded),
                          ("objects.json", make_objects(included))]:
        (tmp_path / name).write_text(json.dumps(payload))
    rc = run_cli(["build-inventory",
                  "--custodian-identity", CUSTODIAN,
                  "--operator-identity", OPERATOR,
                  "--coverage-sha256", sha_hex("coverage"),
                  "--evidence", str(tmp_path / "evidence.json"),
                  "--excluded-days", str(tmp_path / "excluded.json"),
                  "--objects", str(tmp_path / "objects.json"),
                  "--out", str(tmp_path / "inventory.json")])
    assert rc == 2
    assert "custody" in capsys.readouterr().err
    assert not (tmp_path / "inventory.json").exists()


def test_cli_build_inventory_refuses_tracked_out_path(tmp_path, capsys):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(pf.__file__)))
    rc = run_cli(["build-inventory",
                  "--custodian-identity", CUSTODIAN,
                  "--operator-identity", OPERATOR,
                  "--coverage-sha256", sha_hex("coverage"),
                  "--evidence", str(tmp_path / "missing-evidence.json"),
                  "--excluded-days", str(tmp_path / "missing.json"),
                  "--objects", str(tmp_path / "missing.json"),
                  "--out", os.path.join(repo_root, "eval", "inventory.json")])
    assert rc == 2
    assert "--out" in capsys.readouterr().err
    assert not os.path.exists(os.path.join(repo_root, "eval", "inventory.json"))


def test_cli_verify_execution_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pf, "_free_gb", lambda path: 100000.0)
    monkeypatch.chdir(tmp_path)
    assert run_cli(["plan", "--custodian-identity", CUSTODIAN,
                    "--operator-identity", OPERATOR]) == 0
    plan_path = pf.DEFAULT_OUT_DIR + "/g0bn_acquisition_plan.json"
    # Development: days-file content and exact stage concurrency match the plan commitment -> OK.
    assert run_cli(["verify-execution", "--partition", "development",
                    "--stage", "download", "--jobs", "4", "--plan", plan_path]) == 0
    assert run_cli(["verify-execution", "--partition", "development",
                    "--stage", "recon", "--jobs", "1", "--plan", plan_path]) == 0

    # A within-bound download drift and a download/recon mix-up both refuse.
    assert run_cli(["verify-execution", "--partition", "development",
                    "--stage", "download", "--jobs", "3", "--plan", plan_path]) == 2
    assert "verify-execution.jobs" in capsys.readouterr().err
    assert run_cli(["verify-execution", "--partition", "development",
                    "--stage", "recon", "--jobs", "4", "--plan", plan_path]) == 2
    assert "verify-execution.jobs" in capsys.readouterr().err

    # Holdout refuses for anyone who is not the provisioned custodian identity
    # (the packet's run_as is documentation; this is the enforcement).
    rc = run_cli(["verify-execution", "--partition", "holdout",
                  "--stage", "download", "--jobs", "4", "--plan", plan_path])
    assert rc == 2
    assert "custodian" in capsys.readouterr().err
    monkeypatch.setattr(pf.getpass, "getuser", lambda: CUSTODIAN)
    assert run_cli(["verify-execution", "--partition", "holdout",
                    "--stage", "download", "--jobs", "4", "--plan", plan_path]) == 0
    # An edited/stale days file refuses BEFORE any vendor I/O could start.
    days_file = pf.DEV_DAYS_FILE
    with open(days_file, "a") as f:
        f.write("2026-02-01\n")
    rc = run_cli(["verify-execution", "--partition", "development",
                  "--stage", "download", "--jobs", "4", "--plan", plan_path])
    assert rc == 2
    assert "days_file_sha256" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("stage", "jobs", "max_field"),
    [("download", "4", "download_jobs_max"),
     ("recon", "1", "recon_jobs_max")],
)
def test_cli_verify_execution_rejects_malformed_jobs_max(
        tmp_path, monkeypatch, capsys, stage, jobs, max_field):
    monkeypatch.setattr(pf, "_free_gb", lambda path: 100000.0)
    monkeypatch.chdir(tmp_path)
    assert run_cli(["plan", "--custodian-identity", CUSTODIAN,
                    "--operator-identity", OPERATOR]) == 0
    plan_path = pf.DEFAULT_OUT_DIR + "/g0bn_acquisition_plan.json"
    with open(plan_path) as f:
        plan = json.load(f)
    plan["execution"][max_field] = None
    with open(plan_path, "w") as f:
        json.dump(resha(plan), f)

    rc = run_cli(["verify-execution", "--partition", "development",
                  "--stage", stage, "--jobs", jobs, "--plan", plan_path])

    assert rc == 2
    assert f"execution.{max_field}" in capsys.readouterr().err


# ------------------------------------------------------------- network isolation


def test_full_preflight_runs_with_sockets_disabled(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted during no-I/O preflight")

    monkeypatch.setattr(pf, "_free_gb", lambda path: 100000.0)
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert run_cli(["plan", "--custodian-identity", CUSTODIAN,
                    "--operator-identity", OPERATOR,
                    "--out-dir", str(tmp_path / "out")]) == 0
    build_inventory()


def test_module_source_never_imports_network_or_vendor_modules():
    import ast

    tree = ast.parse(open(pf.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"boto3", "botocore", "lakeapi", "requests", "urllib", "urllib3",
                 "http", "socket", "ssl", "ftplib", "aiohttp", "pyarrow",
                 "s3fs", "fsspec", "subprocess"}
    assert not imported & forbidden, imported & forbidden


def test_module_import_loads_no_vendor_sdk():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(pf.__file__)))
    code = (
        "import sys\n"
        "import ingest.g0bn_acquisition_preflight\n"
        "bad = [m for m in ('boto3', 'botocore', 'lakeapi', 'requests',\n"
        "                   'urllib3', 'aiohttp', 's3fs') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == ""


def test_normalized_products_pin_matches_writer_sources():
    plan = build_plan()
    assert plan["custody"]["normalized_products"] == list(G0BN_DATA_SOURCES)
    assert plan["custody"]["raw_products"] == plan["products"]
