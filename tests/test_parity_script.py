"""Drive the parity script's pure core (run_parity_core) and reporting with synthetic,
in-memory inputs — exercises the full production plumbing without any vendor access."""
import datetime as dt
import importlib.util
import json
import math
import pathlib

import numpy as np
import pandas as pd

from recon.coinapi import coinapi_frame_from_rows

# scripts/ is not a package — load the script module by path.
_SPEC = importlib.util.spec_from_file_location(
    "run_coinbase_parity",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_coinbase_parity.py",
)
rcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rcp)

DAY = dt.date(2025, 6, 1)
DAY_OPEN = pd.Timestamp("2025-06-01").value
S = 1_000_000_000


def _lake_df():
    """Real-Lake-schema book_delta_v2 for the same book as _coinapi_rows()."""
    rows = [
        (DAY_OPEN + 1 * S, 1, True, 100.0, 2.0),
        (DAY_OPEN + 1 * S, 2, False, 101.0, 3.0),
        (DAY_OPEN + 2 * S, 3, True, 100.0, 5.0),
        (DAY_OPEN + 3 * S, 4, True, 99.0, 1.0),
    ]
    df = pd.DataFrame(rows, columns=["origin_time", "sequence_number", "side_is_bid",
                                     "price", "size"])
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    return df


def _coinapi_rows():
    return coinapi_frame_from_rows([
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="B", time_exchange_ns=86_399_999_000_000),
        dict(update_type="SNAPSHOT", is_buy=False, entry_px=101.0, entry_sx=3.0,
             order_id="A", time_exchange_ns=86_399_999_000_000),
        dict(update_type="SET", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="B", time_exchange_ns=2 * S),
        dict(update_type="ADD", is_buy=True, entry_px=99.0, entry_sx=1.0,
             order_id="B2", time_exchange_ns=3 * S),
    ])


def test_cli_default_size_policy_is_decrement():
    """The Coinbase parity CLI must default to size_policy=decrement: the 2025-06-01 live gate
    proved MATCH.entry_sx is the traded amount for Coinbase limitbook_full, so 'absolute' crosses
    the book ~100% (docs/data.md §5a). 'absolute' stays selectable as the A/B alternative."""
    assert rcp.parse_args([]).size_policy == "decrement"
    assert rcp.parse_args(["--day", "2025-06-01", "--k", "10"]).size_policy == "decrement"
    assert rcp.parse_args(["--size-policy", "absolute"]).size_policy == "absolute"


def test_build_grid_spans_the_day():
    grid = rcp.build_grid(DAY, grid_ms=1000)
    assert len(grid) == 86400
    assert grid[0] == DAY_OPEN
    assert grid[-1] == DAY_OPEN + 86399 * S


def test_run_parity_core_same_book_reports_zero_divergence():
    report, lake, capi = rcp.run_parity_core(
        _lake_df(), [_coinapi_rows()], day=DAY, k=5, grid_ms=1000, horizons_s=(2, 10))
    p = report["parity"]
    assert report["meta"]["lake_delta_rows"] == 4
    assert report["meta"]["coinapi_event_rows"] == 4
    assert p["mid_diff"]["max"] == 0.0
    assert report["lake_quality"]["crossed_rate"] == 0.0
    assert report["coinapi_quality"]["crossed_rate"] == 0.0
    # both books present from +1s onward; missing only the pre-seed seconds [00:00:00, +1s)
    assert report["parity"]["missing_book"]["either_fraction"] < 0.01


def test_run_parity_core_reports_warmup_block_and_restricts_parity():
    report, lake, capi = rcp.run_parity_core(
        _lake_df(), [_coinapi_rows()], day=DAY, k=5, grid_ms=1000, horizons_s=(2,))
    w = report["warmup"]
    assert w["gated"] is True and w["established"] is True and w["cutoff_ts"] is not None
    assert w["excluded_samples"] >= 1                       # the day-open empty sample(s) excluded
    assert report["parity"]["since_ts"] == w["cutoff_ts"]
    assert report["parity"]["n_grid"] < report["parity"]["n_grid_full"]


def test_run_parity_core_warmup_gate_can_be_disabled():
    report, _, _ = rcp.run_parity_core(
        _lake_df(), [_coinapi_rows()], day=DAY, k=5, grid_ms=1000, horizons_s=(2,),
        gate_warmup=False)
    assert report["warmup"]["gated"] is False
    assert report["parity"]["since_ts"] is None
    assert report["parity"]["n_grid"] == report["parity"]["n_grid_full"] == 86400


def test_run_parity_core_handles_empty_lake_day():
    # A Lake gap day → empty delta frame → Lake book fully missing, but the run still
    # completes and the report is well-formed (no crash).
    report, lake, capi = rcp.run_parity_core(
        pd.DataFrame(), [_coinapi_rows()], day=DAY, k=3, grid_ms=1000, horizons_s=(2,))
    assert report["lake_quality"]["missing_book_fraction"] == 1.0
    assert report["meta"]["lake_delta_rows"] == 0


def test_report_is_strict_json_serializable(tmp_path):
    # Divergent mids → NaN corr / spike rows present → ensure _json_safe yields valid JSON
    # (jq empty contract, AGENTS.md). Use a Lake book that crosses to force edge values.
    report, lake, capi = rcp.run_parity_core(
        _lake_df(), [_coinapi_rows()], day=DAY, k=4, grid_ms=1000)
    paths = rcp.write_report(report, lake, capi, str(tmp_path), DAY, 4, dump_grid=True)
    # round-trips through strict JSON (allow_nan=False already enforced on write)
    loaded = json.loads(pathlib.Path(paths["json"]).read_text())
    assert loaded["meta"]["day"] == "2025-06-01"
    assert pathlib.Path(paths["spikes_csv"]).exists()
    assert pathlib.Path(paths["grid_csv"]).exists()
    # no NaN/Inf leaked into the JSON text
    txt = pathlib.Path(paths["json"]).read_text()
    assert "NaN" not in txt and "Infinity" not in txt


def test_json_safe_sanitizes_non_finite_and_numpy():
    out = rcp._json_safe({"a": float("nan"), "b": np.float64(1.5),
                          "c": [np.int64(3), float("inf")], "d": "x"})
    assert out == {"a": None, "b": 1.5, "c": [3, None], "d": "x"}
    assert math.isfinite(out["b"])


# --------------------------------------------------------------------------- Lake seed/reseed (§5a-Recon)
def _stranded_lake_df():
    """A Lake book_delta_v2 day that, cold-started, strands a level and crosses ~all day:
    bid100/ask101 at +1s, then a bid lands at 102 with NO ask clear → best bid 102 ≥ ask 101."""
    rows = [
        (DAY_OPEN + 1 * S, 1, True, 100.0, 1.0),
        (DAY_OPEN + 1 * S, 2, False, 101.0, 1.0),
        (DAY_OPEN + 2 * S, 3, True, 102.0, 1.0),   # strands ask101 → crossed from +2s onward
    ]
    df = pd.DataFrame(rows, columns=["origin_time", "sequence_number", "side_is_bid",
                                     "price", "size"])
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    return df


def _seed_snapshots():
    from recon.reseed import book_snapshot
    # day-open seed + a later vendor snapshot showing the true (uncrossed) book bid102/ask103.
    return [book_snapshot(DAY_OPEN + 1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
            book_snapshot(DAY_OPEN + 3 * S, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]


def test_run_parity_core_lake_reseed_repairs_crossing_and_reports_before_after():
    report, lake, capi = rcp.run_parity_core(
        _stranded_lake_df(), [_coinapi_rows()], day=DAY, k=1,
        lake_book_snapshots=_seed_snapshots(), reseed=True,
        reseed_after_crossed_s=0.0, seed_min_levels=1)
    lr = report["lake_reseed"]
    assert lr["applied"] is True and lr["seed_accepted"] is True and lr["seed_ts"] is not None
    assert lr["reseed_count"] >= 1
    assert lr["crossed_rate_before"] > 0.9          # cold-start crosses ~the whole day
    assert lr["crossed_rate_after"] < 0.01          # reseed clears the stranded level
    assert lr["crossed_rate_after"] < lr["crossed_rate_before"]
    # lake_quality reports the seeded (post-reseed) FULL-grid crossed rate, not the cold-start one
    assert report["lake_quality"]["crossed_rate"] == lr["crossed_rate_after"]


def test_run_parity_core_no_reseed_ab_arm_seeds_but_does_not_repair():
    report, _, _ = rcp.run_parity_core(
        _stranded_lake_df(), [_coinapi_rows()], day=DAY, k=1,
        lake_book_snapshots=_seed_snapshots(), reseed=False,
        reseed_after_crossed_s=0.0, seed_min_levels=1)
    lr = report["lake_reseed"]
    assert lr["seed_accepted"] is True and lr["reseed_count"] == 0
    assert lr["crossed_rate_after"] > 0.9           # seeded but no intraday repair → still crosses
    # reseed disabled ⇒ the crossing is permanent, NOT a residual awaiting a reseed → not excluded.
    assert lr["excluded_crossed_samples"] == 0


def test_crossed_samples_not_excluded_when_seed_is_rejected():
    # Snapshots present but ALL invalid (crossed) → seed REJECTED → book cold-starts and crosses.
    # Those crossed samples are a genuine reconstruction FAILURE and must NOT be excluded from the
    # gate — excluding them would make a failed Lake reconstruction look clean (Codex P2). This is
    # the crossed-`book`-product day case (e.g. 2026-04-01) that multi-day validation must catch.
    from recon.reseed import book_snapshot
    bad = [book_snapshot(DAY_OPEN + 1, bids=[(101.0, 1.0)], asks=[(100.0, 1.0)])]  # crossed → rejected
    report, _, _ = rcp.run_parity_core(
        _stranded_lake_df(), [_coinapi_rows()], day=DAY, k=1, lake_book_snapshots=bad,
        reseed=True, reseed_after_crossed_s=0.0, seed_min_levels=1)
    lr = report["lake_reseed"]
    assert lr["seed_accepted"] is False
    assert lr["excluded_crossed_samples"] == 0           # failure surfaced, not masked
    assert report["parity"]["n_excluded_crossed"] == 0
    assert lr["crossed_rate_after"] > 0.9                # the 67%-style crossing stays visible


def test_run_parity_core_without_snapshots_is_unchanged_cold_start():
    report, _, _ = rcp.run_parity_core(_lake_df(), [_coinapi_rows()], day=DAY, k=5)
    assert report["lake_reseed"]["applied"] is False
    assert report["lake_reseed"]["seed_accepted"] is False
    assert report["parity"]["mid_diff"]["max"] == 0.0   # same-book parity preserved


def test_seeded_report_is_strict_json_serializable(tmp_path):
    # The lake_reseed block adds nested dicts (snapshot_reason_codes, policy), a list (reseed_ts)
    # and Nones — ensure _json_safe + allow_nan=False still yields valid JSON (jq empty contract).
    report, lake, capi = rcp.run_parity_core(
        _stranded_lake_df(), [_coinapi_rows()], day=DAY, k=1,
        lake_book_snapshots=_seed_snapshots(), reseed=True,
        reseed_after_crossed_s=0.0, seed_min_levels=1)
    paths = rcp.write_report(report, lake, capi, str(tmp_path), DAY, 1, dump_grid=False)
    txt = pathlib.Path(paths["json"]).read_text()
    loaded = json.loads(txt)
    assert loaded["lake_reseed"]["reseed_count"] >= 1
    assert "NaN" not in txt and "Infinity" not in txt


def test_warmup_cutoff_is_clamped_to_the_accepted_seed():
    # Cold-started deltas can look two-sided/uncrossed BEFORE the validated seed lands; the parity
    # cutoff must clamp to seed_ts so the gate never compares pre-seed cold-started Lake state
    # (Codex P2 / §5a-Recon). Here deltas at +1s establish a clean book, but the seed is at +5s.
    from recon.reseed import book_snapshot
    df = pd.DataFrame(
        [(DAY_OPEN + 1 * S, 1, True, 100.0, 1.0), (DAY_OPEN + 1 * S, 2, False, 101.0, 1.0)],
        columns=["origin_time", "sequence_number", "side_is_bid", "price", "size"])
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    seed = [book_snapshot(DAY_OPEN + 5 * S, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    report, _, _ = rcp.run_parity_core(
        df, [_coinapi_rows()], day=DAY, k=1, lake_book_snapshots=seed, reseed=True,
        reseed_after_crossed_s=0.0, seed_min_levels=1, warmup_consecutive=3)
    seed_ts = report["lake_reseed"]["seed_ts"]
    assert seed_ts == DAY_OPEN + 5 * S
    # cold-start alone would establish ~+3s; the clamp pushes the cutoff to the seed.
    assert report["parity"]["since_ts"] == seed_ts
    assert report["warmup"]["cutoff_ts"] == seed_ts


def test_cli_exposes_reseed_flags_with_safe_defaults():
    a = rcp.parse_args([])
    assert a.no_reseed is False and a.no_lake_seed is False   # reseed ON by default
    assert a.reseed_after_crossed_s == 2.0 and a.seed_min_levels >= 1
    assert rcp.parse_args(["--no-reseed"]).no_reseed is True
    assert rcp.parse_args(["--no-lake-seed"]).no_lake_seed is True
