"""Offline tests: Binance quota estimation + broad-pull gate (plan Task 4 / Requirements 7-8).

Pure, no vendor I/O. The `book` seed product is budgeted whenever book_delta_v2 is requested (it is
pulled alongside it). The gate has two levels: a HARD monthly-quota-headroom refusal that overrides
--allow-broad, and a SOFT --max-gb cap that only --allow-broad overrides — both exit 4 (mirrors
run_coinbase_quality_map.quota_decision, raised as SystemExit like _common.check_backfill_gate)."""
import pytest

from ingest import lake_binance as lb


# --------------------------------------------------------------------------- estimate
def test_estimate_includes_book_seed_when_book_delta_selected():
    # selecting book_delta_v2 must ALSO budget its `book` seed product. Assert via the module's own
    # constants — do NOT pin the derived (unmeasured, Requirement 7) seed GB/day, so a Phase-1
    # measured value updates LAKE_GB_PER_DAY without breaking this test.
    feeds = lb.LAKE_GB_PER_DAY[("BINANCE_FUTURES", "BTC-USDT-PERP")]
    assert feeds["book"] > 0                                    # the seed product IS budgeted
    gb = lb.estimate_gb("binance-perp", ["book_delta_v2"], n_days=10)
    assert gb == pytest.approx(10 * (feeds["book_delta_v2"] + feeds["book"]))


def test_estimate_omits_book_seed_when_book_delta_not_selected():
    feeds = lb.LAKE_GB_PER_DAY[("BINANCE_FUTURES", "BTC-USDT-PERP")]
    gb = lb.estimate_gb("binance-perp", ["trades", "funding"], n_days=3)
    assert gb == pytest.approx(3 * (feeds["trades"] + feeds["funding"]))  # no book seed added


def test_estimate_counts_book_seed_once_across_multiple_feeds():
    feeds = lb.LAKE_GB_PER_DAY[("BINANCE_FUTURES", "BTC-USDT-PERP")]
    gb = lb.estimate_gb("binance-perp", ["book_delta_v2", "trades"], n_days=1)
    assert gb == pytest.approx(feeds["book_delta_v2"] + feeds["trades"] + feeds["book"])


def test_estimate_spot_book_delta_budgets_its_book_seed():
    feeds = lb.LAKE_GB_PER_DAY[("BINANCE", "BTC-USDT")]
    gb = lb.estimate_gb("binance-spot", ["book_delta_v2"], n_days=2)
    assert gb == pytest.approx(2 * (feeds["book_delta_v2"] + feeds["book"]))


def test_estimate_rejects_invalid_instrument_feed_pair():
    with pytest.raises(ValueError):
        lb.estimate_gb("binance-spot", ["funding"], n_days=1)  # funding is perp-only


def test_full_pull_gb_per_day_sums_both_instruments_all_feeds_plus_seed():
    total = lb.full_pull_gb_per_day()
    expected = (lb.estimate_gb("binance-perp", lb.INSTRUMENTS["binance-perp"].feeds, 1)
                + lb.estimate_gb("binance-spot", lb.INSTRUMENTS["binance-spot"].feeds, 1))
    assert total == pytest.approx(expected)
    assert total == pytest.approx(1.23, abs=0.05)  # docs §6 derived ~1.23 GB/day


# --------------------------------------------------------------------------- broad-pull gate
def test_broad_gate_blocks_without_allow_broad():
    with pytest.raises(SystemExit) as e:
        lb.check_broad_gate(est_gb=50.0, max_gb=5.0, allow_broad=False,
                            used_gb=0.0, quota_gb=300.0, headroom_gb=10.0)
    assert e.value.code == 4


def test_broad_gate_blocks_over_headroom_even_with_allow_broad():
    with pytest.raises(SystemExit) as e:
        lb.check_broad_gate(est_gb=295.0, max_gb=1e9, allow_broad=True,
                            used_gb=20.0, quota_gb=300.0, headroom_gb=10.0)
    assert e.value.code == 4


def test_broad_pull_allowed_with_allow_broad_within_headroom():
    # a broad pull (> --max-gb) is permitted with --allow-broad AS LONG AS it stays under headroom
    lb.check_broad_gate(est_gb=50.0, max_gb=5.0, allow_broad=True,
                        used_gb=0.0, quota_gb=300.0, headroom_gb=10.0)  # no raise


def test_one_day_allowed():
    lb.check_broad_gate(est_gb=1.23, max_gb=5.0, allow_broad=False,      # one day, all feeds + seed
                        used_gb=0.0, quota_gb=300.0, headroom_gb=10.0)   # no raise


def test_zero_estimate_never_raises():
    # nothing to load (e.g. all days skipped) is always allowed, headroom irrelevant
    lb.check_broad_gate(est_gb=0.0, max_gb=0.0, allow_broad=False,
                        used_gb=299.0, quota_gb=300.0, headroom_gb=10.0)  # no raise


def test_gate_exit_code_constant_is_four():
    assert lb.BROAD_GATE_EXIT == 4
