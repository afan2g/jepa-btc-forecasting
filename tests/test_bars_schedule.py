"""Trailing/as-of-only threshold schedule (plan §A, Codex P2b/#12).

threshold_d = trailing mean of coverage-normalized completed dollar volume over prior
days STRICTLY < d, divided by target_bars_per_day. Using day d's own completed volume
would leak future volume into d's bar boundaries. Per plan §J the causality test
asserts on the THRESHOLD VALUE, not bar boundaries (raw trades legitimately re-bin a
day; only the schedule feed is mutated here)."""
import pytest

from bars.clock import DayThreshold, ThresholdConfig, ThresholdSchedule

CFG = ThresholdConfig(
    target_bars_per_day=100,
    window_days=7,
    warmup_days=2,
    seed_threshold=5_000.0,
)


def _loaded(days_volumes):
    s = ThresholdSchedule(CFG)
    for day, vol in days_volumes:
        s.record_day(day, vol)
    return s


def test_warmup_uses_seed_threshold_and_is_flagged():
    s = _loaded([("2025-01-01", 1_000_000.0)])  # only 1 prior day < warmup_days=2
    got = s.threshold_for("2025-01-02")
    assert got == DayThreshold("2025-01-02", 5_000.0, True)


def test_trailing_mean_over_prior_days_after_warmup():
    s = _loaded([("2025-01-01", 1_000_000.0), ("2025-01-02", 3_000_000.0)])
    got = s.threshold_for("2025-01-03")
    assert got.is_warmup is False
    assert got.threshold == pytest.approx(2_000_000.0 / 100)


def test_threshold_uses_strictly_prior_days_only():
    # Recording day d's own (or a later day's) volume must not change threshold_d.
    s = _loaded([("2025-01-01", 1_000_000.0), ("2025-01-02", 3_000_000.0)])
    before = s.threshold_for("2025-01-03")
    s.record_day("2025-01-03", 50_000_000.0)   # day d's own completed volume
    s.record_day("2025-01-04", 90_000_000.0)   # a future day
    assert s.threshold_for("2025-01-03") == before
    # ... while a later day's threshold does see the new history
    assert s.threshold_for("2025-01-05").threshold != before.threshold


def test_window_excludes_days_older_than_window_days():
    entries = [(f"2025-01-{d:02d}", 1_000_000.0) for d in range(1, 9)]  # Jan 1..8
    s = _loaded(entries)
    s2 = ThresholdSchedule(CFG)
    for day, vol in entries:
        if day == "2025-01-01":
            vol = 999_000_000.0  # huge outlier, but outside [d-7, d) for d = Jan 9
        s2.record_day(day, vol)
    assert s2.threshold_for("2025-01-09") == s.threshold_for("2025-01-09")


def test_calendar_gap_days_simply_have_no_entry():
    # Unusable days are absent from the feed; the mean is over recorded days only.
    s = _loaded([("2025-01-01", 1_000_000.0), ("2025-01-05", 3_000_000.0)])
    got = s.threshold_for("2025-01-06")
    assert got.is_warmup is False
    assert got.threshold == pytest.approx(2_000_000.0 / 100)


def test_low_coverage_day_is_normalized_by_covered_fraction():
    # A ~93%-coverage or CoinAPI-filled day would skew every later threshold whose
    # window includes it (Codex #12); normalize volume to a full-day estimate.
    s = ThresholdSchedule(CFG)
    s.record_day("2025-01-01", 1_000_000.0)
    s.record_day("2025-01-02", 500_000.0, covered_fraction=0.5)  # -> 1_000_000 est.
    got = s.threshold_for("2025-01-03")
    assert got.threshold == pytest.approx(1_000_000.0 / 100)


def test_sub_min_coverage_day_is_excluded_and_does_not_count_toward_warmup():
    cfg = CFG._replace(min_covered_fraction=0.8)
    s = ThresholdSchedule(cfg)
    s.record_day("2025-01-01", 1_000_000.0)
    s.record_day("2025-01-02", 100_000.0, covered_fraction=0.1)  # excluded
    got = s.threshold_for("2025-01-03")
    assert got.is_warmup is True  # only 1 qualifying prior day < warmup_days=2
    assert got.threshold == 5_000.0
    s.record_day("2025-01-03", 2_000_000.0)
    got = s.threshold_for("2025-01-04")
    assert got.is_warmup is False
    assert got.threshold == pytest.approx(1_500_000.0 / 100)  # excluded day absent


def test_threshold_is_invariant_to_history_recording_order():
    # float summation order must not depend on record_day call order, or two builds
    # recording the same history differently could disagree in the last ulp; the
    # magnitude mix below makes (1+1)+1e16 != (1e16+1)+1 if iteration follows
    # insertion order
    entries = [("2025-01-01", 1.0), ("2025-01-02", 1.0), ("2025-01-03", 1.0e16)]
    fwd, rev = ThresholdSchedule(CFG), ThresholdSchedule(CFG)
    for day, vol in entries:
        fwd.record_day(day, vol)
    for day, vol in reversed(entries):
        rev.record_day(day, vol)
    assert fwd.threshold_for("2025-01-08") == rev.threshold_for("2025-01-08")


def test_duplicate_day_recording_fails_closed():
    s = _loaded([("2025-01-01", 1_000_000.0)])
    with pytest.raises(ValueError, match="2025-01-01"):
        s.record_day("2025-01-01", 2_000_000.0)


def test_invalid_history_entries_fail_closed():
    s = ThresholdSchedule(CFG)
    with pytest.raises(ValueError):
        s.record_day("2025-01-01", -1.0)
    with pytest.raises(ValueError):
        s.record_day("2025-01-01", 1.0, covered_fraction=0.0)
    with pytest.raises(ValueError):
        s.record_day("2025-01-01", 1.0, covered_fraction=1.5)
    with pytest.raises(ValueError):
        s.record_day("not-a-day", 1.0)


def test_invalid_config_fails_closed():
    with pytest.raises(ValueError):
        ThresholdSchedule(CFG._replace(target_bars_per_day=0))
    with pytest.raises(ValueError):
        ThresholdSchedule(CFG._replace(seed_threshold=0.0))
    with pytest.raises(ValueError):
        ThresholdSchedule(CFG._replace(warmup_days=0))
    with pytest.raises(ValueError):
        ThresholdSchedule(CFG._replace(window_days=1))  # < warmup_days: permanent warmup
    with pytest.raises(ValueError):
        ThresholdSchedule(CFG._replace(min_covered_fraction=1.5))
