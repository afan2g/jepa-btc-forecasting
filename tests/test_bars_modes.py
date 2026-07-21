"""Source-mode contract (issue #61 / plan §Staged dataset modes): the producer runs in
exactly the staged modes — `coinbase_only` must never open a Binance input,
`cross_venue` may open both, and `binance_single_venue` (the G0-BN arm added by
#67/T9) must never open a Coinbase input. The enforcement is fail-closed so a mode
typo or an unauthorized venue open raises instead of silently building the wrong
arm."""
import pytest

from bars.modes import (
    BINANCE_SINGLE_VENUE,
    COINBASE_ONLY,
    CROSS_VENUE,
    SOURCE_MODES,
    VENUE_BINANCE,
    VENUE_COINBASE,
    allowed_venues,
    require_venue_allowed,
    resolve_source_mode,
)


def test_source_modes_are_exactly_the_staged_modes():
    assert SOURCE_MODES == (COINBASE_ONLY, CROSS_VENUE, BINANCE_SINGLE_VENUE)
    assert COINBASE_ONLY == "coinbase_only"
    assert CROSS_VENUE == "cross_venue"
    assert BINANCE_SINGLE_VENUE == "binance_single_venue"


def test_resolve_source_mode_accepts_known_and_rejects_unknown():
    assert resolve_source_mode("coinbase_only") == COINBASE_ONLY
    assert resolve_source_mode("cross_venue") == CROSS_VENUE
    with pytest.raises(ValueError, match="coinbase_only"):
        resolve_source_mode("coinbase")  # near-miss must not silently map


def test_coinbase_only_opens_no_binance_input():
    assert allowed_venues(COINBASE_ONLY) == (VENUE_COINBASE,)
    require_venue_allowed(COINBASE_ONLY, VENUE_COINBASE)  # no raise
    with pytest.raises(ValueError, match="binance"):
        require_venue_allowed(COINBASE_ONLY, VENUE_BINANCE)


def test_cross_venue_allows_both_venues():
    assert allowed_venues(CROSS_VENUE) == (VENUE_COINBASE, VENUE_BINANCE)
    require_venue_allowed(CROSS_VENUE, VENUE_COINBASE)
    require_venue_allowed(CROSS_VENUE, VENUE_BINANCE)


def test_unknown_venue_fails_closed_in_both_modes():
    for mode in SOURCE_MODES:
        with pytest.raises(ValueError, match="kraken"):
            require_venue_allowed(mode, "kraken")
