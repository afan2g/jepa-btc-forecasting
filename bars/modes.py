"""Staged dataset source modes (plan §Staged dataset modes; umbrella #37).

One producer, two modes: `coinbase_only` may never open a Binance input (the first
G0-CB pilot build), `cross_venue` requires certified coverage from both venues. T1
ships only this contract; T2/T4/T9 route every venue open through
`require_venue_allowed` so an unauthorized open raises instead of silently building
the wrong arm. The pre-E2.5 clock reference stream is Coinbase trades in BOTH modes
(plan §A / decision #1) — the mode governs which venues may be OPENED, not the
clock trigger.
"""
from __future__ import annotations

COINBASE_ONLY = "coinbase_only"
CROSS_VENUE = "cross_venue"
SOURCE_MODES = (COINBASE_ONLY, CROSS_VENUE)

VENUE_COINBASE = "coinbase"
VENUE_BINANCE = "binance"

_ALLOWED_VENUES = {
    COINBASE_ONLY: (VENUE_COINBASE,),
    CROSS_VENUE: (VENUE_COINBASE, VENUE_BINANCE),
}


def resolve_source_mode(mode: str) -> str:
    """Validate a source-mode string, else raise listing the two legal modes."""
    if mode in _ALLOWED_VENUES:
        return mode
    raise ValueError(
        f"unknown source mode {mode!r}; expected one of {COINBASE_ONLY!r}, {CROSS_VENUE!r}"
    )


def allowed_venues(mode: str) -> tuple[str, ...]:
    """The venues a producer stage may open under `mode`."""
    return _ALLOWED_VENUES[resolve_source_mode(mode)]


def require_venue_allowed(mode: str, venue: str) -> None:
    """Fail closed on any venue open the mode does not authorize."""
    if venue not in (VENUE_COINBASE, VENUE_BINANCE):
        raise ValueError(f"unknown venue {venue!r}; expected "
                         f"{VENUE_COINBASE!r} or {VENUE_BINANCE!r}")
    if venue not in allowed_venues(mode):
        raise ValueError(f"source mode {mode!r} does not allow opening {venue!r} inputs")
