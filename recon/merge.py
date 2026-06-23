"""Merge event streams onto one total-ordered engine-time axis."""
from __future__ import annotations
from recon.events import Event, order_key


def merge_sorted(deltas: list[Event], trades: list[Event]) -> list[Event]:
    """Return all events in the single total order defined by recon.events.order_key.
    Deterministic and invariant to input ordering.

    SCOPE: correctness-first and IN-MEMORY — it materializes both streams into one list
    before sorting, so it is for tests and bounded windows only. A real daily
    reconstruction (Binance perp book_delta_v2 is ~109M rows/day, ~4 GB; docs/data.md §7)
    must NOT use this: the production path is a streaming, day-partitioned k-way merge over
    already-sorted/chunked inputs, deferred with the Rust port (spec §3). The
    LiveReconstructor already does a streaming watermark merge; this offline helper stays
    the simple reference the replay-equivalence test pins against."""
    return sorted([*deltas, *trades], key=order_key)
