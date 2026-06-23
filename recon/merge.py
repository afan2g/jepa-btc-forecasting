"""Merge event streams onto one total-ordered engine-time axis."""
from __future__ import annotations
from recon.events import Event, order_key


def merge_sorted(deltas: list[Event], trades: list[Event]) -> list[Event]:
    """Return all events in the single total order defined by recon.events.order_key.
    Deterministic and invariant to input ordering."""
    return sorted([*deltas, *trades], key=order_key)
