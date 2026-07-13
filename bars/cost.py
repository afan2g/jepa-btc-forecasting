"""Causal per-row execution-cost inputs (plan §G, T7; issue #82).

T7 scope only: turn T2's dual target-book reads (`bars.snapshot.BarBookReads`)
into the two per-row cost columns the built evaluator consumes
(`eval/matrix.py:RESERVED` — `cost_bps`, `half_spread_bps`), plus the drift and
slippage diagnostics that explain them:

- **`half_spread_bps`** = `0.5 * (best_ask - best_bid) / observable_mid * 1e4`
  from the OBSERVABLE read only (`received_time <= t_event`, §B/§C.2) — the cost
  is realistic and uses no future state. It stays OUT of `cost_bps`:
  `eval.cost.net_pnl` charges the spread itself, exactly twice per round trip
  (`spread_crossings=2`), so folding it into `cost_bps` would double-count it.
- **`latency_drift_bps`** = `abs(true_t_event_mid / observable_mid - 1) * 1e4`
  — the `target_read_ts -> t_event` move between the last observable book and
  the decision, charged FORWARD as slippage (plan #1 / §G). The true `t_event`
  mid is T2's label-anchor read (a plain origin cut): label `P0` already owns
  it, so the drift is a real entry cost, never a label shift and never an
  observable feature.
- **`slippage_bps`** = `base_slippage_bps + latency_drift_bps`;
  **`cost_bps`** = `2 * taker_fee_bps + slippage_bps` — taker fees are charged
  exactly once per side of the round trip (§G honest-taker discipline).

`CostAssumption` is the explicit, immutable, serializable fee/slippage contract
(§G: "recorded in the manifest, never hidden in code"): venue + product +
source identity, an assumption version, the taker fee and base slippage in bps,
and the drift-policy identity. No production Binance tier is baked in — every
field is required (plan Q1: T10 selects and freezes the real tier; tests use
explicit synthetic values). Binance and Coinbase schedules must never silently
alias: `require_assumption_identity` is the T8/T9 binding gate that rejects an
assumption whose venue/product/source does not exactly match the build's
declared identity.

Fail-closed discipline: T2 already drops one-sided/crossed/invalid/stale books
as `SnapshotRejection` rows, so a malformed read reaching this module means the
pipeline is broken, not that one row is unusable — every violation raises.
T9 obligations: map rows over the T2 stream (rejections pass through upstream,
they are never priced), bind the build's cost assumption once via
`require_assumption_identity` before pricing any row, and persist
`CostAssumption.as_dict()` in the manifest `sources` block (T8).
"""
from __future__ import annotations

import math
import numbers
from typing import NamedTuple

from bars.modes import VENUE_BINANCE, VENUE_COINBASE
from bars.snapshot import BarBookReads

# The one drift policy this module implements (charged-forward absolute
# mid-ratio move). A different policy identity is a different cost model and
# must fail closed here rather than silently repricing rows.
DRIFT_POLICY = "abs_true_over_observable_mid_v1"

# The repo venue vocabulary (bars.modes). An unknown venue must never validate:
# a fee schedule is venue-specific and silent reuse is exactly the aliasing
# §G forbids.
KNOWN_VENUES = (VENUE_BINANCE, VENUE_COINBASE)


class CostAssumption(NamedTuple):
    """Versioned per-venue execution-cost assumption (persist via `as_dict()`).

    Every field is required by construction: a default fee or venue would be an
    invented production tier (plan Q1) or a silent cross-venue alias."""
    venue: str              # bars.modes venue name, e.g. "binance"
    product: str            # instrument identity, e.g. "BTC-USDT-PERP"
    source: str             # normalized source identity the costs are tied to
    version: str            # assumption version tag (frozen by T10 for prod)
    taker_fee_bps: float    # one-way taker fee; charged twice per round trip
    base_slippage_bps: float  # slippage floor added on top of latency drift
    drift_policy: str = DRIFT_POLICY

    def as_dict(self) -> dict:
        """Manifest-ready (JSON-primitive) copy of every field, validated."""
        validate_cost_assumption(self)
        return {
            "venue": self.venue,
            "product": self.product,
            "source": self.source,
            "version": self.version,
            "taker_fee_bps": float(self.taker_fee_bps),
            "base_slippage_bps": float(self.base_slippage_bps),
            "drift_policy": self.drift_policy,
        }


class CostRow(NamedTuple):
    """One priced bar decision: the two consumer columns + their breakdown."""
    t_event: int
    half_spread_bps: float    # observable-book spread; NOT part of cost_bps
    latency_drift_bps: float  # charged target_read_ts -> t_event mid move
    slippage_bps: float       # base_slippage_bps + latency_drift_bps
    cost_bps: float           # 2 * taker_fee_bps + slippage_bps


def _identity_str(name: str, v) -> str:
    if not isinstance(v, str) or not v:
        raise ValueError(f"{name} must be a non-empty string; got {v!r}")
    return v


def _bps_param(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name} must be a real number of bps; got {v!r}")
    f = float(v)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(f"{name} must be finite and >= 0 bps; got {f!r}")
    return f


def validate_cost_assumption(a: CostAssumption) -> None:
    """Fail closed on any unknown, missing, or degenerate assumption field."""
    if not isinstance(a.venue, str) or a.venue not in KNOWN_VENUES:
        raise ValueError(f"unknown venue {a.venue!r}; expected one of "
                         f"{KNOWN_VENUES} — a fee schedule is venue-specific "
                         "and must never be reused across venues")
    _identity_str("product", a.product)
    _identity_str("source", a.source)
    _identity_str("version", a.version)
    _bps_param("taker_fee_bps", a.taker_fee_bps)
    _bps_param("base_slippage_bps", a.base_slippage_bps)
    if a.drift_policy != DRIFT_POLICY:
        raise ValueError(f"unknown drift policy {a.drift_policy!r}; this module "
                         f"implements only {DRIFT_POLICY!r} — a different "
                         "policy needs its own reviewed cost model")


def require_assumption_identity(assumption: CostAssumption, *, venue: str,
                                product: str, source: str) -> None:
    """The T8/T9 anti-aliasing gate: the build's declared venue/product/source
    identity must exactly match the assumption, or pricing must not start —
    a Binance schedule silently costing Coinbase rows (or vice versa) is the
    §G failure mode this exists to stop."""
    validate_cost_assumption(assumption)
    mismatches = [
        f"{name}: assumption {getattr(assumption, name)!r} != declared {want!r}"
        for name, want in (("venue", venue), ("product", product),
                           ("source", source))
        if getattr(assumption, name) != want
    ]
    if mismatches:
        raise ValueError("cost assumption identity mismatch — "
                         + "; ".join(mismatches))


def _int_ts(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise ValueError(f"{name} must be an integer nanosecond timestamp; "
                         f"got {v!r}")
    return int(v)


def _finite_pos(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name} must be a real number; got {v!r}")
    f = float(v)
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {f!r}")
    return f


def cost_row(reads: BarBookReads, *, assumption: CostAssumption) -> CostRow:
    """Price one bar decision from T2's dual reads. Deterministic and finite by
    construction once the inputs validate; every contract violation raises
    (module docstring: per-row drops happened at T2, not here).

    Causality: only `reads.observable` (received-gated, §C.2) enters the
    spread; `reads.label` (the true origin cut at `t_event`) is consumed
    SOLELY as the endpoint of the charged latency drift."""
    validate_cost_assumption(assumption)
    t_event = _int_ts("t_event", reads.t_event)
    obs, lab = reads.observable, reads.label

    bb = _finite_pos("observable best_bid", obs.best_bid)
    ba = _finite_pos("observable best_ask", obs.best_ask)
    _finite_pos("observable best_bid_size", obs.best_bid_size)
    _finite_pos("observable best_ask_size", obs.best_ask_size)
    if bb >= ba:
        # repo convention (recon/parity.py, bars/snapshot.py): locked is not a
        # tradable state either
        raise ValueError(f"crossed observable book: best_bid {bb!r} >= "
                         f"best_ask {ba!r} — T2 rejects these, so this read "
                         "bypassed the snapshot gate")
    obs_mid = _finite_pos("observable mid", obs.mid)
    if obs_mid != (bb + ba) / 2.0:
        # the mid is the denominator of BOTH emitted quantities; a read whose
        # mid is not the arithmetic top-of-book mid is corrupted/incompatible
        raise ValueError(f"observable mid {obs_mid!r} does not equal "
                         f"(best_bid + best_ask) / 2 = {(bb + ba) / 2.0!r}")
    true_mid = _finite_pos("label (true t_event) mid", lab.mid)

    if _int_ts("target_read_ts", obs.target_read_ts) > t_event:
        raise ValueError(f"target_read_ts {obs.target_read_ts} > t_event "
                         f"{t_event} — an observable read cannot postdate the "
                         "decision (broken causality upstream)")
    if _int_ts("label_cut_ts", lab.label_cut_ts) != t_event:
        raise ValueError(f"label cut {lab.label_cut_ts} != t_event {t_event} — "
                         "the drift endpoint must be the true book AT the "
                         "decision instant")

    half_spread_bps = 0.5 * (ba - bb) / obs_mid * 1e4
    latency_drift_bps = abs(true_mid / obs_mid - 1.0) * 1e4
    slippage_bps = float(assumption.base_slippage_bps) + latency_drift_bps
    cost_bps = 2.0 * float(assumption.taker_fee_bps) + slippage_bps
    if not (math.isfinite(latency_drift_bps) and math.isfinite(cost_bps)):
        # half_spread_bps is bounded ((ask-bid)/mid < 2 whenever bid < ask),
        # but the mid RATIO can overflow for individually-finite pathological
        # scales — a nonfinite row must die here, not at validate_matrix
        raise ValueError(
            f"cost outputs are not finite (latency_drift_bps="
            f"{latency_drift_bps!r}, cost_bps={cost_bps!r}) — pathological "
            "observable/true mid scale ratio or parameter magnitude")
    return CostRow(t_event=t_event, half_spread_bps=half_spread_bps,
                   latency_drift_bps=latency_drift_bps,
                   slippage_bps=slippage_bps, cost_bps=cost_bps)
