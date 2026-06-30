"""Order-book state machine. Plain dicts; min/max queries (perf deferred to Rust)."""
from __future__ import annotations
import heapq
from recon.events import Delta


class OrderBook:
    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self._last_seq: int | None = None

    def apply(self, d: Delta) -> bool:
        """Apply a delta. Returns False when `seq` is NOT strictly increasing vs the last
        applied delta (a duplicate/out-of-order anomaly signal), True otherwise.

        This is a partial signal only: it does NOT detect a forward gap (missing
        sequence_numbers, e.g. 5 -> 8), and neither reconstructor currently consumes the
        return value. Full reseed-on-discontinuity (docs/data.md §5a-Recon) and
        forward-gap detection are deferred — they need the real book_delta_v2
        sequence_number increment semantics confirmed by Task 1's Lake capture."""
        ok = self._last_seq is None or d.seq > self._last_seq
        self._last_seq = d.seq
        book = self.bids if d.side == "bid" else self.asks
        if d.size == 0.0:
            book.pop(d.price, None)
        else:
            book[d.price] = d.size
        return ok

    def reseed(self, bids, asks) -> None:
        """Replace the entire book state with a validated full snapshot (docs/data.md §5a-Recon).

        `bids`/`asks` are iterables of `(price, size)` from a Crypto Lake `book` snapshot that the
        caller has already validated (uncrossed, two-sided, finite/positive). A reseed is the cure
        for the cold-start level-stranding failure: it drops every stale/stranded level and restores
        a known-good state. `_last_seq` is cleared because the snapshot breaks delta sequence
        continuity (and seq is per-event, not a row-gap detector — so it is never used to *trigger*
        a reseed; crossing is)."""
        self.bids = {float(p): float(s) for p, s in bids}
        self.asks = {float(p): float(s) for p, s in asks}
        self._last_seq = None

    def copy(self) -> "OrderBook":
        """Shallow-but-independent copy: new level dicts + carried gap state. Used to
        seed a reconstruction without mutating the caller's book (prices/sizes are
        immutable floats, so dict() copies suffice)."""
        ob = OrderBook()
        ob.bids = dict(self.bids)
        ob.asks = dict(self.asks)
        ob._last_seq = self._last_seq
        return ob

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def mid(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    def microprice(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        bs, as_ = self.bids[bb], self.asks[ba]
        return (as_ * bb + bs * ba) / (bs + as_)

    def snapshot(self, k: int) -> dict:
        # Pad missing levels/metrics with NaN (a float) rather than None so every
        # numeric column infers float64 in pandas regardless of book depth or
        # one-sidedness. With None, an all-missing column becomes object dtype and a
        # partially-missing one becomes float64, making the reconstructed schema
        # data-dependent and breaking the offline/live byte-identity guarantee.
        #
        # Top-K via heapq.nlargest/nsmallest, NOT sorted(...)[:k]: these are documented to
        # equal sorted(...)[:k]/sorted(...,reverse=True)[:k] (so the output stays byte-
        # identical) but run in O(N log k) instead of O(N log N). This matters because a
        # parity/label sampler calls snapshot() once per grid point (86,400×/day at a 1 s
        # grid) on a Coinbase book of ~tens of thousands of levels per side; a full sort
        # each time would dominate the one-day run. best bid/ask (and hence mid/microprice)
        # are read off the top-K we already computed, avoiding 4 extra full max/min scans.
        nan = float("nan")
        bids = heapq.nlargest(k, self.bids)   # prices, descending
        asks = heapq.nsmallest(k, self.asks)  # prices, ascending
        bb = bids[0] if bids else None
        ba = asks[0] if asks else None
        if bb is not None and ba is not None:
            bs, as_ = self.bids[bb], self.asks[ba]
            m = (bb + ba) / 2.0
            mp = (as_ * bb + bs * ba) / (bs + as_)
        else:
            m = mp = None
        out: dict = {"mid": nan if m is None else m,
                     "microprice": nan if mp is None else mp}
        for i in range(k):
            out[f"bid_{i}_price"] = bids[i] if i < len(bids) else nan
            out[f"bid_{i}_size"] = self.bids[bids[i]] if i < len(bids) else nan
            out[f"ask_{i}_price"] = asks[i] if i < len(asks) else nan
            out[f"ask_{i}_size"] = self.asks[asks[i]] if i < len(asks) else nan
        return out
