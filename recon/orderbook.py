"""Order-book state machine. Plain dicts; min/max queries (perf deferred to Rust)."""
from __future__ import annotations
from recon.events import Delta


class OrderBook:
    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self._last_seq: int | None = None

    def apply(self, d: Delta) -> bool:
        """Apply a delta. Returns False if a sequence gap is detected (seq not strictly
        increasing within the stream), True otherwise. Caller decides re-snapshot policy."""
        ok = self._last_seq is None or d.seq > self._last_seq
        self._last_seq = d.seq
        book = self.bids if d.side == "bid" else self.asks
        if d.size == 0.0:
            book.pop(d.price, None)
        else:
            book[d.price] = d.size
        return ok

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
        bids = sorted(self.bids, reverse=True)[:k]
        asks = sorted(self.asks)[:k]
        out: dict = {"mid": self.mid(), "microprice": self.microprice()}
        for i in range(k):
            out[f"bid_{i}_price"] = bids[i] if i < len(bids) else None
            out[f"bid_{i}_size"] = self.bids[bids[i]] if i < len(bids) else None
            out[f"ask_{i}_price"] = asks[i] if i < len(asks) else None
            out[f"ask_{i}_size"] = self.asks[asks[i]] if i < len(asks) else None
        return out
