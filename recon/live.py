"""Streaming reconstruction with a bounded-out-of-orderness watermark.

Buffers arrivals; releases events whose ts_engine <= (max_ts_seen - watermark_ns)
in total order, then snapshots at trades exactly as the offline path does.

INVARIANT (load-bearing): `watermark_ns` must STRICTLY exceed the feed's maximum
out-of-orderness. If an event can arrive displaced by up to D ns from its sorted
position, choose watermark_ns > D (e.g. D + 1). Under that condition every event with
ts <= (max_ts - watermark_ns) has provably already arrived, so each release batch is
complete and the released order == the global sorted order — making the output
byte-identical to recon.reconstruct (the E0.1 gate). With watermark_ns == D the
lowest-ts events of a reversed arrival block can be released prematurely, breaking
equivalence."""
from __future__ import annotations
import pandas as pd
from recon.events import Delta, Trade, order_key
from recon.orderbook import OrderBook


class LiveReconstructor:
    def __init__(self, *, k: int, watermark_ns: int) -> None:
        self.k = k
        self.watermark_ns = watermark_ns
        self._buf: list = []
        self._max_ts = None
        self._ob = OrderBook()
        self._rows: list[dict] = []

    def _emit_delta(self, d: Delta) -> None:
        self._ob.apply(d)

    def _emit_trade(self, t: Trade) -> None:
        snap = self._ob.snapshot(self.k)
        snap.update(trade_ts=t.ts_engine, trade_seq=t.seq, trade_side=t.side,
                    trade_price=t.price, trade_amount=t.amount)
        self._rows.append(snap)

    def _release(self, threshold_ts) -> None:
        ready = [e for e in self._buf if e.ts_engine <= threshold_ts]
        if not ready:
            return
        self._buf = [e for e in self._buf if e.ts_engine > threshold_ts]
        for ev in sorted(ready, key=order_key):
            self._emit_delta(ev) if isinstance(ev, Delta) else self._emit_trade(ev)

    def push(self, ev) -> None:
        self._buf.append(ev)
        self._max_ts = ev.ts_engine if self._max_ts is None else max(self._max_ts, ev.ts_engine)
        self._release(self._max_ts - self.watermark_ns)

    def flush(self) -> pd.DataFrame:
        for ev in sorted(self._buf, key=order_key):
            self._emit_delta(ev) if isinstance(ev, Delta) else self._emit_trade(ev)
        self._buf = []
        return pd.DataFrame(self._rows)
