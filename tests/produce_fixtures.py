"""Synthetic single-venue market-day fixtures for the T9 producer tests (issue #94).

Everything here is synthetic and tiny (no vendor I/O, no real January access): a
deterministic zigzag order book plus a steady taker-trade stream, written to the
normalized parquet object contract `bars.produce` consumes. The generator threads
book state across days (each day's snapshot object is the true end-of-prior-day
book, mirroring how a certified normalized feed would seed a day), so multi-day
segments exercise the cross-day carry paths the orchestrator owns.

Timing model (all int ns, absolute UTC):
- deltas every 200 ms and trades every 100 ms inside the active window at the
  start of the day; the rest of the day is quiet (time-cap bars, stale books);
- received_time = origin_time + a deterministic VARYING per-event lag (1 ms to
  ~901 ms, cycling on seq). A constant lag would make the observable and true
  label reads identical on every bar, leaving the received-time observability
  gate and the P0/true-mid discipline untestable — with the cycle, some deltas
  are not yet observable at nearby decisions, so observable state != label state
  on a known subset of bars and latency_drift_bps is strictly positive somewhere;
- the mid zigzags one tick per second (up, up, down, down), giving the trailing
  EWMA a real vol and letting horizontal barriers fire.
"""
from __future__ import annotations

import datetime as _dt
import hashlib

import pyarrow as pa
import pyarrow.parquet as pq

DAY_NS = 86_400 * 10**9
RECEIVED_LAG_NS = 2_000_000  # snapshot/seed lag only: observable before any bar


def event_lag_ns(seq: int) -> int:
    """Deterministic per-event capture lag, cycling 1 ms .. ~901 ms on seq."""
    return 1_000_000 + (seq % 7) * 150_000_000

TICK = 0.01
BASE_MID = 100.0
DEPTH_PER_SIDE = 6          # seeded levels beyond the zigzag churn; top_k=3 always covered
LEVEL_SIZE = 5.0

TRADE_NOTIONAL = 60.0       # price * amount per synthetic taker print
TRADES_PER_SEC = 10
DELTAS_PER_SEC = 5
ACTIVE_SECONDS = 60         # activity at the start of the day; the tail is quiet

# One bar every ~12 trades once the trailing schedule is live (day notional
# 60 * 10 * 60 = 36_000; threshold = 36_000 / 50 = 720).
TARGET_BARS_PER_DAY = 50
SEED_THRESHOLD = 720.0
# 7h deliberately does NOT divide the 24h day: the final [21:00, 28:00) interval is
# truncated by day end, so a late trade yields a real CLOSE_DAY_END bar (T9 masks it).
TIME_CAP_NS = 25_200_000_000_000


def day_open_ns(day: str) -> int:
    d = _dt.date.fromisoformat(day)
    dt = _dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc)
    return int(dt.timestamp()) * 10**9


class SyntheticWorld:
    """Deterministic book/trade generator threading state across days."""

    def __init__(self) -> None:
        self._mid_ticks = round(BASE_MID / TICK)   # mid in integer ticks
        self._books: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}
        self._seq = 0
        self._reseed_book()

    def _reseed_book(self) -> None:
        self._books = {"bid": {}, "ask": {}}
        for i in range(DEPTH_PER_SIDE):
            self._books["bid"][round((self._mid_ticks - 1 - i) * TICK, 10)] = LEVEL_SIZE
            self._books["ask"][round((self._mid_ticks + 1 + i) * TICK, 10)] = LEVEL_SIZE

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def best(self, side: str) -> float:
        prices = self._books[side]
        return max(prices) if side == "bid" else min(prices)

    def snapshot_rows(self, day: str) -> list[dict]:
        """The day-open book seed: full current state, prior-day origin stamps."""
        origin = day_open_ns(day) - 1_000_000_000
        rows = []
        for side in ("bid", "ask"):
            for price, size in sorted(self._books[side].items()):
                rows.append({"origin_time": origin, "received_time": origin + RECEIVED_LAG_NS,
                             "seq": self._next_seq(), "side": side,
                             "price": price, "size": size})
        return rows

    def _delta_row(self, origin: int, side: str, price: float, size: float) -> dict:
        seq = self._next_seq()
        return {"origin_time": origin, "received_time": origin + event_lag_ns(seq),
                "seq": seq, "side": side, "price": price, "size": size}

    def _shift_mid(self, origin: int, up: bool) -> list[dict]:
        """Move the whole top of book one tick via explicit add/remove deltas."""
        rows = []
        self._mid_ticks += 1 if up else -1
        for side, direction in (("bid", -1), ("ask", +1)):
            best = round((self._mid_ticks + direction) * TICK, 10)
            tail = round((self._mid_ticks + direction * DEPTH_PER_SIDE) * TICK, 10)
            book = self._books[side]
            if best not in book:
                book[best] = LEVEL_SIZE
                rows.append(self._delta_row(origin, side, best, LEVEL_SIZE))
            crossing = [p for p in book
                        if ((p > best) if side == "bid" else (p < best))]
            for p in crossing:
                del book[p]
                rows.append(self._delta_row(origin, side, p, 0.0))
            if tail not in book:
                book[tail] = LEVEL_SIZE
                rows.append(self._delta_row(origin, side, tail, LEVEL_SIZE))
        return rows

    def _window_deltas(self, open_ns: int, start_offset_ns: int, seconds: int,
                       first_delta_offset_ns: int) -> list[dict]:
        deltas: list[dict] = []
        step = 10**9 // DELTAS_PER_SEC
        for i in range(seconds * DELTAS_PER_SEC):
            origin = open_ns + start_offset_ns + first_delta_offset_ns + i * step
            second = i // DELTAS_PER_SEC
            if i % DELTAS_PER_SEC == 0 and second > 0:
                deltas.extend(self._shift_mid(origin, up=(second % 4) in (1, 2)))
            else:
                side = "bid" if i % 2 == 0 else "ask"
                price = self.best(side)
                size = LEVEL_SIZE + (0.5 if i % 4 < 2 else -0.5)
                self._books[side][price] = size
                deltas.append(self._delta_row(origin, side, price, size))
        return deltas

    def _window_trades(self, open_ns: int, start_offset_ns: int,
                       seconds: int) -> list[dict]:
        trades: list[dict] = []
        t_step = 10**9 // TRADES_PER_SEC
        for i in range(seconds * TRADES_PER_SEC):
            origin = open_ns + start_offset_ns + 100_000_000 + i * t_step + 1
            side = "buy" if i % 2 == 0 else "sell"
            price = self.best("ask") if side == "buy" else self.best("bid")
            seq = self._next_seq()
            trades.append({"origin_time": origin,
                           "received_time": origin + event_lag_ns(seq),
                           "seq": seq, "side": side,
                           "price": price, "quantity": TRADE_NOTIONAL / price})
        return trades

    def day_events(self, day: str, *, first_delta_offset_ns: int = 100_000_000,
                   late_active: bool = False, late_trade: bool = False,
                   one_sided_snapshot: bool = False, midnight_burst: bool = False
                   ) -> tuple[list[dict], list[dict], list[dict]]:
        """(snapshot_rows, delta_rows, trade_rows) for one day, advancing state.

        late_active adds a second active window at 23:55:00-23:59:00 so bars form
        close enough to the day boundary for per-horizon prefilter/coverage drops;
        late_trade injects one trade at 23:59:30, inside the truncated final cap
        interval, forcing a CLOSE_DAY_END bar; one_sided_snapshot omits the ask
        side from the day's seed object (early bars reject as one_sided_book
        until churn deltas restore the ask ladder); midnight_burst injects a
        threshold-crossing trade run just before midnight whose last capture lag
        crosses into the next day, so the closing bar's monotone watermark lands
        past the day end (the T9 boundary-spillover mask must drop it)."""
        open_ns = day_open_ns(day)
        snapshot = self.snapshot_rows(day)
        if one_sided_snapshot:
            snapshot = [r for r in snapshot if r["side"] == "bid"]
        deltas = self._window_deltas(open_ns, 0, ACTIVE_SECONDS, first_delta_offset_ns)
        trades = self._window_trades(open_ns, 0, ACTIVE_SECONDS)
        if late_active:
            late_start = DAY_NS - 300_000_000_000  # 23:55:00, 240s of activity
            deltas += self._window_deltas(open_ns, late_start, 240, 0)
            trades += self._window_trades(open_ns, late_start, 240)
        if late_trade:
            origin = open_ns + DAY_NS - 30_000_000_000  # 23:59:30: lands in the
            trades.append({"origin_time": origin,       # truncated final interval
                           "received_time": origin + RECEIVED_LAG_NS,
                           "seq": self._next_seq(), "side": "buy",
                           "price": self.best("ask"),
                           "quantity": TRADE_NOTIONAL / self.best("ask")})
        if midnight_burst:
            # enough rapid prints to cross the trailing threshold several times,
            # every one captured ~700ms late: each burst-closing bar's watermark
            # lands past midnight while every origin stays inside the day
            n_burst = 40
            for i in range(n_burst):
                origin = open_ns + DAY_NS - 500_000_000 + i * 10_000_000
                trades.append({"origin_time": origin,
                               "received_time": origin + 700_000_000,
                               "seq": self._next_seq(), "side": "buy",
                               "price": self.best("ask"),
                               "quantity": TRADE_NOTIONAL / self.best("ask")})
        return snapshot, deltas, trades


_L2_SCHEMA = pa.schema([("origin_time", pa.int64()), ("received_time", pa.int64()),
                        ("seq", pa.int64()), ("side", pa.string()),
                        ("price", pa.float64()), ("size", pa.float64())])
_TRADE_SCHEMA = pa.schema([("origin_time", pa.int64()), ("received_time", pa.int64()),
                           ("seq", pa.int64()), ("side", pa.string()),
                           ("price", pa.float64()), ("quantity", pa.float64())])


def _write_rows(rows: list[dict], schema: pa.Schema, path) -> str:
    cols = {name: [r[name] for r in rows] for name in schema.names}
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_day_objects(world: SyntheticWorld, day: str, directory, **day_kwargs
                      ) -> tuple[dict, dict]:
    """Write the day's three normalized objects; return (paths, sha256s) keyed by
    the normalized product names."""
    snapshot, deltas, trades = world.day_events(day, **day_kwargs)
    paths, shas = {}, {}
    for product, rows, schema in (
            ("binance_futures_l2_snapshot", snapshot, _L2_SCHEMA),
            ("binance_futures_l2_delta", deltas, _L2_SCHEMA),
            ("binance_futures_trades", trades, _TRADE_SCHEMA)):
        path = directory / f"{product}-{day}.parquet"
        shas[product] = _write_rows(rows, schema, path)
        paths[product] = path
    return paths, shas
