# Recon: Event-Time Reconstruction + Replay-Equivalence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase-0 recon substrate — a single event-time order-book reconstruction function (shared by offline + live) that produces book-state-at-trade with **no lookahead**, proven by a **byte-identical replay-equivalence test** — plus the `book_delta_v2` schema verification it depends on.

**Architecture:** Pure-Python, correctness-first (the throughput-critical loop is ported to Rust later — spec §3). Raw Crypto Lake rows are normalized into typed `Delta`/`Trade` events by an ingest adapter; events are merged onto one **total-ordered engine-time axis**; an `OrderBook` state machine replays deltas; reconstruction snapshots the book at each trade using **strict-`<` apply-before-read** on the global order key. A `LiveReconstructor` with a bounded-out-of-orderness watermark must reproduce the offline result exactly.

**Tech Stack:** Python 3.12, pandas + pyarrow (already in `.venv`), pytest (added in Task 0), `lakeapi` (already installed) for the one schema-verification pull. No `sortedcontainers` dep — the book uses plain dicts with `min`/`max` (perf deferred to the Rust port).

**Scope:** This plan covers experiment-plan **E0.2** (book_delta_v2 schema verify — Task 1) and **E0.1** (recon + replay-equivalence — Tasks 0, 2–7). E0.3–E0.5 (bars, labels/CV, cost-PnL) are separate later plans.

---

## The order & snapshot convention (THE load-bearing decision — read before coding)

Spec §5.3 demands a single fixed convention applied identically offline and live. This plan fixes it as:

- Every event gets a **total order key**: `(ts_engine, kind, seq)` where `kind = 0` for a book delta and `kind = 1` for a trade. So **at equal `ts_engine`, deltas sort before trades.**
  - Delta: `seq = sequence_number`. Trade: `seq = id`.
- **`book_state_at(trade)` = the OrderBook after applying exactly the deltas whose order key is strictly less than the trade's order key.** The triggering trade's own market impact and all later events are **excluded**; same-timestamp book updates are **included** (because `(ts,0,·) < (ts,1,·)`).
- **No-lookahead property (testable):** deleting every delta with order key `≥` a trade's key must not change that trade's reconstructed snapshot.
- `ts_engine` = exchange/origin time when populated, else capture/received time (decided per-stream by Task 1's §4 check). The *same* column choice is used offline and live.

This convention is encoded once in `recon/events.py::order_key` and reused everywhere.

---

## File structure

- `pyproject.toml` — package + pytest config (Task 0)
- `recon/__init__.py` — package marker (Task 0)
- `recon/synthetic.py` — deterministic synthetic delta/trade stream generator for tests (Task 0)
- `recon/events.py` — `Delta`, `Trade` NamedTuples + `order_key` (Task 2)
- `recon/ingest.py` — raw Lake DataFrame → normalized event lists; schema assertion + origin/received fallback (Task 2)
- `recon/orderbook.py` — `OrderBook` state machine + `snapshot()` (Task 3)
- `recon/merge.py` — `merge_sorted` total-order merge (Task 4)
- `recon/reconstruct.py` — `reconstruct_book_at_trades` offline (Task 5)
- `recon/live.py` — `LiveReconstructor` with watermark (Task 6)
- `scripts/verify_book_delta_v2.py` — the E0.2 schema pull + fixture capture (Task 1)
- `tests/conftest.py` — fixture paths (Task 0)
- `tests/test_schema_book_delta_v2.py` — §4 schema/origin_time assertions (Task 1)
- `tests/test_events.py`, `test_ingest.py`, `test_orderbook.py`, `test_merge.py`, `test_reconstruct_no_lookahead.py`, `test_replay_equivalence.py`, `test_fixture_integration.py` (Tasks 2–7)
- `tests/fixtures/` — captured real samples (Task 1)

---

## Verified schemas (from cached qnt.dat data, 2026-06-22)

- **`trades`** (Binance): `side`(buy/sell), `amount`(float), `price`(float), `id`(Int64), `timestamp`(ns Int64, exchange), `receipt_timestamp`(ns Int64, capture), `exchange`, `symbol`, `dt`. — `timestamp` is populated.
- **`book`** (Binance, 20-level periodic snapshot): `timestamp`, `receipt_timestamp`, `sequence_number`, `bid_0..19_price/size`, `ask_0..19_price/size`, … — used only as an optional reconstruction seed, not the delta source.
- **`book_delta_v2`**: **NOT yet verified — Task 1 captures it.** Expected (Crypto Lake docs / spec §4): incremental L2 with `sequence_number`, per-update `side`/`price`/`size`, and `origin_time` (often empty for order books → fall back to `received_time`). Task 1 confirms exact column names and origin_time population.

---

## Task 0: Project scaffolding + deterministic synthetic generator

**Files:**
- Create: `pyproject.toml`, `recon/__init__.py`, `recon/synthetic.py`, `tests/conftest.py`

- [ ] **Step 1: Create the package config**

`pyproject.toml`:
```toml
[project]
name = "jepa-btc"
version = "0.0.1"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.setuptools.packages.find]
include = ["recon*"]
```

- [ ] **Step 2: Create package markers**

`recon/__init__.py`:
```python
"""Event-time order-book reconstruction (Phase 0, spec §5.3)."""
```

- [ ] **Step 3: Install pytest into the existing venv**

Run: `.venv/bin/python -m pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 4: Write the deterministic synthetic generator**

`recon/synthetic.py`:
```python
"""Deterministic synthetic delta/trade streams for tests (no RNG state leakage).

A 'world' is a list of (kind, ts_engine, seq, side, price, size_or_amount) tuples
in true causal order. We build small, hand-verifiable books so tests can assert
exact reconstructed snapshots.
"""
from __future__ import annotations


def simple_world():
    """A tiny, fully hand-checkable stream.

    Timeline (ts in ns), deltas (kind d) seed/modify a 2-level book, trades (kind t)
    occur between updates. Returns (deltas, trades) as raw dict rows matching the
    NORMALIZED event field names used by recon.events.

    Book evolution:
      ts=10 d bid 100@2 ; ts=10 d ask 101@3       -> mid 100.5
      ts=20 t buy 101@0.5  (sees mid 100.5)
      ts=30 d bid 100@0    (remove) ; d bid  99@1  -> best bid 99
      ts=40 t sell 99@0.7  (sees bid 99 / ask 101)
      ts=50 d ask 101@0 (remove) ; d ask 102@4     -> best ask 102
      ts=60 t buy 102@0.2  (sees bid 99 / ask 102)
    """
    deltas = [
        dict(ts_engine=10, seq=1, side="bid", price=100.0, size=2.0),
        dict(ts_engine=10, seq=2, side="ask", price=101.0, size=3.0),
        dict(ts_engine=30, seq=3, side="bid", price=100.0, size=0.0),
        dict(ts_engine=30, seq=4, side="bid", price=99.0, size=1.0),
        dict(ts_engine=50, seq=5, side="ask", price=101.0, size=0.0),
        dict(ts_engine=50, seq=6, side="ask", price=102.0, size=4.0),
    ]
    trades = [
        dict(ts_engine=20, seq=1001, side="buy", price=101.0, amount=0.5),
        dict(ts_engine=40, seq=1002, side="sell", price=99.0, amount=0.7),
        dict(ts_engine=60, seq=1003, side="buy", price=102.0, amount=0.2),
    ]
    return deltas, trades
```

- [ ] **Step 5: Create the test fixtures config**

`tests/conftest.py`:
```python
import pathlib
FIXTURES = pathlib.Path(__file__).parent / "fixtures"
```

- [ ] **Step 6: Verify pytest collects (no tests yet is fine)**

Run: `.venv/bin/python -m pytest`
Expected: `no tests ran` (exit code 5) — confirms config is valid.

- [ ] **Step 7: Commit**
```bash
git init -q 2>/dev/null; git add pyproject.toml recon/ tests/conftest.py
git commit -m "chore: scaffold recon package + synthetic test world"
```

---

## Task 1: E0.2 — Verify book_delta_v2 schema + capture fixtures

**Files:**
- Create: `scripts/verify_book_delta_v2.py`, `tests/test_schema_book_delta_v2.py`

**Note:** This is the one task needing live Lake access. The script is concrete; the test **skips** if the fixture is absent so the rest of the suite runs offline. Run the script once, record findings (origin_time populated? exact columns? seed-snapshot present?) in the memory file `crypto-lake-access-state`.

- [ ] **Step 1: Write the capture script**

`scripts/verify_book_delta_v2.py`:
```python
"""E0.2: pull a minimal book_delta_v2 + trades sample, verify schema, capture fixtures.

Spec §4 check #1: is origin_time populated for Binance book_delta_v2? If empty
(0/-1), reconstruction falls back to received_time. Writes small parquet fixtures.
"""
import datetime as dt
import pathlib
import boto3
import lakeapi

OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)
sess = boto3.Session(region_name="eu-west-1")

# A 2-minute window AFTER Binance-futures book history start (2022-11-14, spec §4).
start = dt.datetime(2022, 11, 15, 0, 0, 0)
end = dt.datetime(2022, 11, 15, 0, 2, 0)

print("used_data BEFORE:", lakeapi.used_data(sess))

deltas = lakeapi.load_data(
    table="book_delta_v2", start=start, end=end,
    symbols=["BTC-USDT-PERP"], exchanges=["BINANCE_FUTURES"], boto3_session=sess,
)
trades = lakeapi.load_data(
    table="trades", start=start, end=end,
    symbols=["BTC-USDT-PERP"], exchanges=["BINANCE_FUTURES"], boto3_session=sess,
)

print("delta rows:", len(deltas), "cols:", list(deltas.columns))
print("delta dtypes:\n", deltas.dtypes)
print("delta head:\n", deltas.head(5).to_string())

# §4 origin_time population check.
for col in ("origin_time", "received_time", "timestamp", "receipt_timestamp"):
    if col in deltas.columns:
        empty = (deltas[col].astype("int64") <= 0).mean()
        print(f"  {col}: present, fraction<=0 = {empty:.3%}")

# Capture small fixtures (first ~5k delta rows, all trades in window).
deltas.head(5000).to_parquet(OUT / "book_delta_v2_sample.parquet")
trades.to_parquet(OUT / "trades_sample.parquet")
print("WROTE fixtures to", OUT)
print("used_data AFTER:", lakeapi.used_data(sess))
```

- [ ] **Step 2: Run the capture script (records the real schema)**

Run: `.venv/bin/python scripts/verify_book_delta_v2.py`
Expected: prints the real `book_delta_v2` columns + dtypes, the origin_time fraction-empty, and writes two parquet fixtures. **Record the exact column names and origin_time result** — they confirm the `recon/ingest.py` mapping in Task 2.

- [ ] **Step 3: Write the schema-assertion test**

`tests/test_schema_book_delta_v2.py`:
```python
import pytest
import pandas as pd
from tests.conftest import FIXTURES

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "book_delta_v2_sample.parquet").exists(),
    reason="run scripts/verify_book_delta_v2.py first (needs Lake access)",
)

def test_book_delta_v2_has_required_fields():
    df = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    # A usable incremental-L2 stream MUST give us: a sequence, a side, a price,
    # a size, and at least one engine-time column. Exact names confirmed in Task 1;
    # update this set to the observed names if Crypto Lake differs.
    have = set(df.columns)
    assert "sequence_number" in have
    assert {"side"} & have or {"is_bid"} & have, "no side/is_bid column"
    assert "price" in have
    assert "size" in have or "amount" in have, "no size/amount column"
    assert {"origin_time", "received_time", "timestamp", "receipt_timestamp"} & have

def test_engine_time_column_is_populated():
    df = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    # Pick the first engine-time column that is actually populated; the recon
    # adapter (Task 2) must use the SAME choice. Fails loudly if none are usable.
    candidates = [c for c in ("origin_time", "received_time", "timestamp",
                              "receipt_timestamp") if c in df.columns]
    usable = [c for c in candidates if (df[c].astype("int64") > 0).mean() > 0.99]
    assert usable, f"no populated engine-time column among {candidates}"
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_schema_book_delta_v2.py -v`
Expected: PASS (or SKIP if Lake access is unavailable — then revisit before Task 5).

- [ ] **Step 5: Commit**
```bash
git add scripts/verify_book_delta_v2.py tests/test_schema_book_delta_v2.py
git commit -m "feat: E0.2 verify book_delta_v2 schema + capture fixtures"
```

---

## Task 2: Normalized events + ingest adapter

**Files:**
- Create: `recon/events.py`, `recon/ingest.py`, `tests/test_events.py`, `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test for the order-key convention**

`tests/test_events.py`:
```python
from recon.events import Delta, Trade, order_key

def test_delta_sorts_before_trade_at_equal_ts():
    d = Delta(ts_engine=10, seq=2, side="bid", price=100.0, size=2.0)
    t = Trade(ts_engine=10, seq=1001, side="buy", price=101.0, amount=0.5)
    assert order_key(d) < order_key(t)

def test_order_within_kind_is_by_seq():
    d1 = Delta(ts_engine=10, seq=1, side="bid", price=100.0, size=2.0)
    d2 = Delta(ts_engine=10, seq=2, side="ask", price=101.0, size=3.0)
    assert order_key(d1) < order_key(d2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'recon.events'`

- [ ] **Step 3: Implement events**

`recon/events.py`:
```python
"""Normalized events + the single total-order convention (spec §5.3)."""
from __future__ import annotations
from typing import NamedTuple, Union


class Delta(NamedTuple):
    ts_engine: int
    seq: int
    side: str      # "bid" | "ask"
    price: float
    size: float    # absolute size at this level; 0.0 => remove the level


class Trade(NamedTuple):
    ts_engine: int
    seq: int
    side: str      # "buy" | "sell"
    price: float
    amount: float


Event = Union[Delta, Trade]


def order_key(ev: Event) -> tuple[int, int, int]:
    """Total order: (ts_engine, kind, seq). Deltas (kind=0) precede trades (kind=1)
    at equal ts_engine. book_state_at(trade) applies deltas with key < trade's key."""
    kind = 0 if isinstance(ev, Delta) else 1
    return (ev.ts_engine, kind, ev.seq)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_events.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the failing ingest test (synthetic round-trip + fixture)**

`tests/test_ingest.py`:
```python
import pandas as pd
import pytest
from recon.events import Delta, Trade
from recon.ingest import deltas_from_df, trades_from_df
from recon.synthetic import simple_world
from tests.conftest import FIXTURES


def test_deltas_from_df_normalizes_synthetic():
    draw, _ = simple_world()
    df = pd.DataFrame(draw).rename(columns={"ts_engine": "origin_time"})
    out = deltas_from_df(df, engine_time_col="origin_time")
    assert out[0] == Delta(ts_engine=10, seq=1, side="bid", price=100.0, size=2.0)
    assert all(isinstance(d, Delta) for d in out)
    assert [d.ts_engine for d in out] == [10, 10, 30, 30, 50, 50]


def test_trades_from_df_normalizes_synthetic():
    _, traw = simple_world()
    df = pd.DataFrame(traw).rename(columns={"ts_engine": "timestamp", "seq": "id"})
    out = trades_from_df(df, engine_time_col="timestamp")
    assert out[0] == Trade(ts_engine=20, seq=1001, side="buy", price=101.0, amount=0.5)


def test_ingest_rejects_unpopulated_engine_time():
    df = pd.DataFrame([dict(origin_time=0, seq=1, side="bid", price=1.0, size=1.0)])
    with pytest.raises(ValueError, match="engine-time"):
        deltas_from_df(df, engine_time_col="origin_time")


@pytest.mark.skipif(not (FIXTURES / "book_delta_v2_sample.parquet").exists(),
                    reason="needs Task 1 fixture")
def test_ingest_real_fixture_smoke():
    df = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    col = "origin_time" if (df.get("origin_time", pd.Series([0])).astype("int64") > 0).mean() > 0.99 else "received_time"
    out = deltas_from_df(df, engine_time_col=col)
    assert len(out) == len(df)
    assert all(d.ts_engine > 0 for d in out[:100])
```

- [ ] **Step 6: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'recon.ingest'`

- [ ] **Step 7: Implement ingest**

`recon/ingest.py`:
```python
"""Raw Crypto Lake DataFrame -> normalized event lists.

The exact source column names for book_delta_v2 are confirmed by Task 1
(scripts/verify_book_delta_v2.py). This adapter is the SINGLE schema-dependent
seam: update the SIDE_COL / SIZE_COL fallbacks below if Lake differs.
"""
from __future__ import annotations
import pandas as pd
from recon.events import Delta, Trade


def _side_str(v, *, bid_set=("bid", "b", "buy", True, 1, "1")) -> str:
    return "bid" if v in bid_set else "ask"


def _require_populated(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"engine-time column {col!r} not in {list(df.columns)}")
    if not (df[col].astype("int64") > 0).all():
        raise ValueError(f"engine-time column {col!r} has non-populated (<=0) rows")


def deltas_from_df(df: pd.DataFrame, *, engine_time_col: str) -> list[Delta]:
    _require_populated(df, engine_time_col)
    side_col = "side" if "side" in df.columns else "is_bid"
    size_col = "size" if "size" in df.columns else "amount"
    ts = df[engine_time_col].astype("int64").to_numpy()
    seq = df["sequence_number"].astype("int64").to_numpy()
    side = df[side_col].to_numpy()
    price = df["price"].astype("float64").to_numpy()
    size = df[size_col].astype("float64").to_numpy()
    return [Delta(int(ts[i]), int(seq[i]), _side_str(side[i]),
                  float(price[i]), float(size[i])) for i in range(len(df))]


def trades_from_df(df: pd.DataFrame, *, engine_time_col: str) -> list[Trade]:
    _require_populated(df, engine_time_col)
    ts = df[engine_time_col].astype("int64").to_numpy()
    seq = df["id"].astype("int64").to_numpy()
    side = df["side"].astype(str).to_numpy()
    price = df["price"].astype("float64").to_numpy()
    amount = df["amount"].astype("float64").to_numpy()
    return [Trade(int(ts[i]), int(seq[i]), str(side[i]),
                  float(price[i]), float(amount[i])) for i in range(len(df))]
```

- [ ] **Step 8: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: PASS (real-fixture test SKIPs if Task 1 not run)

- [ ] **Step 9: Commit**
```bash
git add recon/events.py recon/ingest.py tests/test_events.py tests/test_ingest.py
git commit -m "feat: normalized events + ingest adapter with engine-time guard"
```

---

## Task 3: OrderBook state machine

**Files:**
- Create: `recon/orderbook.py`, `tests/test_orderbook.py`

- [ ] **Step 1: Write the failing test**

`tests/test_orderbook.py`:
```python
from recon.events import Delta
from recon.orderbook import OrderBook


def apply_all(ob, deltas):
    for d in deltas:
        ob.apply(d)


def test_apply_add_and_best_prices():
    ob = OrderBook()
    apply_all(ob, [Delta(10, 1, "bid", 100.0, 2.0), Delta(10, 2, "ask", 101.0, 3.0)])
    assert ob.best_bid() == 100.0
    assert ob.best_ask() == 101.0
    assert ob.mid() == 100.5


def test_size_zero_removes_level():
    ob = OrderBook()
    apply_all(ob, [Delta(10, 1, "bid", 100.0, 2.0), Delta(20, 2, "bid", 100.0, 0.0),
                   Delta(20, 3, "bid", 99.0, 1.0)])
    assert ob.best_bid() == 99.0


def test_microprice_weights_by_opposite_size():
    ob = OrderBook()
    apply_all(ob, [Delta(10, 1, "bid", 100.0, 1.0), Delta(10, 2, "ask", 101.0, 3.0)])
    # microprice = (ask_sz*bid_px + bid_sz*ask_px)/(bid_sz+ask_sz)
    assert ob.microprice() == (3.0 * 100.0 + 1.0 * 101.0) / 4.0


def test_snapshot_top_k_sorted_and_padded():
    ob = OrderBook()
    apply_all(ob, [Delta(10, 1, "bid", 100.0, 2.0), Delta(10, 2, "bid", 99.0, 1.0),
                   Delta(10, 3, "ask", 101.0, 3.0)])
    snap = ob.snapshot(k=2)
    assert snap["bid_0_price"] == 100.0 and snap["bid_1_price"] == 99.0
    assert snap["ask_0_price"] == 101.0
    assert snap["ask_1_price"] is None  # padded when fewer than k levels


def test_gap_detection_on_nonmonotonic_sequence():
    ob = OrderBook()
    ob.apply(Delta(10, 5, "bid", 100.0, 2.0))
    assert ob.apply(Delta(10, 5, "bid", 100.0, 1.0)) is False  # seq not increasing => gap flag
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orderbook.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'recon.orderbook'`

- [ ] **Step 3: Implement OrderBook**

`recon/orderbook.py`:
```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orderbook.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**
```bash
git add recon/orderbook.py tests/test_orderbook.py
git commit -m "feat: OrderBook state machine (apply/remove/mid/microprice/snapshot/gap)"
```

---

## Task 4: Deterministic total-order merge

**Files:**
- Create: `recon/merge.py`, `tests/test_merge.py`

- [ ] **Step 1: Write the failing test**

`tests/test_merge.py`:
```python
from recon.events import order_key
from recon.merge import merge_sorted
from recon.ingest import deltas_from_df, trades_from_df
from recon.synthetic import simple_world
import pandas as pd


def _events():
    draw, traw = simple_world()
    d = deltas_from_df(pd.DataFrame(draw).rename(columns={"ts_engine": "origin_time"}),
                       engine_time_col="origin_time")
    t = trades_from_df(pd.DataFrame(traw).rename(columns={"ts_engine": "timestamp", "seq": "id"}),
                       engine_time_col="timestamp")
    return d, t


def test_merge_is_globally_sorted_by_order_key():
    d, t = _events()
    merged = merge_sorted(d, t)
    keys = [order_key(e) for e in merged]
    assert keys == sorted(keys)


def test_merge_is_order_invariant_to_input_permutation():
    d, t = _events()
    a = [order_key(e) for e in merge_sorted(d, t)]
    b = [order_key(e) for e in merge_sorted(list(reversed(d)), list(reversed(t)))]
    assert a == b
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'recon.merge'`

- [ ] **Step 3: Implement merge**

`recon/merge.py`:
```python
"""Merge event streams onto one total-ordered engine-time axis."""
from __future__ import annotations
from recon.events import Event, order_key


def merge_sorted(deltas: list[Event], trades: list[Event]) -> list[Event]:
    """Return all events in the single total order defined by recon.events.order_key.
    Deterministic and invariant to input ordering."""
    return sorted([*deltas, *trades], key=order_key)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_merge.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**
```bash
git add recon/merge.py tests/test_merge.py
git commit -m "feat: deterministic total-order merge of delta+trade streams"
```

---

## Task 5: Offline reconstruction + the no-lookahead property

**Files:**
- Create: `recon/reconstruct.py`, `tests/test_reconstruct_no_lookahead.py`

- [ ] **Step 1: Write the failing test (known-truth + no-lookahead)**

`tests/test_reconstruct_no_lookahead.py`:
```python
import pandas as pd
from recon.events import Delta, Trade, order_key
from recon.merge import merge_sorted
from recon.orderbook import OrderBook
from recon.reconstruct import reconstruct_book_at_trades
from recon.synthetic import simple_world
from recon.ingest import deltas_from_df, trades_from_df


def _events():
    draw, traw = simple_world()
    d = deltas_from_df(pd.DataFrame(draw).rename(columns={"ts_engine": "origin_time"}),
                       engine_time_col="origin_time")
    t = trades_from_df(pd.DataFrame(traw).rename(columns={"ts_engine": "timestamp", "seq": "id"}),
                       engine_time_col="timestamp")
    return d, t


def test_reconstruct_matches_hand_computed_snapshots():
    d, t = _events()
    out = reconstruct_book_at_trades(d, t, k=1)
    # trade @20 sees mid 100.5 ; @40 sees bid 99/ask 101 -> mid 100.0 ;
    # @60 sees bid 99/ask 102 -> mid 100.5
    assert list(out["trade_ts"]) == [20, 40, 60]
    assert list(out["mid"]) == [100.5, 100.0, 100.5]
    assert list(out["bid_0_price"]) == [100.0, 99.0, 99.0]
    assert list(out["ask_0_price"]) == [101.0, 101.0, 102.0]


def test_no_lookahead_dropping_future_deltas_is_invariant():
    """For each trade, deleting every delta with order key >= the trade's key must
    not change that trade's reconstructed snapshot (spec §5.3 lookahead guard)."""
    d, t = _events()
    full = reconstruct_book_at_trades(d, t, k=1)
    for i, tr in enumerate(t):
        kept = [x for x in d if order_key(x) < order_key(tr)]
        one = reconstruct_book_at_trades(kept, [tr], k=1)
        assert one["mid"].iloc[0] == full["mid"].iloc[i]
        assert one["bid_0_price"].iloc[0] == full["bid_0_price"].iloc[i]
        assert one["ask_0_price"].iloc[0] == full["ask_0_price"].iloc[i]


def test_trade_does_not_see_same_ts_later_kind_or_its_own_impact():
    # A delta with the SAME ts but kind=trade ordering must be excluded; a same-ts
    # delta (kind=0) must be included.
    d = [Delta(10, 1, "bid", 100.0, 2.0), Delta(10, 2, "ask", 101.0, 3.0),
         Delta(20, 3, "ask", 101.0, 0.0)]  # removes ask AT ts=20, same as the trade
    tr = [Trade(20, 1001, "buy", 101.0, 0.5)]
    out = reconstruct_book_at_trades(d, tr, k=1)
    # delta(20,kind0,seq3) < trade(20,kind1,seq1001) => the ask removal IS applied,
    # so the trade sees NO ask at 101 (best_ask None).
    assert out["ask_0_price"].iloc[0] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reconstruct_no_lookahead.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'recon.reconstruct'`

- [ ] **Step 3: Implement reconstruction**

`recon/reconstruct.py`:
```python
"""Offline event-time reconstruction: book-state-at-trade with strict-< apply-before-read."""
from __future__ import annotations
import pandas as pd
from recon.events import Delta, Trade
from recon.merge import merge_sorted
from recon.orderbook import OrderBook


def reconstruct_book_at_trades(deltas, trades, *, k: int, seed: OrderBook | None = None) -> pd.DataFrame:
    """Replay the merged stream; at each trade, emit the book snapshot AS OF that trade
    (all deltas with order key < the trade's key already applied; the trade's own impact
    and later events excluded). Returns one row per trade."""
    ob = seed if seed is not None else OrderBook()
    rows: list[dict] = []
    for ev in merge_sorted(deltas, trades):
        if isinstance(ev, Delta):
            ob.apply(ev)
        else:  # Trade -> snapshot the book it saw
            snap = ob.snapshot(k)
            snap["trade_ts"] = ev.ts_engine
            snap["trade_seq"] = ev.seq
            snap["trade_side"] = ev.side
            snap["trade_price"] = ev.price
            snap["trade_amount"] = ev.amount
            rows.append(snap)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reconstruct_no_lookahead.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**
```bash
git add recon/reconstruct.py tests/test_reconstruct_no_lookahead.py
git commit -m "feat: offline book-at-trade reconstruction + no-lookahead property tests"
```

---

## Task 6: ⭐ Live reconstructor + the replay-equivalence GATE

**Files:**
- Create: `recon/live.py`, `tests/test_replay_equivalence.py`

This is the E0.1 headline gate: the live path (watermark over out-of-order arrivals) must reproduce the offline reconstruction **exactly**.

- [ ] **Step 1: Write the failing test**

`tests/test_replay_equivalence.py`:
```python
import pandas as pd
from recon.events import Delta, Trade, order_key
from recon.reconstruct import reconstruct_book_at_trades
from recon.live import LiveReconstructor


def _bigger_world():
    """Deterministic stream with same-ts events and interleaving, no RNG."""
    deltas, trades = [], []
    seq_d = seq_t = 0
    price = 100.0
    for step in range(50):
        ts = 10 * (step + 1)
        seq_d += 1; deltas.append(Delta(ts, seq_d, "bid", price - 1, 1.0 + step % 3))
        seq_d += 1; deltas.append(Delta(ts, seq_d, "ask", price + 1, 1.0 + (step + 1) % 3))
        if step % 2 == 0:
            seq_t += 1; trades.append(Trade(ts, 100000 + seq_t, "buy", price + 1, 0.1))
        if step % 5 == 0 and step:  # occasionally move the book
            seq_d += 1; deltas.append(Delta(ts, seq_d, "bid", price - 1, 0.0))
            price += 1
    return deltas, trades


def _arrival_within_watermark(events, window_ns):
    """Permute events so none moves more than `window_ns` from its sorted ts position:
    sort, then reverse each contiguous block whose ts-span <= window_ns."""
    ev = sorted(events, key=order_key)
    out, i = [], 0
    while i < len(ev):
        j = i
        while j + 1 < len(ev) and ev[j + 1].ts_engine - ev[i].ts_engine <= window_ns:
            j += 1
        out.extend(reversed(ev[i:j + 1]))  # deterministic out-of-order within window
        i = j + 1
    return out


def test_live_equals_offline_exactly():
    deltas, trades = _bigger_world()
    offline = reconstruct_book_at_trades(deltas, trades, k=3).reset_index(drop=True)

    window = 30  # max out-of-orderness in the simulated feed
    arrival = _arrival_within_watermark([*deltas, *trades], window_ns=window)
    live = LiveReconstructor(k=3, watermark_ns=window)
    for ev in arrival:
        live.push(ev)
    online = live.flush().reset_index(drop=True)

    # Byte-identical: same columns, same dtypes, same values.
    pd.testing.assert_frame_equal(offline, online, check_dtype=True)


def test_live_handles_in_order_arrival_identically():
    deltas, trades = _bigger_world()
    offline = reconstruct_book_at_trades(deltas, trades, k=3).reset_index(drop=True)
    live = LiveReconstructor(k=3, watermark_ns=30)
    for ev in sorted([*deltas, *trades], key=order_key):
        live.push(ev)
    pd.testing.assert_frame_equal(offline, live.flush().reset_index(drop=True))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay_equivalence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'recon.live'`

- [ ] **Step 3: Implement the live reconstructor**

`recon/live.py`:
```python
"""Streaming reconstruction with a bounded-out-of-orderness watermark.

Buffers arrivals; releases events whose ts_engine <= (max_ts_seen - watermark_ns)
in total order, then snapshots at trades exactly as the offline path does. Given the
feed is never more than `watermark_ns` out of order, the released order == the global
sorted order, so output is byte-identical to recon.reconstruct (the E0.1 gate)."""
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_replay_equivalence.py -v`
Expected: PASS (2 passed) — **this is the E0.1 gate going green.**

- [ ] **Step 5: Commit**
```bash
git add recon/live.py tests/test_replay_equivalence.py
git commit -m "feat: live watermark reconstructor + replay-equivalence gate (E0.1)"
```

---

## Task 7: Real-fixture integration smoke test

**Files:**
- Create: `tests/test_fixture_integration.py`

Proves the pipeline runs end-to-end on the real captured `book_delta_v2` + `trades` sample and yields a sane book (no crossed market, monotone trade times).

- [ ] **Step 1: Write the test**

`tests/test_fixture_integration.py`:
```python
import pandas as pd
import pytest
from tests.conftest import FIXTURES
from recon.ingest import deltas_from_df, trades_from_df
from recon.reconstruct import reconstruct_book_at_trades

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "book_delta_v2_sample.parquet").exists(),
    reason="run scripts/verify_book_delta_v2.py first (needs Lake access)",
)


def _engine_col(df):
    for c in ("origin_time", "received_time", "timestamp", "receipt_timestamp"):
        if c in df.columns and (df[c].astype("int64") > 0).mean() > 0.99:
            return c
    raise AssertionError("no populated engine-time column")


def test_reconstruct_real_sample_is_sane():
    dd = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    tt = pd.read_parquet(FIXTURES / "trades_sample.parquet")
    deltas = deltas_from_df(dd, engine_time_col=_engine_col(dd))
    trades = trades_from_df(tt, engine_time_col=_engine_col(tt))
    out = reconstruct_book_at_trades(deltas, trades, k=10)
    assert len(out) > 0
    valid = out.dropna(subset=["bid_0_price", "ask_0_price"])
    # No crossed book once both sides exist.
    assert (valid["ask_0_price"] > valid["bid_0_price"]).all()
    # Trade timestamps are non-decreasing (total order preserved).
    assert out["trade_ts"].is_monotonic_increasing
```

- [ ] **Step 2: Run (PASS or SKIP without Lake fixture)**

Run: `.venv/bin/python -m pytest tests/test_fixture_integration.py -v`
Expected: PASS if Task 1 fixtures exist, else SKIP.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all PASS (Lake-dependent tests SKIP if fixtures absent).

- [ ] **Step 4: Commit**
```bash
git add tests/test_fixture_integration.py
git commit -m "test: end-to-end recon smoke on real book_delta_v2 fixture"
```

---

## Self-review (coverage of E0.1 / E0.2)

- **E0.2 schema verify** → Task 1 (pull + origin_time check + fixtures + schema assertions). ✓
- **E0.1 single shared recon function** → `reconstruct_book_at_trades` (offline) and `LiveReconstructor` (live) share `OrderBook`, `merge`, `order_key`, `ingest`. ✓
- **Strict-`<` apply-before-read / no lookahead** → `test_no_lookahead_dropping_future_deltas_is_invariant`, `test_trade_does_not_see_same_ts_later_kind_or_its_own_impact`. ✓
- **Replay-equivalence (byte-identical offline vs live)** → `test_replay_equivalence.py` (the gate). ✓
- **Sequence-gap detection** → `OrderBook.apply` return + `test_gap_detection_on_nonmonotonic_sequence`. ✓
- **origin_time→received_time fallback** → `_require_populated` + `engine_time_col` param, chosen by Task 1's finding. ✓

**Deferred (call out, not silently dropped):**
- **Seeding the book** from an initial snapshot: book_delta_v2's seeding semantics (snapshot flag vs accumulate-from-start) are confirmed in Task 1; the `seed` param on `reconstruct_book_at_trades` is the hook. If the real feed needs a `book`-table seed aligned by `sequence_number`, that's a one-task follow-up.
- **Cross-venue merge** (Binance + Coinbase on one axis) reuses `order_key`/`merge_sorted` with a per-stream `engine_time_col`; add when Coinbase (CoinAPI) ingest lands.
- **Performance** (1000+ level book, TB-scale): plain-dict `min`/`max` is correctness-first; the Rust port of `OrderBook`/`reconstruct` is the deferred optimization (spec §3). The replay-equivalence test becomes the cross-language conformance test.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-22-recon-replay-equivalence.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Note: **Task 1 needs live Crypto Lake access**; if the worker can't auth, it captures fixtures separately and the Lake-dependent tests SKIP (the synthetic-data core still fully validates E0.1).
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
