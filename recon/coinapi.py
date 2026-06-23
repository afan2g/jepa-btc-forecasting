"""CoinAPI `limitbook_full` (L3, order-by-order) replay → aggregated top-K L2.

Consumes the Parquet schema written by `ingest/download_coinapi.py` (docs/data.md §4.3):

    seq, time_exchange_ns, time_coinapi_ns, update_type, is_buy, entry_px, entry_sx, order_id

and reconstructs a top-K L2 book on an exchange-time grid for the one-day Coinbase vendor
parity gate (docs/data.md §5a hard gate #1). It is the CoinAPI counterpart of the Crypto
Lake `book_delta_v2` reconstructor (`recon.reconstruct.reconstruct_book_at_samples`), and
shares the SAME `sample_topk_as_of` sampler so the two vendors are sampled identically.

Ordering rule (mandatory — docs/data.md §4.3):
  * Replay strictly in **`seq` order** (file/row order). `seq` is canonical; there is no
    `sequence_number`. We process rows in the order the chunks deliver them and only *check*
    that `seq` is non-decreasing (a `seq_disorder` counter), never re-sort by it.
  * The opening **SNAPSHOT** block (lowest `seq`) is the initial book state for the
    partition day and is applied BEFORE every non-snapshot event, even though its
    `time_exchange` carries the prior-day close (e.g. 23:59:59.999). We therefore clamp the
    SNAPSHOT rows' label time to the partition-day open (`dt` midnight), and use
    `dt + time_exchange_ns` only as the display/label time for non-snapshot events. A
    non-decreasing watermark over the label time drives sampling, so a stray backward
    time stamp can never sort the snapshot to the end of the day.

⚠️ VENDOR-SEMANTICS ASSUMPTION — pending live validation (docs/data.md §5a hard gate).
`update_type` is an OPEN string set {SNAPSHOT, ADD, DELETE, MATCH, SET, SUB, …} (store as
string, never an enum). Their book meaning (CoinAPI L3): SNAPSHOT seeds the book, ADD posts
a resting order, DELETE cancels it, SET sets its state, SUB is a partial fill (size reduced),
MATCH is an execution against it. The exact *size convention* of the reducing events
(SUB/MATCH) is NOT yet confirmed against real Coinbase data, so it is selectable:

  * `size_policy="absolute"` (DEFAULT): every row reports the order's resulting resting size
    at `entry_px` (size 0 ⇒ remove). Self-consistent with ADD/SET/SNAPSHOT and with the Lake
    `book_delta_v2` "size is absolute, 0 removes" rule, and drift-free.
  * `size_policy="decrement"`: SUB/MATCH carry the amount REMOVED (subtract from the order's
    current size; ≤0 ⇒ remove).

The one-day pilot can run BOTH and keep whichever yields an uncrossed, parity-matching book —
that empirical A/B is exactly what this gate exists to resolve. State-defining events
(SNAPSHOT/ADD/SET) on an unseen `order_id` create it; reducing events (SUB/MATCH/DELETE) on
an unseen `order_id` are a missing-seed signal — counted as `missing_order` and skipped, never
fabricated. Unknown update types are counted (`unknown:<TYPE>`) and skipped, or raise under
`on_unknown="raise"`. Memory: chunk-streaming — only per-order state + the tiny per-sample
output is held; a multi-GB day is never materialized.
"""
from __future__ import annotations
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from recon.orderbook import OrderBook
from recon.reconstruct import sample_topk_as_of

# Sizes below EPS are treated as "no liquidity" (level removed). Guards float residue from
# repeated add/subtract on aggregated price levels.
EPS = 1e-9


class L3Book:
    """Order-by-order (L3) book keyed by `order_id`, aggregated to price levels.

    Per-order state lives in `self.orders[order_id] = [side, price, size]`; the aggregated
    price→size levels live in an internal `OrderBook`, so `snapshot(k)` returns the EXACT
    same schema as the Lake reconstructor (byte-identical parity columns). `q` is a Counter
    of data-quality signals (unknown types, missing orders, crossed/regression counts)."""

    def __init__(self, *, size_policy: str = "absolute", on_unknown: str = "count") -> None:
        if size_policy not in ("absolute", "decrement"):
            raise ValueError(f"size_policy must be 'absolute' or 'decrement', got {size_policy!r}")
        if on_unknown not in ("count", "raise"):
            raise ValueError(f"on_unknown must be 'count' or 'raise', got {on_unknown!r}")
        self.size_policy = size_policy
        self.on_unknown = on_unknown
        self.orders: dict[str, list] = {}
        self._l2 = OrderBook()
        self.q: Counter = Counter()

    # -- level math ---------------------------------------------------------------
    def _level(self, side: str) -> dict:
        return self._l2.bids if side == "bid" else self._l2.asks

    def _add_to_level(self, side: str, price: float, delta: float) -> None:
        book = self._level(side)
        new = book.get(price, 0.0) + delta
        if new <= EPS:
            book.pop(price, None)
        else:
            book[price] = new

    # -- per-order ops ------------------------------------------------------------
    def _set_order(self, oid: str, side: str, price: float, size: float) -> None:
        """Set the order to absolute `size` at `price` (size ≤ EPS removes it)."""
        old = self.orders.pop(oid, None)
        if old is not None:
            self._add_to_level(old[0], old[1], -old[2])
        if size > EPS:
            self.orders[oid] = [side, price, size]
            self._add_to_level(side, price, size)

    def _remove_order(self, oid: str) -> bool:
        old = self.orders.pop(oid, None)
        if old is None:
            self.q["missing_order"] += 1
            return False
        self._add_to_level(old[0], old[1], -old[2])
        return True

    def _decrement_order(self, oid: str, amount: float) -> None:
        old = self.orders.get(oid)
        if old is None:
            self.q["missing_order"] += 1
            return
        new = old[2] - amount
        if new <= EPS:
            self._remove_order(oid)
        else:
            self._add_to_level(old[0], old[1], -amount)
            old[2] = new

    # -- dispatch -----------------------------------------------------------------
    def apply(self, ut: str, side: str, price: float, size: float, oid: str) -> None:
        self.q["total_rows"] += 1
        if ut == "SNAPSHOT":
            self.q["snapshot_rows"] += 1
            self._set_order(oid, side, price, size)
        elif ut == "ADD":
            self.q["add"] += 1
            if oid in self.orders:
                self.q["add_existing"] += 1  # duplicate ADD; absolute set still correct
            self._set_order(oid, side, price, size)
        elif ut == "SET":
            self.q["set"] += 1
            if oid not in self.orders:
                self.q["set_missing"] += 1   # state-defining ⇒ create from absolute size
            self._set_order(oid, side, price, size)
        elif ut == "DELETE":
            self.q["delete"] += 1
            self._remove_order(oid)
        elif ut in ("SUB", "MATCH"):
            self.q[ut.lower()] += 1
            if self.size_policy == "absolute":
                # entry_sx = order's resulting resting size (0 ⇒ fully consumed).
                if oid in self.orders:
                    self._set_order(oid, side, price, size)
                else:
                    self.q["missing_order"] += 1  # reducing op on unseen order: skip
            else:  # decrement: entry_sx = amount removed
                self._decrement_order(oid, size)
        else:
            self.q["unknown_total"] += 1
            self.q[f"unknown:{ut}"] += 1
            if self.on_unknown == "raise":
                raise ValueError(
                    f"unknown CoinAPI update_type {ut!r} (open set; on_unknown='raise')"
                )
            # count + skip (book unchanged)

    def snapshot(self, k: int) -> dict:
        return self._l2.snapshot(k)


# ----------------------------------------------------------------------------- replay
def _iter_actions(chunks, book: L3Book, day_open_ns: int):
    """Yield (label_ns, update_type, side, price, size, order_id) in file/`seq` order,
    with SNAPSHOT label clamped to the partition-day open and a non-decreasing watermark on
    the label time. Vectorizes the per-row time/side math per chunk; the apply loop stays
    sequential (the book is a state machine). Updates `book.q` seq/time-order counters."""
    cur = None
    last_seq = None
    for df in chunks:
        if len(df) == 0:
            continue
        seq = df["seq"].to_numpy()
        uts = df["update_type"].astype(str).to_numpy()
        isb = df["is_buy"].to_numpy().astype(bool)
        pxs = df["entry_px"].to_numpy().astype("float64")
        sxs = df["entry_sx"].to_numpy().astype("float64")
        oids = df["order_id"].astype(str).to_numpy()
        txs = df["time_exchange_ns"].to_numpy().astype("int64")
        is_snap = uts == "SNAPSHOT"
        labels = np.where(is_snap, day_open_ns, day_open_ns + txs)
        sides = np.where(isb, "bid", "ask")
        for i in range(len(df)):
            s = int(seq[i])
            if last_seq is not None and s <= last_seq:
                book.q["seq_disorder"] += 1
            last_seq = s
            lab = int(labels[i])
            if cur is not None and lab < cur:
                book.q["time_regressions"] += 1
                lab = cur  # watermark: never sample backwards
            cur = lab if cur is None else max(cur, lab)
            yield (cur, uts[i], sides[i], float(pxs[i]), float(sxs[i]), oids[i])


def reconstruct_coinapi_l2_at_samples(
    chunks, *, k: int, day, sample_ts, size_policy: str = "absolute",
    on_unknown: str = "count",
) -> tuple[pd.DataFrame, dict]:
    """Replay CoinAPI L3 `chunks` → top-K L2 sampled at each ts in `sample_ts` (sorted int
    ns). Returns `(frame, quality)`:

      * `frame`: one row per sample, columns `sample_ts, mid, microprice,
        bid_i_price/size, ask_i_price/size` — identical schema to the Lake reconstructor.
      * `quality`: per-event counters (snapshot_rows/add/set/sub/match/delete, missing_order,
        unknown:*, seq_disorder, time_regressions, …) plus sample-level crossed-book and
        missing-book rates measured on `frame`.

    `chunks` is an iterable of downloader-schema DataFrames in `seq` order (stream Parquet
    row-groups for memory safety), or a single DataFrame. `day` is the partition date
    (`datetime.date`/parseable) used for the day-open clamp and label times."""
    if isinstance(chunks, pd.DataFrame):
        chunks = [chunks]
    book = L3Book(size_policy=size_policy, on_unknown=on_unknown)
    day_open_ns = int(pd.Timestamp(day).value)  # midnight UTC of the partition day, ns
    sample_ts = [int(t) for t in sample_ts]

    frame = sample_topk_as_of(
        sample_ts, _iter_actions(chunks, book, day_open_ns), k=k, book=book,
        apply=lambda b, e: b.apply(e[1], e[2], e[3], e[4], e[5]),
        time_of=lambda e: e[0],
    )

    q = dict(book.q)
    n = len(frame)
    if n:
        bid0, ask0 = frame["bid_0_price"], frame["ask_0_price"]
        both = bid0.notna() & ask0.notna()
        crossed = both & (bid0 >= ask0)
        missing = bid0.isna() | ask0.isna()
        q["crossed_samples"] = int(crossed.sum())
        q["crossed_rate"] = float(crossed.sum() / n)
        q["missing_book_samples"] = int(missing.sum())
        q["missing_book_fraction"] = float(missing.sum() / n)
    else:
        q.update(crossed_samples=0, crossed_rate=0.0,
                 missing_book_samples=0, missing_book_fraction=0.0)
    q["n_samples"] = n
    q["resting_orders_final"] = len(book.orders)
    q["day_open_ns"] = day_open_ns
    q["size_policy"] = size_policy
    return frame, q


def coinapi_frame_from_rows(rows: Iterable[dict]) -> pd.DataFrame:
    """Build a downloader-schema DataFrame from dict rows (test/fixture helper). Fills `seq`
    from row order when omitted and coerces dtypes to match `download_coinapi.py`'s output."""
    df = pd.DataFrame(list(rows))
    if "seq" not in df.columns:
        df["seq"] = range(len(df))
    # Add absent columns AND fill per-row gaps (rows that omit a key) with the schema default
    # so the int casts below never hit a NaN.
    for col, default in (("time_exchange_ns", 0), ("time_coinapi_ns", 0),
                         ("entry_px", 0.0), ("entry_sx", 0.0), ("order_id", ""),
                         ("update_type", ""), ("is_buy", True)):
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    return df.astype({
        "seq": "int64", "time_exchange_ns": "int64", "time_coinapi_ns": "int64",
        "update_type": "string", "is_buy": "bool", "entry_px": "float64",
        "entry_sx": "float64", "order_id": "string",
    })[["seq", "time_exchange_ns", "time_coinapi_ns", "update_type", "is_buy",
        "entry_px", "entry_sx", "order_id"]]
