"""Offline event-time reconstruction: book-state-at-trade with strict-< apply-before-read."""
from __future__ import annotations
from typing import Callable, Iterable
import numpy as np
import pandas as pd
from recon.events import Delta, Trade, order_key
from recon.ingest import _pick, _require_populated, _side_str
from recon.merge import merge_sorted
from recon.orderbook import OrderBook


def sample_topk_as_of(sample_ts, ordered_events, *, k: int, book,
                      apply: Callable, time_of: Callable) -> pd.DataFrame:
    """Emit a top-K book snapshot AS OF each sample timestamp by replaying `ordered_events`.

    This is the SINGLE sampling convention shared by every vendor reconstructor (Crypto
    Lake `book_delta_v2` and CoinAPI `limitbook_full`), so two streams of the same book
    are sampled identically and a parity comparison reflects real divergence, not a
    sampler mismatch.

    Contract:
      * `sample_ts` is sorted ascending (int ns).
      * `ordered_events` is iterable in NON-DECREASING `time_of(ev)` order (the caller is
        responsible for the order — deltas via order_key, CoinAPI via seq + a watermark).
      * `book` exposes `.snapshot(k) -> dict`; `apply(book, ev)` mutates it in place;
        `time_of(ev)` returns the event's int-ns engine time.

    "As of g" = the book reflecting every event with `time_of(ev) <= g`. We emit sample g
    just before applying the first event with `time_of > g`; because events are
    non-decreasing, all events at/below g are already applied (apply-before-read, no
    look-ahead). Trailing samples get the final book."""
    sample_ts = list(sample_ts)
    n = len(sample_ts)
    rows: list[dict] = []
    si = 0
    for ev in ordered_events:
        t = time_of(ev)
        while si < n and sample_ts[si] < t:
            snap = book.snapshot(k)
            snap["sample_ts"] = int(sample_ts[si])
            rows.append(snap)
            si += 1
        apply(book, ev)
    while si < n:
        snap = book.snapshot(k)
        snap["sample_ts"] = int(sample_ts[si])
        rows.append(snap)
        si += 1
    return pd.DataFrame(rows)


def reconstruct_book_at_samples(deltas: Iterable[Delta], sample_ts, *, k: int,
                                seed: OrderBook | None = None) -> pd.DataFrame:
    """Replay `deltas` and emit the top-K L2 book AS OF each timestamp in `sample_ts`
    (sorted, int ns). Returns one row per sample with columns
    `sample_ts, mid, microprice, bid_i_price/size, ask_i_price/size` (i in 0..k-1).

    This is the book-on-a-grid analogue of `reconstruct_book_at_trades`: it snapshots at
    fixed sample times (e.g. a 1 s exchange-time grid) rather than at trades, which is what
    a label clock and a cross-vendor parity check consume. A `seed` book (carried across a
    day/file boundary, docs/data.md §5a-Recon) is COPIED, not mutated. Like the
    trade-snapshot path, this does NOT consume the sequence-gap signal — reseed-on-
    discontinuity is deferred (docs/data.md §5a)."""
    ob = OrderBook() if seed is None else seed.copy()
    ordered = sorted(deltas, key=order_key)
    return sample_topk_as_of(sample_ts, ordered, k=k, book=ob,
                             apply=lambda b, d: b.apply(d), time_of=lambda d: d.ts_engine)


def reconstruct_book_at_trades(deltas, trades, *, k: int, seed: OrderBook | None = None) -> pd.DataFrame:
    """Replay the merged stream; at each trade, emit the book snapshot AS OF that trade
    (all deltas with order key < the trade's key already applied; the trade's own impact
    and later events excluded). Returns one row per trade.

    A `seed` book (e.g. carried across a day/file boundary — docs/data.md §5a-Recon) is
    COPIED, not mutated, so the function is pure w.r.t. its inputs and reusing one seed
    across calls is safe. NOTE: this call does not return the post-replay book, so
    carrying state forward across segments is a deferred follow-up.

    Sequence-gap handling: OrderBook.apply() returns a per-delta monotonicity signal but
    it is intentionally NOT consumed here — reseed-on-discontinuity (docs/data.md §5a) is
    deferred (it needs the real book_delta_v2 sequence_number semantics from Task 1)."""
    ob = OrderBook() if seed is None else seed.copy()
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


def _decode_sides(side_arr: np.ndarray) -> np.ndarray:
    """Vectorized `side_is_bid`/side decode → array of "bid"/"ask". Fast path for the real
    Lake bool column; otherwise map each DISTINCT value through ingest._side_str so an
    unknown encoding still fails loudly (it just validates per unique value, not per row)."""
    if side_arr.dtype == bool:
        return np.where(side_arr, "bid", "ask")
    mapping = {v: _side_str(v) for v in pd.unique(side_arr)}
    return np.array([mapping[v] for v in side_arr], dtype=object)


def reconstruct_lake_l2_at_samples(df: pd.DataFrame, sample_ts, *, k: int,
                                   engine_time_col: str, seed: OrderBook | None = None) -> pd.DataFrame:
    """Memory-safe array path: a Crypto Lake `book_delta_v2` DataFrame → top-K L2 sampled at
    each ts in `sample_ts`, WITHOUT materializing a per-row `Delta` list.

    A single Coinbase `book_delta_v2` day is ~34 M rows (docs/data.md §6); building 34 M
    `Delta` NamedTuples would cost multiple GB. Instead we resolve the same column aliases as
    `recon.ingest.deltas_from_df`, sort by `(engine_time, sequence_number)` with `np.lexsort`
    (matching `recon.events.order_key` for deltas), and feed a LAZY `Delta` generator to the
    shared `sample_topk_as_of` sampler — so only columnar arrays + the index permutation are
    resident. Output schema is identical to `reconstruct_book_at_samples` (the CoinAPI side
    matches it too), which is what makes the cross-vendor parity comparison apples-to-apples."""
    _require_populated(df, engine_time_col)
    seq_col = _pick(df, ("sequence_number", "seq"), field="delta sequence")
    side_col = _pick(df, ("side_is_bid", "side", "is_bid"), field="delta side")
    size_col = _pick(df, ("size", "amount"), field="delta size")
    ts = df[engine_time_col].astype("int64").to_numpy()
    seq = df[seq_col].astype("int64").to_numpy()
    price = df["price"].astype("float64").to_numpy()
    size = df[size_col].astype("float64").to_numpy()
    sides = _decode_sides(df[side_col].to_numpy())
    order = np.lexsort((seq, ts))  # primary key = ts, secondary = seq
    ob = OrderBook() if seed is None else seed.copy()

    def gen():
        for o in order:
            yield Delta(int(ts[o]), int(seq[o]), sides[o], float(price[o]), float(size[o]))

    return sample_topk_as_of(sample_ts, gen(), k=k, book=ob,
                             apply=lambda b, d: b.apply(d), time_of=lambda d: d.ts_engine)
