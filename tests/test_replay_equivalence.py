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

    feed_disorder = 30  # max out-of-orderness in the simulated feed
    arrival = _arrival_within_watermark([*deltas, *trades], window_ns=feed_disorder)
    # The watermark must STRICTLY exceed the feed's max out-of-orderness. An event at
    # the bottom of a reversed block can be displaced by exactly `feed_disorder` ns;
    # a watermark equal to it would release that block's lowest-ts events one-by-one
    # in reversed order (premature release). +1 guarantees whole-block release in
    # total order, so the live output is byte-identical to offline.
    live = LiveReconstructor(k=3, watermark_ns=feed_disorder + 1)
    for ev in arrival:
        live.push(ev)
    online = live.flush().reset_index(drop=True)

    # Byte-identical: same columns, same dtypes, same values.
    pd.testing.assert_frame_equal(offline, online, check_dtype=True)


def test_live_drops_events_behind_the_watermark_without_corrupting():
    """A late event (skew > watermark) must be dropped, never retro-injected into the
    book — otherwise it corrupts every subsequent snapshot (literature-review.md §5.3)."""
    deltas = [Delta(10, 1, "bid", 100.0, 2.0), Delta(10, 2, "ask", 101.0, 3.0),
              Delta(100, 3, "bid", 100.0, 5.0)]
    trades = [Trade(110, 1001, "buy", 101.0, 0.1)]
    ordered = sorted([*deltas, *trades], key=order_key)

    clean = LiveReconstructor(k=2, watermark_ns=30)
    for ev in ordered:
        clean.push(ev)
    expected = clean.flush().reset_index(drop=True)
    assert clean.late_count == 0

    live = LiveReconstructor(k=2, watermark_ns=30)
    for ev in ordered:
        live.push(ev)
    # ts=10 is far behind max_ts=110 (threshold 80); a fresh ask@105 that, if replayed,
    # would wrongly appear as ask level 1 in the already-advanced book.
    live.push(Delta(10, 99, "ask", 105.0, 7.0))
    out = live.flush().reset_index(drop=True)

    assert live.late_count == 1
    pd.testing.assert_frame_equal(expected, out)       # late event had zero effect
    assert pd.isna(out["ask_1_price"]).all()           # corruption (ask 105) did NOT land


def test_live_handles_in_order_arrival_identically():
    deltas, trades = _bigger_world()
    offline = reconstruct_book_at_trades(deltas, trades, k=3).reset_index(drop=True)
    live = LiveReconstructor(k=3, watermark_ns=30)
    for ev in sorted([*deltas, *trades], key=order_key):
        live.push(ev)
    pd.testing.assert_frame_equal(offline, live.flush().reset_index(drop=True))
