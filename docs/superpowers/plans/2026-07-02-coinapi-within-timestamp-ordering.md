# CoinAPI Within-Timestamp Ordering — Resolution Note

**Goal:** Close the docs/data.md §10 open item *"Within-timestamp ordering for CoinAPI (no
`sequence_number`; rely on `seq` + L3 `order_id`)"* — make the ordering of same-timestamp
CoinAPI `limitbook_full` events explicit, deterministic, documented, and regression-tested.

**Status:** Resolved 2026-07-02. No replay reordering was needed; the existing pipeline was
already order-preserving end to end. The change is a pinned policy, a split quality counter,
and eight synthetic regression tests.

## The problem

CoinAPI L3 carries no exchange `sequence_number`. Many events share one `time_exchange_ns`
(an ADD and the MATCH that fills it routinely carry the same stamp), and for Coinbase both
`MATCH` and `SUB` are `decrement` events — so applying same-timestamp events in the wrong
relative order changes the final book (a reducer that precedes its ADD is skipped as
`missing_order` and the size it should have removed stays resting). Determinism and
correctness therefore require a pinned within-timestamp order.

## Canonical ordering policy

1. **File/stream order is canonical.** The downloader (`ingest/download_coinapi.py`) parses
   the vendor CSV sequentially and writes `seq` = row index within the day; the Parquet row
   groups preserve that order; `scripts/run_coinbase_parity.py::iter_coinapi_chunks` streams
   row groups in file order; `recon/coinapi.py::_iter_actions` applies rows exactly as
   delivered and **never re-sorts** (docs/data.md §4.3 mandatory rule — re-sorting by
   wall-clock time would also throw the prior-day-close-stamped SNAPSHOT block to the end of
   the day).
2. **`seq` is a check, not a sort key.** It records file order; the replay verifies it is
   increasing. A strict regression counts `seq_disorder` (the stream was re-sorted upstream —
   a real red flag); a duplicate counts `seq_duplicate` (a tie, broken deterministically by
   rule 3, not an error). Previously both cases landed in `seq_disorder`.
3. **Ties break by original row index.** Same timestamp — and even same `seq` — events keep
   their input row order, including across chunk boundaries (per-chunk vectorization never
   permutes rows; watermark/seq state carries across chunks). Since rows are never re-sorted,
   this stable tie-break is a no-op by construction; the tests pin it so it stays one.
4. **`order_id` is never an ordering key.** Coinbase L3 order ids are UUIDs; their
   lexicographic order carries no market meaning. Sorting by `(time, order_id)` would silently
   reorder cross-order events within a timestamp and is explicitly rejected.

Sampling already interacts safely with ties: `sample_topk_as_of` emits a grid point only when
it meets an event with label time strictly greater, so a same-timestamp group is always
applied atomically — no sample can observe a half-applied group.

## What changed

- `recon/coinapi.py::_iter_actions`: the `s <= last_seq` disorder check is split into
  `s < last_seq` → `seq_disorder` and `s == last_seq` → `seq_duplicate`. No ordering
  behavior changed. Module/function docstrings now state the within-timestamp policy.
- `tests/test_coinapi_within_timestamp_ordering.py` (new): synthetic regression tests —
  same-timestamp same-order pairs where order flips the final book (ADD→MATCH full fill;
  ADD→SET last-write-wins), a same-timestamp ADD/MATCH/SUB/DELETE lifecycle cluster with an
  exact final-book assertion, yield-order-equals-input-order with anti-lexicographic
  order_ids, duplicate-`seq` tie-break, file-order-wins-over-decreasing-`seq` (disorder
  counted, never re-sorted), byte-identical repeated runs, and chunk splits landing inside a
  same-timestamp group.
- Test teeth verified by mutation against the final code: a within-timestamp reversal
  mutation fails 7/8 tests (only the repeated-runs determinism test survives — a
  deterministic mutation is still deterministic); a `(time, order_id)` stable-sort mutation
  fails 3/8 (the lifecycle cluster, the direct yield-order pin, and the chunk-split test —
  such a sort preserves per-order sequences, so only order-sensitive pins catch it); a
  per-chunk `last_seq` reset fails the duplicate-seq test's chunk-split arm (counter
  undercount with an identical frame).
- docs/data.md §10: open item checked off with a pointer here (deliberately minimal — other
  branches touch that file).

## Non-goals / follow-ups

- No defensive re-sort was added at the ingestion boundary. If a future data path delivers
  genuinely disordered rows (nonzero `seq_disorder` on real data), the fix belongs upstream
  (restore file order at write time), not in the replay — re-sorting mid-stream cannot be
  done correctly under chunk streaming and would mask the corruption signal.
- `seq_duplicate`/`seq_disorder` are surfaced in the parity report's `coinapi_quality` block
  like every other counter; no gate consumes them numerically yet. Wiring them into a hard
  gate threshold (expected: both exactly 0 on real downloader output) is a candidate for the
  quality-map follow-up.
