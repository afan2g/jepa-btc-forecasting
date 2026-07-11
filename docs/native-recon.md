# Native reconstruction engine (`recon_native`)

The native Rust/PyO3 replay core for Crypto Lake `book_delta_v2` seed/reseed reconstruction
(docs/data.md §5a-Recon). Python remains the correctness **oracle** and orchestrator; the native path
is the throughput implementation needed before multi-day / multi-year data work is viable. Plan:
`docs/superpowers/plans/2026-07-01-native-recon-engine.md`.

## What it is (and is not)

* **Is:** a drop-in accelerator for `recon.reseed.reconstruct_lake_l2_at_samples_seeded`. The native
  `(frame, meta)` is schema-identical, so `scripts/run_coinbase_quality_map.py` and
  `scripts/run_coinbase_parity.py` select it with `--engine {auto,python,native}` transparently.
* **Is not:** a reimplementation of snapshot validation. Python owns snapshot parse/thin/**classify**
  (`snapshots_from_lake_book_df` + `classify_snapshot`) and hands Rust compact arrays with a
  precomputed `reason_code` + `is_valid` per snapshot. Rust only accumulates those reason codes and
  consults `is_valid` for seed/reseed gating (plan §"Native API"). CoinAPI L3 replay stays Python.

## Design (preserved semantics)

The Rust core (`native/recon_native/src/lib.rs`) matches the Python replay exactly:

* Deltas STABLE-sorted by `(engine_time, sequence_number)` → equal `(ts, seq)` rows keep source order,
  reproducing NumPy `np.lexsort((seq, ts))` (Coinbase duplicates `sequence_number` heavily).
* At equal timestamp a delta is applied **before** a snapshot (a same-ts snapshot is authoritative).
* Samples are "as of" `sample_ts`; `size == 0` removes a level; sizes are absolute.
* The book is keyed by **integer ticks** (`round(price * price_scale)`) for O(log L) best-bid/ask and
  ordered top-K — the algorithmic fix over Python's `max(dict)/min(dict)` full-book touch scans. Each
  level carries its **original source float price**, so emitted `bid_i_price`/`ask_i_price` are
  byte-identical to Python (never reconstructed as `tick / scale`).
* Crossed-duration is accounted only once seeded and the trailing open run closes at
  `max(last_event_ts, final_sample_ts)`.

## Coverage metrics (partial-day fill planning)

Both engines' `meta` carries a compact per-sample `coverage` block (partial-day fill plan
`docs/superpowers/plans/2026-07-02-partial-day-fill-policy.md`, Task 3 — implemented 2026-07-02), so
`scripts/run_coinbase_quality_map.py --engine native` plans partial-day CoinAPI fills without
materializing the 86,400-row top-K frame:

* `invalid_runs_idx` — maximal half-open `[i0, i1)` sample-**index** runs where the sample fails the
  shared stitch-policy validity predicate (both top-of-book prices present, non-NaN, `bid < ask` —
  exactly `recon.stitch_policy.valid_mask_from_frame` at `min_levels_per_side=1`). Index pairs, not
  timestamps: the replay does not know the grid step, so the quality map converts against its own
  grid. The list is complete (`n_invalid_runs` = full count); the report caps its emitted
  `invalid_runs` list, never the meta.
* `present_first_idx` / `present_last_idx` — bound indices of the notna both-tops presence predicate
  behind the report's `lake_present_*` fields (bounds only — that is all `plan_day_stitch` reads).

Because the runs are maximal, their complement reconstructs the exact per-sample validity mask, so
the native path feeds the same `plan_day_stitch` as the Python frame path and emits identical Q7
coverage keys and fill plans. The Python frame remains the correctness oracle: replay coverage is
pinned equal to the frame-derived mask by Python-only tests, native == Python by the meta-equality
conformance tests (`tests/test_native_recon.py`), and the plans are pinned identical at the assess
level (`tests/test_quality_map.py`).

The result-dict contract is versioned: `recon_native.META_ABI` (currently **2**) must equal
`recon.native._META_ABI`, so a stale pre-coverage build is rejected at import — it reports
unavailable with the rebuild hint instead of silently degrading partial-day plans to full-day.

Both engines' array entry points also **reject non-finite delta prices/sizes** (`ValueError`,
`recon.reseed.require_finite_deltas` — the `classify_snapshot` finite-values bar applied to the
delta stream): a NaN price keys the books differently (Python keys the raw float, whose
`max()`/`min()` with a NaN key is insertion-order dependent; native casts NaN to tick 0), so a
single dirty row would otherwise make the engines silently disagree on coverage and fill plans.

### Verified tick contract (required for native)

Native mode needs a verified `(exchange, symbol) → price_scale` where every price is an exact multiple
of the tick (so tick ordering == float ordering and same-ts crossing decisions cannot drift). The
registry lives in `recon/native.py::_TICK_SCALE`:

| exchange        | symbol       | tick    | price_scale | evidence                          |
|-----------------|--------------|---------|-------------|-----------------------------------|
| COINBASE        | BTC-USD      | \$0.01  | 100         | quote increment (native-recon PR) |
| BINANCE_FUTURES | BTC-USDT-PERP| \$0.10  | 10          | #64 tick-scale step (issue #71)   |
| BINANCE         | BTC-USDT     | \$0.01  | 100         | #64 tick-scale step (issue #71)   |

The Binance scales were measured by the preregistered #64 tick-scale step on Lake day `2026-04-01`
(prereg commit `60a2b745`): **zero off-tick prices** across every price-bearing feed — perp
`book_delta_v2` 109,317,254 / `trades` 1,591,574 / `book` 30,724,880 prices, spot `book_delta_v2`
33,892,363 / `trades` 1,024,789 / `book` 29,185,440 prices (report
`data/reports/binance_source_quality/tick_scale.json`, `pass=true`, report_hash
`d5025c58aa48fb6b23d26f8f26cf270c42b4be33e567644f359fafd171a1d7f0`; values reconciled on issue #71).

Unknown symbols **fail** under `--engine native` (before any Lake load) and **fall back to Python**
under `--engine auto`. Extend the registry only after a product's tick size is verified.

## Build

A recent Rust toolchain (built/tested with rustc 1.94) and `maturin` (>=1.7) are required to build the
extension. The plain Python test suite does **not** need it — native tests skip cleanly when absent.

```bash
# one-time: install maturin into the venv
.venv/bin/pip install maturin

# build + install the extension into the active venv (rebuild after editing src/lib.rs)
.venv/bin/maturin develop --release -m native/recon_native/Cargo.toml
```

Confirm:

```bash
.venv/bin/python -c "import recon_native; print(recon_native.N_REASONS)"          # -> 7
.venv/bin/python -c "import recon_native; print(recon_native.META_ABI)"           # -> 2
.venv/bin/python -c "from recon import native; print(native.native_available())"  # -> True
```

Unit-test the pure-Rust core (no Python link needed):

```bash
cargo test --no-default-features --manifest-path native/recon_native/Cargo.toml
```

## Validate

```bash
.venv/bin/python -m pytest -q tests/test_native_recon.py             # native-vs-Python conformance
.venv/bin/python scripts/bench_recon_engine.py --rows 1000000 --samples 10000 --levels 10000 \
    --churn 0.20 --engine both
```

## Benchmark result (synthetic; recorded, not a live claim)

Machine: 12th Gen Intel i5-12400F (12 threads), 31 GiB RAM, Linux 6.17, Python 3.12.3, rustc 1.94.0,
numpy 2.4.6 / pandas 2.3.3. Synthetic 1M-row fixture, 10 000 levels/side, 20% churn, 10 000 samples,
k=10:

| engine | time     | rows/s      | samples/s | peak RSS |
|--------|----------|-------------|-----------|----------|
| python | 244.35 s | 4 093       | 41        | —        |
| native | 0.189 s  | 5 293 140   | 52 931    | ~288 MB  |

**native speedup ≈ 1293×** (floor is 10×); native output matched Python on the fixture. The large
factor is expected: native replaces the Python O(N·L) best-bid/ask scans + per-row object boxing with
an O(N·log L) tick-tree replay.

The live Coinbase quality-map / parity runs are **not** included here — they are quota-gated and
require explicit approval (docs/data.md §5a-QualityMap, §9). This file records only the synthetic
benchmark; no live vendor result is claimed.
