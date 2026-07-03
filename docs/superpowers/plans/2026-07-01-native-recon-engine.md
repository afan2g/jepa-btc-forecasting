# Native Reconstruction Engine - Implementation Plan

**Goal:** Replace the Python hot loop for Crypto Lake `book_delta_v2` seed/reseed reconstruction with a native engine that preserves the current Python semantics exactly, then wire the Coinbase quality-map and parity scripts to use it. Python remains the orchestrator and correctness oracle; the native path is the throughput implementation needed before multi-day or multi-year data work is viable.

**Why now:** The Python quality-map smoke run on 2026-07-01 was not operationally acceptable. The default 2-day run (`2025-06-01`, `2026-04-01`) was still CPU-bound after roughly 4 hours; the `--no-cold-ab` run was still CPU-bound after roughly 1h40m. Both were single-process pure-Python replay over 16.5M and 34.7M Coinbase delta rows. This is fine as a reference implementation, but it cannot support a 12-24 month pipeline, let alone Binance perp at ~109M rows/day.

**Scope:** This plan is for the next Claude branch. Implement the first production-speed native core for the existing Lake-side seed/reseed reconstruction. Do not unlock CoinAPI backfill, do not run broad live pulls, and do not rewrite the bar/model pipeline.

## Bottleneck Diagnosis

The problem is not just "Python is slow"; the current path combines an algorithmic hot spot with Python per-row overhead.

1. **Touch checks are full-book scans.** `OrderBook.best_bid()` / `best_ask()` call `max(self.bids)` / `min(self.asks)` on plain dicts. The seeded replay calls them in `update_crossed(...)` after every delta once the book is established, and `emit(...)` calls them at every grid sample. That makes the current replay closer to `O(N_deltas * L_levels)` than `O(N_deltas)`. On a 34M-row day with thousands to tens of thousands of live levels, this dominates runtime.
2. **Every delta crosses Python object boundaries.** The array path still sorts in NumPy, then boxes each ordered row into a `Delta` NamedTuple through a Python generator, routes through another generator, then does a Python method call and string side comparison per row.
3. **Cold A/B doubles the scan.** With cold A/B enabled, the full stream is replayed a second time. The cold path can be the worst case because stale levels accumulate, increasing the level count behind the full-book scans.

The native design must address both costs. A Rust port that still scans all levels per event, receives rows as Python objects, or runs the cold A/B as a second full pass is not acceptable.

## Architecture

Use a Rust core exposed to Python through a small wrapper.

Preferred implementation:
- Add a Rust/PyO3 extension module, built with `maturin`, named `recon_native`.
- Add `recon/native.py` as the import-safe Python adapter. It should expose capability checks and a function matching the current Python seeded reconstruction API.
- Keep `recon/reseed.py` as the reference implementation. Do not delete or simplify it.
- Add `--engine {auto,python,native}` to `scripts/run_coinbase_quality_map.py` and `scripts/run_coinbase_parity.py`.
- Default `auto` may use native when installed and fall back to Python for tests/dev, but live instructions should use `--engine native` once available.

Why PyO3 first, not a standalone Rust parquet reader:
- The immediate bottleneck is the replay loop, not Lake download.
- The existing Python scripts already handle credentials, quota, calendar context, reports, and `lakeapi`.
- Passing columnar NumPy arrays to Rust avoids a Rust parquet/AWS integration in this PR.
- A later streaming engine can move parquet reading into Rust once the native replay semantics are proven.

Follow-on architecture after this PR:
- **Day-level process parallelism is the main wall-clock lever** for the 12-24 month goal. A single book stream is stateful and mostly sequential, but `(venue, symbol, day)` jobs are independent once day-boundary seed policy is explicit. After the native single-day engine is proven, add quota-aware `--jobs N` orchestration and per-day checkpoints as the next PR.
- **Streaming Arrow/parquet read is the next bottleneck after replay.** This PR may accept pandas/NumPy arrays because it isolates the replay semantics. Once replay is native-fast, `lakeapi -> pandas -> to_numpy()` copies and full-day DataFrame materialization will become the limiting cost, especially for Binance. The planned follow-on is a Rust/Arrow row-group reader that streams batches into the same native replay core and emits compact top-K/bar outputs.

## Non-Goals

- No CoinAPI backfill unlock.
- No bulk Lake archive run.
- No full Binance production pipeline.
- No native CoinAPI L3 replay in this PR.
- No day-level multiprocessing in this PR unless the native single-worker path is already proven and the change stays small.
- No semantic changes to seed/reseed policy, thresholds, or quality classifications except where tests prove an existing bug.

## Required Semantics

The native result must match `recon.reseed.reconstruct_lake_l2_at_samples_seeded` for both frame output and metrics on synthetic fixtures.

Preserve these rules exactly:
- Deltas sort by `(engine_time, sequence_number, original_row_index)` using the same columns resolved by Python. The final row-index tiebreaker is required to preserve Python `np.lexsort((seq, ts))` equal-key behavior; Coinbase duplicates `sequence_number` heavily, and reordering equal `(ts, seq)` absolute-size updates can change the final book.
- For equal timestamp between a delta and a snapshot, apply the delta first, then the snapshot. A same-timestamp snapshot is authoritative and overwrites the delta-updated state.
- Samples are "as of" `sample_ts`: all events with `event_ts <= sample_ts` are reflected.
- `book_delta_v2` sizes are absolute sizes; `size == 0` removes the level.
- Side semantics match Python `_decode_sides`: bool `side_is_bid=True` is bid, false is ask; string aliases must remain handled by Python before native or by native with matching tests.
- Seed snapshots must pass the same validation as Python: finite positive prices/sizes, two-sided, uncrossed, enough levels, optional max-spread guard.
- If no valid seed is accepted, classification remains inconclusive in the quality-map path.
- Reseed triggers only after the reconstructed established book stays crossed for `reseed_after_crossed_s`.
- Reseed events apply at their own timestamp only; no lookahead/backpatching.
- `sequence_number` is not a gap detector.
- Residual crossed samples must remain visible in metrics/reporting. Do not hide them to make a day look usable.
- Native book ordering may use integer ticks, but emitted frame level prices must preserve the source float values. Carry the original source float alongside the tick key and emit that float for `bid_{i}_price` / `ask_{i}_price`; do not reconstruct emitted prices as `tick * scale`. This keeps level-price columns byte-identical to Python where possible. `mid` and `microprice` are computed values and may be compared with tight tolerance.

## Native API

Add a Python wrapper with a shape close to:

```python
from recon.native import native_available, reconstruct_lake_l2_at_samples_seeded_native

frame, meta = reconstruct_lake_l2_at_samples_seeded_native(
    df,
    sample_ts,
    k=10,
    engine_time_col="origin_time",
    snapshots=snapshots,
    policy=policy,
    frame_out=True,
)
```

The wrapper may convert DataFrame columns to contiguous NumPy arrays before calling Rust:
- `ts_ns: int64`
- `seq: int64`
- `side_is_bid: bool` or pre-decoded signed side
- `price: float64`
- `size: float64`
- `sample_ts: int64`
- snapshot columns/arrays as a compact native-friendly structure

Do **not** pass one Python object per delta or nested Python `BookSnapshot` objects into the native hot path. For v1, Python owns snapshot parse/thin/classify by reusing `snapshots_from_lake_book_df(...)` and `classify_snapshot(...)` semantics, then passes compact arrays into Rust: `snapshot_ts`, bid/ask price/size matrices, `reason_code`, and `is_valid`. Rust must not reimplement snapshot validation precedence in v1; it only accumulates the precomputed reason codes and consults `is_valid` for seed/reseed gating. This keeps the Rust surface on the delta hot loop while preserving the non-obvious Python snapshot rules, including NaN padding and rejection-code precedence. Python `BookSnapshot` lists may remain for the reference path and small tests.

Return schema when `frame_out=True` must match the Python frame:
- `sample_ts`
- `mid`
- `microprice`
- `bid_{i}_price`, `bid_{i}_size`
- `ask_{i}_price`, `ask_{i}_size`

`mid` and `microprice` are not needed for metrics-only quality-map classification, but they are required for parity and frame conformance. `recon.parity.compare_topk(...)` reads `mid` for mid diffs, spikes, and label agreement.

Return `meta` keys must match Python where the scripts consume them:
- `seed_accepted`
- `seed_ts`
- `seed_reason`
- `reseed_count`
- `reseed_ts`
- `reseed_blocked_invalid_snapshot`
- `snapshot_reason_codes`
- `crossed_rate`
- `crossed_samples`
- `crossed_sample_ts`
- `missing_book_fraction`
- `thin_depth_fraction`
- `crossed_duration_s`
- `policy`

If `frame_out=False`, the native path should skip materializing the top-K frame and return metrics only, matching Python's cold A/B behavior.

For the quality-map script, native mode should use metrics-only output for the seeded "after" path by default. Classification needs crossed/missing/thin metrics and seed/reseed metadata, not the full top-K frame. Parity still needs top-K frames. This relies on a load-bearing invariant: native `meta.crossed_samples`, `meta.crossed_rate`, and `meta.missing_book_fraction` must equal `recon.parity.frame_quality(frame)` computed from the frame path on the same run. Pin that invariant in Python-only reference tests and native conformance tests, including crossed, missing, and thin samples.

## Rust Data Structures

Use correctness-first native structures, then benchmark.

Recommended:
- `BTreeMap<PriceKey, f64>` for bids and asks initially.
- Use integer tick keys as the primary native representation. For supported live symbols, native mode must have an explicit price scale/tick contract (for example from a small `(exchange, symbol) -> price_scale` registry or a required CLI/config override). Unknown symbols should fail clearly in explicit `--engine native` mode rather than silently falling back to ordered floats for live runs. In `--engine auto`, if native is installed but the symbol has no verified tick scale, warn and fall back to Python before any Lake load.
- Maintain best bid/ask queries without scanning all levels per event.
- Build top-K snapshots from the ordered maps at sample times.

Integer ticks are not optional polish: they avoid ordered-float edge cases, speed comparisons, and make tree keys deterministic. The wrapper must test float-to-tick rounding against the Python oracle on boundary cases so same-timestamp crossing decisions do not drift. Store the source float price with each active level for frame emission; tick keys are for ordering and lookup, not for reconstructing output prices. If a product's true tick size is unknown, keep that product on the Python/reference path until the scale is verified.

Important: a naive Rust port that scans all prices for every sample or event can still be too slow. It is acceptable to use `BTreeMap` first/last access and iteration for top-K at sample points, but not full-book scans on every delta. If the benchmark shows tree maintenance dominates, the next data-structure escalation is lazy-deletion min/max heaps or a dense tick-indexed array with maintained best pointers, not tuning around full scans.

## Sorting And Replay Buffer

Do not require Python to hand Rust a NumPy `lexsort` order array for production native replay. That would make Rust gather columns in random order through a large permutation, which is cache-hostile and keeps a major preprocessing cost in Python.

Native mode should accept raw columnar arrays and either:
- detect already-sorted `(ts, seq)` input and replay it directly, or
- sort inside Rust into a contiguous replay buffer.

The sort/merge semantics must still match Python:
- delta order key is `(engine_time, sequence_number, original_row_index)`;
- equal `(engine_time, sequence_number)` deltas retain source-row order;
- snapshots are sorted by timestamp;
- if a delta and snapshot have the same timestamp, replay the delta first and the snapshot second.

Add tests that would fail if a same-timestamp snapshot is applied before a delta, and a fixture with two equal `(ts, seq, side, price)` updates whose final size depends on stable source-row order.

## Cold A/B Strategy

Cold A/B is diagnostically useful but must not be implemented as a mandatory second full native pass.

Native mode should support one of these, in priority order:
1. **Fused cold metrics in one traversal:** maintain a second cold-start book alongside the seeded/reseeded book and compute `crossed_rate_cold`/missing metrics without materializing a cold frame. The cold book must apply deltas only and must never observe snapshots, matching the Python `snapshots=None` cold pass.
2. **Explicit second pass only when requested:** acceptable for the first implementation if clearly reported and disabled by default for broad sweeps. A second pass over an already sorted/replay-buffered delta stream may be simpler for v1; do not over-invest in fusion before the native single-pass path is benchmarked.

The quality-map integration should keep `--no-cold-ab`, but the native implementation should make the default cold A/B path cheap enough for small diagnostic runs.

## Integration Points

### `scripts/run_coinbase_quality_map.py`

Add:
- `--engine {auto,python,native}`.
- Report `engine` in `meta.policy` or `meta.engine`.
- When `--engine native` is selected and native is unavailable, exit with a clear nonzero error before any Lake load.
- When `--engine auto`, choose native only if the extension is available and the requested `(exchange, symbol)` has a verified native tick scale; otherwise warn and use Python before any Lake load.
- Keep quota checks before any Lake load exactly as today.
- Keep `--no-cold-ab`; do not make cold A/B mandatory for broad sweeps.
- In native mode, classify from native metrics without building the top-K "after" frame. Add a separate debug option only if a frame dump is needed.

### `scripts/run_coinbase_parity.py`

Add the same engine selector for the Lake side only. CoinAPI L3 replay remains Python in this PR.

The parity report must record the Lake engine used so measured results remain auditable.

### Packaging

Current `pyproject.toml` uses setuptools and keeps runtime deps light. Add native build support without making plain Python tests unusable on machines without Rust.

Acceptable options:
- Add `maturin` build metadata and document `pip install -e .` / `maturin develop`.
- Or keep the Rust package under `native/recon_native/` with explicit build instructions and make Python import optional.

Hard requirement:
- `python -m pytest -q` must still run in an environment without the native extension, with native-specific tests skipped when unavailable.
- Native conformance tests must run when the extension is installed.

## Tests

Add tests that compare native to Python on small deterministic fixtures. Use synthetic data only; no vendor access in CI.

Required tests:
- Native import/capability smoke test, skipped when extension is unavailable.
- Valid seed day: native frame and meta match Python.
- Stranded/crossed stream repaired by reseed: native matches Python on `reseed_count`, crossed rate, and top-K frame.
- Same-timestamp delta/snapshot ordering: delta first, snapshot second.
- No valid seed: native returns the same inconclusive-driving meta as Python.
- `frame_out=False`: native metrics match Python and no frame is returned.
- Native metrics-only quality-map path matches Python frame + `frame_quality(...)` classification on synthetic fixtures. The fixture must include crossed, missing, and thin samples.
- Native metrics-only mode produces the same per-day report block shape as the Python frame path: `quality.crossed_samples_after`, `missing_book_fraction`, `thin_depth_fraction`, `crossed_duration_s_after`, and related seed fields.
- Python-only reference test: `_replay_seeded(..., collect_frame=True)` metrics equal `frame_quality(frame)` for crossed and missing counts/fractions, so the oracle itself pins the metrics-only invariant.
- Fused cold A/B metrics, if implemented, match the Python second-pass cold result.
- Snapshot array preprocessing/preclassification matches Python `snapshots_from_lake_book_df(...)` + `classify_snapshot(...)` for valid, crossed, thin, one-sided, bad-value, unsorted, and NaN-padded fixtures. Rust consumes precomputed reason codes and `is_valid`; it should not duplicate validation precedence in v1.
- Price tick conversion has explicit boundary tests around equality/crossing (`best_bid == best_ask`, one tick uncrossed, one tick crossed), and emitted `bid_{i}_price` / `ask_{i}_price` values equal the original source floats rather than `tick * scale` round-trips.
- Native sorting matches Python `np.lexsort((seq, ts))`, including duplicate timestamps, duplicate sequence numbers, and equal-key rows that must retain original row order.
- Trailing crossed-duration close-out: if the book remains crossed after the last event, native `crossed_duration_s` closes the interval at `max(last_event_ts, final_sample_ts)`, matching Python.
- Quality-map script engine selection:
  - `--engine python` uses Python path.
  - `--engine native` fails before live load if native unavailable.
  - `--engine auto` falls back cleanly when native is unavailable or when the symbol lacks a verified tick scale.
- Parity script records selected Lake engine.

Use strict comparisons where possible. For floats, use exact equality on deterministic simple fixtures or tight tolerances where native map ordering/float formatting makes exact equality unrealistic.

## Benchmark

Add a local benchmark script that does not require vendor credentials by default.

Suggested file:
- `scripts/bench_recon_engine.py`

It should generate a deterministic synthetic Lake-like DataFrame with configurable rows/samples and compare Python vs native:

```bash
.venv/bin/python scripts/bench_recon_engine.py --rows 1000000 --samples 10000 --levels 10000 --churn 0.20 --engine both
```

Report:
- rows/sec for replay
- samples/sec
- configured live level width (`--levels`) and churn/delete rate (`--churn`)
- peak-ish RSS if cheap to measure
- whether native output matched Python on the benchmark fixture

Do not put benchmark assertions in normal pytest unless they are tiny and stable. Performance gates should be recorded as local validation in the PR.

The benchmark must model the actual bottleneck. It needs a `--levels` / book-width parameter and a churn/delete rate so the Python baseline exercises large live books instead of a tiny stable top-of-book. A handful-of-levels fixture makes the current Python implementation look artificially cheap and will not catch a Rust port that still scans all levels.

Target for this PR:
- On a 1M-row synthetic benchmark with realistic live width (start with `--levels 10000 --churn 0.20`), native must be at least 10x faster than Python; this is a floor, not a success target. Given the current `O(N * L)` touch scans plus Python object overhead, a correctly implemented native tree/tick path should plausibly be much faster. If the benchmark is only barely above 10x, treat that as a warning sign and inspect for residual boxing, Python-side sorting/gathering, full-book scans, or cache-hostile data movement.
- On the real default Coinbase quality-map days, `--engine native --no-cold-ab` should be plausibly minutes, not hours. If live data is run, keep it to the default 2-day set and report exact runtime and Crypto Lake `used_data` before/after.

Before committing heavily to the Rust implementation, do a short measurement spike:
- Capture `cProfile`/sampling-profiler evidence on a synthetic large-level fixture or one small live day if already cached.
- Optionally run a throwaway Python data-structure spike (`sortedcontainers.SortedDict`, heap-backed touch cache, or equivalent) to verify that eliminating full touch scans materially changes runtime. Do not add a production dependency just for the spike unless the PR explicitly chooses that path.
- Record the result in the PR body. The point is to validate the bottleneck, not to replace the native plan.

## Documentation Updates

Update:
- `docs/data.md` §5a-QualityMap: note that the Python replay path is a correctness reference and native engine is required for operational multi-day runs.
- `docs/data.md` §6/§10 if live benchmark results are collected.
- The new native build instructions, either in this plan or a small `docs/native-recon.md` if the build steps are nontrivial.

Do not claim the multi-day quality map passed unless it actually ran and produced `data/reports/coinbase_quality_map.json`.

## Implementation Tasks

### Task 1: Native package scaffold

- Add Rust package/module for `recon_native`.
- Add import-safe Python wrapper `recon/native.py`.
- Add tests that skip cleanly when native is unavailable.
- Validate that normal Python tests still pass without building native.

### Task 2: Port the order book and snapshot input contract

- Implement native book state and top-K snapshot.
- Implement integer tick conversion/validation for the supported live products.
- Implement the snapshot-array input contract that consumes Python-precomputed reason codes and `is_valid` flags.
- Add adapter/preprocessing tests against Python for valid/crossed/thin/one-sided/bad-value/unsorted/NaN-padded snapshots.

### Task 3: Port seeded/reseeded replay

- Implement native delta sorting or sorted-input detection, snapshot merge, seed, reseed, sample loop, and metrics.
- Support `frame_out=True` and `frame_out=False`.
- Support metrics-only quality-map mode without building a top-K frame.
- Implement fused cold A/B metrics if feasible; otherwise make second-pass cold A/B explicit and easy to disable.
- Add conformance tests against `recon.reseed.reconstruct_lake_l2_at_samples_seeded`.

### Task 4: Wire quality map and parity scripts

- Add `--engine`.
- Record engine in output JSON.
- Ensure `--engine native` fails before Lake load if unavailable.
- Preserve all quota/backfill gates.

### Task 5: Add benchmark script and docs

- Add synthetic benchmark.
- Document build/run commands.
- Update `docs/data.md` with the performance decision and any measured local runtime.

## Validation Commands

Minimum local validation before PR:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile scripts/run_coinbase_quality_map.py scripts/run_coinbase_parity.py scripts/bench_recon_engine.py
git diff --check
```

If Rust is installed and the extension is built:

```bash
maturin develop
.venv/bin/python -m pytest -q
.venv/bin/python scripts/bench_recon_engine.py --rows 1000000 --samples 10000 --levels 10000 --churn 0.20 --engine both
```

Optional live validation, only after synthetic conformance passes:

```bash
.venv/bin/python scripts/run_coinbase_quality_map.py --engine native --no-cold-ab
jq empty data/reports/coinbase_quality_map.json
jq '.summary, .meta.engine, .meta.quota' data/reports/coinbase_quality_map.json
```

Do not run broader live day sets in this PR without explicit user approval.

## PR Requirements

PR body must include:
- Summary of native engine scope.
- Whether native extension was built locally.
- Test results with and without native if both were run.
- Benchmark result and machine context.
- Any live vendor commands run, with exact days, `used_data` before/after, and report path.
- Explicit statement that CoinAPI backfill remains locked and no bulk backfill was run.

## Review Checklist

Reviewer should verify:
- Native and Python semantics match on synthetic fixtures.
- `--engine native` cannot silently fall back after the user explicitly requested native.
- Quota checks still happen before Lake loads.
- No raw data or large reports are committed.
- Build changes do not make ordinary Python tests require Rust.
- `docs/data.md` does not claim a quality-map pass unless there is a real report artifact.

## Execution Handoff

Use a dedicated branch/worktree:

```bash
cd /home/aaron/jepa-btc-forecasting
git checkout master
git pull --ff-only origin master
scripts/new_claude_worktree.sh feat/native-recon-engine
```

Prompt Claude to implement only this plan's scoped native replay engine. If the branch gets too large, split after Task 3: first PR for native conformance, second PR for script integration and benchmark/live docs.
