# experiments/ — CoinAPI snapshot-only seeding evaluation (issue #54)

Experiment-scoped code. **Nothing here is production policy.** The standing
partial-day fill policy (docs/superpowers/plans/2026-07-02-partial-day-fill-policy.md)
prohibits cross-vendor seeding; this experiment measures whether that prohibition could
be relaxed. A GO verdict authorizes a *separate reviewed* implementation/policy PR — it
never silently changes `recon/`, the quality map, the #33 manifest, or the #53
executor. No file under `recon/`, `ingest/`, `eval/`, or `scripts/` (other than the
experiment's own runner) imports from `experiments/`.

## Question

Can a bounded historical CoinAPI order-book snapshot (or minimal bootstrap) seed/reseed
the Coinbase Crypto Lake `book_delta_v2` replay well enough to replace some of the 457
`crossed_seed_source` full-day CoinAPI book fills ($1,037.39 of the #33 manifest's
$1,369.45 book spend; 167 days / $379.09 inside the #34 Milestone-A pilot window)
without weakening reconstruction correctness?

## Design

Everything runs OFFLINE from data already on disk. The three days with both a local
full-day CoinAPI `limitbook_full` parquet and cached Lake data are the complete
fixture set (see `preregistration_54.json`):

| day | class | why it is in the set |
|---|---|---|
| 2025-06-01 | clean control | quality-map `lake_usable`; the §5a clean reference day |
| 2024-12-04 | crossed-seed (mild, 8.4%) | cross-validation FAILED Lake-only rehab |
| 2026-04-01 | crossed-seed (severe, 37.5%) | worst measured seed source; CV failed |

The issue's remaining fixture classes (no-Lake-snapshot day, sparse day, seam day) have
no local CoinAPI reference, so they are **emulated** on real days
(`emulate_degradation`: `leading_gap`, `sparse`; plus running CoinAPI arms without the
Lake `book` product) and labeled as emulations — never presented as the real days.

Trusted snapshots are emulated from the full-day L3 file we already own:

* `coinapi_snapshot_at` — replay L3 events with label ≤ T (the `sample_topk_as_of`
  as-of convention, SNAPSHOT block clamped to day open exactly as `recon.coinapi`
  does) → full-depth L2 state, optionally truncated to top-N. Emulates "a snapshot
  requested at T".
* `snapshots_from_topk_frame` — the day's reference frame rows as 1 s-cadence top-X
  candidates. Emulates the Flat Files `limitbook_snapshot_X` product; `n_changed`
  seconds drive its size/cost projection.
* `frame_snapshot_provider` — on-demand snapshot at the last grid second ≤ request
  (causal; sub-second age). Emulates a REST snapshot response.

Every candidate passes `classify_candidate` before injection: the production
structural gate (`classify_snapshot`: two-sided, finite/positive, deep enough, sorted,
uncrossed) plus cross-vendor checks — **future** (causality), **stale** (`max_age_s`),
**off_tick** (tick-scale conformance, COINBASE BTC-USD = 100 ticks/$). Accepted
candidates are handed UNMODIFIED to the production seeded replay
(`recon.reseed.reconstruct_lake_l2_at_samples_seeded` / native twin) — the experiment
swaps only the snapshot *source*, never replay semantics, same-timestamp ordering
(delta-then-snapshot at equal ts), or sampling.

### Arms (per day)

| arm | seed source | reseed |
|---|---|---|
| `cold_control` | none (production cold start) | — |
| `lake_book_control` | Lake `book` product (production behavior) | on sustained crossing |
| `coinapi_day_open_L20` / `_full` | one CoinAPI snapshot at day open (REST-depth-capped / full) | none available |
| `coinapi_stream_L{5,10,20,50}` | 1 s top-X CoinAPI candidates (`limitbook_snapshot_X` emulation) | on sustained crossing |
| `coinapi_on_demand_L20` | day starts cold; snapshot REQUESTED at each observable sustained-crossing trigger (iterative causal fixed point; request count = vendor cost) | at trigger |

Each arm is compared against the day's full CoinAPI L3→L2 reference on the identical
1 s grid with `run_parity_core`'s exclusion semantics (warm-up clamp to the accepted
seed, residual crossed samples excluded only when a seed was accepted and reseed
active), reporting crossed/missing/thin fractions, crossed duration, mid
median/p95/p99 error, correlation, spike buckets, and 2/10/60 s label agreement — plus
a `parity_guarded` variant that additionally masks `injection_guard_s` (default 60 s)
after every injection, because reference and snapshots share the same source file
(see Limitations).

### Preregistration

`preregistration_54.json` (pinned equal to `PREREGISTERED` by test) fixes the
thresholds BEFORE any real-data run: `lake_usable` day-quality bars, parity bars
placed between the documented clean-reference and the measured crossed-seed
cross-validation failures, clean-control non-regression deltas, and the economic bar
(≤ 25% of the day's full-day fill cost). A `spike_gt50_fraction` failure may be
overridden only by PR-#28-style volatility attribution documented spike-by-spike.

### Determinism / provenance

Every arm reports `frame_hash` (canonical content hash of the replayed top-K frame)
and `report_hash`; every snapshot carries provenance (vendor, product emulated,
method, extraction params, source sha256). Same inputs ⇒ same hashes (pinned by test).

## Running

```bash
flock -w 14400 /tmp/jepa-expensive-compute.lock \
  /home/aaron/jepa-btc-forecasting/.venv/bin/python \
  scripts/run_snapshot_seed_experiment.py --day 2025-06-01 \
  --coinapi-root /home/aaron/jepa-btc-forecasting/data/raw \
  --lake-cache-root /home/aaron/jepa-btc-forecasting/.lake_cache \
  --out-dir data/reports/snapshot_seed --engine auto
```

Reads the local CoinAPI parquet (exit 3 if absent — never downloads) and the existing
lakeapi joblib cache (read-only). Reports land in `data/reports/snapshot_seed/`
(git-ignored): `snapshot_seed_<day>[_<variant>].json` + `_arms.csv`; the expensive
CoinAPI reference frame is cached under `cache/` keyed by source sha256.

## Limitations (bind any GO)

1. **Shared-source circularity.** Emulated snapshots and the parity reference come
   from the same `limitbook_full` file. Injections re-anchor the arm to the reference,
   so parity right after an injection is trivially good; the `parity_guarded` variant
   and the stream arm's `reseed_count` bound this effect, but a real cross-vendor
   snapshot adds vendor-book divergence (documented clean-day p95 $0.48) that this
   experiment cannot measure offline.
2. **n=3 real days.** The fixture set brackets crossed-seed severity but is not a
   sample; degraded classes beyond crossed-seed are emulations.
3. **Product feasibility is documentation-based.** REST historical order book is
   confirmed L2/20-level; `limitbook_snapshot_X` existence is confirmed but its
   COINBASE availability, per-symbol size, and 2026 rates are unverified (flat-file
   pricing pages were bot-gated 2026-07-10; rates match repo-measured 2026-06/07
   billing). Billing granularity unknowns are carried as explicit cost BANDS.
4. **No live calls were made.** Closing the product unknowns requires the bounded,
   pre-approved probe plan posted on #54 — never run from this experiment code.
