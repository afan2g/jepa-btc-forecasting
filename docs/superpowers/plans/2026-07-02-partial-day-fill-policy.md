# Partial-Day / Vendor-Seam Fill Policy — Implementation Plan

> **For agentic workers:** this plan is both a policy spec and an implementation plan. The
> synthetic-testable core (`recon/stitch_policy.py` + `tests/test_stitch_policy.py`) ships with this
> plan's PR; the wiring tasks (§Implementation Tasks 2–5) are follow-up branches. Use
> superpowers:executing-plans (or subagent-driven-development) for the follow-ups.

**Goal:** Remove all ambiguity from partial-day Coinbase fills and Lake↔CoinAPI vendor seams — the
§5a-QualityMap unlock-precondition item (b), tracked in the §10 quality-map open item — by fixing
the day classification rules, the exact
vendor-switch boundary, seam guard bands, seam reseeding, label/feature treatment across seams, and
the quality-map report extensions, with pure-Python helpers and synthetic tests for every rule.

**Why now:** PR #11's expanded quality map found that gap edges bleed into adjacent "present" days as
leading partial days (`docs/data.md` §5a-QualityMap finding 2): `2025-01-07` resumes ~14:45:00Z
(61.46% of the grid missing, clean where present — the canonical leading partial-day fill case) and
`2024-08-05` starts ~16:08:35Z (67.26% missing **and** a 28.78% crossed seed source). PR #13 resolved
crossed-seed-source days to CoinAPI fill (provisional). Partial-day seam handling is the last policy
gap named by the unlock precondition (`docs/data.md` as it read before this PR: "Unlock still
requires at least the partial-day/seam fill policy (2025-01-07, 2024-08-05) …" — this branch
annotates that sentence as defined).

**Scope:** Policy spec + `recon/stitch_policy.py` pure helpers + synthetic tests. No CoinAPI
downloads, no `--allow-backfill`, no full-window quality map, no production stitcher, no native-engine
changes. **Backfill stays LOCKED.**

## Non-Goals

- No real CoinAPI day downloads and no bulk Lake pulls.
- No production stitched-frame builder (the helpers only *plan* segments and masks).
- No changes to seed/reseed semantics, `Thresholds`, or the five-class day taxonomy.
- No native (Rust) engine changes; the new per-day segment metrics for the native path are a
  documented follow-up.
- No feature-manifest schema change in this PR (integration is specified below, implemented with the
  bar builder).

## Definitions

All timestamps are int64 ns on the single exchange-time engine clock (`origin_time`; see
`shared_engine_time_col`). All intervals are half-open `[start_ts, end_ts)`. The grid is
`build_grid(day, grid_ms)`: `day_open_ts = int(pd.Timestamp(day).value)`, 86,400 samples at the
default 1 s, `day_end_ts = day_open_ts + 86_400e9`.

| term | definition |
|---|---|
| **valid sample** | grid sample where the seeded Lake frame has both top-of-book prices, uncrossed (`bid_0_price < ask_0_price`), and ≥ `min_levels_per_side` levels per side — exactly the `good` predicate of `recon/parity.py::lake_warmup_cutoff` at its operative default **`min_levels_per_side = 1`** (the parity gate's `--warmup-min-levels` default; distinct from the seed-validation knob `seed_min_levels = 5`, which gates snapshot acceptance, not sample validity). Depth is deliberately NOT a per-sample validity dimension: top-of-book usability decides segment planning, while depth degradation is governed day-level by `thin_usable_max` + the `THIN_DEPTH_OVER_BAR` full-day override (a thin-failed day never keeps mask-planned Lake spans — Codex P2), and a thin-but-two-sided span inside a trusted segment still emits NaN beyond its real depth, so depth-consuming features are masked naturally rather than fed fabricated liquidity. |
| `lake_present_start_ts` / `lake_present_end_ts` | first / (exclusive) last-plus-one-grid-step grid sample where both top-of-book prices are present. Presence only — no trust implied. `None` when Lake never has a top-of-book. (When no presence mask is supplied to the helper, the fields fall back to the validity mask — strictly narrower, conservative.) |
| first accepted Lake seed | the existing `seed_ts` metric (`recon/reseed.py`): engine ts of the first `classify_snapshot == "ok"` Lake `book` snapshot. |
| `warmup_qualified_ts` (boundary primitive) | the ts of the `warmup_consecutive`-th **consecutive valid** grid sample, counting only samples with `ts >= seed_ts`. `None` if the seeded book never sustains. A strictly conservative refinement of the parity gate's `cutoff = max(lake_warmup_cutoff, seed_ts)` clamp: the run restarts at the seed, so it is `>=` the clamp, equal whenever no valid run straddles the seed. |
| `trusted_lake_start_ts` / `trusted_lake_end_ts` (plan/report fields) | the SURVIVING Lake coverage bounds: start of the first / exclusive end of the last Lake segment after the `min_lake_segment_s` island drop. `None` on every full-day route (including crossed-source days where the book may sustain). Differ from `warmup_qualified_ts` exactly when a qualified island was dropped. |
| `fill_segments` | ordered, non-overlapping, exhaustive partition of `[day_open_ts, day_end_ts)` into segments `{source, start_ts, end_ts, reason}` with `source ∈ {lake, coinapi, excluded}`. |
| **seam** | an internal boundary between adjacent segments with *different* sources. The boundary sample itself belongs to the segment on the right. Lake↔CoinAPI seams are vendor switches; excluded↔lake edges are treated as seams too (conservative). |
| `seam_guard_s` | half-width of the masked window `[seam − guard, seam + guard)` around every seam. |
| `vendor_source` (per sample/bar) | source of the segment containing the sample's `t_event`. |
| `feature_vendor_source` | the set of sources intersecting the feature window `[t_event − lookback, t_event]`. |
| `label_vendor_source` | the set of sources intersecting the label window `(t_event, t_barrier]`. |

## Policy decisions

### Q1. Classifying a day with a missing leading/trailing segment

The five-class taxonomy (`lake_usable` / `lake_present_degraded` / `missing_needs_coinapi` /
`excluded` / `inconclusive`) stays **closed and unchanged**. A partial day classifies exactly as
today: `2025-01-07` is `lake_present_degraded` (missing 0.6146 > 0.02), `2024-08-05` is
`inconclusive` (crossed seed source). Following the PR #13 precedent, the partial-day policy is
encoded in the machine-readable fill contract, **not** by reclassifying: the per-day record gains
sub-day *structure* — coverage timestamps in `quality`, and a `fill_profile` + `fill_segments` block
under `coinapi_fill` (schema in Q7). Rationale: `classification` keeps meaning "what Lake alone
supports"; where the fill comes from and for which window is fill-plan state.

### Q2. Routing: Lake-only vs full-day fill vs partial fill vs exclusion

Decision order (first match wins), applied only after the day-level `coinapi_fill.needs_fill`
decision:

| # | condition | routing | reason code |
|---|---|---|---|
| 1 | `needs_fill == False` (`lake_usable`) | **Lake-only**, no stitch plan | `lake_usable` |
| 2 | no accepted seed (`seed_accepted == False`) | **full-day CoinAPI** | `no_accepted_seed` |
| 3 | crossed/untrusted seed source (`seed_source_crossed_frac > 0.05`, the PR #13 rule) | **full-day CoinAPI** | `crossed_seed_source` |
| 4 | seeded book never warmup-qualifies (`trusted_lake_start_ts is None`) | **full-day CoinAPI** | `lake_never_warmup_qualified` |
| 5 | every trusted Lake segment shorter than `min_lake_segment_s` | **full-day CoinAPI** | `lake_trusted_span_too_short` |
| 6 | invalid fraction *within* the Lake segments > `span_invalid_max` | **full-day CoinAPI** | `quality_over_trusted_span` |
| 7 | otherwise | **partial fill**: CoinAPI covers the missing/invalid windows, Lake covers its trusted spans | per-segment reasons |

Consequences for the two anchor days:

- `2025-01-07` → rule 7: CoinAPI `[00:00, trusted_lake_start_ts)`, Lake
  `[trusted_lake_start_ts, 24:00)` — `leading_partial_fill`.
- `2024-08-05` → rule 3: crossed seed source **dominates** the partial day → full-day CoinAPI fill.
  (This also disposes of its untested crossed-source parity: the day is a fill day either way.)

Exclusion is never a *routing* outcome at day level here — a fill day whose CoinAPI flat file is
absent (`coinapi.fillable == false`) stays a surfaced `needs_fill` day that the fill manifest must
report as unfillable (it drops out of training coverage, mirroring the existing `no_verdict`
never-silently-dropped rule). Sub-day exclusion exists only as (a) `excluded` micro-segments (below)
and (b) seam guard bands.

### Q3. The exact vendor-switch boundary timestamp

**The boundary is the first post-seed warmup-qualified sample**: `warmup_qualified_ts` = the ts of
the `warmup_consecutive`-th (default 3) consecutive valid grid sample counting only samples at/after
`seed_ts`. CoinAPI covers `[day_open_ts, boundary)`; the boundary sample itself is Lake's.

Rejected alternatives, and why:

- **First valid Lake sample** — a cold-started book can look two-sided/uncrossed for the wrong reason
  (stranded levels); this is exactly why the parity gate refuses pre-seed samples
  (`run_coinbase_parity.py`, "pre-seed samples are cold-started state").
- **First accepted Lake seed (`seed_ts`) alone** — the seed proves one good snapshot, not a healthy
  replay; the book can go crossed immediately after (the seed-quality gate in `docs/data.md`
  §5a-Recon requires N consecutive good states before emitting bars).
- **First non-missing/non-crossed sample** — same cold-start objection as "first valid sample".

The chosen boundary is a **strictly conservative refinement** of the composition the parity gate
already validates (`cutoff = max(lake_warmup_cutoff, seed_ts)`): the parity clamp lets a warmup run
that started *before* the seed stand, whereas this boundary restarts the run at the seed — so
`warmup_qualified_ts >= max(lake_warmup_cutoff, seed_ts)`, equal whenever no valid run straddles
the seed (pinned by test, including the divergence case). It also satisfies **no-lookahead**: the
qualification ts is a function only of samples with `ts <=` itself (the qualifying run *ends* at
the boundary), so information after the boundary can never move it earlier — pinned by test. (The
plan-level *segment layout* — e.g. an island dropped for being under `min_lake_segment_s` — is a
data-preparation decision over the whole day and carries no such per-sample claim.)

After an internal CoinAPI fill segment, the same rule re-applies per span: the Lake side resumes at
the `warmup_consecutive`-th consecutive valid sample of that span, and the requalification window
joins the preceding CoinAPI segment.

### Q4. Seam guard band

Two complementary exclusions, both masked to NaN **on the regular grid, never compacted** (the
`compare_topk` exclude/label precedent — compaction horizon-stretches `shift(-step)`):

1. **Guard band**: samples within `[seam − seam_guard_s, seam + seam_guard_s)` are excluded from
   label origins *and* label/feature targets. Default `seam_guard_s = 60` — equal to the longest
   label horizon in the ladder (2/10/60 s), so a guard-clean sample's label window can never lean on
   seam-adjacent book-settling; cost is ≤ 2 minutes of grid per seam.
2. **Window rule** (primary): any feature window or label window that *crosses* a seam is excluded
   regardless of guard distance (Q6). The guard band adds margin for vendor-level book differences
   right at the switch (the two vendors' books are near- but not byte-identical; parity median
   |Δmid| = $0.00 but p95 = $0.48 on the clean day).

Warmup qualification (3 consecutive valid samples) already vets the Lake side's health; the guard
band is belt-and-braces on top, applied symmetrically to both sides of every seam.

### Q5. Reseeding at the seam

**The incoming vendor always establishes its own state; book state never crosses a seam** (the
`docs/data.md` §5a-Recon rule "reseed from the incoming vendor's first full state; do not assume
continuity across the seam", made concrete):

- **CoinAPI→Lake switch** (leading fill, gap end): the Lake replay runs exactly as today — cold
  start at day open, seed from Lake's own first valid `book` snapshot, warmup-qualify. The CoinAPI
  book is never injected into the Lake replay as a seed. The boundary is derived from the Lake
  replay (Q3); no new replay code.
- **Lake→CoinAPI switch** (trailing/internal fill): CoinAPI replays its **whole-day** event stream
  from its opening SNAPSHOT block (label-clamped to 00:00 by `recon/coinapi.py`) in `seq` order, and
  the fill segment is realized purely by restricting `sample_ts` to the segment's window —
  `reconstruct_coinapi_l2_at_samples(chunks, k=k, day=day, sample_ts=<segment grid>,
  size_policy="decrement")`. The CoinAPI book at any mid-day boundary is therefore fully warmed by
  its own replay from day open. `size_policy="decrement"` is mandatory for Coinbase (the library
  default `"absolute"` is for other venues).
- **Prior-day carry**: not allowed across any gap or seam ("Never carry state *through* a gap").
  Prior-day seed carry for *contiguous* clean days remains a separate deferred §10 item and is not
  changed by this policy; until it lands, every day starts cold and seeds from its own vendor state.
- **Snapshot seed vs replayed vendor book**: Lake segments seed from a validated snapshot
  (`classify_snapshot == "ok"`); CoinAPI segments use the replayed vendor book (its SNAPSHOT block is
  the vendor's own full state). Cross-vendor seeding (using a CoinAPI book to seed the Lake replay or
  vice versa) is prohibited — it would silently blend vendor semantics before the seam even starts.

### Q6. Labels/bars whose windows cross a seam

A training row is usable only if **both** of its windows are single-vendor and guard-clean:

- `feature_vendor_source` of `[t_event − max_lookback_ns, t_event]` must be a singleton (`{lake}` or
  `{coinapi}`), and the window must not intersect any seam guard band.
- `label_vendor_source` of `(t_event, t_barrier]` must be a singleton and guard-clean.
- Rows failing either test are masked (NaN label / dropped row) on the regular grid — never
  compacted. Mixed-vendor windows are excluded even though single-day parity is high, because
  cross-vendor level/size semantics have not been validated as *interchangeable within one window*;
  if a future parity study proves a narrower exception, it must be encoded as an explicit relaxation,
  not assumed.
- Bars that aggregate raw events (dollar bars etc.) inherit the same rule via their event span:
  a bar whose `[t_open, t_close]` crosses a seam is excluded.

Helpers: the rule is the composition of two checks — `window_vendor_sources(start, end, segments)`
must be a singleton `{lake}` or `{coinapi}` (single-vendor coverage; also rejects windows touching
`excluded` segments, and windows reaching outside the day's partition carry `uncovered` — a
day-edge label whose target lands past day end never reads as clean; cross-midnight windows
resolve only against the adjacent day's plan, a bar-builder follow-up), AND `label_valid_mask` /
`feature_valid_mask` must pass (seam-crossing + guard geometry). Neither alone suffices: a window
inside an `excluded` segment contains no seam, and a guard-clean window can still touch two
vendors' segments only via a seam (caught by both).
Manifest integration:

- **Now (zero schema change)**: the build-level `sources` list in the feature manifest accepts dict
  entries with extra keys — record per-vendor day/segment coverage there (e.g.
  `{"name": "coinapi/limitbook_full", "days": [...], "segments": ...}`).
- **With the bar builder**: add a per-bar `vendor_source` column declared in `extra_cols` (or
  appended to `RESERVED` if made mandatory). Note the leaky-name screen: a column literally named
  `label_vendor_source` matches the `label` pattern and can never be a feature — provenance columns
  are non-features by design, so declare them reserved/extra.

### Q7. Quality-map report expression

Per-day record extensions (all new keys must also land in `_empty_quality_block` /
`_default_*` blocks — the schema-consistency invariant is pinned by `test_quality_map.py`):

- `quality` gains coverage timestamps:
  `lake_present_start_ts`, `lake_present_end_ts`, `trusted_lake_start_ts`, `trusted_lake_end_ts`,
  `n_invalid_runs`, `invalid_runs` (list of `[start_ts, end_ts]`, capped like `reseed_ts[:100]`).
- `coinapi_fill` gains the fill plan:

```json
{
  "day": "2025-01-07",
  "classification": "lake_present_degraded",
  "coinapi_fill": {
    "needs_fill": true,
    "why": "quality_over_usable_bar",
    "fill_profile": "leading_partial_fill",
    "fill_segments": [
      {"source": "coinapi", "start_ts": 1736208000000000000, "start_iso": "2025-01-07T00:00:00Z",
       "end_ts": 1736261140000000000, "end_iso": "2025-01-07T14:45:40Z",
       "reason": "lake_missing_leading_segment"},
      {"source": "lake", "start_ts": 1736261140000000000, "start_iso": "2025-01-07T14:45:40Z",
       "end_ts": 1736294400000000000, "end_iso": "2025-01-08T00:00:00Z",
       "reason": "trusted_seeded_lake_reconstruction"}
    ],
    "seams": [1736261140000000000],
    "seam_policy": {"seam_guard_s": 60.0, "warmup_consecutive": 3, "fill_min_s": 300.0,
                    "min_lake_segment_s": 3600.0, "span_invalid_max": 0.01,
                    "exclude_labels_crossing_seam": true, "exclude_features_crossing_seam": true}
  }
}
```

(The boundary `14:45:40Z` above is illustrative — the real value is *computed* as
`trusted_lake_start_ts` from the seeded replay, at/after the ~14:45:00Z resume the doc table
records as prose.)

- `fill_profile` ∈ `{lake_only, full_day_fill, leading_partial_fill, trailing_partial_fill,
  internal_gap_fill, mixed_partial_fill}`; days without a stitch plan (no fill, or no verdict) carry
  `fill_profile: null`.
- `summary.coinapi_fill` gains a `partial_fill` day list alongside
  `needs_fill`/`no_fill`/`no_verdict`/`not_in_scope` (`partial_fill ⊆ needs_fill`, so existing
  consumers are unaffected).
- The usable calendar keeps day-granular `coinbase_fill_days` untouched; sub-day spans live only in
  the quality-map report (single source of truth for fill manifests). Fill *downloads* stay
  day-granular regardless (the CoinAPI flat-file downloader and the §5a backfill gate operate on
  whole days; a partial-day fill downloads the whole day and *uses* a window of it — budget seam
  days as full-day downloads, ~$1–2.4/day).

### Q8. Required tests before unlocking real backfill

Synthetic (this PR, `tests/test_stitch_policy.py` — no vendor I/O):

1. Leading missing segment → CoinAPI then Lake; boundary == warmup-qualified-after-seed ts.
2. Trailing missing segment → Lake then CoinAPI.
3. Internal gap ≥ `fill_min_s` → Lake/CoinAPI/Lake with two seams; short blips do NOT split.
4. Crossed seed source → full-day CoinAPI even when the day is also partial (2024-08-05 shape).
5. No accepted seed / never-qualified / span-too-short / span-quality-fail → full-day CoinAPI.
6. Valid-before-seed samples never count toward warmup (seed clamp).
7. Guard band masks samples around each seam, both sides.
8. Labels crossing a seam are excluded; labels inside one segment survive; same for feature windows.
9. No-lookahead: truncating or corrupting all samples after the boundary does not move the boundary.
10. Segments exactly partition `[day_open, day_end)`; every boundary sample belongs to the right
    segment; plan JSON round-trips under `json.dumps(..., allow_nan=False)`.
11. `valid_mask_from_frame` reproduces `lake_warmup_cutoff` on synthetic frames (shared predicate
    pinned).
12. Boundary vs parity clamp: `warmup_qualified_ts >= max(lake_warmup_cutoff, seed_ts)`, with the
    straddle-divergence case pinned explicitly (all-valid book, mid-day seed).
13. Dropped-island semantics: a qualified island under `min_lake_segment_s` beside a surviving span
    — `plan.trusted_lake_start_ts` reports the surviving segment, `warmup_qualified_ts` the raw
    qualification ts.
14. Exact-threshold pins: an invalid run exactly at `fill_min_s` fills (inclusive `>=`), a Lake
    island exactly at `min_lake_segment_s` survives (inclusive `>=`), a span invalid fraction
    exactly at `span_invalid_max` stays partial (strict `>`, the quality-map inclusive-usable
    convention); `DEFAULT_SEAM_POLICY` pinned to the defaults table.
15. Input validation: irregular grid, non-positive `grid_ns` (incl. the n==1 case), and
    `present`-mask length mismatch all raise; `window_vendor_sources` singleton rule pinned,
    including the `uncovered` marker for windows extending past the day partition (Codex P2).

Live/integration (follow-up branches, before unlock):

16. **Seam-day stitch validation on 2025-01-07** (one bounded single-day CoinAPI pull, gate-allowed):
    build the stitched plan from the real quality-map record, replay CoinAPI over the leading
    window, and run `compare_topk` on the *overlap* region (both vendors, post-boundary) — parity in
    the Lake window must meet the 2025-06-01-class bar; report seam guard exclusions transparently.
17. **2024-08-05 full-day verification**: confirm the crossed-source routing (needs_fill=true,
    full_day_fill) against its real record; no parity rehabilitation attempted.
18. A trailing-partial and an internal-gap day located from the broad map (if none exist, synthetic
    coverage stands).
19. Native-engine conformance for the new coverage metrics (present/trusted timestamps, invalid
    runs) once wired — byte-equal Python vs native. **Done 2026-07-02 (Task 3, synthetic):**
    native meta == Python meta and identical assess-level plans/quality blocks pinned in
    `tests/test_native_recon.py` / `tests/test_quality_map.py`.
20. Report schema tests: extended `quality`/`coinapi_fill` blocks schema-consistent across
    assessed/excluded/load-failed days; `jq empty` passes.
21. Fill-manifest budget check: every `needs_fill` day (full or partial) appears with a whole-day
    download cost; partial days must NOT be budgeted as fractional downloads.

## Segment derivation algorithm (normative)

Inputs: the full-day regular grid `sample_ts` (strictly increasing, spacing `grid_ns` — enforced),
per-sample `valid` mask, `seed_accepted`, `seed_ts`, `seed_source_trusted`, `SeamPolicy`.

1. If rules 2–4 of the Q2 table fire → single `coinapi` segment `[day_open, day_end)` with the
   routing reason; `fill_profile = full_day_fill`; no seams.
2. Compute maximal invalid runs; runs with duration ≥ `fill_min_s` are **fill windows**.
3. The complement spans are candidate Lake spans. For each span, the trusted start is the
   `warmup_consecutive`-th consecutive valid sample at/after `max(span_start, seed_ts)`; the
   requalification prefix `[span_start, trusted_start)` joins the preceding fill window (or the
   leading window). A span that never qualifies joins the fill region entirely.
4. Drop trusted Lake spans shorter than `min_lake_segment_s` into the fill region (a small island is
   not worth two seams; the whole day's CoinAPI file is already downloaded).
5. If no Lake span survives → full-day CoinAPI (`lake_trusted_span_too_short` /
   `lake_never_warmup_qualified`).
6. If the invalid fraction *within* the surviving Lake spans (short blips) > `span_invalid_max` →
   full-day CoinAPI (`quality_over_trusted_span`); blips below the bar stay masked samples inside
   the Lake segment (existing crossed-sample exclusion).
7. Merge adjacent non-Lake windows; each merged window becomes `coinapi` if its duration ≥
   `fill_min_s`, else `excluded` (e.g. the few pre-seed seconds on an otherwise clean day — too
   small to fill, not Lake-trustworthy either).
8. Segment reasons: `coinapi` windows are `lake_missing_leading_segment` /
   `lake_missing_internal_segment` / `lake_missing_trailing_segment` by position; Lake segments are
   `trusted_seeded_lake_reconstruction`; `excluded` windows carry `leading_warmup_excluded` — they
   are structurally day-open-only, since every mid-day non-Lake window contains the
   `>= fill_min_s` invalid run that ended the previous Lake segment and therefore routes `coinapi`.
9. Seams = every internal boundary between different-source segments. Profile from the CoinAPI
   window positions (leading/trailing/internal/mixed; none → `lake_only`).

## Defaults and rationale

| knob | default | rationale |
|---|---|---|
| `warmup_consecutive` | 3 | matches the parity gate's `--warmup-consecutive`; same predicate, same value. |
| `seam_guard_s` | 60.0 | equals the longest label horizon (60 s ladder) — a guard-clean origin's window never touches the seam; ≤ 2 min masked per seam. |
| `fill_min_s` | 300.0 | below 5 min, a fill segment buys less data than its two seams + guards mask (2 × 2 min); short outages stay masked samples. |
| `min_lake_segment_s` | 3600.0 | a sub-hour Lake island on a fill day saves nothing (the day's CoinAPI file is already downloaded) and adds two seams of risk. 2025-01-07's ~9.25 h span clears it comfortably. |
| `span_invalid_max` | 0.01 | mirrors `Thresholds.crossed_usable_max` — the trusted span must meet the same bar a usable day meets. |

All knobs are emitted in `seam_policy` in the report (the `Thresholds.as_dict` pattern), so every
artifact records the policy that produced it.

## Worked examples

- **2025-01-07 (leading partial)**: seed source clean, seed accepted after the 14:45:00Z resume,
  crossed 0.000116 where present → rules 2–6 pass → CoinAPI `[00:00, boundary)`, Lake
  `[boundary, 24:00)`, one seam, `leading_partial_fill`. Labels with a 60 s horizon lose origins in
  `[boundary − 120 s, boundary + 60 s)` — the guard band plus every origin whose label window
  touches it or crosses the boundary.
- **2024-08-05 (partial + crossed source)**: `seed_source_crossed_frac = 0.2878 > 0.05` → rule 3 →
  one CoinAPI segment `[00:00, 24:00)`, `full_day_fill`, reason `crossed_seed_source`. The 16:08:35Z
  resume never matters.
- **Hypothetical trailing outage** (Lake dies 21:30, stays dead): trailing invalid run 2.5 h ≥
  `fill_min_s` → Lake `[boundary, 21:30)`, CoinAPI `[21:30, 24:00)`, `trailing_partial_fill`.
- **Hypothetical internal outage** (45 min mid-day): Lake / CoinAPI / Lake, two seams,
  `internal_gap_fill`; the post-gap Lake segment starts at its own requalification sample.

## Integration points

- `recon/stitch_policy.py` (this PR): pure planning/masking helpers; no vendor I/O; numpy plus a
  pandas frame adapter (`valid_mask_from_frame`) only.
- `scripts/run_coinbase_quality_map.py` (follow-up): compute the valid mask on the Python frame path
  (`valid_mask_from_frame`), emit the Q7 `quality` coverage keys and `coinapi_fill` plan block via
  `plan_day_stitch`; extend `_empty_quality_block`; add the `partial_fill` summary list.
- `recon/reseed.py` + native engine (**implemented 2026-07-02**, Task 3): the native path is
  frame-free, so `_replay_seeded`/Rust emit invalid-run boundaries (the `crossed_sample_ts`
  precedent) as the compact `meta["coverage"]` block — conformance-pinned before the broad map
  relies on native coverage metrics.
- Future stitcher/backfill manifest: consumes `coinapi_fill.fill_segments` + `coinapi.fillable`;
  downloads whole days (backfill gate unchanged); realizes CoinAPI segments by restricting
  `sample_ts` with `size_policy="decrement"`; realizes Lake segments through the existing seeded
  replay; applies guard/window masks at bar/label build.
- `docs/feature-manifest.md` / `eval/manifest.py` (with the bar builder): per-bar `vendor_source`
  reserved/extra column; `sources` entries carry vendor+segment coverage today with zero schema
  change; JEPA/LightGBM training manifests inherit the masks through the standard timing columns —
  no learner-specific logic.

## Implementation Tasks

### Task 1 (this PR): policy doc + `recon/stitch_policy.py` + synthetic tests

- This document.
- `recon/stitch_policy.py`: `SeamPolicy`, `Segment`, `DayStitchPlan`, `valid_mask_from_frame`,
  `warmup_qualified_ts`, `plan_day_stitch`, `seam_guard_mask`, `window_crosses_seam`,
  `label_valid_mask`, `feature_valid_mask`, `vendor_source_at`, `window_vendor_sources` —
  implementing the normative algorithm above, stable strings for all JSON-facing codes.
- `tests/test_stitch_policy.py`: the Q8 synthetic list (items 1–15), TDD (tests written first).
- `docs/data.md`: three scoped pointer edits (§5a-QualityMap finding 2, the §5a-QualityMap
  "Unlock still requires …" precondition paragraph, and the §10 quality-map item) marking the
  policy DEFINED with wiring open; backfill stays locked.

### Task 2 (follow-up branch): quality-map runner wiring — **implemented 2026-07-02**

Emit coverage keys + fill plan in the report (Python engine first); schema-consistency +
`jq empty` tests; `partial_fill` summary list. Landed in `scripts/run_coinbase_quality_map.py`
(`coinapi_fill_block` + `_stitch_and_coverage`) with `full_day_plan`/`invalid_runs` helpers in
`recon/stitch_policy.py`; synthetic tests in `tests/test_quality_map.py`.

### Task 3 (follow-up branch): native coverage metrics — **implemented 2026-07-02**

Invalid-run boundaries from the Rust core; conformance tests vs the Python frame path. Landed as a
compact `meta["coverage"]` block — maximal half-open `[i0, i1)` invalid-run sample-index pairs (the
shared `valid_mask_from_frame` predicate at `min_levels_per_side=1`) plus presence bound indices —
emitted by BOTH `recon.reseed._replay_seeded` and the Rust core (`recon_native.META_ABI = 2` rejects
stale builds at import). `scripts/run_coinbase_quality_map.py` reconstructs the masks
(`_masks_from_native_coverage`) and feeds the shared `_stitch_and_coverage_masks`, so
`--engine native` emits the same Q7 coverage keys and partial fill plans as the Python frame path,
with the conservative full-day fallback kept for coverage-less meta. Conformance pinned in
`tests/test_native_recon.py` (Python replay coverage == frame-derived mask; native meta == Python
meta) and `tests/test_quality_map.py` (identical stitch plans/quality blocks at the assess level,
plus the report-cap and fallback paths). See docs/native-recon.md "Coverage metrics".

### Task 4 (follow-up branch): seam-day live validation

Q8 items 12–13 (bounded single-day pulls only; gate untouched); update `docs/data.md` §10.

### Task 5 (with bar builder): masks + `vendor_source` in the dataset build and manifests

## Validation Commands

```bash
.venv/bin/python -m py_compile recon/stitch_policy.py tests/test_stitch_policy.py
.venv/bin/python -m pytest -q tests/test_stitch_policy.py
.venv/bin/python -m pytest -q tests/test_parity.py tests/test_quality_map.py tests/test_reseed.py
git diff --check
```

(Worker note: agent worktrees have no `.venv`; use the main checkout's interpreter
`/home/aaron/jepa-btc-forecasting/.venv/bin/python` from the worktree root.)

## PR Requirements

- Title: `feat: add partial-day Coinbase fill policy`.
- Body: Summary; Policy decisions (Q1–Q8 digest); Tests/validation output; Risks and assumptions
  (provisional PR #13 crossed-source rule, no live seam-day validation yet, knob defaults untested
  against real seam days); **no vendor/API calls run**; **Backfill status: still locked**;
  Follow-ups (Tasks 2–5).
- Commit only docs/source/tests — no data, reports, or secrets.

## Review Checklist

- [ ] Boundary definition is a strictly conservative refinement of the parity gate's clamp
      (`>= max(lake_warmup_cutoff, seed_ts)`, equal when no valid run straddles the seed) — the
      divergence is documented and pinned by test, never silent.
- [ ] Every JSON-facing string (sources, reasons, profiles) is a stable module constant.
- [ ] Segments always partition the day; property pinned by test.
- [ ] No-lookahead property of the boundary pinned by test.
- [ ] Masks operate on the regular grid; nothing compacts.
- [ ] No vendor I/O anywhere in the new module or tests; backfill gate untouched.
