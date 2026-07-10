# Staged Signal Acquisition and Gate Protocol

**Status:** adopted execution policy. This document changes acquisition order and
experiment sequencing; it does not replace the reviewed downloader, reconstruction,
bar/label, or baseline implementation plans.

**Tracks:** GitHub issue #46.

**References:**

- `docs/experiment-plan.md`
- `docs/data.md`
- `docs/superpowers/plans/2026-07-03-bar-label-producer.md`
- `docs/superpowers/plans/2026-07-02-binance-downloader-plan.md`
- `docs/superpowers/plans/2026-06-22-lightgbm-baseline.md`

## 1. Decision

Stage data spend and model evidence in this order:

1. Complete the Coinbase quality/backfill gate for a predeclared pilot window.
2. Build a Coinbase-only ModelMatrix and run a preliminary signal/economics screen
   (`G0-CB`).
3. Acquire and reconstruct six months of Binance data.
4. On one matched row universe, compare Coinbase-only, Binance-only, and combined
   LightGBM arms (`G0-XV`).
5. Acquire and reconstruct the remaining Binance archive only if `G0-XV` authorizes
   the spend.
6. Run the formal full-data G1 and later E2.3 analyses with a separate untouched
   holdout.

The long-term data target remains 12-24 months. This protocol avoids committing the
full Binance quota/storage budget before the cross-venue premise has bounded OOS
evidence.

## 2. Gate Semantics

### G0-CB: Coinbase-only preliminary screen

`G0-CB` validates target-venue data, labels, costs, CPCV, and the lower bound supplied
by Coinbase's own book and trade flow. It uses the existing manifest-driven baseline
ladder and a preregistered gate block; it does not introduce a second evaluator.

- PASS: proceed to the six-month Binance pilot.
- FAIL because the target data, label timing, cost model, or execution economics are
  invalid: stop and repair or record a human-approved pivot before more data spend.
- FAIL only because Coinbase-own-book predictivity is weak: this does **not** falsify
  the Binance-to-Coinbase hypothesis. The default is a documented human decision on
  whether to run the bounded Binance pilot, not an automatic project stop.

`G0-CB` is not formal G1 and must not be reported as the project-defining gate.

### G0-XV: six-month cross-venue spend gate

Build three arms over identical rows, labels, costs, horizons, CPCV splits, and regime
tags:

1. Coinbase-only features.
2. Binance-only signal features, lagged to their observable decision time.
3. Combined Coinbase and Binance features.

The exact numerical gate block is frozen in the three feature manifests before the
pilot OOS month is touched. Authorization for the full Binance archive requires:

- at least one non-naive cross-venue arm to clear the existing net-of-cost G1-style
  gate block (including DSR and PBO); and
- the combined arm to beat the matched Coinbase-only control OOS net-of-cost by more
  than its preregistered bootstrap noise band.

Any post-hoc feature, horizon, cost, or threshold change is another trial and enters
the DSR/PBO trial ledger. A no-verdict or unavailable PBO fails closed. A failed
`G0-XV` blocks the full Binance pull until a documented stop or pivot decision.

`G0-XV` is an acquisition screen, not final E2.3. Six post-ETF months cannot satisfy
E2.3's pre/post-ETF comparison.

### Formal G1 and E2.3

Formal G1 remains the project hard stop defined in `docs/experiment-plan.md`. It runs
only after the approved full-data inputs and a separately frozen final holdout exist.
The pilot OOS month is model-selection evidence and may not be reused as that holdout.

E2.3 still requires the Coinbase-only, Binance-only, and combined ablation on the
full approved coverage. Its pre/post-ETF claim remains blocked unless certified
Coinbase target data extends to both regimes; additional Binance history alone does
not satisfy that requirement.

## 3. Frozen Pilot Window

| Use | Inclusive dates | Treatment |
|---|---|---|
| Six-month pilot | `2025-11-01` through `2026-04-30` | Six complete calendar months; all pilot vendor acquisition is bounded to this range. |
| Development/CPCV | `2025-11-01` through `2026-03-31` | Training, CPCV, calibration, and registered trials. |
| Pilot OOS | `2026-04-01` through `2026-04-30` | Touched once after manifests/configuration are frozen; consumed after G0 decisions. |

The existing `2026-04-01` Binance Stage-1 smoke is inside the pilot window. Coverage
gaps remain explicit exclusions; neither producer nor evaluator may silently shorten
the date range to improve results.

The formal G1 holdout must:

- be outside the pilot window;
- be selected from the certified all-feed calendar using coverage only, never model
  outcomes;
- be frozen and hash-pinned before any full-data tuning or G1 run; and
- remain untouched until the preregistered full-data configuration is final.

## 4. Dataset and Manifest Contract

The producer emits explicit, versioned datasets rather than zero-filling unavailable
venue features:

- `coinbase_only_pilot`: Coinbase clock, book, trade, labels, and costs; the manifest
  lists only Coinbase in `venues` and only Coinbase features in `feature_cols`.
- `cross_venue_pilot`: common matched rows with certified Coinbase and Binance
  coverage. It supports three explicit manifests (Coinbase-only control,
  Binance-only, combined) whose feature lists differ but whose reserved columns,
  labels, costs, row IDs, horizons, and splits are identical.
- `full_cross_venue`: produced only after the archive gate passes.

Every manifest pins source manifests/hashes, usable-calendar hash, stitch policy,
window, exclusions, bar-clock schedule, feature order, gate block, and build ID.
Missing Binance data is an exclusion in cross-venue mode, not a column of zeros.

## 5. Acquisition and Resource Gates

The six-month Binance pilot contains 181 days. The downloader plan's conservative
estimate is approximately `181 * 1.23 GB = 222.63 GB`. The completed `2026-04-01`
smoke measured `687,215,789` bytes, which extrapolates to about 124.4 decimal GB, but
one day is not a quota guarantee.

- Plan with the conservative estimate; reconcile actual manifest bytes after each
  batch.
- Keep the operational target at no more than 250 GB per quota window even though
  the vendor's 300 GB figure is a soft limit.
- At the recorded 156.25 GB usage snapshot, do not launch the entire pilot as one
  batch. Generate deterministic resumable tranches and re-check usage/disk first.
- Coinbase CoinAPI downloads are likewise staged: approve the pilot-window subset of
  the reviewed manifest first; defer remaining full-window fills until the pilot
  decisions authorize them.
- No data runner may infer scope from a directory glob. It consumes reviewed,
  hash-pinned manifests and preserves exclusions.

## 6. Issue and Merge Boundaries

- Coinbase quality-map resolution remains in #33.
- Coinbase backfill #34 gains pilot-first and deferred-remainder milestones.
- Binance Stage-1 #35 and Stage-2 #36 close on the six-month pilot deliverables.
- Bar/label issue #37 owns both producer modes and matched-arm manifests.
- Issue #38 remains formal full-data G1.
- Issue #47 tracks `G0-CB`; #48 tracks `G0-XV`; #49 tracks remaining Binance
  acquisition; #50 tracks full production reconstruction.

Operational downloads and generated reports remain untracked. Any code or durable-doc
change discovered by a pilot run uses its own issue, branch, review, and PR.

## 7. Stop Conditions

Do not start the remaining Binance archive when any of these holds:

- `G0-XV` is FAIL, blocking, or inconclusive;
- pilot source coverage or reconstruction is uncertified;
- the three arms do not share identical rows/splits/labels/costs;
- the pilot OOS was touched before manifests and trials were frozen;
- quota, disk, or spend approval is missing; or
- a policy deviation has not been recorded on the owning issue.
