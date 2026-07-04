# Coinbase backfill review manifest — design

**Date:** 2026-07-03
**Branch:** `feat/backfill-manifest-review`
**Script:** `scripts/review_coinbase_backfill_manifest.py`
**Status:** approved design (brainstorming) → next: implementation plan + TDD

## 1. Purpose

Create the **review / decision layer** that turns completed Coinbase quality-map
outputs into a **human-auditable backfill manifest** before any `--allow-backfill`
CoinAPI pull. It is a **gatekeeping tool, not the backfill downloader**: no vendor
I/O, no downloads, no live API calls. It does **not** unlock or run the backfill and
does not touch the §5a gate in `ingest/download_coinapi.py` / `ingest/_common.py`.

Pipeline position:

```
quality map (per-batch reports)  ─┐
batch plan manifest              ─┼─► reviewed backfill manifest ─► human approval
usable calendar (measured GB)    ─┘        (this tool)             + spend controls
                                                                   ─► CoinAPI backfill
```

## 2. Inputs (a deterministic join of three artifacts)

1. **Batch-plan `manifest.json`** — output of
   `scripts/plan_coinbase_quality_map_batches.py`
   (`data/tmp/coinbase_quality_map_batches/manifest.json`). The **authoritative
   batch registry**: `summary.n_batches`; `batches[]` with `file`, `report_dir`,
   `n_days`, `first_day`, `last_day`; `meta.input_calendar`, `meta.out_dir`;
   `batched_trade_only_fill_days[]`; and `skipped.{fill_days_book_gap,
   excluded_days_by_reason, days_dropped_as_excluded_or_book_gap}`.
2. **Per-batch quality-map reports** — loaded from each
   `batches[i].report_dir/coinbase_quality_map.json`
   (`scripts/run_coinbase_quality_map.py` output). **Never directory-globbed** —
   the plan drives which reports must exist. Each report has `meta`, `summary`,
   `days[]`; every `days[]` record carries a machine-readable `coinapi_fill` block.
3. **`usable_calendar.json`** — pinned by `plan.meta.input_calendar`
   (`ingest/verify_trades_and_calendar.py` output). The **only** source of the
   **trade-fill** and **book-gap** dimensions (reports are `book_delta_v2`-only),
   plus **measured per-day sizes** in `fill_status[day].{book,trades}.mb`.
   Relevant fields: `lake_all_days[]`, `usable_days[]`,
   `coinbase_fill_days{day:{book,trades}}`, `excluded_days_by_reason{day:[reason]}`,
   `fill_status{day:{book:{present,mb,ok}|null, trades:{present,mb,ok}|null,
   error, reason, ok}}`, `fill_days_unfillable[]`, `fill_days_probe_error[]`,
   `backfill_verified`, `anchor_end`.

The tool is a **pure aggregation layer** over `assess_lake_day` outputs: it
**preserves each day's `coinapi_fill` decision verbatim** (`fill_profile`,
`full_day_reason`, `fill_segments`, `seams`, `seam_policy`, `trusted_lake_*`) and
**never recomputes a stitch plan or reinterprets reason prose**.

### Modes (mutually exclusive)
- **Readiness mode** (`--plan-manifest`) — the gate. Plan-driven completeness checks
  can reach `status="ready"`; otherwise fail-closed `status="blocking"`.
- **Inspection mode** (`--report ...`, no plan manifest) — for eyeballing one or more
  reports. `scope_complete` is forced `false` and `status` is always `report_only`
  (a lone report is never the full backfill scope). It does **not** gate.

`--plan-manifest` and `--report` are mutually exclusive; exactly one is required.

## 3. Output manifest schema (v1, stable)

Written to `data/reports/backfill/coinbase_backfill_manifest.json` (git-ignored),
strict JSON (`json.dump(indent=2, allow_nan=False)` + trailing newline),
byte-deterministic given an injected `generated_utc`.

```jsonc
{
  "manifest_version": 1,
  "meta": {
    "kind": "coinbase_backfill_review",
    "tool": "scripts/review_coinbase_backfill_manifest.py",
    "generated_utc": "2026-07-03T12:00:00Z",
    "status": "ready" | "blocking" | "report_only",
    "scope_complete": true | false,
    "exchange": "COINBASE",
    "symbol": "BTC-USD",
    "thresholds": { ... },              // pinned from reports; must agree across all
    "inputs": {                          // §fix-3 input identity pinning
      "plan_manifest": { "path": "...", "sha256": "..." },
      "usable_calendar": { "path": "...", "sha256": "...", "anchor_end": "..." },
      "batch_reports": [
        { "report_dir": "...", "path": "...", "sha256": "...",
          "batch_file": "batch_001_days.txt", "n_days": 604 }
      ],
      "n_batches": 2,
      "plan_generated_utc": "..."
    },
    "cost_model": {
      "book_usd_per_gb": 1.0,
      "trades_usd_per_gb": 3.0,
      "est_book_gb_per_day": 2.27,       // §2.2/§6 nominal L3 (conservative)
      "est_trades_gb_per_day": 0.05,     // §8 2.6 GB / 52 days
      "credit_usd": 25.0,                // §8 flat-files trial pool
      "partial_day_charged_as_full_day": true,
      "tiered_discount_applied": false,  // §2.2 unconfirmed; flat $1/GB
      "notes": "measured fill_status.mb where available; nominal per-day rate for
                quality-map-added present days; flat $1/GB until measured."
    }
  },

  // canonical per-day records — ONE per relevant calendar-universe day (§fix-1):
  // Lake-present (usable/degraded/inconclusive), calendar book-gap, trade-only,
  // excluded, and unresolved days all appear here. Sections index into this list.
  "days": [
    {
      "day": "2024-12-04",
      "classification": "inconclusive" | "lake_usable" | "lake_present_degraded"
                        | "missing_needs_coinapi" | "excluded" | null,
                        // one of the five quality-map classes for report-backed days;
                        // null for calendar-only days (sources without "quality_map").
                        // The classification enum check applies ONLY to report-backed days.
      "sources": ["quality_map"],        // which inputs contributed: quality_map|calendar_gap|calendar_trade|calendar_excluded
      "calendar": {                       // authoritative, from usable_calendar
        "in_lake_all_days": true,
        "in_usable_days": true,
        "is_coinbase_fill_day": false,
        "book_gap": false,                // coinbase_fill_days[day].book
        "trades_gap": false,              // coinbase_fill_days[day].trades
        "excluded_reason": null
      },
      "book_fill": {
        "needed": true,
        "source": "quality_map" | "calendar_gap" | "both" | null,   // §fix-4
        "kind": "full_day" | "partial" | null,   // from fill_profile
        "why": "crossed_seed_source_cross_validated_2026-07-01",     // verbatim coinapi_fill.why, or "calendar_book_gap"
        "fill_profile": "full_day_fill",          // verbatim
        "full_day_reason": "crossed_seed_source", // verbatim
        "fill_segments": [ { "source": "coinapi", "start_ts": ..., "start_iso": "...",
                             "end_ts": ..., "end_iso": "...", "reason": "..." } ], // verbatim
        "seams": [],                              // verbatim (int ns)
        "seam_policy": { ... },                   // verbatim
        "trusted_lake_start_ts": null,
        "trusted_lake_end_ts": null,
        "gb": 2.27, "gb_basis": "estimated", "usd": 2.27
      },
      "trade_fill": {
        "needed": false,
        "source": null,                  // "calendar" when needed
        "measured_mb": null,
        "gb": 0.0, "gb_basis": "measured" | "estimated", "usd": 0.0
      },
      "excluded": null,                    // { "reason": [...] } when classification=="excluded"/calendar-excluded
      "unresolved": null,                  // { "why": "no_verdict", "classification": "inconclusive", "reasons": [...] } when blocking
      "notes": []                          // e.g. "native fallback full-day (meta.coverage absent)"
    }
  ],

  // day-list VIEWS into days[] — the seven required buckets (§scope)
  "sections": {
    "full_day_book_fills":  ["..."],       // book_fill.needed && kind=="full_day" (calendar_gap + quality_map + both)
    "partial_day_book_fills": ["..."],     // book_fill.needed && kind=="partial"
    "trade_fills":          ["..."],       // trade_fill.needed (calendar-sourced)
    "lake_usable_days":     ["..."],       // classification=="lake_usable"
    "lake_present_degraded_days": ["..."], // classification=="lake_present_degraded" (NORMAL section, not blocking)
    "excluded_days":        ["..."],       // classification=="excluded" or calendar-excluded (out-of-scope, non-Coinbase)
    "unresolved_days":      ["..."]        // unresolved != null (BLOCKING; == blockers.unresolved_days)
  },

  "cost_summary": {
    "book_fill_days": 51, "full_book_fill_days": 49, "partial_book_fill_days": 2,
    "trade_fill_days": 52,
    "book_gb_measured": 84.6, "book_gb_estimated": 4.6, "book_gb_total": 89.2,
    "trades_gb_measured": 2.6, "trades_gb_estimated": 0.0, "trades_gb_total": 2.6,
    "book_usd": 89.2, "trades_usd": 7.8, "gross_usd": 97.0,
    "credit_usd": 25.0, "net_usd": 72.0,
    "calendar_gap_baseline_usd": 92.0,     // COMPUTED from measured fill_status over calendar book+trade gap days
    "quality_map_addition_usd": 5.0,       // gross − computed calendar-gap baseline
    "docs_reference_usd": 92.0,            // docs §8 reference tied to the 2026-06-22 calendar snapshot (reconciliation only)
    "band": { "low_usd": 92.0, "high_usd": 97.0 }  // measured-only low vs +estimate high
  },

  "blockers": {                            // populated iff status=="blocking"; else all empty
    "structural": [], "missing_keys": [], "coverage_gaps": [],
    "inconsistencies": [], "unresolved_days": [], "batch_incomplete": [],
    "book_fill_unavailable": [], "trade_fill_unavailable": [], "calendar_drift": []
  }
}
```

## 4. Fail-closed rules

**Default is fail-closed.** In **readiness mode**, any condition below sets
`meta.status="blocking"` and exits **3**. `--report-only` keeps the honest `status`
(`ready`/`blocking`) and the full `blockers`, but forces **exit 0** — an explicit,
logged opt-in so the gate never silently fails open. **Inspection mode** (`--report`,
no plan) always sets `status="report_only"` and exits 0; it does not gate. The
`report_only` status value therefore denotes inspection mode; a `--report-only`
readiness run keeps its true `blocking`/`ready` status.

**Structural / input errors** → **exit 2**, `ERROR: ...` on stderr (PlanError style):
plan manifest or any required report missing, unreadable, or not valid JSON / not a
JSON object; `meta.input_calendar` missing or unreadable.

**Blocking verdict** (`status="blocking"`) → **exit 3**:

- **missing_keys** — a report lacks `meta`/`summary`/`days`, or a `days[]` record
  lacks `day`/`classification`/`coinapi_fill` (with `needs_fill`/`why`/`fill_profile`).
  Strict: absent required keys block, never assumed.
- **coverage_gaps** —
  - a planned `batches[i]` has no report at its `report_dir` (`planned_but_no_report`);
  - a planned batch day (union of `batch_NNN_days.txt` ≡ `select_days(cal).batch_days`)
    absent from every report's `days[]` (`day_not_mapped`);
  - a day appearing in more than one batch report (`duplicate_across_batches`);
  - a calendar book-gap day (`coinbase_fill_days[d].book==true`) that is neither
    represented as `missing_needs_coinapi` in some report nor listed in
    `plan.skipped.fill_days_book_gap` (`gap_day_unmapped`);
  - `plan.skipped.days_dropped_as_excluded_or_book_gap` non-empty (contradictory
    calendar underlies the batch day-set).
- **batch_incomplete** (§fix-5) — a batch report whose **completion evidence does not
  indicate it actually ran**: `meta.quota` missing required keys, or `meta.quota.reason`
  in {`quota_headroom`, `exceeds_auto_cap`} (the pull was refused → 0 days loaded), or
  the report's mapped day count does not reconcile with the batch's planned day count.
  Implemented as "required quota/report completion fields must be present and indicate
  the batch ran", with strict missing-key handling — **not** a bare `quota.ok` check.
- **inconsistencies** —
  - unknown enum (**report-backed days only**): `classification` ∉ the five classes,
    `coinapi_fill.why` ∉ the six why-codes, or `fill_profile` ∉ the seven profile
    values. A calendar-only day carries `classification=null` by construction (its
    `sources` excludes `quality_map`) and is exempt from the classification enum check;
  - contradiction: `needs_fill==true` with `fill_profile` ∈ {`null`,`lake_only`}
    (**degraded/fill day with no fill policy**, §fix-2), or `fill_profile=="full_day_fill"`
    with `full_day_reason==null`, or a partial profile with `full_day_reason!=null`,
    or a full-day route with non-null `trusted_lake_*`;
  - recomputed per-class `counts` / `fill_counts` disagree with the report's `summary`
    (tool trusts `days[]` as primary, `summary` as cross-check);
  - cross-batch meta drift: `exchange`, `symbol`, or `thresholds{}` differ between reports.
- **calendar_drift** — a mapped day's report `calendar` context
  (`in_lake_all_days`/`is_coinbase_fill_day`/`excluded_reason`) contradicts the loaded
  `usable_calendar` (reports carry no calendar hash, so this per-day cross-check is the
  drift guard).
- **unresolved_days** (global block) — any day with `coinapi_fill.needs_fill==null` and
  `why=="no_verdict"` (unresolved `inconclusive` — no seed / rejected seed / load
  failure). Blocks **all** backfill readiness, not just its own batch.
- **trade_fill_unavailable** — a trade-fill day (`coinbase_fill_days[d].trades==true`)
  whose `fill_status[d].trades` is absent/not present/not `ok`, or
  `fill_status[d].error==true`, or `d` ∈ `fill_days_unfillable`/`fill_days_probe_error`.
- **book_fill_unavailable** — symmetric for a calendar **book-gap** day
  (`coinbase_fill_days[d].book==true`) whose `fill_status[d].book` is absent/not
  present/not `ok`, or `fill_status[d].error==true`, or `d` ∈
  `fill_days_unfillable`/`fill_days_probe_error`. Calendar-derived book fills are part of
  the complete backfill spec, and `ingest/verify_trades_and_calendar.py` records both
  products in `fill_status`, so a book gap must be verifiably fillable too.

**Ready** (`status="ready"`, `scope_complete=true`, exit 0) requires: every planned
batch has a ran report; planned day-set == union of report days exactly (no missing,
extra, or cross-batch duplicate); every book-gap and trade-fill day represented **and
verifiably fillable** (`fill_status`); no unresolved days; no inconsistencies; no
calendar drift. A lone report (inspection mode) can never satisfy the plan-driven
checks → never `ready`.

### Degraded vs unresolved (§fix-2)
`lake_present_degraded` is a **normal** classification that maps to a book fill
(`needs_fill=true`, a full-day or partial plan). It is **not** a blocker. Only these
block: `no_verdict` days; unknown/contradictory `coinapi_fill`; a degraded/fill day
whose `coinapi_fill` carries **no** fill policy (the contradiction case above); and
missing report/calendar evidence.

## 5. Cost model

- **Measured** `fill_status[day].{book,trades}.mb / 1000` (GB) for calendar-gap /
  trade-fill days — reproduces the §8 ~$92 figure exactly.
- **Estimated** nominal per-day rate for quality-map-added present days that have no
  measured `fill_status`: book **2.27 GB/day** (§2.2/§6 L3), trades **0.05 GB/day**.
- Prices **$1.00/GB** book (`limitbook_full`), **$3.00/GB** trades (§2.2).
- **Partial book fill is charged as one full L3 day-file** (whole-file S3 GET; no
  proration by segment duration — the conservative choice, and the actual download cost).
- Credit **$25** (flat-files pool) subtracted once from the book+trades gross.
- Tiered discount ($1→$0.10/GB above 512 GB, §2.2) **unconfirmed** → flat $1/GB.
- `cost_summary` keeps **measured vs estimated** GB/$ explicit and emits a low/high
  band. The **calendar-gap baseline is COMPUTED** from measured `fill_status` over the
  calendar book+trade gap days, so it tracks the actual input calendar (which may be
  regenerated with a different `anchor_end`). The docs §8 **$92** figure is carried
  separately as `docs_reference_usd` — a reconciliation reference tied to the
  2026-06-22 snapshot, **not** a hard-coded baseline. Spend Management should be sized
  to the **high** band.

## 6. CLI

```
scripts/review_coinbase_backfill_manifest.py

  # exactly one mode (mutually exclusive, one required):
  --plan-manifest PATH        # READINESS mode: authoritative batch registry; can reach status=ready
  --report REPORT ...         # INSPECTION mode: one or more reports, no plan; status is always report_only

  --usable-calendar PATH      # default: plan.meta.input_calendar (required in readiness mode)
  --out PATH                  # default data/reports/backfill/coinbase_backfill_manifest.json
  --report-only               # readiness mode: downgrade a blocking verdict to exit 0 (keeps honest status + blockers)
  --generated-utc ISO         # injectable for deterministic tests
```

Terminal summary (concise): status line, per-section counts, cost band, and the top
blockers when blocking. **No runnable `--allow-backfill` commands are emitted** — the
manifest is a data spec for a future runner; it surfaces gate context (multi-day pull
⇒ needs `--allow-backfill` + CoinAPI Spend Management, §8) only as an advisory note.
The §5a gate stays owned by `ingest/_common.check_backfill_gate`.

## 7. Testing (synthetic fixtures, `tmp_path`, no vendor I/O)

Mirrors `tests/test_plan_quality_map_batches.py`: load the script by path; build tiny
synthetic plan + per-batch reports + calendar JSON. Cases:

- **all-clear → `ready`** (exit 0), sections + costs correct;
- **incomplete batch set** (a planned batch has no report) → `blocking`/exit 3;
- **full-day fill day** (missing / crossed-seed) routed + costed;
- **partial-day fill day** — `fill_segments`/`seams`/`seam_policy` preserved **verbatim**;
- **unresolved day** (`no_verdict`) → `blocking`, global;
- **degraded day with a valid fill** → **not** blocking (normal section);
- **degraded/fill day with no fill policy** (contradiction) → `blocking`;
- **excluded day** kept separate from `missing_needs_coinapi`;
- **trade-fill dimension** from the calendar (incl. a day needing **both** book+trade);
- **cost/GB aggregation** reconciles to a hand-computed figure (measured + estimated band);
- **fail-closed default** vs **`--report-only`** downgrade;
- **input identity** — `meta.inputs.*.sha256` present and matches file bytes;
- **calendar drift** → `blocking`.

Commands: `python -m py_compile scripts/review_coinbase_backfill_manifest.py` and the
targeted pytest module.

## 8. Docs

Add a `docs/data.md` §5a-QualityMap subsection describing
**quality map → reviewed backfill manifest → human approval/spend controls → CoinAPI
backfill**, stating explicitly this step **does not unlock or run the backfill**.

## 9. Invariants (must hold)

1. Stitch decisions copied **verbatim** — never recompute a plan or reinterpret prose.
2. Timestamps stay **int64 ns** exactly as emitted in machine fields.
3. `fill_segments` exactly partition `[day_open_ts, day_end_ts)`; seam sample belongs
   to the right segment; segment `source` ∈ {lake, coinapi, excluded}.
4. `needs_fill==true` ⇒ `fill_profile` ∈ {full_day_fill} ∪ partials; `needs_fill` ∈
   {false, null} ⇒ `fill_profile` is null.
5. `full_day_reason` non-null **iff** `fill_profile=="full_day_fill"`; full-day route ⇒
   `trusted_lake_*` null.
6. `classification` is a Lake-only verdict — never re-derived; routing lives only in
   `coinapi_fill`.
7. Book fill GB charged **per whole day-file**; partial == full-day GB.
8. Trade fills are an **independent** calendar-sourced dimension; a day may need both.
9. Fill scope is a **superset** of calendar gaps + degraded + crossed-seed present days
   — never restricted to calendar-gap-only.
10. Use `quality.n_invalid_runs` (full count), not `len(invalid_runs)` (capped at 100).
11. Strict JSON under a git-ignored path; byte-deterministic given injected `generated_utc`.
12. This PR does **not** unlock or run the backfill.

## 10. Risks / open items (surface, don't block)

- `summary.coinapi_fill.fill_counts.crossed_source_full_day` vs
  `full_day_reason_counts['crossed_seed_source']` can diverge when a native
  coverage-missing fallback also stamps `crossed_seed_source` — cross-check but do
  **not** hard-fail on that pair alone.
- Native-engine days lacking `meta.coverage` fall back to a conservative full-day plan;
  a genuinely partial-fillable degraded day may be over-routed to full-day (extra spend,
  not data loss) — surface in `notes`, don't block.
- Tiered discount unconfirmed; the $25 credit is a flat-files-only pool (don't also
  credit REST usage).
- Reports carry no calendar hash; per-day `calendar` cross-check is the only drift guard
  short of adding a hash upstream (noted as a possible follow-up).
