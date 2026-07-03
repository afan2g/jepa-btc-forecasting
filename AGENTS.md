# AGENTS.md

This repository uses agent-authored pull requests and Codex review. Treat this
file as the durable reviewer and repository-contract guidance.

## Project Architecture

- The project forecasts short-horizon Coinbase BTC-USD mid moves using Binance
  futures/spot market structure as the primary signal source.
- Current source-of-truth planning docs:
  - `jepa_btc_forecasting_spec.md`: product and modeling spec.
  - `docs/data.md`: vendor, coverage, cost, and data-quality decisions.
  - `docs/experiment-plan.md`: implementation phases and gates.
  - `docs/feature-manifest.md`: the normalized bar/feature manifest contract for
    modeling data (schema, leakage/timing rules, how training consumes it).
  - `docs/superpowers/plans/`: task-level implementation plans.
- `ingest/` contains vendor verification and download scripts. These scripts may
  touch paid/vendor APIs, so do not run bulk or live verification commands unless
  the user explicitly asks.
- Generated/raw market data belongs under ignored data paths. Do not commit raw
  parquet/csv.gz data or secrets.

## Coding Standards

- Prefer small, reviewable changes scoped to one task.
- Preserve existing file style and naming before introducing new abstractions.
- Use explicit manifests/contracts for modeling data. Do not infer feature
  columns from "all non-reserved columns" unless a plan explicitly says so.
- Keep vendor-specific schema knowledge at ingestion boundaries. Downstream code
  should consume normalized internal contracts.
- Comments should explain non-obvious domain or safety decisions, not restate
  the code.
- Do not add production dependencies without calling them out in the PR.

## Testing Requirements

- For Python syntax-only changes, run `python -m py_compile` on changed scripts
  when practical.
- For JSON artifacts, run `jq empty <file>` and reconcile important counts when
  the artifact is used as evidence.
- For implementation plans, verify that expected commands, test counts, and
  self-review claims match the snippets in the plan.
- For ingest/vendor changes, separate cheap local checks from billable/live
  vendor checks. State clearly which were not run.
- For modeling/evaluation changes, include synthetic PASS/FAIL tests for gates
  and explicit leakage/timing checks when relevant.

## Performance Constraints

- Market data can be multi-GB per day. Avoid loading full days into memory unless
  the script is explicitly designed for that.
- Prefer streaming, chunked, or day-partitioned processing for CoinAPI and Crypto
  Lake data.
- Keep CoinAPI calls throttled and bounded. Respect `COINAPI_RPM`, spend caps,
  and documented billing risks.
- Avoid broad scans over raw data in CI.

## Data And Secrets Rules

- Never commit `.env`, credentials, raw vendor data, or large local caches.
- Keep `data/raw/`, `.lake_cache/`, `.coinapi_cache/`, and rate-limit research
  artifacts out of git unless a specific small audit artifact is intentionally
  tracked.
- Treat date anchors and coverage numbers as snapshot-specific. If a doc says
  "latest" or "current," verify the date and source before updating claims.
- Do not claim a data coverage result is verified unless the exact products
  needed by the pipeline were checked.

## Builder Rules

- One agent, one task, one branch.
- Use Conventional Branch purpose-prefixed names like `feat/<topic>`, `fix/<topic>`, or `chore/<topic>`; do not use agent/vendor prefixes such as `ai/`, `claude/`, or `codex/` for worker branches.
- Keep work in the branch's own worktree. Do not edit another agent's worktree.
- Do not merge your own PR.
- Do not force-push a branch after review unless the PR explicitly says why.
- PRs must include summary, validation, risks, and follow-ups.
- If Codex or CI finds an issue, fix it with the smallest scoped commit and
  request re-review.

## Review Guidelines

- Findings first, ordered by severity.
- Focus on correctness, regressions, data leakage, reproducibility, performance,
  security, test coverage, and claim/code mismatches.
- Cite exact files and lines.
- Avoid style-only comments unless they affect behavior or maintainability.
- Check docs against scripts and generated artifacts. If a doc claims a measured
  number, verify the number from the artifact when possible.
- Call out tests or checks that were not run.

## Deep Review Guidelines

When a PR comment asks for a "deep review", "long review", or "architecture
review", provide a fuller review than the default GitHub code-review pass:

- Include P1/P2/P3 findings, not only merge-blocking defects.
- Review the PR against the relevant spec/plan/docs, not just local code style.
- Check architecture fit, data/modeling assumptions, reproducibility,
  evaluation methodology, CI/test adequacy, and integration risks.
- Mention "no findings" explicitly if no issues are found.
- After findings, include open questions/assumptions and residual risks.
- Keep summaries brief, but do not omit important non-blocking risks.
- Do not approve or merge the PR.

## Integration Review Guidelines

- Integration review answers: "Do the reviewed PRs work together?"
- Use a disposable `integration/<date-or-topic>` branch rebuilt from latest
  `master` plus selected PR heads.
- Look for conflicts, duplicated abstractions, incompatible assumptions, stale
  docs, test gaps across PR boundaries, and CI failures.
- Prefer commenting back on source PRs over hiding broad fixes in the integration
  branch.
- If an integration fix is genuinely needed, make it a small separate PR.
