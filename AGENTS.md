# AGENTS.md

This repository uses agent-authored pull requests and Codex review. Treat this
file as the durable reviewer and repository-contract guidance.

## Project Architecture

- The first project-defining gate forecasts Binance BTC-USDT perpetual mid
  moves from that instrument's own L2 book and trades. Coinbase transfer,
  cross-exchange features, Binance spot/derivatives state, multi-asset inputs,
  and JEPA are conditional increments after the single-venue `G0-BN` gate.
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

- Plans and specs remain the technical source of truth. GitHub issues are
  authoritative for scope, dependencies, acceptance criteria, and execution
  state.
- The [JEPA BTC Forecasting Roadmap](https://github.com/users/afan2g/projects/2)
  is an operational visualization, not an authority override. Its planning
  dates are estimates, never technical commitments or approval for vendor/live
  work. Follow the canonical [agent workflow](docs/agent-workflow.md#authority-and-roadmap)
  for Project fields, views, automation, and lifecycle transitions.
- Start implementation only from an unblocked issue whose scope and acceptance
  criteria are clear. A new top-level issue must first be in the Project with
  Roadmap Stage, Planning Start, Planning Target, and Status assigned. Split
  umbrella issues into reviewable subissues before assigning workers.
- Keep the active issue label and Project Status aligned: `status:ready` with
  Ready, `status:in-progress` with In Progress, and `blocked` with Blocked. Use
  Done only for a completed issue.
- One agent, one issue, one branch/worktree.
- Use Conventional Branch purpose-prefixed names like `feat/<topic>`, `fix/<topic>`, or `chore/<topic>`; do not use agent/vendor prefixes such as `ai/`, `claude/`, or `codex/` for worker branches.
- Keep work in the branch's own worktree. Do not edit another agent's worktree.
- Read the assigned issue and every linked plan/spec before editing. Keep the PR
  within that issue's acceptance criteria and call out any necessary scope
  change before implementing it.
- Link every worker PR to its issue. Use `Closes #N` only when the PR satisfies
  the whole issue; use `Part of #N` or `Refs #N` for partial work or umbrella
  issues. A partial PR must not close the issue or move it to Done.
- Operational work such as approved vendor downloads or report generation does
  not need a branch/worktree unless it also changes tracked code or docs. Record
  commands, results, costs, and blockers on the issue.
- Do not merge your own PR.
- Do not force-push a branch after review unless the PR explicitly says why.
- PRs must include summary, validation, risks, and follow-ups.
- If Codex or CI finds an issue, fix it with the smallest scoped commit and
  request re-review.
- After human merge, remove the completed PR's worktree and branch. Close the
  issue and use Project Status Done only when all acceptance criteria are met;
  otherwise keep the issue open and set its state from the remaining work.

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
