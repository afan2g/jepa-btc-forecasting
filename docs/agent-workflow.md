# Agent Workflow

This repo is set up for multiple local Claude Code workers and Codex PR review.

Branches and worktrees are temporary execution environments created only for
ready implementation work. This document is the canonical operational workflow
for agents and maintainers.

## Authority And Roadmap

The repository uses three layers with different responsibilities:

1. Plans and specs remain the technical source of truth. They define product,
   data, modeling, timing, leakage, evaluation, and implementation contracts.
2. GitHub issues remain authoritative for work scope, dependencies, acceptance
   criteria, and execution state. An issue may link to a plan or spec but cannot
   silently override it.
3. The [JEPA BTC Forecasting Roadmap](https://github.com/users/afan2g/projects/2)
   is an operational visualization of workflow state, roadmap stage, planning
   dates, linked PRs, and dependency sequencing. It does not override an issue,
   plan, or spec.

If these layers disagree, stop and reconcile the issue or Project entry with the
applicable plan/spec before implementation. Do not interpret a Project value as
permission to expand scope or bypass an acceptance criterion.

### Project Fields

Every implementation item uses these Project fields:

| Field | Values or purpose |
| --- | --- |
| Status | `Ready`, `In Progress`, `Blocked`, or `Done` |
| Roadmap Stage | The item's current roadmap planning stage |
| Planning Start | Estimated start date for roadmap coordination |
| Planning Target | Estimated target date for roadmap coordination |

`Planning Start` and `Planning Target` are estimates. They are never technical
commitments, delivery guarantees, vendor approval, spend approval, or permission
to run live, paid, bulk, or compute-heavy work. The issue and repository safety
rules still govern execution.

### Project Views

- **Work Register** is the inventory view. Use it to confirm that an issue is in
  the Project and that its stage, dates, Status, and other planning metadata are
  populated.
- **Execution Board** is the workflow-state view. Use it to see Ready, In
  Progress, Blocked, and Done work and to verify state transitions.
- **Critical Path** is the roadmap sequencing view. Use it to review stages,
  estimated dates, and dependencies; it does not create a technical dependency
  or a delivery commitment.

### Inclusion And Automation Boundary

Before implementation, every new top-level issue must be added to the Project
and assigned Roadmap Stage, Planning Start, Planning Target, and Status.
Associating the repository with the Project does not automatically add every
top-level issue, so the issue creator or maintainer must add and triage it.

The currently enabled Project workflows cover automatic subissue addition,
item-added and item-closed events, linked and merged PR events, and issue
closure. This automation reduces bookkeeping but is not the authority for
completion. Workers must verify the resulting issue state, issue labels, linked
PRs, and Project fields after every transition. In particular:

- Confirm that an auto-added subissue received the intended stage, dates, and
  Status before implementation.
- Correct missing or stale Project values rather than assuming an event workflow
  ran successfully.
- Do not let a linked or merged partial PR close its issue or set Status to Done.

## Roles

- Claude worker: implements one assigned issue on one branch in one worktree.
- Codex reviewer: reviews each PR for serious correctness, testing, security,
  reproducibility, and data/modeling issues.
- Claude integration agent: checks whether multiple reviewed PRs work together.
- Human reviewer: owns final judgment and merge.

## Branches

- Worker branches: Conventional Branch purpose-prefixed names, e.g. `feat/<topic>`, `fix/<topic>`, `chore/<topic>`; do not use agent/vendor prefixes such as `ai/` or `claude/`.
- Integration branches: `integration/<date-or-topic>`
- Base branch: `master`

One agent owns one issue on one worker branch in one worktree. Never share a
worker branch between worktrees. Git branches are mutable refs and should have
one active owner at a time.

## Issues And Milestones

Use an issue to record the objective, acceptance criteria, dependencies, linked
plans/specs, validation expectations, and live/vendor constraints. Assign it to
the relevant phase milestone and area label. For a new top-level issue, add it
to the Project and populate all four Project fields before implementation.

An issue is ready for a worker only when it is unblocked and small enough for one
reviewable PR. Split broad umbrella issues into subissues; keep the umbrella open
until all acceptance criteria are complete. A worktree is not a placeholder for
backlog work.

Use these linkage forms in PR bodies:

- `Closes #N` when the PR completes every acceptance criterion.
- `Part of #N` or `Refs #N` for a partial PR or umbrella issue.

Operational work such as approved downloads, quality-map runs, and report
generation normally runs from the clean main checkout and does not need a branch.
Record the command, artifact location, measured cost/usage, outcome, and blockers
on the issue. Create a worker branch only when tracked code or documentation must
change.

## Labels

Use these labels in GitHub:

Issue planning:

- `phase:0-data`
- `phase:1-baseline`
- `area:coinbase`
- `area:binance`
- `area:modeling-data`
- `operations`
- `priority:high`
- `status:ready`
- `status:in-progress`
- `blocked`

PR/review routing:

- `agent:claude`
- `codex-blocked`
- `integration-ready`
- `integration-conflict`
- `human-review`
- `needs-codex-review`

You enabled native Codex automatic review on PR open. That means labels are for
state, filtering, and human/agent routing; they are not the normal Codex review
trigger. Keep `needs-codex-review` only as an exception label for PRs where the
automatic review did not run or where you want to request a manual `@codex
review`.

Use exactly one of `status:ready`, `status:in-progress`, or `blocked` on active
implementation issues. Remove status labels when an issue closes. Milestones
represent phase ownership, not execution status.

Keep issue labels and Project Status aligned:

| Issue state or label | Project Status |
| --- | --- |
| `status:ready` | `Ready` |
| `status:in-progress` | `In Progress` |
| `blocked` | `Blocked` |
| Completed issue | `Done` |

An issue is completed only when all acceptance criteria are satisfied. A merged
PR, closed PR, elapsed planning date, or completed subtask is not sufficient on
its own.

## Issue Lifecycle

Every transition follows the issue's acceptance criteria and actual execution
state, never a planning date or automation event alone.

1. **Create and triage.** Create or refine the issue from the task issue form.
   Link the relevant plans/specs, make dependencies explicit, and define
   acceptance criteria and validation. Assign phase/area/priority labels and a
   milestone. For a top-level issue, add it to the Project and assign Roadmap
   Stage, Planning Start, Planning Target, and Status.
2. **Mark ready or blocked.** Use `status:ready` plus Project Status Ready only
   when the issue is unblocked, clear, and small enough for one reviewable
   worker. Otherwise use `blocked` plus Blocked, record the concrete blocker and
   unblock condition, and do not begin implementation.
3. **Start work.** Assign one agent to the issue and create one Conventional
   Branch branch/worktree. Replace `status:ready` with `status:in-progress` and
   change Project Status from Ready to In Progress. Verify both updates before
   relying on the Execution Board.
4. **Block active work.** Stop at the safety or dependency boundary, record what
   remains and what will unblock it, replace `status:in-progress` with `blocked`,
   and set Project Status to Blocked. Do not use roadmap dates as authorization
   to continue through a vendor, spend, data, or compute blocker.
5. **Link the PR.** Use `Closes #N` only when merge will satisfy every acceptance
   criterion. Use `Part of #N` or `Refs #N` for a partial PR or umbrella issue.
   Verify that the PR appears on the Project item.
6. **Handle partial completion.** A partial PR must leave the issue open and must
   not set Project Status to Done. After merge, choose the state from the actual
   remaining work: In Progress if the same owned work continues, Ready if a
   clear unblocked slice awaits a worker, or Blocked if a dependency prevents
   progress. Keep the corresponding issue label aligned.
7. **Merge.** Only a human merges. A merge event may trigger Project automation,
   but merge alone does not prove issue completion. Recheck every acceptance
   criterion and correct the issue label, issue state, and Project Status after
   automation runs. A PR closed without merge does not complete the issue.
8. **Close and complete.** Close the issue and set Project Status to Done only
   after all acceptance criteria and required validation are complete. Remove
   active status labels. If a linked PR or issue-closure workflow produces Done
   prematurely, reopen or correct the item and restore the state required by the
   remaining work.
9. **Clean up.** After the human merge (or other final disposition) of the PR,
   remove its worker worktree and branch. Cleanup is not evidence that the issue
   is complete: a partial PR's parent issue remains open and non-Done until its
   remaining acceptance criteria are satisfied.

## Worker Flow

1. Read the issue and perform the Project preflight in Work Register. Verify its
   authority links, dependencies, acceptance criteria, Roadmap Stage, planning
   dates, and Status. Confirm the status label matches. Then create a worktree.

   ```bash
   gh issue view <number>
   scripts/new_claude_worktree.sh feat/<topic>
   cd ../jepa-agent-worktrees/feat-<topic>
   claude
   ```

2. Give Claude a task brief.

   ```text
   You are working on branch feat/<topic>.
   Your assigned issue is #<number>. Read it and every linked plan/spec first.
   Read CLAUDE.md, AGENTS.md, and docs/agent-workflow.md first.
   Verify the issue's Project fields and synchronized status before editing.
   Keep the change scoped to the issue's acceptance criteria.
   Run relevant checks.
   Commit, push, and open a PR.
   Do not merge.
   ```

3. Open a PR and add labels:

   ```bash
   gh pr create --fill --label agent:claude
   ```

4. Wait for Codex auto-review and CI.

5. Ask the same Claude worker to fix comments in the same worktree.

   ```text
   Address the Codex review comments with minimal changes.
   Preserve the PR scope.
   Run the relevant checks and push follow-up commits.
   ```

## Integration Flow

1. Select source PRs that are individually reviewed and labeled
   `integration-ready`.

2. Create a disposable integration worktree.

   ```bash
   scripts/new_integration_branch.sh <date-topic> feat/one feat/two
   cd ../jepa-integration-worktrees/<date-topic>
   claude
   ```

3. Ask the integration agent to check cross-PR behavior.

   ```text
   You are the integration reviewer.
   Read CLAUDE.md and AGENTS.md.
   Review the combined branch for conflicts, incompatible assumptions, duplicated
   abstractions, stale docs, and missing cross-PR tests.
   Prefer comments back on source PRs over broad integration changes.
   ```

4. Run full checks if implementation code exists.

5. Either send fixes back to source PRs or open a small integration-fix PR.

## Codex Review Options

Current path:

1. Enable Codex code review for the repository in Codex settings.
2. Keep automatic review enabled for PR open.
3. Keep exhaustive review disabled for normal PRs so comments stay focused.
4. Add `AGENTS.md` review guidance.
5. Use `@codex review` only when you need a manual re-review or a focused pass.

For a longer review, use the explicit deep-review path:

```bash
scripts/request_codex_deep_review.sh <PR_NUMBER>
```

This posts a detailed `@codex` PR comment asking for a broader architecture,
testing, reproducibility, and integration-risk review. Use it on larger PRs or
PRs implementing a plan/spec. Keep the automatic PR-open review enabled as the
default fast pass.

## Merge Policy

- Claude agents do not merge.
- Integration branches are disposable and rebuildable.
- Human review is required after Codex review and CI.
- Squash or rebase merge is fine, but preserve PR discussion and validation in
  the final merge record.
- Delete the merged worker worktree and branch, then verify the issue and Project
  state.
- Close an issue and use Status Done only when its acceptance criteria are
  complete. Partial PRs update but do not close their parent issue or mark it
  Done.

## Local Resource Scheduling

- Prefer at most two implementation agents on this workstation. Run one live
  data job alongside them only while their active checks are lightweight.
- Do not run multiple full test suites or other CPU/RAM-heavy checks
  concurrently.
- Pause broad vendor downloads while agents run intensive integration or native
  reconstruction tests.
- Give paid/live operations priority over speculative background checks so
  failures and spend remain attributable.
