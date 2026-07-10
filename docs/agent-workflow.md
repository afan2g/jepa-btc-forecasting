# Agent Workflow

This repo is set up for multiple local Claude Code workers and Codex PR review.

GitHub issues are the durable backlog and status record. Branches and worktrees
are temporary execution environments created only for ready implementation work.
The linked specs/plans remain authoritative for technical behavior.

## Roles

- Claude worker: implements one task on one branch in one worktree.
- Codex reviewer: reviews each PR for serious correctness, testing, security,
  reproducibility, and data/modeling issues.
- Claude integration agent: checks whether multiple reviewed PRs work together.
- Human reviewer: owns final judgment and merge.

## Branches

- Worker branches: Conventional Branch purpose-prefixed names, e.g. `feat/<topic>`, `fix/<topic>`, `chore/<topic>`; do not use agent/vendor prefixes such as `ai/` or `claude/`.
- Integration branches: `integration/<date-or-topic>`
- Base branch: `master`

Never share a worker branch between worktrees. Git branches are mutable refs and
should have one active owner at a time.

## Issues And Milestones

Use an issue to record the objective, acceptance criteria, dependencies, linked
plans/specs, validation expectations, and live/vendor constraints. Assign it to
the relevant phase milestone and area label.

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

## Issue Lifecycle

1. Create or refine the issue from the task issue form.
2. Link the relevant plans/specs and make dependencies explicit.
3. Assign phase/area/priority labels and a milestone.
4. Mark it `blocked` or `status:ready`.
5. When a worker starts, replace `status:ready` with `status:in-progress` and
   create one branch/worktree.
6. Link the PR with `Closes`, `Part of`, or `Refs` as appropriate.
7. Record material blockers and live/manual results on the issue.
8. After merge, remove the worktree/branch and close the issue only if all
   acceptance criteria are complete.

## Worker Flow

1. Create a worktree.

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
   Read CLAUDE.md and AGENTS.md first.
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
- Delete the merged worker worktree and branch.
- Close an issue only when its acceptance criteria are complete; partial PRs
  update but do not close their parent issue.

## Local Resource Scheduling

- Prefer at most two implementation agents on this workstation. Run one live
  data job alongside them only while their active checks are lightweight.
- Do not run multiple full test suites or other CPU/RAM-heavy checks
  concurrently.
- Pause broad vendor downloads while agents run intensive integration or native
  reconstruction tests.
- Give paid/live operations priority over speculative background checks so
  failures and spend remain attributable.
