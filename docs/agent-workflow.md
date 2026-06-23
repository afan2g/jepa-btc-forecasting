# Agent Workflow

This repo is set up for multiple local Claude Code workers and Codex PR review.

## Roles

- Claude worker: implements one task on one branch in one worktree.
- Codex reviewer: reviews each PR for serious correctness, testing, security,
  reproducibility, and data/modeling issues.
- Claude integration agent: checks whether multiple reviewed PRs work together.
- Human reviewer: owns final judgment and merge.

## Branches

- Worker branches: `ai/claude/<issue-or-topic>`
- Integration branches: `integration/<date-or-topic>`
- Base branch: `master`

Never share a worker branch between worktrees. Git branches are mutable refs and
should have one active owner at a time.

## Labels

Use these labels in GitHub:

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

## Worker Flow

1. Create a worktree.

   ```bash
   scripts/new_claude_worktree.sh <topic>
   cd ../jepa-agent-worktrees/<topic>
   claude
   ```

2. Give Claude a task brief.

   ```text
   You are working on branch ai/claude/<topic>.
   Read CLAUDE.md and AGENTS.md first.
   Keep the change scoped to <task>.
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
   scripts/new_integration_branch.sh <date-topic> ai/claude/one ai/claude/two
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

## Merge Policy

- Claude agents do not merge.
- Integration branches are disposable and rebuildable.
- Human review is required after Codex review and CI.
- Squash or rebase merge is fine, but preserve PR discussion and validation in
  the final merge record.
