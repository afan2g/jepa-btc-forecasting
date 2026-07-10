# CLAUDE.md

Follow `AGENTS.md`. This file adds role-specific instructions for Claude Code
worker and integration-agent sessions.

## Claude Worker Agents

- Every worker session must have one assigned GitHub issue or subissue. Read the
  issue, its acceptance criteria, dependencies, and every linked plan/spec before
  editing. If the issue is blocked or is an umbrella without an implementation
  slice, stop and report that instead of creating a broad branch.
- Work in one git worktree and one branch only.
- Branch naming: Conventional Branch purpose prefixes only, e.g. `feat/<issue-or-short-topic>`, `fix/<issue-or-short-topic>`, or `chore/<issue-or-short-topic>`. Do not use `ai/`, `claude/`, or other agent/vendor prefixes.
- Start from latest `origin/master` unless the user names another base.
- Keep the diff scoped to the assigned task.
- Do not edit unrelated files, rewrite another worker's branch, or merge to
  `master`.
- Commit focused changes with clear messages.
- Run relevant cheap checks before opening a PR. If live vendor checks or bulk
  downloads are needed, ask first and document that they were not run.
- Open a PR with `gh pr create` when ready.
- In the PR body, include:
  - Issue link and whether the PR closes it or is only part of it
  - Summary
  - Validation
  - Risks and assumptions
  - Follow-ups or deferred work
- Use `Closes #N` only when every acceptance criterion is satisfied. Use
  `Part of #N` or `Refs #N` for partial PRs; never close an umbrella issue from
  one implementation slice.
- Comment on the issue with material blockers or live/manual validation results
  that are not durable in the PR. Do not paste secrets or raw vendor data.
- After Codex review, address comments with minimal follow-up commits and request
  re-review.

## Claude Integration Agent

- Only combine PRs that are labeled `integration-ready` or explicitly selected
  by the user.
- Rebuild the integration branch from latest `origin/master`; do not treat it as
  a permanent development branch.
- Branch naming: `integration/<date-or-topic>`.
- Merge or cherry-pick selected PR heads, run the relevant checks, and report
  cross-PR issues.
- Prefer commenting back on the source PRs. Do not silently rewrite worker
  changes in the integration branch.
- If integration-only glue is necessary, open a small integration-fix PR.

## Suggested Local Commands

Create a worker worktree:

```bash
scripts/new_claude_worktree.sh feat/data-calendar
cd ../jepa-agent-worktrees/feat-data-calendar
claude
```

Create a disposable integration branch/worktree from selected branches:

```bash
scripts/new_integration_branch.sh 2026-06-23-data feat/data-calendar feat/baseline
cd ../jepa-integration-worktrees/2026-06-23-data
claude
```

## Claude PR Lifecycle

```text
Claude worker
  -> branch/worktree
  -> PR
  -> smoke CI
  -> Codex review
  -> Claude fixes comments
  -> CI
  -> Codex re-review if material changes
  -> human review
  -> merge
```

## Integration Lifecycle

```text
Claude worker PRs
  -> Codex review
  -> label integration-ready
  -> disposable integration branch
  -> Claude integration review
  -> source PR fixes or small integration-fix PR
  -> full CI
  -> human merge decision
```
