#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required. Install GitHub CLI and run: gh auth login" >&2
  exit 2
fi

upsert_label() {
  local name="$1"
  local color="$2"
  local description="$3"

  if gh label list --limit 200 --json name --jq '.[].name' | grep -Fxq "$name"; then
    gh label edit "$name" --color "$color" --description "$description"
  else
    gh label create "$name" --color "$color" --description "$description"
  fi
}

upsert_label "agent:claude" "7057ff" "PR authored by a Claude worker agent"
upsert_label "codex-blocked" "d73a4a" "Codex found an issue that should be addressed before merge"
upsert_label "integration-ready" "0e8a16" "PR is ready to include in a disposable integration branch"
upsert_label "integration-conflict" "fbca04" "Cross-PR conflict or incompatible assumption found"
upsert_label "human-review" "1d76db" "Ready for human review and merge decision"
upsert_label "needs-codex-review" "b60205" "Manual Codex review or re-review needed"

# Issue planning and execution. Keep this list aligned with docs/agent-workflow.md.
upsert_label "phase:0-data" "1d76db" "Phase 0 data integrity and measurement harness"
upsert_label "phase:1-baseline" "5319e7" "Phase 1 LightGBM signal-existence gate"
upsert_label "area:coinbase" "0052cc" "Coinbase target-venue data and reconstruction"
upsert_label "area:binance" "f9d0c4" "Binance signal-source data and reconstruction"
upsert_label "area:modeling-data" "0e8a16" "Bars, features, labels, costs, and ModelMatrix"
upsert_label "operations" "c5def5" "Live data acquisition or operational workflow"
upsert_label "priority:high" "d93f0b" "High-priority work on the critical path"
upsert_label "status:ready" "0e8a16" "Unblocked and ready for a worker"
upsert_label "status:in-progress" "fbca04" "Owned by an active worker or operation"
upsert_label "blocked" "b60205" "Blocked by another issue, gate, or external condition"

echo "GitHub labels are configured."
