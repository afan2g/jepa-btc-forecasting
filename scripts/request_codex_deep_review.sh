#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/request_codex_deep_review.sh <pr-number-or-url> [focus]

Posts a detailed @codex review request for a longer PR review.

Examples:
  scripts/request_codex_deep_review.sh 2
  scripts/request_codex_deep_review.sh 2 "focus on leakage and PBO/G1 gate math"
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required. Install GitHub CLI and run: gh auth login" >&2
  exit 2
fi

pr="$1"
focus="${2:-}"

body=$(cat <<'EOF'
@codex review

Please do a deep review of this PR, in the style of a senior engineering design/code review.

Use AGENTS.md, especially the Review Guidelines and Deep Review Guidelines.

Please go beyond the default high-priority-only pass:
- include P1/P2/P3 findings where they matter;
- compare the implementation against the relevant spec, plan, and docs;
- check correctness, data leakage, reproducibility, evaluation methodology, tests, performance, security, and integration risk;
- cite exact files and lines;
- put findings first, ordered by severity;
- then include open questions/assumptions and residual risks;
- say clearly if there are no findings.

Do not approve, merge, or make changes.
EOF
)

if [[ -n "$focus" ]]; then
  body="$body

Additional focus: $focus"
fi

gh pr comment "$pr" --body "$body"
echo "Requested deep Codex review for PR $pr."
