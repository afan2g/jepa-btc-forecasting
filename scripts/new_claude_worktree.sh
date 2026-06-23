#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/new_claude_worktree.sh <topic-slug> [base-branch]

Creates a Claude worker branch and git worktree:
  branch: ai/claude/<topic-slug>
  path:   ../jepa-agent-worktrees/<topic-slug>

Environment:
  AGENT_WORKTREE_DIR  Override parent directory for worker worktrees.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

topic="$1"
base="${2:-master}"

if [[ ! "$topic" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "topic-slug must contain only letters, numbers, dot, underscore, or hyphen" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
parent="${AGENT_WORKTREE_DIR:-$(dirname "$repo_root")/jepa-agent-worktrees}"
branch="ai/claude/$topic"
path="$parent/$topic"

if [[ -e "$path" ]]; then
  echo "worktree path already exists: $path" >&2
  exit 2
fi

if git show-ref --verify --quiet "refs/heads/$branch"; then
  echo "branch already exists: $branch" >&2
  exit 2
fi

mkdir -p "$parent"
git fetch origin "$base"
git worktree add -b "$branch" "$path" "origin/$base"

cat <<EOF
Created Claude worker worktree.

Branch: $branch
Path:   $path

Next:
  cd "$path"
  claude
EOF
