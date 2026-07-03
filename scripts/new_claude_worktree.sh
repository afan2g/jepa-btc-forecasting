#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/new_claude_worktree.sh <type>/<description> [base-branch]

Creates a Claude worker branch and git worktree:
  branch: <type>/<description>
  path:   ../jepa-agent-worktrees/<type>-<description>

Branch names follow Conventional Branch purpose prefixes only:
  feat/, feature/, fix/, bugfix/, hotfix/, release/, chore/

Environment:
  AGENT_WORKTREE_DIR  Override parent directory for worker worktrees.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

branch="$1"
base="${2:-master}"

if [[ ! "$branch" =~ ^(feat|feature|fix|bugfix|hotfix|release|chore)/[a-z0-9][a-z0-9.-]*[a-z0-9]$ && ! "$branch" =~ ^(feat|feature|fix|bugfix|hotfix|release|chore)/[a-z0-9]$ ]]; then
  echo "branch must follow Conventional Branch purpose form: <type>/<description>" >&2
  echo "allowed types: feat, feature, fix, bugfix, hotfix, release, chore" >&2
  echo "description: lowercase letters, numbers, hyphens, and dots only" >&2
  exit 2
fi

desc="${branch#*/}"
if [[ "$desc" == *"--"* || "$desc" == *".."* || "$desc" == *".-"* || "$desc" == *"-."* \
      || "$desc" == .* || "$desc" == *. || "$desc" == -* || "$desc" == *- ]]; then
  echo "branch description must not contain consecutive separators or leading/trailing separators" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
parent="${AGENT_WORKTREE_DIR:-$(dirname "$repo_root")/jepa-agent-worktrees}"
path="$parent/${branch//\//-}"

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
