#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/new_integration_branch.sh <topic-slug> <branch-or-ref>...

Creates a disposable integration branch/worktree from latest origin/master and
merges the selected worker branches.

  branch: integration/<topic-slug>
  path:   ../jepa-integration-worktrees/<topic-slug>

Environment:
  INTEGRATION_WORKTREE_DIR  Override parent directory for integration worktrees.
  BASE_BRANCH               Override base branch, default: master.

If the integration branch/worktree already exists, remove it manually first.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
  usage
  exit 0
fi

topic="$1"
shift
base="${BASE_BRANCH:-master}"

if [[ ! "$topic" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "topic-slug must contain only letters, numbers, dot, underscore, or hyphen" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
parent="${INTEGRATION_WORKTREE_DIR:-$(dirname "$repo_root")/jepa-integration-worktrees}"
branch="integration/$topic"
path="$parent/$topic"

if [[ -e "$path" ]]; then
  echo "integration worktree path already exists: $path" >&2
  exit 2
fi

if git show-ref --verify --quiet "refs/heads/$branch"; then
  echo "integration branch already exists: $branch" >&2
  echo "Delete it explicitly before rebuilding: git branch -D $branch" >&2
  exit 2
fi

mkdir -p "$parent"
git fetch origin "$base"
git worktree add -b "$branch" "$path" "origin/$base"

(
  cd "$path"
  for ref in "$@"; do
    merge_ref="$ref"
    if ! git rev-parse --verify --quiet "$merge_ref^{commit}" >/dev/null; then
      if git ls-remote --exit-code --heads origin "$ref" >/dev/null; then
        git fetch origin "$ref:refs/remotes/origin/$ref"
        merge_ref="origin/$ref"
      else
        echo "cannot resolve branch/ref: $ref" >&2
        exit 2
      fi
    fi
    git merge --no-edit --no-ff "$merge_ref"
  done
)

cat <<EOF
Created integration worktree.

Branch: $branch
Path:   $path
Merged: $*

Next:
  cd "$path"
  claude
EOF
