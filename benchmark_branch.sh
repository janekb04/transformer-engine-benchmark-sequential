#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <branch-name> <path-to-git-repo>" >&2
  exit 1
fi

BRANCH="$1"
REPO_PATH="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$REPO_PATH/.git" ]; then
  echo "Error: $REPO_PATH is not a valid Git repository." >&2
  exit 1
fi

cd "$REPO_PATH"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# --- Cleanup function ---
cleanup() {
  echo "" >&2
  echo "Interrupted. Restoring original branch: $CURRENT_BRANCH" >&2
  git checkout --quiet "$CURRENT_BRANCH"
  exit 1
}

# --- Trap SIGINT (Ctrl+C) ---
trap cleanup SIGINT

echo "Fetching latest changes..." >&2
git fetch origin >&2

BASE_COMMIT=$(git merge-base origin/main origin/"$BRANCH")
COMMITS=$(git rev-list --reverse "$BASE_COMMIT"..origin/"$BRANCH")
COMMITS="$BASE_COMMIT"$'\n'"$COMMITS"

# --- Output header (stdout) ---
echo "|Commit|Fused BF16|Fused FP8|Sequential BF16|Sequential FP8|Builtin BF16|Builtin FP8|"
echo "|-|-|-|-|-|-|-|"

for COMMIT in $COMMITS; do
  echo "Checking out commit: $COMMIT" >&2
  git checkout --quiet "$COMMIT"

  FULL_HASH=$(git rev-parse HEAD)
  echo -n "|$FULL_HASH|"

  echo "Running benchmark for commit: $FULL_HASH" >&2
  python3 "$SCRIPT_DIR/benchmark_sequential.py"
done

echo "Restoring original branch: $CURRENT_BRANCH" >&2
git checkout --quiet "$CURRENT_BRANCH"
