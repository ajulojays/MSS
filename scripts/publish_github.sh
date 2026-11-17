#!/bin/bash
# Safely publish this repository to GitHub with the gh CLI.
#   scripts/publish_github.sh [owner/repo]     (default ajulojays/MSS)
# Creates a PRIVATE repo if it does not exist. If it exists, pushes only when it is empty
# or when this repo's history already contains the remote's (fast-forward). Never forces.
set -euo pipefail
REPO="${1:-ajulojays/MSS}"
command -v gh >/dev/null || { echo "gh CLI not found" >&2; exit 2; }
gh auth status >/dev/null || { echo "run: gh auth login" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
[[ -z "$(git status --porcelain)" ]] || { echo "commit your changes first" >&2; exit 2; }
if ! gh repo view "$REPO" >/dev/null 2>&1; then
  gh repo create "$REPO" --private --source . --remote origin --push
  exit 0
fi
URL="https://github.com/$REPO.git"
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$URL"
[[ "$(git remote get-url origin)" == "$URL" ]] || { echo "origin points elsewhere: $(git remote get-url origin)" >&2; exit 2; }
if [[ "$(gh repo view "$REPO" --json isEmpty -q .isEmpty)" == "true" ]]; then
  git push -u origin HEAD:main
else
  git fetch origin
  BR="$(gh repo view "$REPO" --json defaultBranchRef -q .defaultBranchRef.name)"
  git merge-base --is-ancestor "origin/$BR" HEAD || { echo "remote has commits not in local history; refusing (no force push)" >&2; exit 3; }
  git push origin "HEAD:$BR"
fi
gh repo view "$REPO" --json url,visibility
