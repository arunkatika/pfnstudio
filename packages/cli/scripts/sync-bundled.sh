#!/usr/bin/env bash
# Pre-build sync — copies the repo-root schemas/ and templates/ into
# packages/cli/priorstudio/_bundled/ so they're inside the package
# tree at `python -m build` time. The CLI's `validate` and `init`
# commands read from this _bundled path at runtime.
#
# Run automatically by `python -m build` is not yet supported; for now,
# call this script manually before each build:
#
#   packages/cli/scripts/sync-bundled.sh && cd packages/cli && python -m build
#
# Idempotent. Safe to re-run; rsync handles "newer source overwrites".

set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
BUNDLED="$PKG_DIR/priorstudio/_bundled"

mkdir -p "$BUNDLED"

for src in schemas templates; do
  if [[ ! -d "$REPO_ROOT/$src" ]]; then
    echo "ERROR: $REPO_ROOT/$src not found" >&2
    exit 1
  fi
  rm -rf "$BUNDLED/$src"
  cp -R "$REPO_ROOT/$src" "$BUNDLED/$src"
  echo "✓ Synced $src/ → priorstudio/_bundled/$src/"
done
