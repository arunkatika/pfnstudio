#!/usr/bin/env bash
# Reproduce the do-pfn study end-to-end.
# Expected runtime: ~25-35 min on H100/H200, ~6-8h on CPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "→ Validating prior + model + run spec..."
pfnstudio validate "studies/do-pfn/priors/do_pfn_scm/"

echo
echo "→ Training (12000 steps, ~25-35 min on H100/H200)..."
pfnstudio run "studies/do-pfn/runs/v0_1.yaml"

echo
echo "→ Done. Checkpoint at: $REPO_ROOT/checkpoint/model.pt"
echo "→ Evaluate CID + CATE recovery against the oracle by running:"
echo "    pfnstudio eval studies/do-pfn/evals/cid_recovery.yaml"
