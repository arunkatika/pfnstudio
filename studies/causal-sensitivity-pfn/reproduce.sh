#!/usr/bin/env bash
# Reproduce the causal-sensitivity-pfn study end-to-end.
# Expected runtime: ~12-15 min on H100/H200, ~3-4h on CPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "→ Validating prior + model + run spec..."
pfnstudio validate "studies/causal-sensitivity-pfn/priors/mlp_sensitivity_scm/"

echo
echo "→ Training (8000 steps, ~12-15 min on H100/H200)..."
pfnstudio run "studies/causal-sensitivity-pfn/runs/v0_1.yaml"

echo
echo "→ Done. Checkpoint at: $REPO_ROOT/checkpoint/model.pt"
echo "→ Compare against naive baseline + oracle by running:"
echo "    pfnstudio eval studies/causal-sensitivity-pfn/evals/cate_recovery.yaml"
