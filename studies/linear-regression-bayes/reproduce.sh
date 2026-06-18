#!/usr/bin/env bash
# Reproduce the linear-regression-bayes study end-to-end.
# Expected runtime: ~50 seconds on a modern CPU.

set -euo pipefail

# Resolve repo root regardless of where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "→ Validating prior + model + run spec..."
pfnstudio validate "studies/linear-regression-bayes/priors/bayesian_linear/"

echo
echo "→ Training (2000 steps, ~47s on CPU)..."
pfnstudio run "studies/linear-regression-bayes/runs/v0_1.yaml"

echo
echo "→ Done. Checkpoint at: $REPO_ROOT/checkpoint/model.pt"
echo "→ Compare against the OLS baseline by running:"
echo "    python examples/01_linear_regression.py"
