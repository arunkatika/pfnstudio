#!/usr/bin/env bash
# Reproduce the tabpfn-style-classification study end-to-end.
# Expected runtime: ~15 minutes on a modern laptop CPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "→ Validating prior + model + run + eval specs..."
pfnstudio validate "studies/tabpfn-style-classification/priors/random_binary_classification/"

echo
echo "→ Training (10000 steps, ~14 min on CPU) + scoring against breast-cancer..."
pfnstudio run "studies/tabpfn-style-classification/runs/v0_1.yaml"

echo
echo "→ Done. Checkpoint at: $REPO_ROOT/checkpoint/model.pt"
echo "→ Headline metric (pfn_accuracy, pfn_auc) is in the run output's evals.breast_cancer_vs_logreg.metrics block."
