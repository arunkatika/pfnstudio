#!/usr/bin/env bash
# Reproduce the custom-block-demo study: define a custom architecture block
# as project code and train + score a model that uses it. ~10s on CPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "→ Validating..."
pfnstudio validate "studies/custom-block-demo/priors/bayesian_linear/"
echo "→ check-model (assembles the custom gated_residual block)..."
pfnstudio check-model "studies/custom-block-demo" demo_model
echo "→ Training the sanity run..."
pfnstudio run "studies/custom-block-demo/runs/sanity_run.yaml"
